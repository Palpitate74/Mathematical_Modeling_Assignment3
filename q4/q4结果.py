import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


seed = 128
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# 1. 读取数据
data = pd.read_csv(r"C:\Users\Lenovo\WPSDrive\1628475303\WPS企业云盘\同济大学\我的企业文档\我的企业文档\大二下\数学建模\作业3\q4_11-22数据.csv")
# 转换时间戳为datetime类型
data['日期'] = pd.to_datetime(data['日期'], errors='coerce')

# 提取日期和天数
data['date'] = data['日期'].dt.date  # 提取日期（不包括时间）
data['day'] = data['日期'].dt.dayofyear  # 提取一年中的天数

# 2. 构建训练数据
def create_sequence1(data, window_size, target_user_ids, exclude_users):
    X, y = [], []
    filtered_data = data[~data['用户ID (User ID)'].isin(exclude_users)]
    for user_id in filtered_data['用户ID (User ID)'].unique():  # 对剩余的用户进行遍历
        print(user_id)
        user_data = filtered_data[filtered_data['用户ID (User ID)'] == user_id]
        # 确保目标日期的数据存在
        start_date = pd.to_datetime('2024-07-11').date()
        end_date = pd.to_datetime('2024-07-14').date()
        label_date = pd.to_datetime('2024-07-15').date()
        # 获取期间的特征（第2列到最后一列）
        features = user_data[(user_data['date'] >= start_date) & (user_data['date'] <= end_date)].iloc[:, 3:171].values
        X.append(features)

        # 获取 2024/7/15 号的数据作为标签
        label = user_data[user_data['date'] == label_date].iloc[:, 3:171].values
        y.append(label)

    return np.array(X), np.array(y)

def create_sequence2(data, window_size, target_user_ids, exclude_users):
    X, y = [], []
    filtered_data = data[~data['用户ID (User ID)'].isin(exclude_users)]
    for user_id in filtered_data['用户ID (User ID)'].unique():  # 对剩余的用户进行遍历
        print(user_id)
        user_data = filtered_data[filtered_data['用户ID (User ID)'] == user_id]
        # 确保目标日期的数据存在
        start_date = pd.to_datetime('2024-07-16').date()
        end_date = pd.to_datetime('2024-07-19').date()
        label_date = pd.to_datetime('2024-07-20').date()
        # 获取期间的特征（第2列到最后一列）
        features = user_data[(user_data['date'] >= start_date) & (user_data['date'] <= end_date)].iloc[:, 3:171].values
        X.append(features)

        # 获取 2024/7/20 号的数据作为标签
        label = user_data[user_data['date'] == label_date].iloc[:, 3:171].values
        y.append(label)

    return np.array(X), np.array(y)


# 选择目标用户（'U9', 'U22405', 'U16', 'U48420'）
exclude_user_ids = ['U9', 'U22405', 'U16', 'U48420']
window_size = 10  # 使用过去10天的数据作为时间窗
X1, y1 = create_sequence1(data, window_size, target_user_ids=None, exclude_users=exclude_user_ids)
X2, y2 = create_sequence2(data, window_size, target_user_ids=None, exclude_users=exclude_user_ids)
X = np.vstack((X1, X2))
y = np.concatenate((y1, y2))

# 3. 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

# 4. 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# 转换为 PyTorch 张量
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)  # 标签直接是一个向量
y_test = torch.tensor(y_test, dtype=torch.float32)


# 5. 构建LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # 输出标签的形状与特征数相同

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), hidden_size).to(x.device)  # 修改为 num_layers = 2
        c0 = torch.zeros(2, x.size(0), hidden_size).to(x.device)  # 修改为 num_layers = 2
        out, _ = self.lstm(x, (h0, c0))  # LSTM 层输出
        out = self.fc(out[:, -1, :])  # 获取最后时刻的输出
        return out  # 输出为连续的标签值


# 初始化LSTM模型
input_size = X_train.shape[2]  # 每个时间步的特征数量
hidden_size = 64  # LSTM隐藏层大小
num_layers = 2  # LSTM层数
output_size = y_train.shape[2]  # 标签的维度
model = LSTMModel(input_size, hidden_size, num_layers, output_size)
torch.backends.mkldnn.enabled = False
# 6. 训练模型
criterion = nn.MSELoss()  # 使用均方误差损失函数，因为标签是连续值
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 将训练数据和标签打包成一个数据集
train_dataset = TensorDataset(X_train, y_train)

# 创建 DataLoader 用于批量加载数据
batch_size = 32  # 你可以调整批大小
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


epochs = 20
for epoch in range(epochs):
    model.train()
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# 7. 使用训练好的模型来预测目标用户在2024/7/21的在线情况
model.eval()
with torch.no_grad():
    # 为目标用户构建预测数据（2024/7/12到2024/7/18,2024/7/13到2024/7/19,2024/7/14到2024/7/20）
    target_features = []
    start_date = pd.to_datetime('2024-07-11').date()
    end_date = pd.to_datetime('2024-07-22').date()
    for user_id in exclude_user_ids:
        user_data = data[data['用户ID (User ID)'] == user_id]
        features = user_data[(user_data['date'] >= start_date) & (user_data['date'] <= end_date)].iloc[:,
                   3:171].values
        target_features.append(features)

    target_features = np.array(target_features)
    target_features_scaled = scaler.transform(target_features.reshape(-1, target_features.shape[-1])).reshape(
        target_features.shape)
    target_features_tensor = torch.tensor(target_features_scaled, dtype=torch.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    target_features_tensor = target_features_tensor.to(device)
    # 预测
    predictions = model(target_features_tensor)
    categories = torch.floor(predictions * 5).long()  # 将连续的预测值映射为 0, 1, 2, 3, 4
    # 将输出限制在 0-4 范围内
    categories = torch.clamp(categories, min=0, max=4)
    # 输出预测结果
    for i, user_id in enumerate(exclude_user_ids):
        print(f"Predicted online status for {user_id} on 2024/7/21: {categories[i].detach().cpu().numpy()}")
    categories = categories.cpu().numpy()
    df = pd.DataFrame(categories)
    df.to_csv('q4互动结果.csv', index=False)