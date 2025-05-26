import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

seed = 128
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# 1. 读取数据
data = pd.read_csv(r"C:\Users\Lenovo\WPSDrive\1628475303\WPS企业云盘\同济大学\我的企业文档\我的企业文档\大二下\数学建模\作业3\q4_所有用户在线特征_天.csv")
# 转换时间戳为datetime类型
data['时间 (Time)'] = pd.to_datetime(data['时间 (Time)'], errors='coerce')

# 3. 选择用户ID、时间和互动次数等相关列
data['date'] = data['时间 (Time)'].dt.date  # 提取日期（不包括时间）
data['day'] = data['时间 (Time)'].dt.dayofyear  # 提取一年中的天数


# 3. 构建训练数据：使用过去12天的数据作为特征，2024/7/23的在线情况作为标签
def create_sequence(data, window_size, target_user_ids, exclude_users):
    X, y = [], []
    filtered_data = data[~data['用户ID (User ID)'].isin(exclude_users)]
    for user_id in filtered_data['用户ID (User ID)'].unique():
        user_data = filtered_data[filtered_data['用户ID (User ID)'] == user_id]

        # 生成时间窗特征
        features = []
        features.append(user_data.iloc[0:window_size][
                                ['在线频率', '观看次数', '点赞次数', '评论次数', '关注次数',
                                 'day']].values)
        # 标签是2024/7/23的在线情况
        label = user_data.iloc[-1]['在线情况']  # 使用2024/7/20的在线情况作为标签
        X.append(features)
        y.append(label)

    return np.array(X), np.array(y)


# 选择目标用户（'U9', 'U22405', 'U16', 'U48420'）
exclude_user_ids = ['U10', 'U1951', 'U1833', 'U26447']
window_size = 12  # 使用过去12天的数据作为时间窗
X, y = create_sequence(data, window_size, target_user_ids=None, exclude_users=exclude_user_ids)

# 4. 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

# 5. 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# 转换为 PyTorch 张量
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # 适应二分类标签
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
X_train = X_train.squeeze(1)  # 去掉第二个维度，变成 (batch_size, sequence_length, input_size)

print(X_train.shape)
# 6. 构建LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # 输出一个值，表示是否在线（互动次数）

    def forward(self, x):
        # batch_size = x.size(0)
        h0 = torch.zeros(2, x.size(0), hidden_size).to(x.device)  # 修改为 num_layers = 2
        c0 = torch.zeros(2, x.size(0), hidden_size).to(x.device)  # 修改为 num_layers = 2
        out, _ = self.lstm(x, (h0, c0))  # LSTM 层输出
        out = self.fc(out[:, -1, :])  # 获取最后时刻的输出
        return torch.sigmoid(out)  # Sigmoid 激活函数用于二分类


# 初始化LSTM模型
input_size = X_train.shape[2]  # 每个时间步的特征数量
hidden_size = 64  # LSTM隐藏层大小
num_layers = 2  # LSTM层数
model = LSTMModel(input_size, hidden_size, num_layers)

# 7. 训练模型
# 损失函数和优化器
criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 将训练数据和标签打包成一个数据集
train_dataset = TensorDataset(X_train, y_train)

# 创建 DataLoader 用于批量加载数据
batch_size = 32  # 你可以调整批大小
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
# 训练循环
epochs = 200
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

# 8. 使用训练好的模型来预测目标用户在2024/7/23的在线情况

model.eval()
with torch.no_grad():
    # 为目标用户构建预测数据
    target_features = []
    for user_id in exclude_user_ids:
        user_data = data[data['用户ID (User ID)'] == user_id]


        features = []
        features.append(user_data.iloc[1:window_size][
                            ['在线频率', '观看次数', '点赞次数', '评论次数', '关注次数',
                             'day']].values)
        target_features.append(features)

    target_features = np.array(target_features)
    target_features_scaled = scaler.transform(target_features.reshape(-1, target_features.shape[-1])).reshape(
        target_features.shape)
    target_features_tensor = torch.tensor(target_features_scaled, dtype=torch.float32)
    target_features_tensor = target_features_tensor.squeeze(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    target_features_tensor = target_features_tensor.to(device)
    # 预测
    predictions = model(target_features_tensor)
    predictions_label = (predictions > 0.5).float()  # 转换为二分类标签：在线为1，离线为0

# 输出预测结果
for i, user_id in enumerate(exclude_user_ids):
    print(f"Predicted online status for {user_id} on 2024/7/23: {predictions_label[i].item()}")

