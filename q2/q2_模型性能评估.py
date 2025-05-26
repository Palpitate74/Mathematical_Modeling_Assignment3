import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# 读取数据
df = pd.read_csv(r"C:\Users\Lenovo\WPSDrive\1628475303\WPS企业云盘\同济大学\我的企业文档\我的企业文档\大二下\数学建模\作业3\C1.csv")

# 确保 '时间 (Time)' 列为 datetime 类型
df['时间 (Time)'] = pd.to_datetime(df['时间 (Time)'])

# 提取时间部分，按小时分段
df['时间段 (Time Period)'] = df['时间 (Time)'].dt.hour
df = df.drop(columns=['时间 (Time)'])

# 使用 get_dummies 转换类别变量为独热编码
df = pd.get_dummies(df, columns=['博主ID (Blogger ID)', '时间段 (Time Period)', '用户行为 (User behaviour)'],
                    prefix=['博主', '时间段', '行为'], sparse=True)

# # 进行随机下采样
# df = df.sample(frac=1)
# fraud_df = df.loc[df['行为_4'] == 1]   # 欺诈样本
# non_fraud_df = df.loc[df['行为_4'] == 0][:len(fraud_df)]   # 非欺诈样本
# normal_distributed_df = pd.concat([fraud_df, non_fraud_df])    # 将27875个正样本和27875个负样本拼在一起
# df = normal_distributed_df.sample(frac=1, random_state=128)   # 再把这个新的27875个样本打散

# 将特征列和标签列分开
X = df.drop(columns=['行为_4', '用户ID (User ID)'])  # 假设行为_4是标签
y = df['行为_4']  # 标签列

# 数据分割：先分为训练集和临时集（80%和20%），然后再分临时集为验证集和测试集（各占10%）
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=128)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=128)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.fit_transform(X_val)
X_test_scaled = scaler.fit_transform(X_test)
# X_val_scaled = scaler.transform(X_val)
# X_test_scaled = scaler.transform(X_test)
# 转换为 PyTorch Tensor
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# 读取新的 CSV 文件（需要进行预测的数据集）
new_data = pd.read_csv(r"C:\Users\Lenovo\WPSDrive\1628475303\WPS企业云盘\同济大学\我的企业文档\我的企业文档\大二下\数学建模\作业3\q2\q2_22feature.csv")  # 更改为你的新数据集路径

# 对新数据进行相同的预处理（包括标准化）
new_data_scaled = scaler.transform(new_data.drop(columns=['Unnamed: 0', '用户ID (User ID)']))  # 使用训练集的标准化参数

# 检查是否有 GPU 可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 模型字典
model = {
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=128),
    'CatBoost': CatBoostClassifier(iterations=1000, depth=10, learning_rate=0.1, random_state=128, verbose=0),
    'Neural Network': nn.Sequential(
        nn.Linear(X_train_scaled.shape[1], 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),  # 输出层
        nn.Sigmoid()  # 二分类问题使用 Sigmoid 激活函数
    )
}

# 将神经网络模型移动到 GPU（如果可用）
model['Neural Network'] = model['Neural Network'].to(device)

# 训练和评估每个模型
# 创建一个 Excel writer 对象
with pd.ExcelWriter('model_predictions.xlsx', engine='openpyxl') as writer:
    for name, model_instance in model.items():
        print(f"Training {name}...")

        # 训练神经网络
        if name == 'Neural Network':
            # 设置训练的超参数
            criterion = nn.BCELoss()  # 二分类交叉熵损失
            optimizer = optim.Adam(model_instance.parameters(), lr=0.001)

            # 训练神经网络
            num_epochs = 10
            for epoch in range(num_epochs):
                model_instance.train()
                optimizer.zero_grad()

                # 将数据移动到 GPU（如果可用）
                inputs, targets = X_train_tensor.to(device), y_train_tensor.to(device)

                output = model_instance(inputs)
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()

            # 测试神经网络
            model_instance.eval()
            with torch.no_grad():
                # 对新数据进行预测
                inputs = torch.tensor(new_data_scaled, dtype=torch.float32).to(device)
                output = model_instance(inputs)
                predicted = (output > 0.5).float()  # 二分类阈值设为 0.5

                # 保存预测结果到 DataFrame
                predictions = pd.DataFrame(predicted.cpu().numpy(), columns=['Prediction'])
                predictions.to_excel(writer, sheet_name=f'{name}_predictions', index=False)

        else:
            # 其他模型使用scikit-learn进行训练
            model_instance.fit(X_train_scaled, y_train)

            # 预测并保存结果
            y_pred = model_instance.predict(new_data_scaled)
            predictions = pd.DataFrame(y_pred, columns=['Prediction'])

            # 保存预测结果到 DataFrame
            predictions.to_excel(writer, sheet_name=f'{name}_predictions', index=False)

            print(f"Model {name} predictions saved.")