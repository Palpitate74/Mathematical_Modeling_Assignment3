import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.gaussian_process import GaussianProcessRegressor
import xgboost as xgb
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# 1. 读取数据
df = pd.read_csv(r"C:\Users\Lenovo\WPSDrive\1628475303\WPS企业云盘\同济大学\我的企业文档\我的企业文档\大二下\数学建模\作业3\pivot_df.csv")

# 2. 处理时间数据，确保日期格式正确，并提取日期部分
df['时间 (Time)'] = pd.to_datetime(df['时间 (Time)']).dt.date

def create_lag_features(df, lag=7):
    # 为每个博主创建过去7天的数据特征
    for i in range(1, lag + 1):
        df[f'1_lag_{i}'] = df.groupby('博主ID (Blogger ID)')['1'].shift(i)
        df[f'2_lag_{i}'] = df.groupby('博主ID (Blogger ID)')['2'].shift(i)
        df[f'3_lag_{i}'] = df.groupby('博主ID (Blogger ID)')['3'].shift(i)
    return df


# 创建过去7天的特征
df = create_lag_features(df, lag=7)
# 4. 删除缺失数据（由于创建了滞后特征，前7天的数据会有缺失）
df = df.dropna()

# 5. 特征和标签
X = df[['1_lag_1', '1_lag_2', '1_lag_3', '1_lag_4', '1_lag_5', '1_lag_6', '1_lag_7',
        '2_lag_1', '2_lag_2', '2_lag_3', '2_lag_4', '2_lag_5', '2_lag_6', '2_lag_7',
        '3_lag_1', '3_lag_2', '3_lag_3', '3_lag_4', '3_lag_5', '3_lag_6', '3_lag_7']]  # 滞后特征
y = df['4']  # 标签数据列（你要预测的 '4'）

# 6. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. 定义模型字典
models = {
    'GaussianProcessRegressor': GaussianProcessRegressor(),  # 添加 GaussianProcessRegressor 模型
}

# 8. 针对每个博主的预测
results = []

for model_name, model in models.items():
    # 训练模型
    model.fit(X_train, y_train)

    # 对每个博主的所有数据进行预测
    for blogger_id in df['博主ID (Blogger ID)'].unique():
        X_blogger = df[df['博主ID (Blogger ID)'] == blogger_id]

        target_date = pd.to_datetime('2024-07-20').date()  # 将目标日期转换为 datetime 类型
        target_date_datetime = pd.to_datetime(target_date)
        # 为了预测，使用2024-7-20的数据作为输入特征（即2024-7-20的滞后特征）
        X_blogger = X_blogger[X_blogger['时间 (Time)'] == target_date_datetime]  # 使用2024-7-20的特征数据

        X_blogger_features = X_blogger.dropna()[['1_lag_1', '1_lag_2', '1_lag_3', '1_lag_4', '1_lag_5', '1_lag_6', '1_lag_7',
                                        '2_lag_1', '2_lag_2', '2_lag_3', '2_lag_4', '2_lag_5', '2_lag_6', '2_lag_7',
                                        '3_lag_1', '3_lag_2', '3_lag_3', '3_lag_4', '3_lag_5', '3_lag_6', '3_lag_7']]

        # 获取该博主的预测值
        predictions = model.predict(X_blogger_features)

        # 将预测结果保存到列表
        for i, pred in enumerate(predictions):
            results.append({
                '博主ID (Blogger ID)': blogger_id,
                '模型': model_name,
                '预测值': pred
            })

# 9. 保存结果到CSV文件
results_df = pd.DataFrame(results)
results_df.to_csv('博主预测结果.csv', index=False)

print("预测结果已保存到 '博主预测结果.csv' 文件中。")