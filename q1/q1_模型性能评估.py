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
# 删除缺失数据（由于创建了滞后特征，前7天的数据会有缺失）
df = df.dropna()

# 5. 特征和标签
X = df[['1_lag_1', '1_lag_2', '1_lag_3', '1_lag_4', '1_lag_5', '1_lag_6', '1_lag_7',
        '2_lag_1', '2_lag_2', '2_lag_3', '2_lag_4', '2_lag_5', '2_lag_6', '2_lag_7',
        '3_lag_1', '3_lag_2', '3_lag_3', '3_lag_4', '3_lag_5', '3_lag_6', '3_lag_7']]  # 滞后特征
y = df['4']  # 标签数据列（你要预测的 '4'）

# 6. 数据分割 - 训练集：2024-07-11 到 2024-07-20
train_data = df[(df['时间 (Time)'] >= pd.to_datetime('2024-07-11').date()) &
                (df['时间 (Time)'] <= pd.to_datetime('2024-07-20').date())]

X_train = train_data[['1_lag_1', '1_lag_2', '1_lag_3', '1_lag_4', '1_lag_5', '1_lag_6', '1_lag_7',
                      '2_lag_1', '2_lag_2', '2_lag_3', '2_lag_4', '2_lag_5', '2_lag_6', '2_lag_7',
                      '3_lag_1', '3_lag_2', '3_lag_3', '3_lag_4', '3_lag_5', '3_lag_6', '3_lag_7']]
y_train = train_data['4']

# 7. 定义模型字典
models = {
    'RandomForest': RandomForestRegressor(),
    'NeuralNetwork': MLPRegressor(),
    'SVR': SVR(),
    'LinearRegression': LinearRegression(),
    'Lasso': Lasso(),
    'GaussianProcessRegressor': GaussianProcessRegressor(),
    'XGBoost': xgb.XGBRegressor()
}

# 8. 对每个模型进行训练并计算误差
results = []

for model_name, model in models.items():
    # 训练模型
    model.fit(X_train, y_train)

    # 使用训练集预测
    y_pred = model.predict(X_train)

    # 计算误差
    mse = mean_squared_error(y_train, y_pred)
    mae = mean_absolute_error(y_train, y_pred)
    rmse = np.sqrt(mse)

    # 保存每个模型的误差结果
    results.append({
        '模型': model_name,
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse
    })

# 9. 将误差结果保存到 CSV 文件
results_df = pd.DataFrame(results)
results_df.to_csv('模型误差结果.csv', index=False)

print("每个模型的误差结果已保存到 '模型误差结果.csv' 文件中。")
