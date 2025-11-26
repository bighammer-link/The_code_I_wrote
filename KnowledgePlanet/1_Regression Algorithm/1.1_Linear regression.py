import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

# 加载加州房价数据集
california = fetch_california_housing()
data = pd.DataFrame(california.data, columns=california.feature_names)
data['MedHouseVal'] = california.target


# 分离特征变量和目标变量
X = data.drop('MedHouseVal', axis=1)
y = data['MedHouseVal']
#
# # 拆分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # 构建线性回归模型
# model = LinearRegression()
# model.fit(X_train, y_train) # 进行拟合，也就是训练的过程
# # 预测
# y_train_pred = model.predict(X_train)
# y_test_pred = model.predict(X_test)
# # 真实值 vs 预测值
# plt.figure(figsize=(10, 5))# 创建画布
# plt.scatter(y_train, y_train_pred, color='blue', label='Train data') # 绘制散点图，标签为Train data
# plt.scatter(y_test, y_test_pred, color='red', label='Test data')
# plt.xlabel('True Values') # 设置轴的标签
# plt.ylabel('Predictions')
# plt.legend()# 添加图例
# plt.title('True Values vs Predictions')
# plt.show()
# # 绘制流程 先创建画布，然后绘制散点图，然后设置轴的标签，然后添加图例和标题，最后show
# # 残差分布
# plt.figure(figsize=(10, 5))# 创建画布
# sns.histplot((y_train - y_train_pred), bins=50, kde=True, label='Train data', color='blue')
# sns.histplot((y_test - y_test_pred), bins=50, kde=True, label='Test data', color='red')
# plt.legend()
# plt.title('Residuals Distribution')
# plt.show()
#
#
# # 评估训练集性能
# mse_train = mean_squared_error(y_train, y_train_pred)
# r2_train = r2_score(y_train, y_train_pred)
#
# # 评估测试集性能
# mse_test = mean_squared_error(y_test, y_test_pred)
# r2_test = r2_score(y_test, y_test_pred)
#
# print(f'Train MSE: {mse_train}, Train R2: {r2_train}')
# print(f'Test MSE: {mse_test}, Test R2: {r2_test}')



#通过特征选择和标准化来优化模型。
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

# 特征选择
selector = SelectKBest(f_regression, k=8)
X_new = selector.fit_transform(X, y)
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_new)
X_test_scaled = scaler.transform(X_test_new)

# 构建新模型
model_new = LinearRegression()
model_new.fit(X_train_scaled, y_train_new)

# 预测
y_train_pred_new = model_new.predict(X_train_scaled)
y_test_pred_new = model_new.predict(X_test_scaled)

# 评估新模型性能
mse_train_new = mean_squared_error(y_train_new, y_train_pred_new)
r2_train_new = r2_score(y_train_new, y_train_pred_new)
mse_test_new = mean_squared_error(y_test_new, y_test_pred_new)
r2_test_new = r2_score(y_test_new, y_test_pred_new)

print(f'Train MSE (new): {mse_train_new}, Train R2 (new): {r2_train_new}')
print(f'Test MSE (new): {mse_test_new}, Test R2 (new): {r2_test_new}')