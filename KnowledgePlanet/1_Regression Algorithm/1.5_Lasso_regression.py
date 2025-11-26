import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# 加载数据
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['MedHouseVal'] = data.target

# 初步查看数据
print(df.head())
print(df.describe())
# 检查缺失值
print(df.isnull().sum())

# 分离特征和目标变量
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']
# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 特征缩放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# 特征相关性
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.show()
# 建立 Lasso 模型
lasso = Lasso()
# 定义超参数网格
param_grid = {'alpha': np.logspace(-4, 4, 50)}
# 网格搜索
grid_search = GridSearchCV(lasso, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)
# 最佳超参数
best_alpha = grid_search.best_params_['alpha']
print(f'Best alpha: {best_alpha}')
# 训练最终模型
lasso_opt = Lasso(alpha=best_alpha)
lasso_opt.fit(X_train_scaled, y_train)
# 预测
y_pred_train = lasso_opt.predict(X_train_scaled)
y_pred_test = lasso_opt.predict(X_test_scaled)
# 评估
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
print(f'MSE (Train): {mse_train}')
print(f'MSE (Test): {mse_test}')
print(f'R^2 (Train): {r2_train}')
print(f'R^2 (Test): {r2_test}')
# 可视化
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, alpha=0.6, color='b')
plt.plot([0, 5], [0, 5], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted House Prices')
plt.show()