import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score


# 加载 California 房价数据集
housing = fetch_california_housing(as_frame=True)
data = housing.frame
# 打印数据基本信息
print("数据集维度：", data.shape)
print("数据集特征：\n", data.columns)

# 选择特征与目标变量
X = data.drop("MedHouseVal", axis=1)  # 房屋中位数作为目标变量
y = data["MedHouseVal"]

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化 KNN 回归器
knn = KNeighborsRegressor()

# 设置超参数搜索空间：邻居数从 1 到 30
param_grid = {'n_neighbors': np.arange(1, 31)}

# 使用 GridSearchCV 进行超参数优化，采用 5 折交叉验证
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# 输出最优参数与对应的得分
best_k = grid_search.best_params_['n_neighbors']
best_score = -grid_search.best_score_
print("最优邻居数 k =", best_k)
print("最佳均方误差 (MSE) =", best_score)

# 使用最优参数训练模型
best_knn = grid_search.best_estimator_
best_knn.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = best_knn.predict(X_test)

# 计算测试集上的 MSE 和 R2 分数
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("测试集均方误差 (MSE) =", mse)
print("测试集 R2 score =", r2)

# 可视化：真实值 vs 预测值
plt.figure(figsize=(10, 6))
# 绘制散点图，可视化预测值和真实值
plt.scatter(y_test, y_pred, alpha=0.6, color='mediumspringgreen', edgecolor='k')
# 绘制理想情况下的对角线，预测值等于真实值
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("True Values", fontsize=14)
plt.ylabel("Predicted Values", fontsize=14)
plt.title("True vs Predicted Values", fontsize=16)
plt.tight_layout()
plt.show()

# 可视化：残差分布
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=50, kde=True, color='hotpink')
plt.xlabel("Residuals", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.title("Residual Distribution", fontsize=16)
plt.tight_layout()
plt.show()

# 可视化：GridSearchCV 中不同 k 值的均方误差热力图
# 提取所有搜索结果
results = pd.DataFrame(grid_search.cv_results_)
results['mean_test_mse'] = -results['mean_test_score']

plt.figure(figsize=(12, 6))
plt.plot(results['param_n_neighbors'], results['mean_test_mse'], marker='o', linestyle='-', color='dodgerblue')
plt.xlabel("Number of Neighbors (k)", fontsize=14)
plt.ylabel("Mean Squared Error (MSE)", fontsize=14)
plt.title("Grid Search MSE vs k", fontsize=16)
plt.xticks(results['param_n_neighbors'])
plt.grid(True)
plt.tight_layout()
plt.show()