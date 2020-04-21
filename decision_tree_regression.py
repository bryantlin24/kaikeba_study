from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

data = pd.read_csv("data_reg.csv")
x = data[["AT", "V", "AP", "RH"]]
y = data[["PE"]]

# 预处理
sc = StandardScaler()
x_pre = sc.fit_transform(x)

# 数据分割
x_train, x_test, y_train, y_test = train_test_split(x_pre, y, test_size=0.3, random_state=123)

# SVR
svr = SVR()
svr.fit(x_train, y_train)
score = svr.score(x_test, y_test)
print("SVM:", score)

# 决策树
decision_tree_regressor = DecisionTreeRegressor(criterion="mse", max_depth=50)
decision_tree_regressor.fit(x_train, y_train)
score = decision_tree_regressor.score(x_test, y_test)
print("决策树：", score)


