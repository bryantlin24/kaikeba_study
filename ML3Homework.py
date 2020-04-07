import pandas as pd
import  matplotlib.pyplot as plt
import  numpy as np
from sklearn.model_selection import  train_test_split
from  sklearn.linear_model import LinearRegression
from sklearn import metrics

def getTrainSetAndTestSet(DataPath):
    data = pd.read_csv(DataPath)
    X = data[['AT','V','AP','RH']]
    X = np.array(X)
    Y =data[['PE']]
    Y = np.array(Y)
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state=1)
    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)
    return  X_train,X_test,Y_train,Y_test


def TrainLinearRegression(X_train,Y_train):
    model = LinearRegression()
    model.fit(X_train,Y_train)
    print("w:"+str(model.coef_))
    print("b"+str(model.intercept_))
    return  model

def EvaluationLinearRegression(model,X_test,Y_test):
    Y_pred = model.predict(X_test)
    print("均方误差MSE:"+str(metrics.mean_squared_error(Y_test,Y_pred)))
    print("均方根误差RMSE:"+str(np.sqrt(metrics.mean_squared_error(Y_test,Y_pred))))
    return Y_pred


def Visualization(Y_test,Y_pred):
    plt.scatter(Y_test,Y_pred)
    plt.plot([Y_test.min(),Y_test.max()],[Y_test.min(),Y_test.max()],'k--',lw=6)
    plt.ylabel("Predicted")
    plt.show()


if __name__ == "__main__":
    DataPath = 'data(1).csv'
    X_train,X_test,Y_train,Y_test = getTrainSetAndTestSet(DataPath)
    linreg = TrainLinearRegression(X_train,Y_train)
    Y_pred = EvaluationLinearRegression(linreg,X_test,Y_test)
    Visualization(Y_test,Y_pred)
