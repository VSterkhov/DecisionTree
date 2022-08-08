from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.model_selection import GridSearchCV

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.datasets as ds

# Проверяем наличие выбросов целевой переменной
def show_boxplot(df, columns=[]):
    df = df.loc[:, columns]
    sns.boxplot(x="variable", y="value", data=pd.melt(df))    
    plt.figure(figsize=(16,9))
    plt.show()

california_bunch = ds.fetch_california_housing()

data = np.c_[california_bunch.data, california_bunch.target]
columns = np.append(california_bunch.feature_names, california_bunch.target_names)
calif_df = pd.DataFrame(data, columns=columns)

corr = calif_df['Population'].corr(calif_df['MedHouseVal'])
print('Correlation \'Population\' to Target: ',corr)
show_boxplot(calif_df, ['Population'])
calif_df['Population'] = calif_df['Population'].median()



X = calif_df.loc[:, calif_df.columns != 'MedHouseVal']
Y = calif_df['MedHouseVal']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

linearRegression = LinearRegression()
linearRegression.fit(X_train, Y_train)
print('Score linear regression: ', linearRegression.score(X_test, Y_test))

clf = DecisionTreeRegressor(max_depth=3)
clf.fit(X_train, Y_train)
print('Score decision tree. Depth=3: ', clf.score(X_test, Y_test))


show_boxplot(calif_df, columns=['AveOccup','AveRooms','AveBedrms'])


# Удаляем выбросы по среднему кол-ву комнат больше 15

calif_df = calif_df[calif_df['AveOccup']<=200]
show_boxplot(calif_df, columns=['AveOccup'])

X = calif_df.loc[:, calif_df.columns != 'MedHouseVal']
Y = calif_df['MedHouseVal']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf, 
                   feature_names=california_bunch.feature_names,  
                   class_names=california_bunch.target_names,
                   filled=True)

parametrs = { 'max_depth': range (1,13, 2),
              'min_samples_leaf': range (1,8),
              'min_samples_split': range (2,10,2) }

clf = GridSearchCV(DecisionTreeRegressor(), parametrs, cv=5)
clf.fit(X_train, Y_train)
best_params = clf.best_params_
print('Best params for decision tree: ', best_params)
print('Score best params decision tree: ', clf.score(X_test, Y_test))
