import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

# Importing the dataset
housing = pd.read_csv('Housing.csv')

#Data cleaning
fig, axs = plt.subplots(2,3, figsize = (10,5))
plt1 = sns.boxplot(housing['price'], ax = axs[0,0])
plt2 = sns.boxplot(housing['area'], ax = axs[0,1])
plt3 = sns.boxplot(housing['bedrooms'], ax = axs[0,2])
plt1 = sns.boxplot(housing['bathrooms'], ax = axs[1,0])
plt2 = sns.boxplot(housing['stories'], ax = axs[1,1])
plt3 = sns.boxplot(housing['parking'], ax = axs[1,2])

plt.tight_layout()
plt.show()

# outlier treatment for price
plt.boxplot(housing.price)
Q1 = housing.price.quantile(0.25)
Q3 = housing.price.quantile(0.75)
IQR = Q3 - Q1
housing = housing[(housing.price >= Q1 - 1.5*IQR) & (housing.price <= Q3 + 1.5*IQR)]
plt.show()

# outlier treatment for area
plt.boxplot(housing.area)
Q1 = housing.area.quantile(0.25)
Q3 = housing.area.quantile(0.75)
IQR = Q3 - Q1
housing = housing[(housing.area >= Q1 - 1.5*IQR) & (housing.area <= Q3 + 1.5*IQR)]
plt.show()

# Outlier Analysis
fig, axs = plt.subplots(2,3, figsize = (10,5))
plt1 = sns.boxplot(housing['price'], ax = axs[0,0])
plt2 = sns.boxplot(housing['area'], ax = axs[0,1])
plt3 = sns.boxplot(housing['bedrooms'], ax = axs[0,2])
plt1 = sns.boxplot(housing['bathrooms'], ax = axs[1,0])
plt2 = sns.boxplot(housing['stories'], ax = axs[1,1])
plt3 = sns.boxplot(housing['parking'], ax = axs[1,2])

plt.tight_layout()
plt.show()

#Drop unused data
housing = housing.drop(housing.columns[[3,5,6,7,8,9,10,11,12]],axis=1)
price = housing.iloc[:, 0].values
area = housing.iloc[:, 1].values
bedrooms = housing.iloc[:, 2].values
stories = housing.iloc[:, 3].values

#Check out data
print(housing.head(),"\n------------------------------------")
print(housing.describe(),"\n------------------------------------")
print(housing.columns,"\n------------------------------------")

sns.pairplot(housing)
plt.show()

sns.heatmap(housing.corr(), annot=True)
plt.title('House Attributes Correlation')
plt.show()

#Training linear regression model
#X and y arrays
X = housing[['area', 'bedrooms', 'stories']]
y = housing[['price']]

#Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn import metrics
from sklearn.model_selection import cross_val_score

def cross_val(model):
    pred = cross_val_score(model, X, y, cv=10)
    return pred.mean()


def print_evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)
    print('__________________________________')


def evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    return mae, mse, rmse, r2_square

#Calculate time at start time
start_time = time.time()

#Linear regression
print("Linear Regression\n-------------------------------------")
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)

#Calculate time at and time
end_time = time.time()
elapsed_time = end_time - start_time
print ("Run time: {0}".format(elapsed_time) + "[sec]")

#Model evaluation
print("Linear regression intercept:", lin_reg.intercept_)
print("Linear regression coefficient:")
print("Area:", lin_reg.coef_.ravel()[0])
print("Bedrooms:", lin_reg.coef_.ravel()[1])
print("Story:", lin_reg.coef_.ravel()[2])

#Prediction from model
pred = lin_reg.predict(X_test)
plt.scatter(y_test, pred, color='red')
plt.axis('scaled')
plt.title('Predicted value VS True value (Linear Regression)')
plt.xlabel('True Value')
plt.ylabel('Predicted value')
plt.show()

plt.hist(y_test-pred)
plt.title("Residual Histogram (Linear Regression)")
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.show()

test_pred = lin_reg.predict(X_test)
train_pred = lin_reg.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)
print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)


#-----------------------------------------------------------------------------

#Calculate time at start time
start_time2 = time.time()

#Polynomial regression
print("Polynomial Regression\n--------------------------------------")
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=2)

X_train_2_d = poly_reg.fit_transform(X_train)
X_test_2_d = poly_reg.transform(X_test)

lin_reg2 = LinearRegression()
lin_reg2.fit(X_train_2_d,y_train)

#Model evaluation
print("Polynomial regression intercept:", lin_reg2.intercept_)
print("Polynomial regression coefficient:", lin_reg2.coef_)

#Calculate time at and time
end_time2 = time.time()
elapsed_time2 = end_time2 - start_time2
print ("Run time: {0}".format(elapsed_time2) + "[sec]")

#Prediction from model
pred2 = lin_reg2.predict(X_test_2_d)
plt.scatter(y_test, pred2, color='red')
plt.title('Predicted value VS True value (Polynomial Regression)')
plt.xlabel('True Value')
plt.ylabel('Predicted value')
plt.show()

plt.hist(y_test-pred2)
plt.title("Residual Histogram (Polynomial Regression)")
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.show()

test_pred2 = lin_reg2.predict(X_test_2_d)
train_pred2 = lin_reg2.predict(X_train_2_d)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred2)
print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred2)