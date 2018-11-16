## Data and Visual Analytics - Homework 4
## Georgia Institute of Technology
## Applying ML algorithms to detect seizure

import numpy as np
import pandas as pd
import time

from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, normalize

######################################### Reading and Splitting the Data ###############################################
# XXX
# TODO: Read in all the data. Replace the 'xxx' with the path to the data set.
# XXX
data = pd.read_csv('seizure_dataset.csv')

# Separate out the x_data and y_data.
x_data = data.loc[:, data.columns != "y"]
y_data = data.loc[:, "y"]

# The random state to use while splitting the data.
random_state = 100

# XXX
# TODO: Split 70% of the data into training and 30% into test sets. Call them x_train, x_test, y_train and y_test.
# Use the train_test_split method in sklearn with the paramater 'shuffle' set to true and the 'random_state' set to 100.
# XXX

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=random_state)

# ############################################### Linear Regression ###################################################
# XXX
# TODO: Create a LinearRegression classifier and train it.
# XXX

reg = LinearRegression().fit(X_train, y_train)

# XXX
# TODO: Test its accuracy (on the training set) using the accuracy_score method.
y_predict=reg.predict(X_train)
print('Test its accuracy of LinearRegression(train): ', str(round(accuracy_score(y_train, y_predict.round()),2)*100)+'%')
# TODO: Test its accuracy (on the testing set) using the accuracy_score method.
y_predict=reg.predict(X_test)
print('Test its accuracy of LinearRegression(test): ', str(round(accuracy_score(y_test, y_predict.round()),2)*100)+'%')
# Note: Use y_predict.round() to get 1 or 0 as the output.
# XXX


# ############################################### Multi Layer Perceptron #################################################
# XXX
# TODO: Create an MLPClassifier and train it.
clf = MLPClassifier(random_state=random_state).fit(X_train, y_train)
# XXX


# XXX
# TODO: Test its accuracy on the training set using the accuracy_score method.
y_predict=clf.predict(X_train)
print('Test its accuracy of MLPClassifier(train): ', str(round(accuracy_score(y_train, y_predict.round()),2)*100)+'%')
# TODO: Test its accuracy on the test set using the accuracy_score method.
y_predict=clf.predict(X_test)
print('Test its accuracy of MLPClassifier(test): ', str(round(accuracy_score(y_test, y_predict.round()),2)*100)+'%')
# XXX





# ############################################### Random Forest Classifier ##############################################
# XXX
# TODO: Create a RandomForestClassifier and train it.
clf = RandomForestClassifier(random_state=random_state).fit(X_train,y_train)
# XXX

# XXX
# TODO: Test its accuracy on the training set using the accuracy_score method.
y_predict=clf.predict(X_train)
print('Test its accuracy of RandomForestClassifier(train): ', str(round(accuracy_score(y_train, y_predict.round()),2)*100)+'%')
# TODO: Test its accuracy on the test set using the accuracy_score method.
y_predict=clf.predict(X_test)
print('Test its accuracy of RandomForestClassifier(test): ',str(round(accuracy_score(y_test, y_predict.round()),2)*100)+'%')
# XXX

# XXX
# TODO: Tune the hyper-parameters 'n_estimators' and 'max_depth'.
tuned_parameters = {
    'n_estimators': [100, 200, 500],
    'max_depth': [5,10,50,100]
}
clf = GridSearchCV(estimator = clf, param_grid=tuned_parameters, cv=10, scoring='accuracy')
clf.fit(X_test, y_test)
#       Print the best params, using .best_params_, and print the best score, using .best_score_.
print('Best Params of RandomForestClassifier: ', clf.best_params_)
print('Best Score of RandomForestClassifier after tunning: ', str(round(clf.best_score_,2)*100)+'%')
# XXX


# ############################################ Support Vector Machine ###################################################
# XXX
# TODO: Pre-process the data to standardize or normalize it, otherwise the grid search will take much longer
scaler=StandardScaler()
X_train = scaler.fit(X_train).transform(X_train)
X_test = scaler.transform(X_test)
# TODO: Create a SVC classifier and train it.
clf = SVC().fit(X_train, y_train)
# XXX

# XXX
# TODO: Test its accuracy on the training set using the accuracy_score method.
y_predict=clf.predict(X_train)
print('Test its accuracy of SVC(train): ', str(round(accuracy_score(y_train, y_predict.round()),2)*100)+'%')
# TODO: Test its accuracy on the test set using the accuracy_score method.
y_predict=clf.predict(X_test)
print('Test its accuracy of SVC(test): ', str(round(accuracy_score(y_test, y_predict.round()),2)*100)+'%')
# XXX

# XXX
# TODO: Tune the hyper-parameters 'C' and 'kernel' (use rbf and linear).
tuned_parameters = {
    'kernel': ['rbf', 'linear'],
    'C': [0.0001, 0.1, 100]
}
clf = GridSearchCV(estimator = clf, param_grid=tuned_parameters, cv=10, scoring='accuracy')
clf.fit(X_test, y_test)
#       Print the best params, using .best_params_, and print the best score, using .best_score_.
print('Best Params of SVM: ', clf.best_params_)
print('Best Score of SVM after tunning: ', str(round(clf.best_score_,2)*100)+'%')
print('Mean training score, Mean testing score and Mean fit time for the best combination of hyperparameter values: ',clf.cv_results_)
# XXX