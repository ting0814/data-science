# Kernel SVM

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
t_start = time.process_time()

print('loading data ...')
dataset = pd.read_excel('timewindow+ outsample.xlsx', sheet_name='80-15')
X_train=dataset.iloc[:, 2:9].values
y_train= dataset.iloc[:,9].values
dataset_test = pd.read_excel('timewindow+ outsample.xlsx', sheet_name='16')
X_test= dataset_test.iloc[:, 2:9].values
y_test= dataset_test.iloc[:, 9].values


# Splitting the dataset into the Training set and Test set
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =0)
# Splitting the dataset into the Training set and Test set
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
print('standardizing data ...')
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print('fitting model')
# Fitting classifier to the Training set
from sklearn.svm import SVC
classifier = SVC( kernel = 'rbf', degree= 3, random_state = 0)
classifier.fit(X_train, y_train)
print('predicting model')
# Predicting the Test set resultsx
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix( y_pred, y_test)

print("重要性：",importances)
from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred)
t_stop = time.process_time()
print('takes',t_stop-t_start,"sec")