import pandas as pd
import numpy as np
import seaborn as sn
from sklearn.datasets import make_blobs
from sklearn import decomposition, tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier

# grid search solver for lda
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from numpy import arange
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import plot_confusion_matrix

import sys

data = pd.read_csv('TrainOnMe.csv')
data_evaluate = pd.read_csv('EvaluateOnMe.csv')

#replace weird stuffs
data = data.replace(to_replace ="Bayesian Interference",
                 value ="Bayesian Inference")
data = data.replace(to_replace ="Flase",
                 value ="FALSE")
data = data.replace(to_replace ="False",
                 value ="FALSE")
data = data.replace(to_replace ="True",
                 value ="TRUE")

#drop the ones where no value in x6
#data = data.drop([446, 614])

#make categorical variables into dummy
data = pd.get_dummies(data, columns=['x6'])
data = pd.get_dummies(data, columns=['x12'])

#drop the ones with huge x1
data = data.drop([830,742])

S = data[data.y == "Shoogee"]
A = data[data.y == "Atsuto"]
J = data[data.y == "Jorg"]
B = data[data.y == "Bob"]

data = S.append(A.append(J.append(B)))
classes = np.unique(data.y)
corrMatrix = pd.DataFrame(data, columns= ['x1','x2','x3','x4','x5','x7','x8','x9','x10','x11']).corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()

#variables are selected based on no multicollinearity and model improvements
X = pd.DataFrame(data, columns= ['x1','x7','x8','x9','x10','x11'])
Y = pd.DataFrame(data, columns= ['y'])
Y = Y.replace(to_replace ="Shoogee",
                 value =4)
Y = Y.replace(to_replace ="Atsuto",
                 value =1)
Y = Y.replace(to_replace ="Jorg",
                 value =3)
Y = Y.replace(to_replace ="Bob",
                 value =2)

Y = Y.y.values

#train test split
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2, random_state = 34)

st_x = preprocessing.StandardScaler() #for SVM, gaussianNB
x_train = st_x.fit_transform(x_train)
x_test = st_x.transform(x_test)


#final solution:
#test accuracy: 0.72
'''
m = 0
s = 0

ETrf = ExtraTreesClassifier(n_estimators = 100, random_state = 5)
svm = SVC(random_state = 1, kernel='rbf', probability=True)
nb = GaussianNB()

for i in range(100):
    cv = KFold(n_splits=5, random_state=i, shuffle=True)
    scores = cross_val_score(ETrf, X, Y, scoring='accuracy', cv=cv, n_jobs=-1)
    print("Accuracy of algo cv: ",scores)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    m = m + scores.mean()
    s = s + scores.std()
    
print(m/100)
print(s/100)
'''
X_evaluate = pd.DataFrame(data_evaluate, columns= ['x1','x7','x8','x9','x10','x11'])
ETrf = ExtraTreesClassifier(n_estimators = 100, random_state = 5)
ETrf.fit(X, Y)
Y_evaluate = ETrf.predict(X_evaluate)
print(Y_evaluate)


original_stdout = sys.stdout # Save a reference to the original standard output

with open('labels.txt', 'w') as f:
    sys.stdout = f # Change the standard output to the file we created.

    for i in range(len(Y_evaluate)):
        if Y_evaluate[i] == 1:
            print('Atsuto')
        elif Y_evaluate[i] == 2:
            print('Bob')
        elif Y_evaluate[i] == 3:
            print('Jorg')
        elif Y_evaluate[i] == 4:
            print('Shoogee')
        else:
            print('error') #shouldn't happen

sys.stdout = original_stdout # Reset the standard output to its original value
