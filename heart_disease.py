import pandas as pd #for reading dataset
import numpy as np # array handling functions
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, r2_score


def graph_comparision():
   X, y = make_classification(n_samples=1000, n_features=10, random_state=42)              
   # Split the dataset into training and testing sets
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   # Train the three models
   knn = KNeighborsClassifier()
   knn.fit(X_train, y_train)
   rf = RandomForestClassifier()
   rf.fit(X_train, y_train)

   # Predict the probabilities of the positive class for the testing data

   knn_probs = knn.predict_proba(X_test)[:, 1]
   rf_probs = rf.predict_proba(X_test)[:, 1]

   # Calculate the false positive rate (FPR), true positive rate (TPR), and threshold values for each model


   knn_fpr, knn_tpr, knn_thresholds = roc_curve(y_test, knn_probs)
   rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf_probs)

   # Plot the ROC curve for each model

   plt.plot(knn_fpr, knn_tpr, label='K NEAREST NEIGHBOUR')
   plt.plot(rf_fpr, rf_tpr, label='RANDOM FOREST')
   plt.xlabel('False Positive Rate')
   plt.ylabel('True Positive Rate')
   plt.title('ROC Curve')
   plt.legend()
   plt.show()

   # Calculate the area under the ROC curve (AUC) for each model


   knn_auc = roc_auc_score(y_test, knn_probs)
   rf_auc = roc_auc_score(y_test, rf_probs)



   print("k nearest neighbour AUC: {:.2f}".format(knn_auc))
   print("Random Forest AUC: {:.2f}".format(rf_auc))


dataset = pd.read_csv("heart.csv")#reading dataset
#print(dataset) # printing dataset

x = dataset.iloc[:,:-1].values #locating inputs
y = dataset.iloc[:,-1].values #locating outputs

#printing X and Y
print("x=",x)
print("y=",y)

from sklearn.model_selection import train_test_split # for splitting dataset
x_train,x_test,y_train,y_test = train_test_split(x ,y, test_size = 0.25 ,random_state = 0)
#printing the spliited dataset
print("x_train=",x_train)
print("x_test=",x_test)
print("y_train=",y_train)
print("y_test=",y_test)

# KNN ALGORITHM TRAINING
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)


y_pred=knn.predict(x_test) #testing model
print("y_pred",y_pred) # predicted output
print("Testing Accuracy")

# RANDOM FOREST ALGORITHM

rf = RandomForestClassifier(n_estimators= 2)
rf.fit(x_train, y_train)






age = float(input('ENTER THE AGE'))
sex = float(input('ENTER THE sex'))
chest_pain = float(input('ENTER THE chest_pain'))
resting_bp = float(input('ENTER THE resting_bp'))
cholesterol = float(input('ENTER THE cholesterol'))
fasting_blood = float(input('ENTER THE fasting_blood'))
resting_ecg = float(input('ENTER THE resting_ecg'))
max_heart_rate = float(input('ENTER THE max_heart_rate'))
excercise_angina = float(input('ENTER THE excercise_angina'))
old_peak = float(input('ENTER THE old_peak'))
st_slope = float(input('ENTER THE st_slope'))

output = knn.predict([[age,sex,chest_pain,resting_bp,cholesterol,fasting_blood,resting_ecg,max_heart_rate,excercise_angina,old_peak,st_slope]])
print("K - NEAREST NEIGHBOUR ALGORITHM OUTPUT IS  :",output)

output1 = rf.predict([[age,sex,chest_pain,resting_bp,cholesterol,fasting_blood,resting_ecg,max_heart_rate,excercise_angina,old_peak,st_slope]])
print("RANDOM FOREST ALGORITHM OUTPUT IS  :",output1)


if output == 0:
   print('NO HEART DISEASE')
else:
   print('HEART DISEASE PREDICTED')

graph_comparision()











