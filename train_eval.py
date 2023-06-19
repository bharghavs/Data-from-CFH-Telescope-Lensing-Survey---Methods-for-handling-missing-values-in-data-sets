# Name: Bharghav Srikhakollu
# Date: 01-21-2023
#######################################################################################################
# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.impute import IterativeImputer
from sklearn.metrics import accuracy_score

# Reference Citation:
# https://scikit-learn.org/stable/modules/impute.html - For Impute method
# https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html#sklearn.impute.KNNImputer - For KNNImputer method
# https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html - For Iterative Imputer method
#######################################################################################################
# Step - 3
#######################################################################################################
# Read CSV file using Pandas: dataframe (df)
# Convert the missing values "99.0000" and "-99.0000" as "NaN", read only the 9 feature columns
df = pd.read_csv('cfhtlens.csv', na_values =(99.0000, -99.0000), usecols=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

XY = df.to_numpy()

X = XY[:, 1:]
Y = XY[:, 0] >= 0.5
missing = np.sum(np.isnan(X), axis = 1) > 0

# Items with "no missing values"
X_use = X[~missing]
Y_use = Y[~missing]

# Split train and test data
X_train, X_test, Y_train, Y_test = train_test_split(X_use, Y_use, train_size = 3000, random_state = 0)
#######################################################################################################
# Step - 4
#######################################################################################################
# Test with remaining 2000 items including missing value items
X_test_full = np.concatenate((X_test, X[missing]))
Y_test_full = np.concatenate((Y_test, Y[missing]))

# Train the model using Support Vector Machine
svm = SVC(C=1.0, kernel = "linear", random_state = 40)
svm.fit(X_train, Y_train)
#######################################################################################################
# Step 5
#######################################################################################################
# (A) Do not classify items with missing values (abstain). Abstentions count as errors.
Y_predict = svm.predict(X_test)
cm = confusion_matrix(Y_test, Y_predict)
# cm[0][0] indicate: TP and cm[1][1] indicate: TN
pred_score_A_a = (cm[0][0] + cm[1][1]) / len(Y_test_full)
print("5(A): a) Classification Accuracy with entire test set is : ", pred_score_A_a*100, "%")
#pred_score_A_x = ~Y[missing] / Y[missing]
#print("5(A): a) Classification Accuracy with entire test set is : ", pred_score_A_x*100, "%")
#######################################################################################################
# (B) Predict the majority class (based on training set) for items with missing values
Y_test_full_pred = Y_test_full
df_Y_test_full_pred = pd.DataFrame(Y_test_full_pred)
df_Y_test_full_new = df_Y_test_full_pred.replace([True], False)
Y_test_full_new = df_Y_test_full_new.to_numpy()
print("5(B): a) Classification Accuracy with entire test set is : ", accuracy_score(Y_test_full, Y_test_full_new)*100, "%")

Y_missing_pred = Y[missing]
df_Y_missing_pred = pd.DataFrame(Y_missing_pred)
df_Y_missing_pred_new = df_Y_missing_pred.replace([True], False)
Y_missing_pred_new = df_Y_missing_pred_new.to_numpy()
print("5(B): b) Classification Accuracy with test items having missing values is : ", accuracy_score(Y[missing], Y_missing_pred_new)*100, "%")
#######################################################################################################
# (C) Omit any features (train and test) with missing values
# Deleting features "MAG_u", "MAG_g", "MAG_r", "MAG_i", "MAG_z" which have missing values
X_train_C = np.delete(X_train, np.s_[4:9], axis = 1)
X_test_full_C = np.delete(X_test_full, np.s_[4:9], axis = 1)

# Re-train using the remaining features
svm_C = SVC(C=1.0, kernel = "linear", random_state = 40)
svm_C.fit(X_train_C, Y_train)

# Calculate the Test Accuracy with test_full dataset (dropped missing values features)
pred_score_C_a = svm_C.score(X_test_full_C, Y_test_full)
print("5(C): a) Classification Accuracy with entire test set is : ", pred_score_C_a*100, "%")
#######################################################################################################
# (D) Impute the missing values using "average(mean)" for the feature from training set
svm_D = SVC(C=1.0, kernel = "linear", random_state = 40)
svm_D.fit(X_train, Y_train)

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(X_train)
X_test_full_D = imp.transform(X_test_full)
pred_score_D_a = svm_D.score(X_test_full_D, Y_test_full)
print("5(D): a) Classification Accuracy with entire test set is : ", pred_score_D_a*100, "%")

X_missing_D = imp.transform(X[missing])
pred_score_D_b = svm_D.score(X_missing_D, Y[missing])
print("5(D): b) Classification Accuracy with test items having missing values is : ", pred_score_D_b*100, "%")
#######################################################################################################
# (E) Impute the missing values using IterativeImputer
svm_E = SVC(C=1.0, kernel = "linear", random_state = 40)
svm_E.fit(X_train, Y_train)

imputer = IterativeImputer(random_state=40, imputation_order="descending")
imputer.fit(X_train)
X_test_full_E = imputer.transform(X_test_full)
pred_score_E_a = svm_E.score(X_test_full_E, Y_test_full)
print("5(E): a) Classification Accuracy with entire test set is : ", pred_score_E_a*100, "%")

X_missing_E = imputer.transform(X[missing])
pred_score_E_b = svm_E.score(X_missing_E, Y[missing])
print("5(E): b) Classification Accuracy with test items having missing values is : ", pred_score_E_b*100, "%")
#######################################################################################################
# Step 6
#######################################################################################################
df_6 = pd.read_csv('result_p07pu67fdkrd9pig.csv', na_values =(99.0000, -99.0000), usecols=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
XY_6 = df_6.to_numpy()

X_6 = XY_6[:, 1:]

X_6_new = imputer.transform(X_6)
Y_pred = svm_E.predict(X_6_new)
result = ""
if Y_pred == 0:
    result = "Not a Star - it is a galaxy"
else:
    result = "It is a Star"
print("6: The prediction of my own sky object is: ", Y_pred, ": ", result)
#######################################################################################################
