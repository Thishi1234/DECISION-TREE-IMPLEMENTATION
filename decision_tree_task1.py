# Importing required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Loading the breast cancer dataset from CSV file
df = pd.read_csv('/content/breastcancer - breastcancer - breastcancer - breastcancer.csv')

# Basic info about the dataset
print('Length of the dataset : ', len(df))
print('Shape of the dataset : ', df.shape)
print(df.head(3))  # Display first 3 rows

# Splitting dataset into features (X) and target (y)
x = df.iloc[:, 2:]  # All columns from 3rd onwards are features
y = df.iloc[:, 1]   # The 2nd column is assumed to be the target class

# Splitting data into training and testing sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=40)

# Feature scaling to normalize the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)  # Fit on training data and transform
x_test = sc.transform(x_test)        # Only transform test data

# Checking shapes of transformed data
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# ---------------- GINI CRITERIA ---------------- #
print("DECISION TREE WITH GINI CRITERIA")

# Creating Decision Tree classifier using 'gini' impurity
dc = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)

# Training the model
dc.fit(x_train, y_train)

# Predicting test set results
y_pred = dc.predict(x_test)
print(y_pred.shape)
print(y_pred)

# Model evaluation using accuracy, confusion matrix and classification report
print("Accuracy Score : ", accuracy_score(y_test, y_pred))
cfm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix : \n", cfm)
print("Classification Report : \n", classification_report(y_test, y_pred))

# Visualizing the confusion matrix using heatmap
import seaborn as sns
import matplotlib.pyplot as plt  # Ensure plt is imported only once at the top

plt.figure(figsize=(6, 4))
sns.heatmap(cfm, annot=True, fmt='d', cmap='Blues')  # Heatmap for better visual understanding
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Visualizing the trained decision tree (Gini)
print("Decision Tree : ")
plt.figure(figsize=(20,10))
plot_tree(dc, filled=True, feature_names=x.columns, class_names=['malignant', 'benign'])
plt.show()

# ---------------- ENTROPY CRITERIA ---------------- #
print("DECISION TREE WITH ENTROPY CRITERIA")

# Creating Decision Tree classifier using 'entropy' impurity
dc = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)

# Training the model again with entropy criterion
dc.fit(x_train, y_train)

# Predicting test set results
y_pred = dc.predict(x_test)
print(y_pred.shape)
print(y_pred)

# Model evaluation again
print("Accuracy Score : ", accuracy_score(y_test, y_pred))
cfm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix : \n", cfm)
print("Classification Report : \n", classification_report(y_test, y_pred))

# Visualizing the new confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cfm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Visualizing the trained decision tree (Entropy)
print("Decision Tree : ")
plt.figure(figsize=(20,10))
plot_tree(dc, filled=True, feature_names=x.columns, class_names=['malignant', 'benign'])
plt.show()
