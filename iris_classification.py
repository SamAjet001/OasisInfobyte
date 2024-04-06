# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris
iris = load_iris()

# Load the datasets into a panda DataFrame
import os

file_path = os.path.join(os.path.expanduser("~"), "Desktop", "iris.csv")
data = pd.read_csv(file_path)

# Separate features and target variable
X = data.drop('Species', axis=1)
Y = data['Species']

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create a Decision Tree Classifier
model = DecisionTreeClassifier()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# import the classification report function from sklearn.metrics
from sklearn.metrics import classification_report

#train and make predictions
report = classification_report(y_test, y_pred, target_names=iris.target_names)
print(report)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', yticklabels=iris.target_names)  # Removed sticklabels
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix')
plt.show()

from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

log_loss = log_loss(y_test, model.predict_proba(X_test))
print(f"Log loss: {log_loss}")

auc_proc + roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
print(f"AUC-ROC: {auc_roc}")

