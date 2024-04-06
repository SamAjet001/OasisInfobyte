import pandas as pd

# Load the dataset
df = pd.read_csv('Advertising(1).csv')

# Display the first few rows of the dataframe
df.head()

# Set the first column as the index and drop it from the dataframe
df.set_index('Unnamed: 0', inplace=True)

# Check for missing values and data types
df_info = df.info()

# Display the info
print(df_info)

# Check for missing values
missing_values = df.isnull().sum()
print(missing_values)

import matplotlib.pyplot as plt
import seaborn as sns

# Set the background color to white for visibility
plt.figure(facecolor='white')

# Pairplot to visualize the distributions and relationships
sns.pairplot(df)
plt.show()

# Correlation matrix
corr_matrix = df.corr()
print(corr_matrix)

# Selecting features and target variable
X = df[['TV', 'Radio']]  # Features
y = df['Sales']  # Target variable

# Display the selected features and target variable
print('Features selected for the model:')
print(X.head())
print('\
Target variable:')
print(y.head())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing the Linear Regression model
model = LinearRegression()

# Training the model
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Calculating the model performance metrics
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

# Display the performance metrics
print(f'MSE: {mse:.2f}, RMSE: {rmse:.2f}, R^2: {r2:.2f}')