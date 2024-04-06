import pandas as pd

# Load the dataset
df = pd.read_csv('C:\\Users\\HP\\Downloads\\spam.csv', encoding='latin-1')

# Drop any unnecessary columns
# The dataset might contain extra columns due to the encoding format. We'll remove them.
df.drop(columns=[col for col in df.columns if 'Unnamed' in col], inplace=True)

# Display the first few rows of the dataframe
df.head()

# Check for missing values and the distribution of spam vs. non-spam emails
df_info = df.isnull().sum().to_frame('Missing Values')
df_info['Distribution'] = df['v1'].value_counts()

# Display the information
df_info

# Correcting the previous code to properly display missing values and distribution
missing_values = df.isnull().sum()
distribution = df['v1'].value_counts()

print('Missing Values:\
', missing_values)
print('\
Distribution of Spam vs. Non-Spam:\
', distribution)

import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Text normalization and cleaning
# Convert to lowercase
# Remove punctuation and numbers
# Tokenize
# Remove stop words
# Stemming

# Function to preprocess text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Stemming
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(word) for word in tokens]
    return ' '.join(stemmed)

# Apply preprocessing to the email texts
df['processed_text'] = df['v2'].apply(preprocess_text)

# Display the first few rows of the dataframe to see the processed text
df[['v2', 'processed_text']].head()

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the processed text to create TF-IDF features
X = vectorizer.fit_transform(df['processed_text'])

# The target variable is whether the email is spam or not
y = df['v1'].map({'ham': 0, 'spam': 1})

# Display the shape of the features and target variable
print('Shape of X (features):', X.shape)
print('Shape of y (target):', y.shape)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Display the evaluation metrics
print('Accuracy:', accuracy)
print('Confusion Matrix:\
', conf_matrix)
print('Classification Report:\
', class_report)