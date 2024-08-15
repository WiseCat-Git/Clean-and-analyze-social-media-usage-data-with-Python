import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import gensim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned data
df_cleaned = pd.read_csv("cleaned_data_with_date.csv")

# Split the data into features (X) and target (y)
X = df_cleaned['text']
y = df_cleaned['target']

# Tokenize text data
X_tokenized = [gensim.utils.simple_preprocess(text) for text in X]

# Load pre-trained Word2Vec model (GoogleNews-vectors-negative300.bin)
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

# Create a function to average word vectors for a sentence
def average_word_vectors(words, model, num_features):
    feature_vec = np.zeros((num_features,), dtype="float32")
    num_words = 0
    for word in words:
        if word in model.key_to_index:  # Access the vectors directly from the model
            num_words += 1
            feature_vec = np.add(feature_vec, model[word])
    if num_words > 0:
        feature_vec = np.divide(feature_vec, num_words)
    return feature_vec

# Generate word vectors for all documents
X_word2vec = np.array([average_word_vectors(sentence, word2vec_model, 300) for sentence in X_tokenized])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_word2vec, y, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression model
model = LogisticRegression(max_iter=100)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Displaying accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plotting the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
