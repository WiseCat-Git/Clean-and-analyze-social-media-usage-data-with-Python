import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned data
df_cleaned = pd.read_csv("cleaned_data.csv")

# Function to extract n-grams
def get_top_n_grams(corpus, n_gram_range, n=None):
    vec = CountVectorizer(ngram_range=n_gram_range, stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]

# Get top 10 bigrams for each sentiment
for sentiment in df_cleaned['target'].unique():
    print(f"\nTop 10 Bigrams for Sentiment {sentiment}")
    top_bigrams = get_top_n_grams(df_cleaned[df_cleaned['target'] == sentiment]['text'], (2, 2), 10)
    for bigram, freq in top_bigrams:
        print(f"{bigram}: {freq}")
    
    # Plotting the bigrams
    bigram_df = pd.DataFrame(top_bigrams, columns=['Bigram', 'Frequency'])
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Frequency', y='Bigram', data=bigram_df, palette='viridis')
    plt.title(f"Top 10 Bigrams for Sentiment {sentiment}")
    plt.show()

# Get top 10 trigrams for each sentiment
for sentiment in df_cleaned['target'].unique():
    print(f"\nTop 10 Trigrams for Sentiment {sentiment}")
    top_trigrams = get_top_n_grams(df_cleaned[df_cleaned['target'] == sentiment]['text'], (3, 3), 10)
    for trigram, freq in top_trigrams:
        print(f"{trigram}: {freq}")
    
    # Plotting the trigrams
    trigram_df = pd.DataFrame(top_trigrams, columns=['Trigram', 'Frequency'])
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Frequency', y='Trigram', data=trigram_df, palette='viridis')
    plt.title(f"Top 10 Trigrams for Sentiment {sentiment}")
    plt.show()
