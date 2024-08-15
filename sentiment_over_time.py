import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned data with the date column
df_cleaned = pd.read_csv("cleaned_data_with_date.csv")

# Convert the 'date' column to datetime (again to ensure correct type after loading)
df_cleaned['date'] = pd.to_datetime(df_cleaned['date'])

# Set the date column as the index
df_cleaned.set_index('date', inplace=True)

# Resample the data to get weekly sentiment trends
sentiment_trend = df_cleaned.resample('W')['target'].mean()

# Plotting the sentiment trend over time
plt.figure(figsize=(12, 6))
sentiment_trend.plot()
plt.title("Sentiment Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Average Sentiment")
plt.show()
