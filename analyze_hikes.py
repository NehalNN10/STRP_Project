import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('data/english_average_data.csv')

# Identify price hikes (positive changes in petrol or diesel prices)
price_hikes = df[(df['petrol_change'] > 0) | (df['diesel_change'] > 0)].copy()

print('Total records:', len(df))
print('Records with price hikes:', len(price_hikes))
print()

# Calculate sentiment differences (text - headline) for price hike periods
price_hikes['sentiment_diff'] = price_hikes['openai_text_overall_sentiment'] - price_hikes['openai_headline_overall_sentiment']

# Calculate averages during price hikes
headline_avg_hikes = price_hikes['openai_headline_overall_sentiment'].mean()
text_avg_hikes = price_hikes['openai_text_overall_sentiment'].mean()
sentiment_diff_avg = price_hikes['sentiment_diff'].mean()

print('During price hikes:')
print('Average headline sentiment:', round(headline_avg_hikes, 6))
print('Average text sentiment:', round(text_avg_hikes, 6))
print('Average difference (text - headline):', round(sentiment_diff_avg, 6))
print()

# Compare with overall averages
overall_headline_avg = df['openai_headline_overall_sentiment'].mean()
overall_text_avg = df['openai_text_overall_sentiment'].mean()
overall_diff_avg = (df['openai_text_overall_sentiment'] - df['openai_headline_overall_sentiment']).mean()

print('Overall averages:')
print('Average headline sentiment:', round(overall_headline_avg, 6))
print('Average text sentiment:', round(overall_text_avg, 6))
print('Average difference (text - headline):', round(overall_diff_avg, 6))
print()

# Analysis
print('Analysis:')
if text_avg_hikes < headline_avg_hikes:
    print('During price hikes, text sentiment is MORE NEGATIVE than headline sentiment')
else:
    print('During price hikes, text sentiment is LESS NEGATIVE than headline sentiment')

print('Text vs Headline during hikes:', round(text_avg_hikes, 3), 'vs', round(headline_avg_hikes, 3))
print('Difference magnitude during hikes:', round(abs(sentiment_diff_avg), 6))

# Also analyze price drops for comparison
price_drops = df[(df['petrol_change'] < 0) | (df['diesel_change'] < 0)].copy()
price_drops['sentiment_diff'] = price_drops['openai_text_overall_sentiment'] - price_drops['openai_headline_overall_sentiment']

headline_avg_drops = price_drops['openai_headline_overall_sentiment'].mean()
text_avg_drops = price_drops['openai_text_overall_sentiment'].mean()
sentiment_diff_avg_drops = price_drops['sentiment_diff'].mean()

print('\nComparison with price drops:')
print('Records with price drops:', len(price_drops))
print('During price drops:')
print('Average headline sentiment:', round(headline_avg_drops, 6))
print('Average text sentiment:', round(text_avg_drops, 6))
print('Average difference (text - headline):', round(sentiment_diff_avg_drops, 6))
