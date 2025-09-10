import pandas as pd
import numpy as np

# Read the Urdu CSV file
df = pd.read_csv('data/urdu_average_data.csv')

print("URDU ARTICLES - BASIC SENTIMENT AVERAGES")
print("=" * 50)

# Calculate basic averages
headline_avg = df['openai_headline_overall_sentiment'].mean()
text_avg = df['openai_text_overall_sentiment'].mean()

print(f"Total records: {len(df)}")
print(f"Average openai_headline_overall_sentiment: {headline_avg:.6f}")
print(f"Average openai_text_overall_sentiment: {text_avg:.6f}")
print(f"Difference (text - headline): {text_avg - headline_avg:.6f}")
print()

# Quick sensitivity check
petrol_headline_corr = df['petrol_change'].corr(df['openai_headline_overall_sentiment'])
petrol_text_corr = df['petrol_change'].corr(df['openai_text_overall_sentiment'])
diesel_headline_corr = df['diesel_change'].corr(df['openai_headline_overall_sentiment'])
diesel_text_corr = df['diesel_change'].corr(df['openai_text_overall_sentiment'])

print("CORRELATION ANALYSIS:")
print(f"Petrol vs Headlines: {petrol_headline_corr:.4f}")
print(f"Petrol vs Text: {petrol_text_corr:.4f}")
print(f"Diesel vs Headlines: {diesel_headline_corr:.4f}")
print(f"Diesel vs Text: {diesel_text_corr:.4f}")

# Determine which is more sensitive
avg_petrol_sens = (abs(petrol_headline_corr) + abs(petrol_text_corr)) / 2
avg_diesel_sens = (abs(diesel_headline_corr) + abs(diesel_text_corr)) / 2

print(f"\nSENSITIVITY COMPARISON:")
print(f"Average petrol sensitivity: {avg_petrol_sens:.4f}")
print(f"Average diesel sensitivity: {avg_diesel_sens:.4f}")

if avg_petrol_sens > avg_diesel_sens:
    print("✓ OpenAI sentiment is MORE sensitive to PETROL price changes")
    print(f"Difference: {avg_petrol_sens - avg_diesel_sens:.4f}")
else:
    print("✓ OpenAI sentiment is MORE sensitive to DIESEL price changes")
    print(f"Difference: {avg_diesel_sens - avg_petrol_sens:.4f}")
