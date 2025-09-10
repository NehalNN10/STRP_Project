import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load the Urdu data
df = pd.read_csv('data/urdu_average_data.csv')

print("=== URDU GEMINI SECTION STATISTICS ===")
print("=" * 50)

# Filter out rows where price changes are NaN or 0
df_filtered = df.dropna(subset=['petrol_change', 'diesel_change'])
df_changes = df_filtered[(df_filtered['petrol_change'] != 0) | (df_filtered['diesel_change'] != 0)]

# Combined fuel price changes (petrol and diesel together)
combined_data = []
for _, row in df_changes.iterrows():
    if row['petrol_change'] != 0:
        combined_data.append({
            'fuel_type': 'petrol',
            'price_change': row['petrol_change'],
            'headline_sentiment': row['gemini_headline_overall_sentiment'],
            'text_sentiment': row['gemini_text_overall_sentiment'],
            'date': row['date']
        })
    if row['diesel_change'] != 0:
        combined_data.append({
            'fuel_type': 'diesel', 
            'price_change': row['diesel_change'],
            'headline_sentiment': row['gemini_headline_overall_sentiment'],
            'text_sentiment': row['gemini_text_overall_sentiment'],
            'date': row['date']
        })

combined_df = pd.DataFrame(combined_data)

# 1. HEADLINE vs BODY TEXT POLARITY DURING PRICE DROPS AND HIKES
print("1. HEADLINE vs BODY TEXT POLARITY")
print("-" * 35)

# Price drops (negative changes)
price_drops = combined_df[combined_df['price_change'] < 0]
price_hikes = combined_df[combined_df['price_change'] > 0]

print(f"Price drops: {len(price_drops)} instances")
if len(price_drops) > 0:
    avg_headline_drops = price_drops['headline_sentiment'].mean()
    avg_text_drops = price_drops['text_sentiment'].mean()
    difference_drops = avg_headline_drops - avg_text_drops
    
    print(f"  Headlines during drops: {avg_headline_drops:.3f}")
    print(f"  Body text during drops: {avg_text_drops:.3f}")
    print(f"  Difference (headline - text): {difference_drops:.3f}")

print(f"\nPrice hikes: {len(price_hikes)} instances")
if len(price_hikes) > 0:
    avg_headline_hikes = price_hikes['headline_sentiment'].mean()
    avg_text_hikes = price_hikes['text_sentiment'].mean()
    difference_hikes = avg_headline_hikes - avg_text_hikes
    
    print(f"  Headlines during hikes: {avg_headline_hikes:.3f}")
    print(f"  Body text during hikes: {avg_text_hikes:.3f}")
    print(f"  Difference (headline - text): {difference_hikes:.3f}")

# 2. SPECIFIC EXAMPLES FOR ASYMMETRIC RESPONSE
print("\n2. ASYMMETRIC RESPONSE EXAMPLES")
print("-" * 30)

# Find examples of price increases around 15 PKR
price_increases_15 = combined_df[(combined_df['price_change'] > 10) & (combined_df['price_change'] < 20)]
if len(price_increases_15) > 0:
    print("Price increases between 10-20 PKR:")
    for _, row in price_increases_15.iterrows():
        print(f"  {row['date']}: {row['price_change']:.2f} PKR increase")
        print(f"    Headline: {row['headline_sentiment']:.3f}, Text: {row['text_sentiment']:.3f}")

# Find subsequent price cuts around 18.50 PKR
price_decreases_18 = combined_df[(combined_df['price_change'] < -15) & (combined_df['price_change'] > -25)]
if len(price_decreases_18) > 0:
    print("\nPrice decreases between 15-25 PKR:")
    for _, row in price_decreases_18.iterrows():
        print(f"  {row['date']}: {row['price_change']:.2f} PKR decrease")
        print(f"    Headline: {row['headline_sentiment']:.3f}, Text: {row['text_sentiment']:.3f}")

# 3. SENTIMENT RESPONSE TABLE DATA
print("\n3. SENTIMENT RESPONSE TABLE DATA")
print("-" * 35)

def calculate_sentiment_metrics(data, fuel_type):
    fuel_data = data[data['fuel_type'] == fuel_type]
    
    increases = fuel_data[fuel_data['price_change'] > 0]
    decreases = fuel_data[fuel_data['price_change'] < 0]
    large_increases = fuel_data[fuel_data['price_change'] >= 10]
    large_decreases = fuel_data[fuel_data['price_change'] <= -10]
    
    metrics = {}
    
    # Sentiment per unit change (correlation-based approach)
    if len(increases) > 0:
        # Calculate average sentiment per PKR increase
        metrics['sentiment_per_unit_increase_headline'] = increases['headline_sentiment'].sum() / increases['price_change'].sum()
        metrics['sentiment_per_unit_increase_text'] = increases['text_sentiment'].sum() / increases['price_change'].sum()
    
    if len(decreases) > 0:
        # Calculate average sentiment per PKR decrease (absolute values)
        metrics['sentiment_per_unit_decrease_headline'] = decreases['headline_sentiment'].sum() / abs(decreases['price_change'].sum())
        metrics['sentiment_per_unit_decrease_text'] = decreases['text_sentiment'].sum() / abs(decreases['price_change'].sum())
    
    # Average sentiment for large changes
    if len(large_increases) > 0:
        metrics['avg_sentiment_large_increase_headline'] = large_increases['headline_sentiment'].mean()
        metrics['avg_sentiment_large_increase_text'] = large_increases['text_sentiment'].mean()
    
    if len(large_decreases) > 0:
        metrics['avg_sentiment_large_decrease_headline'] = large_decreases['headline_sentiment'].mean()
        metrics['avg_sentiment_large_decrease_text'] = large_decreases['text_sentiment'].mean()
    
    return metrics, len(increases), len(decreases), len(large_increases), len(large_decreases)

# Calculate for petrol
petrol_metrics, petrol_inc, petrol_dec, petrol_large_inc, petrol_large_dec = calculate_sentiment_metrics(combined_df, 'petrol')

print("PETROL:")
print(f"  Price increases: {petrol_inc}, decreases: {petrol_dec}")
print(f"  Large increases (>=10 PKR): {petrol_large_inc}, large decreases (<=10 PKR): {petrol_large_dec}")
if 'sentiment_per_unit_increase_headline' in petrol_metrics:
    print(f"  Sentiment per unit increase - Headlines: {petrol_metrics['sentiment_per_unit_increase_headline']:.3f}")
    print(f"  Sentiment per unit increase - Text: {petrol_metrics['sentiment_per_unit_increase_text']:.3f}")
if 'sentiment_per_unit_decrease_headline' in petrol_metrics:
    print(f"  Sentiment per unit decrease - Headlines: {petrol_metrics['sentiment_per_unit_decrease_headline']:.3f}")
    print(f"  Sentiment per unit decrease - Text: {petrol_metrics['sentiment_per_unit_decrease_text']:.3f}")
if 'avg_sentiment_large_increase_headline' in petrol_metrics:
    print(f"  Average sentiment per large increase - Headlines: {petrol_metrics['avg_sentiment_large_increase_headline']:.3f}")
    print(f"  Average sentiment per large increase - Text: {petrol_metrics['avg_sentiment_large_increase_text']:.3f}")
if 'avg_sentiment_large_decrease_headline' in petrol_metrics:
    print(f"  Average sentiment per large decrease - Headlines: {petrol_metrics['avg_sentiment_large_decrease_headline']:.3f}")
    print(f"  Average sentiment per large decrease - Text: {petrol_metrics['avg_sentiment_large_decrease_text']:.3f}")

# Calculate for diesel
diesel_metrics, diesel_inc, diesel_dec, diesel_large_inc, diesel_large_dec = calculate_sentiment_metrics(combined_df, 'diesel')

print("\nDIESEL:")
print(f"  Price increases: {diesel_inc}, decreases: {diesel_dec}")
print(f"  Large increases (>=10 PKR): {diesel_large_inc}, large decreases (<=10 PKR): {diesel_large_dec}")
if 'sentiment_per_unit_increase_headline' in diesel_metrics:
    print(f"  Sentiment per unit increase - Headlines: {diesel_metrics['sentiment_per_unit_increase_headline']:.3f}")
    print(f"  Sentiment per unit increase - Text: {diesel_metrics['sentiment_per_unit_increase_text']:.3f}")
if 'sentiment_per_unit_decrease_headline' in diesel_metrics:
    print(f"  Sentiment per unit decrease - Headlines: {diesel_metrics['sentiment_per_unit_decrease_headline']:.3f}")
    print(f"  Sentiment per unit decrease - Text: {diesel_metrics['sentiment_per_unit_decrease_text']:.3f}")
if 'avg_sentiment_large_increase_headline' in diesel_metrics:
    print(f"  Average sentiment per large increase - Headlines: {diesel_metrics['avg_sentiment_large_increase_headline']:.3f}")
    print(f"  Average sentiment per large increase - Text: {diesel_metrics['avg_sentiment_large_increase_text']:.3f}")
if 'avg_sentiment_large_decrease_headline' in diesel_metrics:
    print(f"  Average sentiment per large decrease - Headlines: {diesel_metrics['avg_sentiment_large_decrease_headline']:.3f}")
    print(f"  Average sentiment per large decrease - Text: {diesel_metrics['avg_sentiment_large_decrease_text']:.3f}")

# 4. SPECIFIC DATE EXAMPLES
print("\n4. SPECIFIC DATE EXAMPLES")
print("-" * 25)

# Look for September 2023 example mentioned in text
sept_2023_data = combined_df[combined_df['date'].str.contains('2023-09', na=False)]
if len(sept_2023_data) > 0:
    print("September 2023 price updates:")
    for _, row in sept_2023_data.iterrows():
        print(f"  {row['date']}: {row['price_change']:.2f} PKR change ({row['fuel_type']})")
        print(f"    Headline: {row['headline_sentiment']:.3f}, Text: {row['text_sentiment']:.3f}")

# Look for July 2022 examples
july_2022_data = combined_df[combined_df['date'].str.contains('2022-07', na=False)]
if len(july_2022_data) > 0:
    print("\nJuly 2022 price updates:")
    for _, row in july_2022_data.iterrows():
        print(f"  {row['date']}: {row['price_change']:.2f} PKR change ({row['fuel_type']})")
        print(f"    Headline: {row['headline_sentiment']:.3f}, Text: {row['text_sentiment']:.3f}")

print("\n" + "=" * 50)
print("SECTION STATISTICS COMPLETE")
print("=" * 50)
