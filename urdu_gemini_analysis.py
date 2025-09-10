import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load the Urdu data
df = pd.read_csv('data/urdu_average_data.csv')

print("=== URDU GEMINI SENTIMENT ANALYSIS ===")
print("=" * 50)

# Basic statistics
print("\n1. BASIC SENTIMENT STATISTICS")
print("-" * 30)
print(f"Gemini Headlines - Mean: {df['gemini_headline_overall_sentiment'].mean():.3f}")
print(f"Gemini Headlines - Std: {df['gemini_headline_overall_sentiment'].std():.3f}")
print(f"Gemini Text - Mean: {df['gemini_text_overall_sentiment'].mean():.3f}")
print(f"Gemini Text - Std: {df['gemini_text_overall_sentiment'].std():.3f}")

# Filter out rows where price changes are NaN or 0
df_filtered = df.dropna(subset=['petrol_change', 'diesel_change'])
df_changes = df_filtered[(df_filtered['petrol_change'] != 0) | (df_filtered['diesel_change'] != 0)]

print(f"\nTotal observations with price changes: {len(df_changes)}")
print(f"Petrol price changes: {(df_changes['petrol_change'] != 0).sum()}")
print(f"Diesel price changes: {(df_changes['diesel_change'] != 0).sum()}")

# 2. CORRELATION ANALYSIS
print("\n2. CORRELATION WITH PRICE CHANGES")
print("-" * 35)

# Petrol correlations
petrol_data = df_changes[df_changes['petrol_change'] != 0]
if len(petrol_data) > 0:
    corr_petrol_headline = stats.pearsonr(petrol_data['petrol_change'], 
                                         petrol_data['gemini_headline_overall_sentiment'])
    corr_petrol_text = stats.pearsonr(petrol_data['petrol_change'], 
                                     petrol_data['gemini_text_overall_sentiment'])
    
    print(f"Petrol vs Gemini Headlines: r = {corr_petrol_headline[0]:.3f}, p = {corr_petrol_headline[1]:.3f}")
    print(f"Petrol vs Gemini Text: r = {corr_petrol_text[0]:.3f}, p = {corr_petrol_text[1]:.3f}")

# Diesel correlations
diesel_data = df_changes[df_changes['diesel_change'] != 0]
if len(diesel_data) > 0:
    corr_diesel_headline = stats.pearsonr(diesel_data['diesel_change'], 
                                         diesel_data['gemini_headline_overall_sentiment'])
    corr_diesel_text = stats.pearsonr(diesel_data['diesel_change'], 
                                     diesel_data['gemini_text_overall_sentiment'])
    
    print(f"Diesel vs Gemini Headlines: r = {corr_diesel_headline[0]:.3f}, p = {corr_diesel_headline[1]:.3f}")
    print(f"Diesel vs Gemini Text: r = {corr_diesel_text[0]:.3f}, p = {corr_diesel_text[1]:.3f}")

# 3. SENSITIVITY COMPARISON
print("\n3. SENSITIVITY COMPARISON (Absolute Correlations)")
print("-" * 45)
if len(petrol_data) > 0 and len(diesel_data) > 0:
    petrol_sensitivity_headline = abs(corr_petrol_headline[0])
    petrol_sensitivity_text = abs(corr_petrol_text[0])
    diesel_sensitivity_headline = abs(corr_diesel_headline[0])
    diesel_sensitivity_text = abs(corr_diesel_text[0])
    
    print(f"Petrol Sensitivity - Headlines: {petrol_sensitivity_headline:.3f}")
    print(f"Petrol Sensitivity - Text: {petrol_sensitivity_text:.3f}")
    print(f"Diesel Sensitivity - Headlines: {diesel_sensitivity_headline:.3f}")
    print(f"Diesel Sensitivity - Text: {diesel_sensitivity_text:.3f}")
    
    print(f"\nGemini is more sensitive to {'PETROL' if petrol_sensitivity_headline > diesel_sensitivity_headline else 'DIESEL'} in headlines")
    print(f"Gemini is more sensitive to {'PETROL' if petrol_sensitivity_text > diesel_sensitivity_text else 'DIESEL'} in text")

# 4. PRICE DROP vs PRICE HIKE ANALYSIS
print("\n4. PRICE DROP vs PRICE HIKE ANALYSIS")
print("-" * 35)

# Petrol price drops and hikes
petrol_drops = petrol_data[petrol_data['petrol_change'] < 0]
petrol_hikes = petrol_data[petrol_data['petrol_change'] > 0]

print("PETROL ANALYSIS:")
print(f"Price drops: {len(petrol_drops)} observations")
if len(petrol_drops) > 0:
    print(f"  Headlines during drops: {petrol_drops['gemini_headline_overall_sentiment'].mean():.3f}")
    print(f"  Text during drops: {petrol_drops['gemini_text_overall_sentiment'].mean():.3f}")

print(f"Price hikes: {len(petrol_hikes)} observations")
if len(petrol_hikes) > 0:
    print(f"  Headlines during hikes: {petrol_hikes['gemini_headline_overall_sentiment'].mean():.3f}")
    print(f"  Text during hikes: {petrol_hikes['gemini_text_overall_sentiment'].mean():.3f}")

# Diesel price drops and hikes
diesel_drops = diesel_data[diesel_data['diesel_change'] < 0]
diesel_hikes = diesel_data[diesel_data['diesel_change'] > 0]

print("\nDIESEL ANALYSIS:")
print(f"Price drops: {len(diesel_drops)} observations")
if len(diesel_drops) > 0:
    print(f"  Headlines during drops: {diesel_drops['gemini_headline_overall_sentiment'].mean():.3f}")
    print(f"  Text during drops: {diesel_drops['gemini_text_overall_sentiment'].mean():.3f}")

print(f"Price hikes: {len(diesel_hikes)} observations")
if len(diesel_hikes) > 0:
    print(f"  Headlines during hikes: {diesel_hikes['gemini_headline_overall_sentiment'].mean():.3f}")
    print(f"  Text during hikes: {diesel_hikes['gemini_text_overall_sentiment'].mean():.3f}")

# 5. LARGE PRICE CHANGE ANALYSIS
print("\n5. LARGE PRICE CHANGE ANALYSIS")
print("-" * 30)

# Define large changes (greater than 1 standard deviation)
petrol_std = petrol_data['petrol_change'].std()
diesel_std = diesel_data['diesel_change'].std()

large_petrol_changes = petrol_data[abs(petrol_data['petrol_change']) > petrol_std]
large_diesel_changes = diesel_data[abs(diesel_data['diesel_change']) > diesel_std]

print(f"Large petrol changes (>{petrol_std:.1f}): {len(large_petrol_changes)} observations")
if len(large_petrol_changes) > 0:
    print(f"  Average headline sentiment: {large_petrol_changes['gemini_headline_overall_sentiment'].mean():.3f}")
    print(f"  Average text sentiment: {large_petrol_changes['gemini_text_overall_sentiment'].mean():.3f}")

print(f"Large diesel changes (>{diesel_std:.1f}): {len(large_diesel_changes)} observations")
if len(large_diesel_changes) > 0:
    print(f"  Average headline sentiment: {large_diesel_changes['gemini_headline_overall_sentiment'].mean():.3f}")
    print(f"  Average text sentiment: {large_diesel_changes['gemini_text_overall_sentiment'].mean():.3f}")

# 6. ASYMMETRIC RESPONSE ANALYSIS
print("\n6. ASYMMETRIC RESPONSE ANALYSIS")
print("-" * 30)

# Check if responses to price increases vs decreases are symmetric
if len(petrol_drops) > 0 and len(petrol_hikes) > 0:
    petrol_drop_response_headline = abs(petrol_drops['gemini_headline_overall_sentiment'].mean())
    petrol_hike_response_headline = abs(petrol_hikes['gemini_headline_overall_sentiment'].mean())
    petrol_drop_response_text = abs(petrol_drops['gemini_text_overall_sentiment'].mean())
    petrol_hike_response_text = abs(petrol_hikes['gemini_text_overall_sentiment'].mean())
    
    print("PETROL ASYMMETRY:")
    print(f"  Headlines: Drops {petrol_drop_response_headline:.3f} vs Hikes {petrol_hike_response_headline:.3f}")
    print(f"  Text: Drops {petrol_drop_response_text:.3f} vs Hikes {petrol_hike_response_text:.3f}")
    print(f"  Headlines more sensitive to: {'HIKES' if petrol_hike_response_headline > petrol_drop_response_headline else 'DROPS'}")
    print(f"  Text more sensitive to: {'HIKES' if petrol_hike_response_text > petrol_drop_response_text else 'DROPS'}")

if len(diesel_drops) > 0 and len(diesel_hikes) > 0:
    diesel_drop_response_headline = abs(diesel_drops['gemini_headline_overall_sentiment'].mean())
    diesel_hike_response_headline = abs(diesel_hikes['gemini_headline_overall_sentiment'].mean())
    diesel_drop_response_text = abs(diesel_drops['gemini_text_overall_sentiment'].mean())
    diesel_hike_response_text = abs(diesel_hikes['gemini_text_overall_sentiment'].mean())
    
    print("\nDIESEL ASYMMETRY:")
    print(f"  Headlines: Drops {diesel_drop_response_headline:.3f} vs Hikes {diesel_hike_response_headline:.3f}")
    print(f"  Text: Drops {diesel_drop_response_text:.3f} vs Hikes {diesel_hike_response_text:.3f}")
    print(f"  Headlines more sensitive to: {'HIKES' if diesel_hike_response_headline > diesel_drop_response_headline else 'DROPS'}")
    print(f"  Text more sensitive to: {'HIKES' if diesel_hike_response_text > diesel_drop_response_text else 'DROPS'}")

# 7. SPECIFIC EXAMPLES
print("\n7. SPECIFIC EXAMPLES")
print("-" * 20)

# Find extreme examples
if len(petrol_data) > 0:
    max_petrol_increase = petrol_data.loc[petrol_data['petrol_change'].idxmax()]
    max_petrol_decrease = petrol_data.loc[petrol_data['petrol_change'].idxmin()]
    
    print("PETROL EXAMPLES:")
    print(f"Largest increase: {max_petrol_increase['petrol_change']:.2f} on {max_petrol_increase['date']}")
    print(f"  Headline sentiment: {max_petrol_increase['gemini_headline_overall_sentiment']:.3f}")
    print(f"  Text sentiment: {max_petrol_increase['gemini_text_overall_sentiment']:.3f}")
    
    print(f"Largest decrease: {max_petrol_decrease['petrol_change']:.2f} on {max_petrol_decrease['date']}")
    print(f"  Headline sentiment: {max_petrol_decrease['gemini_headline_overall_sentiment']:.3f}")
    print(f"  Text sentiment: {max_petrol_decrease['gemini_text_overall_sentiment']:.3f}")

if len(diesel_data) > 0:
    max_diesel_increase = diesel_data.loc[diesel_data['diesel_change'].idxmax()]
    max_diesel_decrease = diesel_data.loc[diesel_data['diesel_change'].idxmin()]
    
    print("\nDIESEL EXAMPLES:")
    print(f"Largest increase: {max_diesel_increase['diesel_change']:.2f} on {max_diesel_increase['date']}")
    print(f"  Headline sentiment: {max_diesel_increase['gemini_headline_overall_sentiment']:.3f}")
    print(f"  Text sentiment: {max_diesel_increase['gemini_text_overall_sentiment']:.3f}")
    
    print(f"Largest decrease: {max_diesel_decrease['diesel_change']:.2f} on {max_diesel_decrease['date']}")
    print(f"  Headline sentiment: {max_diesel_decrease['gemini_headline_overall_sentiment']:.3f}")
    print(f"  Text sentiment: {max_diesel_decrease['gemini_text_overall_sentiment']:.3f}")

print("\n" + "=" * 50)
print("ANALYSIS COMPLETE")
print("=" * 50)
