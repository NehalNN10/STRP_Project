import pandas as pd
import numpy as np
from scipy import stats

def analyze_urdu_sentiment():
    """
    Comprehensive analysis of Urdu sentiment data for fuel price changes
    Reproduces the findings similar to the English analysis
    """
    # Read the Urdu CSV file
    df = pd.read_csv('data/urdu_average_data.csv')
    
    print("=" * 80)
    print("URDU ARTICLES - OPENAI SENTIMENT ANALYSIS")
    print("Fuel Price Changes vs Sentiment (January 2021 - December 2024)")
    print("=" * 80)
    print(f"Total records analyzed: {len(df)}")
    print()
    
    # 1. BASIC AVERAGES
    print("1. OVERALL AVERAGE SENTIMENT SCORES")
    print("-" * 50)
    headline_avg = df['openai_headline_overall_sentiment'].mean()
    text_avg = df['openai_text_overall_sentiment'].mean()
    print(f"Average openai_headline_overall_sentiment: {headline_avg:.6f}")
    print(f"Average openai_text_overall_sentiment: {text_avg:.6f}")
    print(f"Overall difference (text - headline): {text_avg - headline_avg:.6f}")
    print()
    
    # 2. PRICE HIKES vs PRICE DROPS ANALYSIS
    print("2. HEADLINE vs BODY TEXT POLARITY ANALYSIS")
    print("-" * 50)
    
    # Price drops analysis
    price_drops = df[(df['petrol_change'] < 0) | (df['diesel_change'] < 0)].copy()
    price_drops['sentiment_diff'] = price_drops['openai_text_overall_sentiment'] - price_drops['openai_headline_overall_sentiment']
    
    # Price hikes analysis
    price_hikes = df[(df['petrol_change'] > 0) | (df['diesel_change'] > 0)].copy()
    price_hikes['sentiment_diff'] = price_hikes['openai_text_overall_sentiment'] - price_hikes['openai_headline_overall_sentiment']
    
    print(f"Price Drops Analysis ({len(price_drops)} instances):")
    if len(price_drops) > 0:
        headline_drops_avg = price_drops['openai_headline_overall_sentiment'].mean()
        text_drops_avg = price_drops['openai_text_overall_sentiment'].mean()
        drops_diff = text_drops_avg - headline_drops_avg
        print(f"  Headlines average sentiment: {headline_drops_avg:.3f}")
        print(f"  Body text average sentiment: {text_drops_avg:.3f}")
        print(f"  Difference (text - headline): {drops_diff:.3f}")
    
    print(f"\nPrice Hikes Analysis ({len(price_hikes)} instances):")
    if len(price_hikes) > 0:
        headline_hikes_avg = price_hikes['openai_headline_overall_sentiment'].mean()
        text_hikes_avg = price_hikes['openai_text_overall_sentiment'].mean()
        hikes_diff = text_hikes_avg - headline_hikes_avg
        print(f"  Headlines average sentiment: {headline_hikes_avg:.3f}")
        print(f"  Body text average sentiment: {text_hikes_avg:.3f}")
        print(f"  Difference (text - headline): {hikes_diff:.3f}")
    print()
    
    # 3. SENSITIVITY ANALYSIS (Similar to Table in the paper)
    print("3. FUEL PRICE CHANGE SENSITIVITY ANALYSIS")
    print("-" * 50)
    
    # Define large changes (10 PKR or more)
    large_threshold = 10
    
    # Petrol analysis
    petrol_increases = df[df['petrol_change'] > 0]
    petrol_decreases = df[df['petrol_change'] < 0]
    large_petrol_increases = df[df['petrol_change'] >= large_threshold]
    large_petrol_decreases = df[df['petrol_change'] <= -large_threshold]
    
    # Diesel analysis
    diesel_increases = df[df['diesel_change'] > 0]
    diesel_decreases = df[df['diesel_change'] < 0]
    large_diesel_increases = df[df['diesel_change'] >= large_threshold]
    large_diesel_decreases = df[df['diesel_change'] <= -large_threshold]
    
    print("PETROL SENSITIVITY:")
    print("  Headlines:")
    if len(petrol_increases) > 0:
        petrol_headline_per_increase = petrol_increases['openai_headline_overall_sentiment'].mean() / petrol_increases['petrol_change'].mean()
        print(f"    Sentiment per unit increase: {petrol_headline_per_increase:.3f}")
    
    if len(petrol_decreases) > 0:
        petrol_headline_per_decrease = petrol_decreases['openai_headline_overall_sentiment'].mean() / abs(petrol_decreases['petrol_change'].mean())
        print(f"    Sentiment per unit decrease: {petrol_headline_per_decrease:.3f}")
    
    if len(large_petrol_increases) > 0:
        print(f"    Average sentiment per large increase: {large_petrol_increases['openai_headline_overall_sentiment'].mean():.3f}")
    
    if len(large_petrol_decreases) > 0:
        print(f"    Average sentiment per large decrease: {large_petrol_decreases['openai_headline_overall_sentiment'].mean():.3f}")
    
    print("  Body Text:")
    if len(petrol_increases) > 0:
        petrol_text_per_increase = petrol_increases['openai_text_overall_sentiment'].mean() / petrol_increases['petrol_change'].mean()
        print(f"    Sentiment per unit increase: {petrol_text_per_increase:.3f}")
    
    if len(petrol_decreases) > 0:
        petrol_text_per_decrease = petrol_decreases['openai_text_overall_sentiment'].mean() / abs(petrol_decreases['petrol_change'].mean())
        print(f"    Sentiment per unit decrease: {petrol_text_per_decrease:.3f}")
    
    if len(large_petrol_increases) > 0:
        print(f"    Average sentiment per large increase: {large_petrol_increases['openai_text_overall_sentiment'].mean():.3f}")
    
    if len(large_petrol_decreases) > 0:
        print(f"    Average sentiment per large decrease: {large_petrol_decreases['openai_text_overall_sentiment'].mean():.3f}")
    
    print("\nDIESEL SENSITIVITY:")
    print("  Headlines:")
    if len(diesel_increases) > 0:
        diesel_headline_per_increase = diesel_increases['openai_headline_overall_sentiment'].mean() / diesel_increases['diesel_change'].mean()
        print(f"    Sentiment per unit increase: {diesel_headline_per_increase:.3f}")
    
    if len(diesel_decreases) > 0:
        diesel_headline_per_decrease = diesel_decreases['openai_headline_overall_sentiment'].mean() / abs(diesel_decreases['diesel_change'].mean())
        print(f"    Sentiment per unit decrease: {diesel_headline_per_decrease:.3f}")
    
    if len(large_diesel_increases) > 0:
        print(f"    Average sentiment per large increase: {large_diesel_increases['openai_headline_overall_sentiment'].mean():.3f}")
    
    if len(large_diesel_decreases) > 0:
        print(f"    Average sentiment per large decrease: {large_diesel_decreases['openai_headline_overall_sentiment'].mean():.3f}")
    
    print("  Body Text:")
    if len(diesel_increases) > 0:
        diesel_text_per_increase = diesel_increases['openai_text_overall_sentiment'].mean() / diesel_increases['diesel_change'].mean()
        print(f"    Sentiment per unit increase: {diesel_text_per_increase:.3f}")
    
    if len(diesel_decreases) > 0:
        diesel_text_per_decrease = diesel_decreases['openai_text_overall_sentiment'].mean() / abs(diesel_decreases['diesel_change'].mean())
        print(f"    Sentiment per unit decrease: {diesel_text_per_decrease:.3f}")
    
    if len(large_diesel_increases) > 0:
        print(f"    Average sentiment per large increase: {large_diesel_increases['openai_text_overall_sentiment'].mean():.3f}")
    
    if len(large_diesel_decreases) > 0:
        print(f"    Average sentiment per large decrease: {large_diesel_decreases['openai_text_overall_sentiment'].mean():.3f}")
    
    print()
    
    # 4. CORRELATION ANALYSIS
    print("4. CORRELATION ANALYSIS")
    print("-" * 50)
    
    # Calculate correlations
    petrol_headline_corr = df['petrol_change'].corr(df['openai_headline_overall_sentiment'])
    petrol_text_corr = df['petrol_change'].corr(df['openai_text_overall_sentiment'])
    diesel_headline_corr = df['diesel_change'].corr(df['openai_headline_overall_sentiment'])
    diesel_text_corr = df['diesel_change'].corr(df['openai_text_overall_sentiment'])
    
    print(f"Petrol vs Headlines correlation: {petrol_headline_corr:.4f}")
    print(f"Petrol vs Body Text correlation: {petrol_text_corr:.4f}")
    print(f"Diesel vs Headlines correlation: {diesel_headline_corr:.4f}")
    print(f"Diesel vs Body Text correlation: {diesel_text_corr:.4f}")
    
    # Overall sensitivity comparison
    avg_petrol_sensitivity = (abs(petrol_headline_corr) + abs(petrol_text_corr)) / 2
    avg_diesel_sensitivity = (abs(diesel_headline_corr) + abs(diesel_text_corr)) / 2
    
    print(f"\nAverage petrol sensitivity: {avg_petrol_sensitivity:.4f}")
    print(f"Average diesel sensitivity: {avg_diesel_sensitivity:.4f}")
    
    if avg_petrol_sensitivity > avg_diesel_sensitivity:
        print("✓ Sentiment is MORE sensitive to PETROL price changes")
    else:
        print("✓ Sentiment is MORE sensitive to DIESEL price changes")
    
    print()
    
    # 5. ASYMMETRIC RESPONSE ANALYSIS
    print("5. ASYMMETRIC SENTIMENT RESPONSE")
    print("-" * 50)
    
    # Find examples of large price changes for asymmetric analysis
    if len(large_petrol_increases) > 0 and len(large_petrol_decreases) > 0:
        print("Example of Asymmetric Response (Petrol):")
        # Find a representative increase and decrease
        increase_example = large_petrol_increases.iloc[0]
        decrease_example = large_petrol_decreases.iloc[0]
        
        print(f"Price increase of {increase_example['petrol_change']:.1f} PKR:")
        print(f"  Headline sentiment: {increase_example['openai_headline_overall_sentiment']:.2f}")
        print(f"  Body text sentiment: {increase_example['openai_text_overall_sentiment']:.2f}")
        
        print(f"Price decrease of {abs(decrease_example['petrol_change']):.1f} PKR:")
        print(f"  Headline sentiment: {decrease_example['openai_headline_overall_sentiment']:.2f}")
        print(f"  Body text sentiment: {decrease_example['openai_text_overall_sentiment']:.2f}")
    
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON WITH ENGLISH FINDINGS")
    print("=" * 80)
    
    # Compare key metrics with the provided English findings
    print("Key Findings for Urdu Articles:")
    
    if len(price_drops) > 0:
        urdu_headline_drops = price_drops['openai_headline_overall_sentiment'].mean()
        urdu_text_drops = price_drops['openai_text_overall_sentiment'].mean()
        urdu_drops_diff = urdu_text_drops - urdu_headline_drops
        print(f"• During price drops: Headlines avg {urdu_headline_drops:.2f}, Text avg {urdu_text_drops:.2f} (diff: {urdu_drops_diff:.2f})")
    
    if len(price_hikes) > 0:
        urdu_headline_hikes = price_hikes['openai_headline_overall_sentiment'].mean()
        urdu_text_hikes = price_hikes['openai_text_overall_sentiment'].mean()
        urdu_hikes_diff = urdu_text_hikes - urdu_headline_hikes
        print(f"• During price hikes: Headlines avg {urdu_headline_hikes:.2f}, Text avg {urdu_text_hikes:.2f} (diff: {urdu_hikes_diff:.2f})")
    
    print(f"• Petrol sensitivity (avg): {avg_petrol_sensitivity:.3f}")
    print(f"• Diesel sensitivity (avg): {avg_diesel_sensitivity:.3f}")

if __name__ == "__main__":
    try:
        analyze_urdu_sentiment()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the urdu_average_data.csv file exists in the data/ directory")
