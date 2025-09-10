import pandas as pd
import numpy as np

def analyze_gemini_sentiment_urdu():
    """
    Comprehensive analysis of Urdu Gemini sentiment data for fuel price changes
    """
    # Read the Urdu CSV file
    df = pd.read_csv('data/urdu_average_data.csv')
    
    print("=" * 80)
    print("URDU ARTICLES - GEMINI SENTIMENT ANALYSIS")
    print("Fuel Price Changes vs Sentiment (January 2021 - December 2024)")
    print("=" * 80)
    print(f"Total records analyzed: {len(df)}")
    print()
    
    # 1. BASIC AVERAGES
    print("1. OVERALL AVERAGE GEMINI SENTIMENT SCORES")
    print("-" * 50)
    headline_avg = df['gemini_headline_overall_sentiment'].mean()
    text_avg = df['gemini_text_overall_sentiment'].mean()
    print(f"Average gemini_headline_overall_sentiment: {headline_avg:.6f}")
    print(f"Average gemini_text_overall_sentiment: {text_avg:.6f}")
    print(f"Overall difference (text - headline): {text_avg - headline_avg:.6f}")
    print()
    
    # 2. PRICE HIKES vs PRICE DROPS ANALYSIS
    print("2. HEADLINE vs BODY TEXT POLARITY ANALYSIS")
    print("-" * 50)
    
    # Price drops analysis
    price_drops = df[(df['petrol_change'] < 0) | (df['diesel_change'] < 0)].copy()
    price_hikes = df[(df['petrol_change'] > 0) | (df['diesel_change'] > 0)].copy()
    
    print(f"Price Drops Analysis ({len(price_drops)} instances):")
    if len(price_drops) > 0:
        headline_drops_avg = price_drops['gemini_headline_overall_sentiment'].mean()
        text_drops_avg = price_drops['gemini_text_overall_sentiment'].mean()
        drops_diff = headline_drops_avg - text_drops_avg
        print(f"  Headlines average sentiment: {headline_drops_avg:.3f}")
        print(f"  Body text average sentiment: {text_drops_avg:.3f}")
        print(f"  Difference (headline - text): {drops_diff:.3f}")
        
        if headline_drops_avg > text_drops_avg:
            print("  → Headlines are MORE POSITIVE than body text during price drops")
        else:
            print("  → Body text is MORE POSITIVE than headlines during price drops")
    
    print(f"\nPrice Hikes Analysis ({len(price_hikes)} instances):")
    if len(price_hikes) > 0:
        headline_hikes_avg = price_hikes['gemini_headline_overall_sentiment'].mean()
        text_hikes_avg = price_hikes['gemini_text_overall_sentiment'].mean()
        hikes_diff = headline_hikes_avg - text_hikes_avg
        print(f"  Headlines average sentiment: {headline_hikes_avg:.3f}")
        print(f"  Body text average sentiment: {text_hikes_avg:.3f}")
        print(f"  Difference (headline - text): {hikes_diff:.3f}")
        
        if headline_hikes_avg < text_hikes_avg:
            print("  → Headlines are MORE NEGATIVE than body text during price hikes")
        else:
            print("  → Body text is MORE NEGATIVE than headlines during price hikes")
    print()
    
    # 3. CORRELATION ANALYSIS
    print("3. CORRELATION ANALYSIS")
    print("-" * 50)
    
    # Calculate correlations
    petrol_headline_corr = df['petrol_change'].corr(df['gemini_headline_overall_sentiment'])
    petrol_text_corr = df['petrol_change'].corr(df['gemini_text_overall_sentiment'])
    diesel_headline_corr = df['diesel_change'].corr(df['gemini_headline_overall_sentiment'])
    diesel_text_corr = df['diesel_change'].corr(df['gemini_text_overall_sentiment'])
    
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
        print("✓ Gemini sentiment is MORE sensitive to PETROL price changes")
        print(f"  Difference: {avg_petrol_sensitivity - avg_diesel_sensitivity:.4f}")
    else:
        print("✓ Gemini sentiment is MORE sensitive to DIESEL price changes")
        print(f"  Difference: {avg_diesel_sensitivity - avg_petrol_sensitivity:.4f}")
    
    print()
    
    # 4. COMPARISON WITH OPENAI (URDU)
    print("4. COMPARISON WITH OPENAI FINDINGS (URDU)")
    print("-" * 50)
    
    # OpenAI correlations for comparison
    openai_petrol_headline = abs(df['petrol_change'].corr(df['openai_headline_overall_sentiment']))
    openai_petrol_text = abs(df['petrol_change'].corr(df['openai_text_overall_sentiment']))
    openai_diesel_headline = abs(df['diesel_change'].corr(df['openai_headline_overall_sentiment']))
    openai_diesel_text = abs(df['diesel_change'].corr(df['openai_text_overall_sentiment']))
    
    openai_avg_petrol = (openai_petrol_headline + openai_petrol_text) / 2
    openai_avg_diesel = (openai_diesel_headline + openai_diesel_text) / 2
    
    print("OpenAI vs Gemini Sensitivity Comparison (Urdu):")
    print(f"  OpenAI petrol sensitivity: {openai_avg_petrol:.4f}")
    print(f"  Gemini petrol sensitivity: {avg_petrol_sensitivity:.4f}")
    print(f"  OpenAI diesel sensitivity: {openai_avg_diesel:.4f}")
    print(f"  Gemini diesel sensitivity: {avg_diesel_sensitivity:.4f}")
    
    if avg_petrol_sensitivity > openai_avg_petrol:
        print(f"  → Gemini is MORE sensitive to petrol changes than OpenAI")
    else:
        print(f"  → OpenAI is MORE sensitive to petrol changes than Gemini")
    
    if avg_diesel_sensitivity > openai_avg_diesel:
        print(f"  → Gemini is MORE sensitive to diesel changes than OpenAI")
    else:
        print(f"  → OpenAI is MORE sensitive to diesel changes than Gemini")
    
    # Price drops/hikes comparison
    openai_price_drops = df[(df['petrol_change'] < 0) | (df['diesel_change'] < 0)].copy()
    openai_price_hikes = df[(df['petrol_change'] > 0) | (df['diesel_change'] > 0)].copy()
    
    if len(openai_price_drops) > 0:
        openai_headline_drops = openai_price_drops['openai_headline_overall_sentiment'].mean()
        openai_text_drops = openai_price_drops['openai_text_overall_sentiment'].mean()
        print(f"\nPrice drops comparison:")
        print(f"  OpenAI: Headlines {openai_headline_drops:.2f}, Text {openai_text_drops:.2f}")
        print(f"  Gemini: Headlines {headline_drops_avg:.2f}, Text {text_drops_avg:.2f}")
    
    if len(openai_price_hikes) > 0:
        openai_headline_hikes = openai_price_hikes['openai_headline_overall_sentiment'].mean()
        openai_text_hikes = openai_price_hikes['openai_text_overall_sentiment'].mean()
        print(f"\nPrice hikes comparison:")
        print(f"  OpenAI: Headlines {openai_headline_hikes:.2f}, Text {openai_text_hikes:.2f}")
        print(f"  Gemini: Headlines {headline_hikes_avg:.2f}, Text {text_hikes_avg:.2f}")

if __name__ == "__main__":
    try:
        analyze_gemini_sentiment_urdu()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
