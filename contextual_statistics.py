import pandas as pd
import numpy as np

def generate_contextual_statistics():
    """
    Generate specific statistics to support the contextual analysis paragraph
    """
    # Read the Urdu dataset (since the paragraph seems to be about Urdu articles)
    df = pd.read_csv('data/urdu_average_data.csv')
    
    print("=" * 80)
    print("CONTEXTUAL STATISTICS FOR URDU SENTIMENT ANALYSIS")
    print("=" * 80)
    
    # Price drops and hikes analysis
    price_drops = df[(df['petrol_change'] < 0) | (df['diesel_change'] < 0)].copy()
    price_hikes = df[(df['petrol_change'] > 0) | (df['diesel_change'] > 0)].copy()
    
    print(f"Dataset: {len(df)} total records")
    print(f"Price drops: {len(price_drops)} instances")
    print(f"Price hikes: {len(price_hikes)} instances")
    print()
    
    # 1. HEADLINE vs BODY TEXT POLARITY ANALYSIS
    print("1. HEADLINE vs BODY TEXT POLARITY DETAILED ANALYSIS")
    print("-" * 60)
    
    # Price drops analysis
    if len(price_drops) > 0:
        headline_drops_avg = price_drops['openai_headline_overall_sentiment'].mean()
        text_drops_avg = price_drops['openai_text_overall_sentiment'].mean()
        drops_diff = text_drops_avg - headline_drops_avg
        
        print(f"PRICE DROPS ({len(price_drops)} instances):")
        print(f"  Headlines average: {headline_drops_avg:.2f}")
        print(f"  Body text average: {text_drops_avg:.2f}")
        print(f"  Difference (text - headline): {drops_diff:.2f}")
        
        # Check for exceptions (where headlines are more positive than text)
        exceptions_drops = price_drops[price_drops['openai_headline_overall_sentiment'] > price_drops['openai_text_overall_sentiment']]
        print(f"  Exceptions (headlines more positive): {len(exceptions_drops)} out of {len(price_drops)}")
        
        if len(exceptions_drops) > 0:
            print("  Exception dates:")
            for idx, row in exceptions_drops.iterrows():
                print(f"    {row['date']}: H={row['openai_headline_overall_sentiment']:.2f}, T={row['openai_text_overall_sentiment']:.2f}")
        else:
            print("  No exceptions found - body text is always more positive during price drops")
    
    print()
    
    # Price hikes analysis
    if len(price_hikes) > 0:
        headline_hikes_avg = price_hikes['openai_headline_overall_sentiment'].mean()
        text_hikes_avg = price_hikes['openai_text_overall_sentiment'].mean()
        hikes_diff = headline_hikes_avg - text_hikes_avg
        
        print(f"PRICE HIKES ({len(price_hikes)} instances):")
        print(f"  Headlines average: {headline_hikes_avg:.2f}")
        print(f"  Body text average: {text_hikes_avg:.2f}")
        print(f"  Difference (headline - text): {hikes_diff:.2f}")
        
        # Check for exceptions (where body text is more negative than headlines)
        exceptions_hikes = price_hikes[price_hikes['openai_text_overall_sentiment'] < price_hikes['openai_headline_overall_sentiment']]
        print(f"  Exceptions (body text more negative): {len(exceptions_hikes)} out of {len(price_hikes)}")
        
        if len(exceptions_hikes) > 0:
            print("  Exception examples:")
            for idx, row in exceptions_hikes.head(5).iterrows():
                petrol_change = row['petrol_change'] if not pd.isna(row['petrol_change']) and row['petrol_change'] != 0 else ''
                diesel_change = row['diesel_change'] if not pd.isna(row['diesel_change']) and row['diesel_change'] != 0 else ''
                print(f"    {row['date']}: H={row['openai_headline_overall_sentiment']:.2f}, T={row['openai_text_overall_sentiment']:.2f}, P={petrol_change}, D={diesel_change}")
        
        # Look specifically for November 5th or similar date
        nov_dates = price_hikes[price_hikes['date'].str.contains('2021-11-05', na=False)]
        if len(nov_dates) > 0:
            print("  November 5th, 2021 analysis:")
            for idx, row in nov_dates.iterrows():
                print(f"    Date: {row['date']}")
                print(f"    Petrol change: {row['petrol_change']:.1f} PKR")
                print(f"    Headline sentiment: {row['openai_headline_overall_sentiment']:.2f}")
                print(f"    Text sentiment: {row['openai_text_overall_sentiment']:.2f}")
    
    print()
    
    # 2. ASYMMETRIC RESPONSE ANALYSIS
    print("2. ASYMMETRIC SENTIMENT RESPONSE ANALYSIS")
    print("-" * 60)
    
    # Look for large price changes for asymmetric examples
    large_increases = df[df['petrol_change'] >= 20]  # Looking for ~26 PKR increases
    large_decreases = df[df['petrol_change'] <= -30]  # Looking for ~40 PKR decreases
    
    print("LARGE PRICE INCREASES (≥20 PKR):")
    if len(large_increases) > 0:
        for idx, row in large_increases.iterrows():
            print(f"  {row['date']}: +{row['petrol_change']:.1f} PKR")
            print(f"    Headline: {row['openai_headline_overall_sentiment']:.2f}")
            print(f"    Text: {row['openai_text_overall_sentiment']:.2f}")
            
            # Look for a subsequent large decrease
            date_obj = pd.to_datetime(row['date'])
            next_month = date_obj + pd.DateOffset(months=1)
            next_two_months = date_obj + pd.DateOffset(months=2)
            
            # Find decreases in the next 1-2 months
            subsequent_decreases = df[
                (pd.to_datetime(df['date']) >= next_month) & 
                (pd.to_datetime(df['date']) <= next_two_months) & 
                (df['petrol_change'] < -30)
            ]
            
            if len(subsequent_decreases) > 0:
                print(f"    Subsequent decrease found:")
                for _, dec_row in subsequent_decreases.iterrows():
                    print(f"      {dec_row['date']}: {dec_row['petrol_change']:.1f} PKR")
                    print(f"        Headline: {dec_row['openai_headline_overall_sentiment']:.2f}")
                    print(f"        Text: {dec_row['openai_text_overall_sentiment']:.2f}")
    
    print("\nLARGE PRICE DECREASES (≤-30 PKR):")
    if len(large_decreases) > 0:
        for idx, row in large_decreases.iterrows():
            print(f"  {row['date']}: {row['petrol_change']:.1f} PKR")
            print(f"    Headline: {row['openai_headline_overall_sentiment']:.2f}")
            print(f"    Text: {row['openai_text_overall_sentiment']:.2f}")
    
    # 3. COMPARISON WITH ENGLISH ARTICLES
    print("\n3. COMPARISON WITH ENGLISH ARTICLES")
    print("-" * 60)
    
    # Load English data for comparison
    english_df = pd.read_csv('data/english_average_data.csv')
    english_drops = english_df[(english_df['petrol_change'] < 0) | (english_df['diesel_change'] < 0)]
    english_hikes = english_df[(english_df['petrol_change'] > 0) | (english_df['diesel_change'] > 0)]
    
    if len(english_drops) > 0 and len(english_hikes) > 0:
        eng_headline_drops = english_drops['openai_headline_overall_sentiment'].mean()
        eng_text_drops = english_drops['openai_text_overall_sentiment'].mean()
        eng_headline_hikes = english_hikes['openai_headline_overall_sentiment'].mean()
        eng_text_hikes = english_hikes['openai_text_overall_sentiment'].mean()
        
        print("English vs Urdu Asymmetry Comparison:")
        print(f"  English - Price drops: H={eng_headline_drops:.2f}, T={eng_text_drops:.2f}")
        print(f"  Urdu - Price drops: H={headline_drops_avg:.2f}, T={text_drops_avg:.2f}")
        print(f"  English - Price hikes: H={eng_headline_hikes:.2f}, T={eng_text_hikes:.2f}")
        print(f"  Urdu - Price hikes: H={headline_hikes_avg:.2f}, T={text_hikes_avg:.2f}")
        
        # Calculate asymmetry measures
        eng_asymmetry = abs(eng_headline_drops - eng_headline_hikes) - abs(eng_text_drops - eng_text_hikes)
        urdu_asymmetry = abs(headline_drops_avg - headline_hikes_avg) - abs(text_drops_avg - text_hikes_avg)
        
        print(f"  English asymmetry measure: {eng_asymmetry:.3f}")
        print(f"  Urdu asymmetry measure: {urdu_asymmetry:.3f}")
        
        if abs(urdu_asymmetry) > abs(eng_asymmetry):
            print("  → Urdu shows MORE pronounced asymmetry than English")
        else:
            print("  → English shows MORE pronounced asymmetry than Urdu")
    
    # 4. SPECIFIC STATISTICS FOR PARAGRAPH
    print("\n4. SPECIFIC STATISTICS FOR PARAGRAPH ENHANCEMENT")
    print("-" * 60)
    
    # Percentage calculations
    drops_pct_text_more_positive = (len(price_drops[price_drops['openai_text_overall_sentiment'] > price_drops['openai_headline_overall_sentiment']]) / len(price_drops)) * 100
    hikes_pct_headlines_more_negative = (len(price_hikes[price_hikes['openai_headline_overall_sentiment'] < price_hikes['openai_text_overall_sentiment']]) / len(price_hikes)) * 100
    
    print(f"During price drops:")
    print(f"  {drops_pct_text_more_positive:.1f}% of instances have body text more positive than headlines")
    print(f"  Average magnitude of difference: {abs(drops_diff):.2f}")
    
    print(f"\nDuring price hikes:")
    print(f"  {hikes_pct_headlines_more_negative:.1f}% of instances have headlines more negative than body text")
    print(f"  Average magnitude of difference: {abs(hikes_diff):.2f}")
    
    # Standard deviations for variability
    print(f"\nVariability in sentiment responses:")
    print(f"  Headlines std dev: {df['openai_headline_overall_sentiment'].std():.3f}")
    print(f"  Text std dev: {df['openai_text_overall_sentiment'].std():.3f}")

if __name__ == "__main__":
    try:
        generate_contextual_statistics()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
