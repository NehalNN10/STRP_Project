import pandas as pd
import numpy as np

def analyze_urdu_detailed():
    """
    Detailed analysis reproducing the findings from the English paper for Urdu articles
    """
    # Read the Urdu CSV file
    df = pd.read_csv('data/urdu_average_data.csv')
    
    print("=" * 80)
    print("URDU ARTICLES - DETAILED SENTIMENT ANALYSIS")
    print("Reproducing findings similar to English analysis")
    print("=" * 80)
    
    # 1. Basic Statistics
    print(f"Dataset: {len(df)} records (January 2021 - December 2024)")
    print(f"Overall headline sentiment average: {df['openai_headline_overall_sentiment'].mean():.3f}")
    print(f"Overall text sentiment average: {df['openai_text_overall_sentiment'].mean():.3f}")
    print()
    
    # 2. Price Drops vs Price Hikes Analysis
    price_drops = df[(df['petrol_change'] < 0) | (df['diesel_change'] < 0)].copy()
    price_hikes = df[(df['petrol_change'] > 0) | (df['diesel_change'] > 0)].copy()
    
    print("HEADLINE vs BODY TEXT POLARITY:")
    print("-" * 40)
    
    # Price drops analysis
    if len(price_drops) > 0:
        headline_drops_avg = price_drops['openai_headline_overall_sentiment'].mean()
        text_drops_avg = price_drops['openai_text_overall_sentiment'].mean()
        drops_diff = headline_drops_avg - text_drops_avg  # Note: headline - text for comparison
        
        print(f"Price Drops ({len(price_drops)} instances):")
        print(f"  Headlines average: {headline_drops_avg:.2f}")
        print(f"  Body text average: {text_drops_avg:.2f}")
        print(f"  Difference (headline - text): {drops_diff:.2f}")
        
        if headline_drops_avg > text_drops_avg:
            print("  → Headlines are MORE POSITIVE than body text during price drops")
        else:
            print("  → Body text is MORE POSITIVE than headlines during price drops")
    
    # Price hikes analysis
    if len(price_hikes) > 0:
        headline_hikes_avg = price_hikes['openai_headline_overall_sentiment'].mean()
        text_hikes_avg = price_hikes['openai_text_overall_sentiment'].mean()
        hikes_diff = headline_hikes_avg - text_hikes_avg  # Note: headline - text for comparison
        
        print(f"\nPrice Hikes ({len(price_hikes)} instances):")
        print(f"  Headlines average: {headline_hikes_avg:.2f}")
        print(f"  Body text average: {text_hikes_avg:.2f}")
        print(f"  Difference (headline - text): {hikes_diff:.2f}")
        
        if headline_hikes_avg < text_hikes_avg:
            print("  → Headlines are MORE NEGATIVE than body text during price hikes")
        else:
            print("  → Body text is MORE NEGATIVE than headlines during price hikes")
    
    print()
    
    # 3. Detailed Sensitivity Analysis (Table reproduction)
    print("SENTIMENT RESPONSE TO PETROL AND DIESEL PRICE CHANGES:")
    print("-" * 60)
    
    # Define large changes threshold
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
    
    print("PETROL:")
    print("  Headlines:")
    if len(petrol_increases) > 0:
        sentiment_per_increase = petrol_increases['openai_headline_overall_sentiment'].mean() / petrol_increases['petrol_change'].mean()
        print(f"    Sentiment per unit increase: {sentiment_per_increase:.3f}")
    
    if len(petrol_decreases) > 0:
        sentiment_per_decrease = petrol_decreases['openai_headline_overall_sentiment'].mean() / abs(petrol_decreases['petrol_change'].mean())
        print(f"    Sentiment per unit decrease: {sentiment_per_decrease:.3f}")
    
    if len(large_petrol_increases) > 0:
        avg_sentiment_large_inc = large_petrol_increases['openai_headline_overall_sentiment'].mean()
        print(f"    Average sentiment per large increase: {avg_sentiment_large_inc:.3f}")
    
    if len(large_petrol_decreases) > 0:
        avg_sentiment_large_dec = large_petrol_decreases['openai_headline_overall_sentiment'].mean()
        print(f"    Average sentiment per large decrease: {avg_sentiment_large_dec:.3f}")
    
    print("  Body Text:")
    if len(petrol_increases) > 0:
        sentiment_per_increase = petrol_increases['openai_text_overall_sentiment'].mean() / petrol_increases['petrol_change'].mean()
        print(f"    Sentiment per unit increase: {sentiment_per_increase:.3f}")
    
    if len(petrol_decreases) > 0:
        sentiment_per_decrease = petrol_decreases['openai_text_overall_sentiment'].mean() / abs(petrol_decreases['petrol_change'].mean())
        print(f"    Sentiment per unit decrease: {sentiment_per_decrease:.3f}")
    
    if len(large_petrol_increases) > 0:
        avg_sentiment_large_inc = large_petrol_increases['openai_text_overall_sentiment'].mean()
        print(f"    Average sentiment per large increase: {avg_sentiment_large_inc:.3f}")
    
    if len(large_petrol_decreases) > 0:
        avg_sentiment_large_dec = large_petrol_decreases['openai_text_overall_sentiment'].mean()
        print(f"    Average sentiment per large decrease: {avg_sentiment_large_dec:.3f}")
    
    print("\nDIESEL:")
    print("  Headlines:")
    if len(diesel_increases) > 0:
        sentiment_per_increase = diesel_increases['openai_headline_overall_sentiment'].mean() / diesel_increases['diesel_change'].mean()
        print(f"    Sentiment per unit increase: {sentiment_per_increase:.3f}")
    
    if len(diesel_decreases) > 0:
        sentiment_per_decrease = diesel_decreases['openai_headline_overall_sentiment'].mean() / abs(diesel_decreases['diesel_change'].mean())
        print(f"    Sentiment per unit decrease: {sentiment_per_decrease:.3f}")
    
    if len(large_diesel_increases) > 0:
        avg_sentiment_large_inc = large_diesel_increases['openai_headline_overall_sentiment'].mean()
        print(f"    Average sentiment per large increase: {avg_sentiment_large_inc:.3f}")
    
    if len(large_diesel_decreases) > 0:
        avg_sentiment_large_dec = large_diesel_decreases['openai_headline_overall_sentiment'].mean()
        print(f"    Average sentiment per large decrease: {avg_sentiment_large_dec:.3f}")
    
    print("  Body Text:")
    if len(diesel_increases) > 0:
        sentiment_per_increase = diesel_increases['openai_text_overall_sentiment'].mean() / diesel_increases['diesel_change'].mean()
        print(f"    Sentiment per unit increase: {sentiment_per_increase:.3f}")
    
    if len(diesel_decreases) > 0:
        sentiment_per_decrease = diesel_decreases['openai_text_overall_sentiment'].mean() / abs(diesel_decreases['diesel_change'].mean())
        print(f"    Sentiment per unit decrease: {sentiment_per_decrease:.3f}")
    
    if len(large_diesel_increases) > 0:
        avg_sentiment_large_inc = large_diesel_increases['openai_text_overall_sentiment'].mean()
        print(f"    Average sentiment per large increase: {avg_sentiment_large_inc:.3f}")
    
    if len(large_diesel_decreases) > 0:
        avg_sentiment_large_dec = large_diesel_decreases['openai_text_overall_sentiment'].mean()
        print(f"    Average sentiment per large decrease: {avg_sentiment_large_dec:.3f}")
    
    print()
    
    # 4. Correlation and Sensitivity Summary
    print("FUEL PRICE CHANGE SENSITIVITY SUMMARY:")
    print("-" * 45)
    
    petrol_headline_corr = abs(df['petrol_change'].corr(df['openai_headline_overall_sentiment']))
    petrol_text_corr = abs(df['petrol_change'].corr(df['openai_text_overall_sentiment']))
    diesel_headline_corr = abs(df['diesel_change'].corr(df['openai_headline_overall_sentiment']))
    diesel_text_corr = abs(df['diesel_change'].corr(df['openai_text_overall_sentiment']))
    
    print(f"Petrol sensitivity - Headlines: {petrol_headline_corr:.4f}")
    print(f"Petrol sensitivity - Body Text: {petrol_text_corr:.4f}")
    print(f"Diesel sensitivity - Headlines: {diesel_headline_corr:.4f}")
    print(f"Diesel sensitivity - Body Text: {diesel_text_corr:.4f}")
    
    avg_petrol_sens = (petrol_headline_corr + petrol_text_corr) / 2
    avg_diesel_sens = (diesel_headline_corr + diesel_text_corr) / 2
    
    print(f"\nOverall petrol sensitivity: {avg_petrol_sens:.4f}")
    print(f"Overall diesel sensitivity: {avg_diesel_sens:.4f}")
    
    if avg_petrol_sens > avg_diesel_sens:
        print(f"✓ Sentiment is MORE sensitive to PETROL price changes")
        print(f"  Difference: {avg_petrol_sens - avg_diesel_sens:.4f}")
    else:
        print(f"✓ Sentiment is MORE sensitive to DIESEL price changes")
        print(f"  Difference: {avg_diesel_sens - avg_petrol_sens:.4f}")
    
    print()
    
    # 5. Asymmetric Response Example
    print("ASYMMETRIC SENTIMENT RESPONSE EXAMPLE:")
    print("-" * 45)
    
    # Find examples of large price changes
    if len(large_petrol_increases) > 0:
        increase_example = large_petrol_increases.iloc[0]
        print(f"Large petrol increase example ({increase_example['date']}):")
        print(f"  Price change: +{increase_example['petrol_change']:.1f} PKR")
        print(f"  Headline sentiment: {increase_example['openai_headline_overall_sentiment']:.2f}")
        print(f"  Body text sentiment: {increase_example['openai_text_overall_sentiment']:.2f}")
    
    if len(large_petrol_decreases) > 0:
        decrease_example = large_petrol_decreases.iloc[0]
        print(f"\nLarge petrol decrease example ({decrease_example['date']}):")
        print(f"  Price change: {decrease_example['petrol_change']:.1f} PKR")
        print(f"  Headline sentiment: {decrease_example['openai_headline_overall_sentiment']:.2f}")
        print(f"  Body text sentiment: {decrease_example['openai_text_overall_sentiment']:.2f}")
    
    print("\n" + "=" * 80)
    print("COMPARISON WITH ENGLISH FINDINGS")
    print("=" * 80)
    print("English findings (from your paper):")
    print("• Price drops: Headlines +0.53, Text +0.46 (diff +0.07)")
    print("• Price hikes: Headlines -0.90, Text -0.78 (diff -0.12)")
    print("• More sensitive to petrol than diesel")
    print("• Asymmetric response to price changes")
    
    print(f"\nUrdu findings:")
    if len(price_drops) > 0:
        urdu_h_drops = price_drops['openai_headline_overall_sentiment'].mean()
        urdu_t_drops = price_drops['openai_text_overall_sentiment'].mean()
        print(f"• Price drops: Headlines {urdu_h_drops:.2f}, Text {urdu_t_drops:.2f} (diff {urdu_h_drops-urdu_t_drops:.2f})")
    
    if len(price_hikes) > 0:
        urdu_h_hikes = price_hikes['openai_headline_overall_sentiment'].mean()
        urdu_t_hikes = price_hikes['openai_text_overall_sentiment'].mean()
        print(f"• Price hikes: Headlines {urdu_h_hikes:.2f}, Text {urdu_t_hikes:.2f} (diff {urdu_h_hikes-urdu_t_hikes:.2f})")
    
    if avg_petrol_sens > avg_diesel_sens:
        print(f"• More sensitive to petrol ({avg_petrol_sens:.3f}) than diesel ({avg_diesel_sens:.3f})")
    else:
        print(f"• More sensitive to diesel ({avg_diesel_sens:.3f}) than petrol ({avg_petrol_sens:.3f})")

if __name__ == "__main__":
    try:
        analyze_urdu_detailed()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
