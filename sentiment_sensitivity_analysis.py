import pandas as pd
import numpy as np
from scipy import stats

def analyze_sentiment_sensitivity():
    """
    Analyze whether OpenAI sentiment is more sensitive to petrol or diesel price changes
    """
    # Read the CSV file
    df = pd.read_csv('data/english_average_data.csv')
    
    print("=" * 60)
    print("OPENAI SENTIMENT SENSITIVITY TO FUEL PRICE CHANGES")
    print("=" * 60)
    print(f"Total records analyzed: {len(df)}")
    print()
    
    # Remove rows with zero price changes for correlation analysis
    df_changes = df[(df['petrol_change'] != 0) | (df['diesel_change'] != 0)].copy()
    print(f"Records with price changes: {len(df_changes)}")
    print()
    
    # Analysis for both headline and text sentiment
    sentiment_types = [
        ('openai_headline_overall_sentiment', 'Headline Sentiment'),
        ('openai_text_overall_sentiment', 'Text Sentiment')
    ]
    
    for sentiment_col, sentiment_name in sentiment_types:
        print(f"\n{sentiment_name.upper()} ANALYSIS")
        print("-" * 40)
        
        # 1. Correlation Analysis
        petrol_corr = df['petrol_change'].corr(df[sentiment_col])
        diesel_corr = df['diesel_change'].corr(df[sentiment_col])
        
        print(f"Correlation with petrol price changes: {petrol_corr:.4f}")
        print(f"Correlation with diesel price changes: {diesel_corr:.4f}")
        
        # 2. Statistical significance test
        petrol_stat, petrol_p = stats.pearsonr(df['petrol_change'], df[sentiment_col])
        diesel_stat, diesel_p = stats.pearsonr(df['diesel_change'], df[sentiment_col])
        
        print(f"Petrol correlation p-value: {petrol_p:.4f}")
        print(f"Diesel correlation p-value: {diesel_p:.4f}")
        
        # 3. Analyze sentiment response to large price changes
        large_petrol_increases = df[df['petrol_change'] > 10]
        large_diesel_increases = df[df['diesel_change'] > 10]
        large_petrol_decreases = df[df['petrol_change'] < -10]
        large_diesel_decreases = df[df['diesel_change'] < -10]
        
        print(f"\nLarge price changes (>10 or <-10):")
        print(f"Large petrol increases: {len(large_petrol_increases)} records")
        if len(large_petrol_increases) > 0:
            print(f"  Average sentiment: {large_petrol_increases[sentiment_col].mean():.4f}")
        
        print(f"Large diesel increases: {len(large_diesel_increases)} records")
        if len(large_diesel_increases) > 0:
            print(f"  Average sentiment: {large_diesel_increases[sentiment_col].mean():.4f}")
        
        print(f"Large petrol decreases: {len(large_petrol_decreases)} records")
        if len(large_petrol_decreases) > 0:
            print(f"  Average sentiment: {large_petrol_decreases[sentiment_col].mean():.4f}")
        
        print(f"Large diesel decreases: {len(large_diesel_decreases)} records")
        if len(large_diesel_decreases) > 0:
            print(f"  Average sentiment: {large_diesel_decreases[sentiment_col].mean():.4f}")
        
        # 4. Calculate sensitivity ratios
        print(f"\nSensitivity Analysis:")
        
        # For increases
        petrol_increases = df[df['petrol_change'] > 0]
        diesel_increases = df[df['diesel_change'] > 0]
        
        if len(petrol_increases) > 0:
            petrol_sentiment_per_increase = petrol_increases[sentiment_col].mean() / petrol_increases['petrol_change'].mean()
            print(f"Sentiment per unit petrol increase: {petrol_sentiment_per_increase:.6f}")
        
        if len(diesel_increases) > 0:
            diesel_sentiment_per_increase = diesel_increases[sentiment_col].mean() / diesel_increases['diesel_change'].mean()
            print(f"Sentiment per unit diesel increase: {diesel_sentiment_per_increase:.6f}")
        
        # For decreases
        petrol_decreases = df[df['petrol_change'] < 0]
        diesel_decreases = df[df['diesel_change'] < 0]
        
        if len(petrol_decreases) > 0:
            petrol_sentiment_per_decrease = petrol_decreases[sentiment_col].mean() / abs(petrol_decreases['petrol_change'].mean())
            print(f"Sentiment per unit petrol decrease: {petrol_sentiment_per_decrease:.6f}")
        
        if len(diesel_decreases) > 0:
            diesel_sentiment_per_decrease = diesel_decreases[sentiment_col].mean() / abs(diesel_decreases['diesel_change'].mean())
            print(f"Sentiment per unit diesel decrease: {diesel_sentiment_per_decrease:.6f}")
    
    # Overall comparison
    print(f"\n{'='*60}")
    print("OVERALL SENSITIVITY COMPARISON")
    print(f"{'='*60}")
    
    # Calculate absolute correlations for comparison
    headline_petrol_abs = abs(df['petrol_change'].corr(df['openai_headline_overall_sentiment']))
    headline_diesel_abs = abs(df['diesel_change'].corr(df['openai_headline_overall_sentiment']))
    text_petrol_abs = abs(df['petrol_change'].corr(df['openai_text_overall_sentiment']))
    text_diesel_abs = abs(df['diesel_change'].corr(df['openai_text_overall_sentiment']))
    
    print(f"Headline sentiment - Petrol sensitivity: {headline_petrol_abs:.4f}")
    print(f"Headline sentiment - Diesel sensitivity: {headline_diesel_abs:.4f}")
    print(f"Text sentiment - Petrol sensitivity: {text_petrol_abs:.4f}")
    print(f"Text sentiment - Diesel sensitivity: {text_diesel_abs:.4f}")
    
    print(f"\nCONCLUSIONS:")
    
    # Compare petrol vs diesel sensitivity
    avg_petrol_sensitivity = (headline_petrol_abs + text_petrol_abs) / 2
    avg_diesel_sensitivity = (headline_diesel_abs + text_diesel_abs) / 2
    
    if avg_petrol_sensitivity > avg_diesel_sensitivity:
        print(f"✓ OpenAI sentiment is MORE sensitive to PETROL price changes")
        print(f"  Average petrol sensitivity: {avg_petrol_sensitivity:.4f}")
        print(f"  Average diesel sensitivity: {avg_diesel_sensitivity:.4f}")
        print(f"  Difference: {avg_petrol_sensitivity - avg_diesel_sensitivity:.4f}")
    else:
        print(f"✓ OpenAI sentiment is MORE sensitive to DIESEL price changes")
        print(f"  Average diesel sensitivity: {avg_diesel_sensitivity:.4f}")
        print(f"  Average petrol sensitivity: {avg_petrol_sensitivity:.4f}")
        print(f"  Difference: {avg_diesel_sensitivity - avg_petrol_sensitivity:.4f}")
    
    # Compare headline vs text sensitivity
    avg_headline_sensitivity = (headline_petrol_abs + headline_diesel_abs) / 2
    avg_text_sensitivity = (text_petrol_abs + text_diesel_abs) / 2
    
    if avg_headline_sensitivity > avg_text_sensitivity:
        print(f"\n✓ HEADLINE sentiment is more sensitive to price changes overall")
        print(f"  Average headline sensitivity: {avg_headline_sensitivity:.4f}")
        print(f"  Average text sensitivity: {avg_text_sensitivity:.4f}")
    else:
        print(f"\n✓ TEXT sentiment is more sensitive to price changes overall")
        print(f"  Average text sensitivity: {avg_text_sensitivity:.4f}")
        print(f"  Average headline sensitivity: {avg_headline_sensitivity:.4f}")

if __name__ == "__main__":
    try:
        analyze_sentiment_sensitivity()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure pandas and scipy are installed: pip install pandas scipy")
