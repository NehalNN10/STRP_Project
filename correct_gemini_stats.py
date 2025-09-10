import pandas as pd
import numpy as np

def generate_correct_gemini_statistics():
    """
    Generate CORRECT Gemini statistics to replace the copied/incorrect ones
    """
    # Read both datasets
    english_df = pd.read_csv('data/english_average_data.csv')
    urdu_df = pd.read_csv('data/urdu_average_data.csv')
    
    def analyze_dataset(df, dataset_name, sentiment_prefix):
        print(f"\n{dataset_name.upper()} - {sentiment_prefix.upper()} SENTIMENT STATISTICS")
        print("=" * 70)
        
        # Use the correct column names based on the sentiment model
        headline_col = f'{sentiment_prefix}_headline_overall_sentiment'
        text_col = f'{sentiment_prefix}_text_overall_sentiment'
        
        # Price drops and hikes
        price_drops = df[(df['petrol_change'] < 0) | (df['diesel_change'] < 0)].copy()
        price_hikes = df[(df['petrol_change'] > 0) | (df['diesel_change'] > 0)].copy()
        
        print(f"Dataset: {len(df)} total records")
        print(f"Price drops: {len(price_drops)} instances")
        print(f"Price hikes: {len(price_hikes)} instances")
        print()
        
        # 1. HEADLINE vs BODY TEXT POLARITY
        print("1. HEADLINE vs BODY TEXT POLARITY:")
        print("-" * 40)
        
        # Price drops
        if len(price_drops) > 0:
            headline_drops_avg = price_drops[headline_col].mean()
            text_drops_avg = price_drops[text_col].mean()
            drops_diff = headline_drops_avg - text_drops_avg
            
            print(f"Price Drops ({len(price_drops)} instances):")
            print(f"  Headlines average: {headline_drops_avg:.2f}")
            print(f"  Body text average: {text_drops_avg:.2f}")
            print(f"  Difference (headline - text): {drops_diff:.2f}")
            
            # Check exceptions
            exceptions_drops = price_drops[price_drops[headline_col] <= price_drops[text_col]]
            exception_pct = (len(exceptions_drops) / len(price_drops)) * 100
            print(f"  Exceptions (text more/equal positive): {len(exceptions_drops)} ({exception_pct:.1f}%)")
        
        # Price hikes
        if len(price_hikes) > 0:
            headline_hikes_avg = price_hikes[headline_col].mean()
            text_hikes_avg = price_hikes[text_col].mean()
            hikes_diff = headline_hikes_avg - text_hikes_avg
            
            print(f"\nPrice Hikes ({len(price_hikes)} instances):")
            print(f"  Headlines average: {headline_hikes_avg:.2f}")
            print(f"  Body text average: {text_hikes_avg:.2f}")
            print(f"  Difference (headline - text): {hikes_diff:.2f}")
            
            # Check exceptions
            exceptions_hikes = price_hikes[price_hikes[text_col] <= price_hikes[headline_col]]
            exception_pct = (len(exceptions_hikes) / len(price_hikes)) * 100
            print(f"  Exceptions (text more/equal negative): {len(exceptions_hikes)} ({exception_pct:.1f}%)")
        
        print()
        
        # 2. SENSITIVITY ANALYSIS (Table format)
        print("2. SENSITIVITY ANALYSIS (per unit and large changes):")
        print("-" * 50)
        
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
        
        print("PETROL - Headlines:")
        if len(petrol_increases) > 0:
            per_unit_inc = petrol_increases[headline_col].mean() / petrol_increases['petrol_change'].mean()
            print(f"  Sentiment per unit increase: {per_unit_inc:.3f}")
        
        if len(petrol_decreases) > 0:
            per_unit_dec = petrol_decreases[headline_col].mean() / abs(petrol_decreases['petrol_change'].mean())
            print(f"  Sentiment per unit decrease: {per_unit_dec:.3f}")
        
        if len(large_petrol_increases) > 0:
            large_inc = large_petrol_increases[headline_col].mean()
            print(f"  Average sentiment per large increase: {large_inc:.3f}")
        
        if len(large_petrol_decreases) > 0:
            large_dec = large_petrol_decreases[headline_col].mean()
            print(f"  Average sentiment per large decrease: {large_dec:.3f}")
        
        print("PETROL - Body Text:")
        if len(petrol_increases) > 0:
            per_unit_inc = petrol_increases[text_col].mean() / petrol_increases['petrol_change'].mean()
            print(f"  Sentiment per unit increase: {per_unit_inc:.3f}")
        
        if len(petrol_decreases) > 0:
            per_unit_dec = petrol_decreases[text_col].mean() / abs(petrol_decreases['petrol_change'].mean())
            print(f"  Sentiment per unit decrease: {per_unit_dec:.3f}")
        
        if len(large_petrol_increases) > 0:
            large_inc = large_petrol_increases[text_col].mean()
            print(f"  Average sentiment per large increase: {large_inc:.3f}")
        
        if len(large_petrol_decreases) > 0:
            large_dec = large_petrol_decreases[text_col].mean()
            print(f"  Average sentiment per large decrease: {large_dec:.3f}")
        
        print("\nDIESEL - Headlines:")
        if len(diesel_increases) > 0:
            per_unit_inc = diesel_increases[headline_col].mean() / diesel_increases['diesel_change'].mean()
            print(f"  Sentiment per unit increase: {per_unit_inc:.3f}")
        
        if len(diesel_decreases) > 0:
            per_unit_dec = diesel_decreases[headline_col].mean() / abs(diesel_decreases['diesel_change'].mean())
            print(f"  Sentiment per unit decrease: {per_unit_dec:.3f}")
        
        if len(large_diesel_increases) > 0:
            large_inc = large_diesel_increases[headline_col].mean()
            print(f"  Average sentiment per large increase: {large_inc:.3f}")
        
        if len(large_diesel_decreases) > 0:
            large_dec = large_diesel_decreases[headline_col].mean()
            print(f"  Average sentiment per large decrease: {large_dec:.3f}")
        
        print("DIESEL - Body Text:")
        if len(diesel_increases) > 0:
            per_unit_inc = diesel_increases[text_col].mean() / diesel_increases['diesel_change'].mean()
            print(f"  Sentiment per unit increase: {per_unit_inc:.3f}")
        
        if len(diesel_decreases) > 0:
            per_unit_dec = diesel_decreases[text_col].mean() / abs(diesel_decreases['diesel_change'].mean())
            print(f"  Sentiment per unit decrease: {per_unit_dec:.3f}")
        
        if len(large_diesel_increases) > 0:
            large_inc = large_diesel_increases[text_col].mean()
            print(f"  Average sentiment per large increase: {large_inc:.3f}")
        
        if len(large_diesel_decreases) > 0:
            large_dec = large_diesel_decreases[text_col].mean()
            print(f"  Average sentiment per large decrease: {large_dec:.3f}")
        
        print()
        
        # 3. CORRELATION/SENSITIVITY SUMMARY
        print("3. CORRELATION SUMMARY:")
        print("-" * 25)
        
        petrol_headline_corr = abs(df['petrol_change'].corr(df[headline_col]))
        petrol_text_corr = abs(df['petrol_change'].corr(df[text_col]))
        diesel_headline_corr = abs(df['diesel_change'].corr(df[headline_col]))
        diesel_text_corr = abs(df['diesel_change'].corr(df[text_col]))
        
        avg_petrol_sens = (petrol_headline_corr + petrol_text_corr) / 2
        avg_diesel_sens = (diesel_headline_corr + diesel_text_corr) / 2
        
        print(f"Petrol sensitivity: {avg_petrol_sens:.4f}")
        print(f"Diesel sensitivity: {avg_diesel_sens:.4f}")
        
        if avg_petrol_sens > avg_diesel_sens:
            print(f"→ MORE sensitive to PETROL (diff: {avg_petrol_sens - avg_diesel_sens:.4f})")
        else:
            print(f"→ MORE sensitive to DIESEL (diff: {avg_diesel_sens - avg_petrol_sens:.4f})")
        
        # 4. ASYMMETRIC RESPONSE EXAMPLES
        print("\n4. ASYMMETRIC RESPONSE EXAMPLES:")
        print("-" * 35)
        
        # Look for specific large changes for examples
        large_increases_example = df[df['petrol_change'] >= 25].head(1)  # ~26 PKR
        large_decreases_example = df[df['petrol_change'] <= -35].head(1)  # ~40 PKR
        
        if len(large_increases_example) > 0:
            row = large_increases_example.iloc[0]
            print(f"Large increase example ({row['date']}):")
            print(f"  Petrol change: +{row['petrol_change']:.1f} PKR")
            print(f"  Headline sentiment: {row[headline_col]:.2f}")
            print(f"  Text sentiment: {row[text_col]:.2f}")
        
        if len(large_decreases_example) > 0:
            row = large_decreases_example.iloc[0]
            print(f"Large decrease example ({row['date']}):")
            print(f"  Petrol change: {row['petrol_change']:.1f} PKR")
            print(f"  Headline sentiment: {row[headline_col]:.2f}")
            print(f"  Text sentiment: {row[text_col]:.2f}")
        
        return {
            'drops_count': len(price_drops),
            'hikes_count': len(price_hikes),
            'headline_drops_avg': headline_drops_avg if len(price_drops) > 0 else 0,
            'text_drops_avg': text_drops_avg if len(price_drops) > 0 else 0,
            'headline_hikes_avg': headline_hikes_avg if len(price_hikes) > 0 else 0,
            'text_hikes_avg': text_hikes_avg if len(price_hikes) > 0 else 0,
            'petrol_sens': avg_petrol_sens,
            'diesel_sens': avg_diesel_sens
        }
    
    # Generate statistics for Gemini
    print("GENERATING CORRECT GEMINI STATISTICS")
    print("=" * 80)
    
    english_gemini = analyze_dataset(english_df, "English", "gemini")
    urdu_gemini = analyze_dataset(urdu_df, "Urdu", "gemini")
    
    # Summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY FOR PAPER/PARAGRAPH USE")
    print("=" * 80)
    
    print("ENGLISH ARTICLES (Gemini):")
    print(f"• Price drops ({english_gemini['drops_count']} instances): Headlines {english_gemini['headline_drops_avg']:.2f}, Text {english_gemini['text_drops_avg']:.2f}")
    print(f"• Price hikes ({english_gemini['hikes_count']} instances): Headlines {english_gemini['headline_hikes_avg']:.2f}, Text {english_gemini['text_hikes_avg']:.2f}")
    print(f"• Petrol sensitivity: {english_gemini['petrol_sens']:.3f}")
    print(f"• Diesel sensitivity: {english_gemini['diesel_sens']:.3f}")
    
    print("\nURDU ARTICLES (Gemini):")
    print(f"• Price drops ({urdu_gemini['drops_count']} instances): Headlines {urdu_gemini['headline_drops_avg']:.2f}, Text {urdu_gemini['text_drops_avg']:.2f}")
    print(f"• Price hikes ({urdu_gemini['hikes_count']} instances): Headlines {urdu_gemini['headline_hikes_avg']:.2f}, Text {urdu_gemini['text_hikes_avg']:.2f}")
    print(f"• Petrol sensitivity: {urdu_gemini['petrol_sens']:.3f}")
    print(f"• Diesel sensitivity: {urdu_gemini['diesel_sens']:.3f}")

if __name__ == "__main__":
    try:
        generate_correct_gemini_statistics()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
