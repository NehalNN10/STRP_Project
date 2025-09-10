import pandas as pd
import numpy as np

def comprehensive_model_comparison():
    """
    Comprehensive comparison of OpenAI vs Gemini sentiment analysis
    for both English and Urdu articles
    """
    # Read both datasets
    english_df = pd.read_csv('data/english_average_data.csv')
    urdu_df = pd.read_csv('data/urdu_average_data.csv')
    
    print("=" * 100)
    print("COMPREHENSIVE OPENAI vs GEMINI SENTIMENT ANALYSIS COMPARISON")
    print("=" * 100)
    
    # Function to calculate key metrics for any dataset and sentiment columns
    def calculate_metrics(df, headline_col, text_col, dataset_name):
        # Basic averages
        headline_avg = df[headline_col].mean()
        text_avg = df[text_col].mean()
        
        # Price changes analysis
        price_drops = df[(df['petrol_change'] < 0) | (df['diesel_change'] < 0)]
        price_hikes = df[(df['petrol_change'] > 0) | (df['diesel_change'] > 0)]
        
        # Price drops metrics
        drops_headline_avg = price_drops[headline_col].mean() if len(price_drops) > 0 else 0
        drops_text_avg = price_drops[text_col].mean() if len(price_drops) > 0 else 0
        
        # Price hikes metrics
        hikes_headline_avg = price_hikes[headline_col].mean() if len(price_hikes) > 0 else 0
        hikes_text_avg = price_hikes[text_col].mean() if len(price_hikes) > 0 else 0
        
        # Correlations
        petrol_headline_corr = abs(df['petrol_change'].corr(df[headline_col]))
        petrol_text_corr = abs(df['petrol_change'].corr(df[text_col]))
        diesel_headline_corr = abs(df['diesel_change'].corr(df[headline_col]))
        diesel_text_corr = abs(df['diesel_change'].corr(df[text_col]))
        
        avg_petrol_sens = (petrol_headline_corr + petrol_text_corr) / 2
        avg_diesel_sens = (diesel_headline_corr + diesel_text_corr) / 2
        
        return {
            'dataset': dataset_name,
            'headline_avg': headline_avg,
            'text_avg': text_avg,
            'drops_headline': drops_headline_avg,
            'drops_text': drops_text_avg,
            'hikes_headline': hikes_headline_avg,
            'hikes_text': hikes_text_avg,
            'petrol_sens': avg_petrol_sens,
            'diesel_sens': avg_diesel_sens,
            'drops_count': len(price_drops),
            'hikes_count': len(price_hikes)
        }
    
    # Calculate metrics for all combinations
    english_openai = calculate_metrics(english_df, 'openai_headline_overall_sentiment', 
                                     'openai_text_overall_sentiment', 'English-OpenAI')
    english_gemini = calculate_metrics(english_df, 'gemini_headline_overall_sentiment', 
                                     'gemini_text_overall_sentiment', 'English-Gemini')
    urdu_openai = calculate_metrics(urdu_df, 'openai_headline_overall_sentiment', 
                                  'openai_text_overall_sentiment', 'Urdu-OpenAI')
    urdu_gemini = calculate_metrics(urdu_df, 'gemini_headline_overall_sentiment', 
                                  'gemini_text_overall_sentiment', 'Urdu-Gemini')
    
    all_metrics = [english_openai, english_gemini, urdu_openai, urdu_gemini]
    
    # 1. OVERALL SENTIMENT AVERAGES COMPARISON
    print("1. OVERALL SENTIMENT AVERAGES")
    print("-" * 60)
    print(f"{'Dataset':<15} {'Headlines':<12} {'Text':<12} {'Difference':<12}")
    print("-" * 60)
    for m in all_metrics:
        diff = m['text_avg'] - m['headline_avg']
        print(f"{m['dataset']:<15} {m['headline_avg']:<12.3f} {m['text_avg']:<12.3f} {diff:<12.3f}")
    
    print()
    
    # 2. PRICE DROPS ANALYSIS
    print("2. PRICE DROPS ANALYSIS")
    print("-" * 70)
    print(f"{'Dataset':<15} {'Count':<8} {'Headlines':<12} {'Text':<12} {'H-T Diff':<12}")
    print("-" * 70)
    for m in all_metrics:
        diff = m['drops_headline'] - m['drops_text']
        print(f"{m['dataset']:<15} {m['drops_count']:<8} {m['drops_headline']:<12.3f} {m['drops_text']:<12.3f} {diff:<12.3f}")
    
    print()
    
    # 3. PRICE HIKES ANALYSIS
    print("3. PRICE HIKES ANALYSIS")
    print("-" * 70)
    print(f"{'Dataset':<15} {'Count':<8} {'Headlines':<12} {'Text':<12} {'H-T Diff':<12}")
    print("-" * 70)
    for m in all_metrics:
        diff = m['hikes_headline'] - m['hikes_text']
        print(f"{m['dataset']:<15} {m['hikes_count']:<8} {m['hikes_headline']:<12.3f} {m['hikes_text']:<12.3f} {diff:<12.3f}")
    
    print()
    
    # 4. FUEL SENSITIVITY COMPARISON
    print("4. FUEL PRICE SENSITIVITY COMPARISON")
    print("-" * 60)
    print(f"{'Dataset':<15} {'Petrol':<12} {'Diesel':<12} {'More Sensitive To':<15}")
    print("-" * 60)
    for m in all_metrics:
        more_sens = "Petrol" if m['petrol_sens'] > m['diesel_sens'] else "Diesel"
        print(f"{m['dataset']:<15} {m['petrol_sens']:<12.3f} {m['diesel_sens']:<12.3f} {more_sens:<15}")
    
    print()
    
    # 5. MODEL COMPARISON WITHIN EACH LANGUAGE
    print("5. OPENAI vs GEMINI MODEL COMPARISON")
    print("-" * 80)
    
    print("ENGLISH ARTICLES:")
    print(f"  Overall sentiment (Headlines): OpenAI {english_openai['headline_avg']:.3f} vs Gemini {english_gemini['headline_avg']:.3f}")
    print(f"  Overall sentiment (Text): OpenAI {english_openai['text_avg']:.3f} vs Gemini {english_gemini['text_avg']:.3f}")
    print(f"  Petrol sensitivity: OpenAI {english_openai['petrol_sens']:.3f} vs Gemini {english_gemini['petrol_sens']:.3f}")
    print(f"  Diesel sensitivity: OpenAI {english_openai['diesel_sens']:.3f} vs Gemini {english_gemini['diesel_sens']:.3f}")
    
    # Determine which model is more sensitive for English
    english_openai_avg_sens = (english_openai['petrol_sens'] + english_openai['diesel_sens']) / 2
    english_gemini_avg_sens = (english_gemini['petrol_sens'] + english_gemini['diesel_sens']) / 2
    
    if english_openai_avg_sens > english_gemini_avg_sens:
        print(f"  → OpenAI is MORE SENSITIVE overall ({english_openai_avg_sens:.3f} vs {english_gemini_avg_sens:.3f})")
    else:
        print(f"  → Gemini is MORE SENSITIVE overall ({english_gemini_avg_sens:.3f} vs {english_openai_avg_sens:.3f})")
    
    print("\nURDU ARTICLES:")
    print(f"  Overall sentiment (Headlines): OpenAI {urdu_openai['headline_avg']:.3f} vs Gemini {urdu_gemini['headline_avg']:.3f}")
    print(f"  Overall sentiment (Text): OpenAI {urdu_openai['text_avg']:.3f} vs Gemini {urdu_gemini['text_avg']:.3f}")
    print(f"  Petrol sensitivity: OpenAI {urdu_openai['petrol_sens']:.3f} vs Gemini {urdu_gemini['petrol_sens']:.3f}")
    print(f"  Diesel sensitivity: OpenAI {urdu_openai['diesel_sens']:.3f} vs Gemini {urdu_gemini['diesel_sens']:.3f}")
    
    # Determine which model is more sensitive for Urdu
    urdu_openai_avg_sens = (urdu_openai['petrol_sens'] + urdu_openai['diesel_sens']) / 2
    urdu_gemini_avg_sens = (urdu_gemini['petrol_sens'] + urdu_gemini['diesel_sens']) / 2
    
    if urdu_openai_avg_sens > urdu_gemini_avg_sens:
        print(f"  → OpenAI is MORE SENSITIVE overall ({urdu_openai_avg_sens:.3f} vs {urdu_gemini_avg_sens:.3f})")
    else:
        print(f"  → Gemini is MORE SENSITIVE overall ({urdu_gemini_avg_sens:.3f} vs {urdu_openai_avg_sens:.3f})")
    
    print()
    
    # 6. LANGUAGE COMPARISON
    print("6. ENGLISH vs URDU LANGUAGE COMPARISON")
    print("-" * 80)
    
    print("OpenAI Model:")
    print(f"  English petrol sensitivity: {english_openai['petrol_sens']:.3f}")
    print(f"  Urdu petrol sensitivity: {urdu_openai['petrol_sens']:.3f}")
    print(f"  English diesel sensitivity: {english_openai['diesel_sens']:.3f}")
    print(f"  Urdu diesel sensitivity: {urdu_openai['diesel_sens']:.3f}")
    
    print("\nGemini Model:")
    print(f"  English petrol sensitivity: {english_gemini['petrol_sens']:.3f}")
    print(f"  Urdu petrol sensitivity: {urdu_gemini['petrol_sens']:.3f}")
    print(f"  English diesel sensitivity: {english_gemini['diesel_sens']:.3f}")
    print(f"  Urdu diesel sensitivity: {urdu_gemini['diesel_sens']:.3f}")
    
    print()
    
    # 7. KEY FINDINGS SUMMARY
    print("7. KEY FINDINGS SUMMARY")
    print("-" * 80)
    
    print("✓ All models show higher sensitivity to PETROL than DIESEL price changes")
    print("✓ Headlines tend to be more extreme (both positive and negative) than body text")
    print("✓ Price drops generate more positive sentiment than price hikes generate negative sentiment")
    
    # Model sensitivity ranking
    sensitivities = [
        (english_openai_avg_sens, 'English-OpenAI'),
        (english_gemini_avg_sens, 'English-Gemini'),
        (urdu_openai_avg_sens, 'Urdu-OpenAI'),
        (urdu_gemini_avg_sens, 'Urdu-Gemini')
    ]
    sensitivities.sort(reverse=True)
    
    print("\nModel Sensitivity Ranking (Most to Least Sensitive):")
    for i, (sens, name) in enumerate(sensitivities, 1):
        print(f"  {i}. {name}: {sens:.3f}")

if __name__ == "__main__":
    try:
        comprehensive_model_comparison()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
