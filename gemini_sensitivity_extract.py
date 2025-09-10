import pandas as pd
import numpy as np

def extract_gemini_sensitivity_metrics():
    """
    Extract specific Gemini sentiment per unit metrics for both English and Urdu
    """
    # Read both datasets
    english_df = pd.read_csv('data/english_average_data.csv')
    urdu_df = pd.read_csv('data/urdu_average_data.csv')
    
    def calculate_sensitivity_metrics(df, dataset_name):
        print(f"\n{dataset_name.upper()} - GEMINI SENTIMENT METRICS")
        print("=" * 60)
        
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
        
        print("PETROL SENSITIVITY:")
        print("  Headlines:")
        if len(petrol_increases) > 0:
            sentiment_per_increase = petrol_increases['gemini_headline_overall_sentiment'].mean() / petrol_increases['petrol_change'].mean()
            print(f"    Sentiment per unit increase: {sentiment_per_increase:.3f}")
        
        if len(petrol_decreases) > 0:
            sentiment_per_decrease = petrol_decreases['gemini_headline_overall_sentiment'].mean() / abs(petrol_decreases['petrol_change'].mean())
            print(f"    Sentiment per unit decrease: {sentiment_per_decrease:.3f}")
        
        if len(large_petrol_increases) > 0:
            avg_sentiment_large_inc = large_petrol_increases['gemini_headline_overall_sentiment'].mean()
            print(f"    Average sentiment per large increase: {avg_sentiment_large_inc:.3f}")
        
        if len(large_petrol_decreases) > 0:
            avg_sentiment_large_dec = large_petrol_decreases['gemini_headline_overall_sentiment'].mean()
            print(f"    Average sentiment per large decrease: {avg_sentiment_large_dec:.3f}")
        
        print("  Body Text:")
        if len(petrol_increases) > 0:
            sentiment_per_increase = petrol_increases['gemini_text_overall_sentiment'].mean() / petrol_increases['petrol_change'].mean()
            print(f"    Sentiment per unit increase: {sentiment_per_increase:.3f}")
        
        if len(petrol_decreases) > 0:
            sentiment_per_decrease = petrol_decreases['gemini_text_overall_sentiment'].mean() / abs(petrol_decreases['petrol_change'].mean())
            print(f"    Sentiment per unit decrease: {sentiment_per_decrease:.3f}")
        
        if len(large_petrol_increases) > 0:
            avg_sentiment_large_inc = large_petrol_increases['gemini_text_overall_sentiment'].mean()
            print(f"    Average sentiment per large increase: {avg_sentiment_large_inc:.3f}")
        
        if len(large_petrol_decreases) > 0:
            avg_sentiment_large_dec = large_petrol_decreases['gemini_text_overall_sentiment'].mean()
            print(f"    Average sentiment per large decrease: {avg_sentiment_large_dec:.3f}")
        
        print("\nDIESEL SENSITIVITY:")
        print("  Headlines:")
        if len(diesel_increases) > 0:
            sentiment_per_increase = diesel_increases['gemini_headline_overall_sentiment'].mean() / diesel_increases['diesel_change'].mean()
            print(f"    Sentiment per unit increase: {sentiment_per_increase:.3f}")
        
        if len(diesel_decreases) > 0:
            sentiment_per_decrease = diesel_decreases['gemini_headline_overall_sentiment'].mean() / abs(diesel_decreases['diesel_change'].mean())
            print(f"    Sentiment per unit decrease: {sentiment_per_decrease:.3f}")
        
        if len(large_diesel_increases) > 0:
            avg_sentiment_large_inc = large_diesel_increases['gemini_headline_overall_sentiment'].mean()
            print(f"    Average sentiment per large increase: {avg_sentiment_large_inc:.3f}")
        
        if len(large_diesel_decreases) > 0:
            avg_sentiment_large_dec = large_diesel_decreases['gemini_headline_overall_sentiment'].mean()
            print(f"    Average sentiment per large decrease: {avg_sentiment_large_dec:.3f}")
        
        print("  Body Text:")
        if len(diesel_increases) > 0:
            sentiment_per_increase = diesel_increases['gemini_text_overall_sentiment'].mean() / diesel_increases['diesel_change'].mean()
            print(f"    Sentiment per unit increase: {sentiment_per_increase:.3f}")
        
        if len(diesel_decreases) > 0:
            sentiment_per_decrease = diesel_decreases['gemini_text_overall_sentiment'].mean() / abs(diesel_decreases['diesel_change'].mean())
            print(f"    Sentiment per unit decrease: {sentiment_per_decrease:.3f}")
        
        if len(large_diesel_increases) > 0:
            avg_sentiment_large_inc = large_diesel_increases['gemini_text_overall_sentiment'].mean()
            print(f"    Average sentiment per large increase: {avg_sentiment_large_inc:.3f}")
        
        if len(large_diesel_decreases) > 0:
            avg_sentiment_large_dec = large_diesel_decreases['gemini_text_overall_sentiment'].mean()
            print(f"    Average sentiment per large decrease: {avg_sentiment_large_dec:.3f}")
        
        print(f"\nCounts:")
        print(f"  Petrol increases: {len(petrol_increases)}, Large: {len(large_petrol_increases)}")
        print(f"  Petrol decreases: {len(petrol_decreases)}, Large: {len(large_petrol_decreases)}")
        print(f"  Diesel increases: {len(diesel_increases)}, Large: {len(large_diesel_increases)}")
        print(f"  Diesel decreases: {len(diesel_decreases)}, Large: {len(large_diesel_decreases)}")
    
    # Calculate for both datasets
    calculate_sensitivity_metrics(english_df, "English")
    calculate_sensitivity_metrics(urdu_df, "Urdu")

if __name__ == "__main__":
    extract_gemini_sensitivity_metrics()
