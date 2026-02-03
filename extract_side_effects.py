"""
Extract Side Effects from Drug Reviews
Analyzes negative reviews to identify common side effects
"""

import pandas as pd
import re
from collections import Counter, defaultdict
import joblib

# Common side effect keywords and patterns
SIDE_EFFECT_KEYWORDS = [
    # Physical symptoms
    'headache', 'migraine', 'dizziness', 'dizzy', 'nausea', 'vomiting', 'vomit',
    'diarrhea', 'constipation', 'stomach pain', 'abdominal pain', 'cramps',
    'fatigue', 'tired', 'exhausted', 'weakness', 'drowsy', 'sleepy', 'insomnia',
    'weight gain', 'weight loss', 'appetite loss', 'increased appetite',
    'dry mouth', 'sweating', 'night sweats', 'hot flashes', 'chills',
    'rash', 'itching', 'hives', 'skin reaction', 'acne', 'hair loss',
    'blurred vision', 'vision problems', 'eye problems',
    'chest pain', 'heart palpitations', 'rapid heartbeat', 'irregular heartbeat',
    'shortness of breath', 'breathing problems', 'cough',
    'muscle pain', 'joint pain', 'back pain', 'body aches',
    'tremors', 'shaking', 'numbness', 'tingling',
    'swelling', 'bloating', 'gas', 'indigestion',
    
    # Mental/Emotional
    'anxiety', 'anxious', 'panic attacks', 'nervousness', 'restless',
    'depression', 'depressed', 'mood swings', 'irritability', 'irritable',
    'anger', 'aggressive', 'confusion', 'memory loss', 'brain fog',
    'nightmares', 'vivid dreams', 'hallucinations',
    
    # Sexual
    'erectile dysfunction', 'impotence', 'low libido', 'decreased libido',
    'sexual dysfunction', 'difficulty orgasm',
    
    # Severe
    'suicidal thoughts', 'suicidal', 'seizures', 'convulsions',
    'liver damage', 'kidney problems', 'bleeding', 'blood pressure'
]

def extract_side_effects_from_review(review):
    """Extract side effects mentioned in a review"""
    if pd.isna(review):
        return []
    
    review_lower = str(review).lower()
    found_effects = []
    
    for keyword in SIDE_EFFECT_KEYWORDS:
        # Check if keyword is present
        if keyword in review_lower:
            found_effects.append(keyword)
    
    return found_effects

def analyze_side_effects():
    """Analyze all reviews and extract side effects per drug"""
    print("Loading training data...")
    df = pd.read_csv('drugsComTrain_raw.csv')
    
    print(f"Total reviews: {len(df)}")
    
    # Focus on negative reviews (rating <= 6) for side effects
    negative_reviews = df[df['rating'] <= 6].copy()
    print(f"Negative reviews (rating <= 6): {len(negative_reviews)}")
    
    # Extract side effects for each drug
    drug_side_effects = defaultdict(list)
    
    print("\nExtracting side effects from reviews...")
    for idx, row in negative_reviews.iterrows():
        if idx % 5000 == 0:
            print(f"Processed {idx} reviews...")
        
        drug_name = row['drugName']
        review = row['review']
        
        side_effects = extract_side_effects_from_review(review)
        if side_effects:
            drug_side_effects[drug_name].extend(side_effects)
    
    print(f"\nFound side effects for {len(drug_side_effects)} drugs")
    
    # Calculate frequency and percentage for each drug
    results = []
    
    for drug_name, effects_list in drug_side_effects.items():
        # Count occurrences
        effect_counts = Counter(effects_list)
        
        # Get total reviews for this drug
        total_drug_reviews = len(negative_reviews[negative_reviews['drugName'] == drug_name])
        
        # Get top 10 most common side effects
        top_effects = effect_counts.most_common(10)
        
        for effect, count in top_effects:
            percentage = (count / total_drug_reviews) * 100
            
            results.append({
                'drugName': drug_name,
                'side_effect': effect,
                'mention_count': count,
                'total_negative_reviews': total_drug_reviews,
                'percentage': round(percentage, 2)
            })
    
    # Create DataFrame
    side_effects_df = pd.DataFrame(results)
    
    # Save to CSV
    side_effects_df.to_csv('drug_side_effects.csv', index=False)
    print(f"\nSaved {len(side_effects_df)} side effect records to 'drug_side_effects.csv'")
    
    # Print some statistics
    print("\n=== Statistics ===")
    print(f"Unique drugs with side effects: {side_effects_df['drugName'].nunique()}")
    print(f"Unique side effects found: {side_effects_df['side_effect'].nunique()}")
    print("\nTop 10 most common side effects overall:")
    top_overall = side_effects_df.groupby('side_effect')['mention_count'].sum().sort_values(ascending=False).head(10)
    for effect, count in top_overall.items():
        print(f"  {effect}: {count} mentions")
    
    return side_effects_df

if __name__ == "__main__":
    print("=" * 60)
    print("Side Effects Extraction Tool")
    print("=" * 60)
    
    side_effects_df = analyze_side_effects()
    
    print("\n✅ Side effects extraction completed!")
    print("File created: drug_side_effects.csv")
