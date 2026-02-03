"""
Resume Training - Only train missing models
Checks which models already exist and trains only the remaining ones
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import warnings
import os
warnings.filterwarnings('ignore')

# Initialize
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Simple text preprocessing"""
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    
    return ' '.join(words)

def check_existing_models():
    """Check which models already exist"""
    models_status = {
        'Decision Tree': os.path.exists('decision_tree_model.pkl'),
        'SVM': os.path.exists('svm_model.pkl'),
        'Random Forest': os.path.exists('random_forest_model.pkl')
    }
    
    vectorizer_exists = os.path.exists('vectorizer.pkl')
    
    return models_status, vectorizer_exists

def load_and_prepare_data():
    """Load and prepare the dataset"""
    print("Loading data...")
    
    train_df = pd.read_csv('drugsComTrain_raw.csv')
    test_df = pd.read_csv('drugsComTest_raw.csv')
    
    df = pd.concat([train_df, test_df], ignore_index=True)
    
    print(f"Total records: {len(df)}")
    
    df = df.dropna(subset=['review', 'condition', 'rating'])
    
    print(f"Records after cleaning: {len(df)}")
    
    df['sentiment'] = (df['rating'] >= 6).astype(int)
    
    print(f"Positive reviews: {df['sentiment'].sum()}")
    print(f"Negative reviews: {len(df) - df['sentiment'].sum()}")
    
    print("\nPreprocessing text...")
    df['cleaned_review'] = df['review'].apply(preprocess_text)
    
    return df

if __name__ == "__main__":
    print("="*50)
    print("RESUME TRAINING - TRAIN REMAINING MODELS")
    print("="*50)
    
    # Check existing models
    models_status, vectorizer_exists = check_existing_models()
    
    print("\nExisting Models:")
    for name, exists in models_status.items():
        status = "✅ Found" if exists else "❌ Missing"
        print(f"{name}: {status}")
    
    print(f"Vectorizer: {'✅ Found' if vectorizer_exists else '❌ Missing'}")
    
    # Check if all models exist
    if all(models_status.values()) and vectorizer_exists:
        print("\n🎉 All models already trained!")
        print("Run 'python app.py' to start the application")
        exit(0)
    
    # Load data
    df = load_and_prepare_data()
    
    # Prepare features
    X = df['cleaned_review']
    y = df['sentiment']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")
    
    # Load or create vectorizer
    if vectorizer_exists:
        print("\nLoading existing vectorizer...")
        vectorizer = joblib.load('vectorizer.pkl')
        X_train_vec = vectorizer.transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
    else:
        print("\nCreating new vectorizer...")
        vectorizer = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.8)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        joblib.dump(vectorizer, 'vectorizer.pkl')
        print("Vectorizer saved!")
    
    # Define all models
    all_models = {
        'Decision Tree': DecisionTreeClassifier(max_depth=20, random_state=42),
        'SVM': CalibratedClassifierCV(LinearSVC(max_iter=1000, random_state=42)),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, 
            max_depth=20, 
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
    }
    
    # Train only missing models
    print("\n" + "="*50)
    print("TRAINING MISSING MODELS")
    print("="*50)
    
    for name, model in all_models.items():
        if models_status[name]:
            print(f"\n⏭️  Skipping {name} (already exists)")
            continue
        
        print(f"\n{'='*50}")
        print(f"Training {name}...")
        print(f"{'='*50}")
        
        model.fit(X_train_vec, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_vec)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nAccuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
        
        # Save model
        filename = name.lower().replace(' ', '_') + '_model.pkl'
        joblib.dump(model, filename)
        print(f"✅ Model saved as {filename}")
    
    # Create recommendation data if needed
    if not os.path.exists('recommendation_data.csv'):
        print("\n" + "="*50)
        print("CREATING RECOMMENDATION DATA")
        print("="*50)
        
        recommendation_df = df.groupby(['condition', 'drugName']).agg({
            'sentiment': 'mean',
            'rating': 'mean',
            'usefulCount': 'sum'
        }).reset_index()
        
        recommendation_df.columns = ['condition', 'drugName', 'positive_ratio', 'avg_rating', 'total_votes']
        
        recommendation_df['rec_score'] = (
            recommendation_df['positive_ratio'] * 0.5 + 
            (recommendation_df['avg_rating'] / 10) * 0.3 +
            (recommendation_df['total_votes'] / recommendation_df['total_votes'].max()) * 0.2
        )
        
        recommendation_df.to_csv('recommendation_data.csv', index=False)
        print(f"✅ Recommendation data saved with {len(recommendation_df)} records")
    else:
        print("\n⏭️  Recommendation data already exists")
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    
    # Show final status
    print("\nFinal Status:")
    models_status, vectorizer_exists = check_existing_models()
    for name, exists in models_status.items():
        status = "✅" if exists else "❌"
        print(f"{status} {name}")
    
    print(f"✅ Vectorizer")
    print(f"✅ Recommendation Data")
    
    print("\n🚀 Ready to run: python app.py")
