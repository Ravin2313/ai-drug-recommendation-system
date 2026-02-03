"""
Simple Flask App for Drug Recommendation
Main model: Random Forest
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

app = Flask(__name__)

# Initialize
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Load models and data
print("Loading models...")
rf_model = joblib.load('random_forest_model.pkl')  # Main model
dt_model = joblib.load('decision_tree_model.pkl')

# Try to load SVM (optional)
try:
    svm_model = joblib.load('svm_model.pkl')
    has_svm = True
    print("SVM model loaded!")
except:
    svm_model = None
    has_svm = False
    print("SVM model not found (skipped)")

vectorizer = joblib.load('vectorizer.pkl')
recommendation_data = pd.read_csv('recommendation_data.csv')

# Load side effects data
try:
    side_effects_data = pd.read_csv('drug_side_effects.csv')
    has_side_effects = True
    print("Side effects data loaded!")
except:
    side_effects_data = None
    has_side_effects = False
    print("Side effects data not found (skipped)")

print("Models loaded successfully!")

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

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_page')
def predict_page():
    return render_template('predict.html')

@app.route('/recommend_page')
def recommend_page():
    # Get unique conditions for dropdown
    conditions = sorted(recommendation_data['condition'].unique())
    return render_template('recommend.html', conditions=conditions)

@app.route('/compare_page')
def compare_page():
    """Show drug comparison page"""
    conditions = sorted(recommendation_data['condition'].unique())
    return render_template('compare.html', conditions=conditions)

@app.route('/get_drugs_by_condition', methods=['POST'])
def get_drugs_by_condition():
    """Get list of drugs for a specific condition"""
    try:
        condition = request.json.get('condition', '').lower()
        
        if not condition:
            return jsonify({'error': 'Please provide a condition'})
        
        # Filter drugs by condition
        condition_drugs = recommendation_data[
            recommendation_data['condition'].str.lower() == condition
        ].copy()
        
        if len(condition_drugs) == 0:
            return jsonify({'drugs': []})
        
        # Get unique drug names sorted by recommendation score
        condition_drugs = condition_drugs.sort_values('rec_score', ascending=False)
        drug_names = condition_drugs['drugName'].unique().tolist()
        
        return jsonify({'drugs': drug_names})
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    """Predict sentiment using Random Forest (main model) with negation handling"""
    try:
        review_text = request.form.get('review_text', '')
        
        if not review_text:
            return jsonify({'error': 'Please provide review text'})
        
        # Rule-based negation check
        review_lower = review_text.lower()
        
        # Strong negative indicators
        strong_negative_words = [
            'terrible', 'horrible', 'worst', 'awful', 'useless', 'waste',
            'disappointed', 'disappointing', 'regret', 'never again',
            'do not recommend', 'would not recommend', 'avoid', 'bad experience'
        ]
        
        # Negation patterns
        negation_patterns = [
            'did not work', 'does not work', 'didn\'t work', 'doesn\'t work',
            'not effective', 'not helpful', 'no relief', 'no improvement',
            'not recommend', 'would not', 'did not help', 'made worse',
            'caused', 'side effects'
        ]
        
        # Count negative indicators
        negative_score = 0
        for word in strong_negative_words:
            if word in review_lower:
                negative_score += 2
        
        for pattern in negation_patterns:
            if pattern in review_lower:
                negative_score += 1
        
        # Preprocess
        cleaned_review = preprocess_text(review_text)
        
        # Vectorize
        review_vec = vectorizer.transform([cleaned_review])
        
        # Predict using all available models
        rf_pred = rf_model.predict(review_vec)[0]
        dt_pred = dt_model.predict(review_vec)[0]
        
        # Get probabilities from Random Forest
        rf_proba = rf_model.predict_proba(review_vec)[0]
        
        # Apply rule-based correction to ALL models
        # If strong negative indicators present, override ML predictions
        if negative_score >= 3:
            # Force all models to predict negative
            rf_pred = 0
            dt_pred = 0
            confidence = 85.0  # High confidence due to clear negative language
        else:
            confidence = max(rf_proba) * 100
        
        # Main prediction from Random Forest (with correction)
        main_prediction = "Positive" if rf_pred == 1 else "Negative"
        
        result = {
            'main_prediction': main_prediction,
            'confidence': f"{confidence:.2f}%",
            'random_forest': "Positive" if rf_pred == 1 else "Negative",
            'decision_tree': "Positive" if dt_pred == 1 else "Negative"
        }
        
        # Add SVM prediction if available (with correction)
        if has_svm:
            svm_pred = svm_model.predict(review_vec)[0]
            # Apply same rule-based correction
            if negative_score >= 3:
                svm_pred = 0
            result['svm'] = "Positive" if svm_pred == 1 else "Negative"
        else:
            result['svm'] = "Not Available"
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/recommend', methods=['POST'])
def recommend():
    """Recommend top drugs for multiple conditions - shows common drugs first"""
    try:
        condition1 = request.form.get('condition1', '').strip().lower()
        condition2 = request.form.get('condition2', '').strip().lower()
        condition3 = request.form.get('condition3', '').strip().lower()
        
        if not condition1:
            return jsonify({'error': 'Please select at least one condition'})
        
        # Collect all conditions
        conditions = [condition1]
        if condition2:
            conditions.append(condition2)
        if condition3:
            conditions.append(condition3)
        
        # Get recommendations for each condition
        all_recommendations = {}
        condition_drug_sets = {}  # To find common drugs
        
        for condition in conditions:
            # Filter by condition
            condition_drugs = recommendation_data[
                recommendation_data['condition'].str.lower() == condition
            ].copy()
            
            if len(condition_drugs) == 0:
                all_recommendations[condition] = []
                condition_drug_sets[condition] = set()
            else:
                # Sort by recommendation score
                condition_drugs = condition_drugs.sort_values('rec_score', ascending=False)
                
                # Get top 10 drugs for finding common ones
                top_drugs = condition_drugs.head(10)
                
                drugs_list = []
                drug_names = set()
                
                for _, row in top_drugs.iterrows():
                    drug_name = row['drugName']
                    
                    # Get side effects for this drug
                    side_effects = []
                    if has_side_effects:
                        drug_side_effects = side_effects_data[
                            side_effects_data['drugName'] == drug_name
                        ].sort_values('mention_count', ascending=False).head(5)
                        
                        for _, se_row in drug_side_effects.iterrows():
                            side_effects.append({
                                'name': se_row['side_effect'],
                                'percentage': se_row['percentage'],
                                'count': se_row['mention_count']
                            })
                    
                    drugs_list.append({
                        'name': drug_name,
                        'positive_ratio': f"{row['positive_ratio']*100:.1f}%",
                        'avg_rating': f"{row['avg_rating']:.1f}/10",
                        'total_votes': int(row['total_votes']),
                        'rec_score': f"{row['rec_score']:.3f}",
                        'rec_score_float': row['rec_score'],
                        'side_effects': side_effects
                    })
                    drug_names.add(drug_name)
                
                all_recommendations[condition] = drugs_list
                condition_drug_sets[condition] = drug_names
        
        # Find common drugs across all conditions
        common_drugs = []
        if len(conditions) > 1:
            # Find intersection of all drug sets
            common_drug_names = condition_drug_sets[conditions[0]]
            for condition in conditions[1:]:
                if len(condition_drug_sets[condition]) > 0:
                    common_drug_names = common_drug_names.intersection(condition_drug_sets[condition])
            
            # Get details of common drugs with average scores
            if len(common_drug_names) > 0:
                for drug_name in common_drug_names:
                    drug_info = {
                        'name': drug_name,
                        'conditions': [],
                        'avg_positive_ratio': 0,
                        'avg_rating': 0,
                        'total_votes': 0,
                        'avg_rec_score': 0,
                        'side_effects': []
                    }
                    
                    total_positive = 0
                    total_rating = 0
                    total_rec_score = 0
                    count = 0
                    
                    for condition in conditions:
                        for drug in all_recommendations[condition]:
                            if drug['name'] == drug_name:
                                drug_info['conditions'].append(condition)
                                total_positive += float(drug['positive_ratio'].rstrip('%'))
                                total_rating += float(drug['avg_rating'].split('/')[0])
                                total_rec_score += drug['rec_score_float']
                                drug_info['total_votes'] += drug['total_votes']
                                
                                # Collect side effects (avoid duplicates)
                                for se in drug['side_effects']:
                                    if se not in drug_info['side_effects']:
                                        drug_info['side_effects'].append(se)
                                
                                count += 1
                                break
                    
                    if count > 0:
                        drug_info['avg_positive_ratio'] = f"{total_positive / count:.1f}%"
                        drug_info['avg_rating'] = f"{total_rating / count:.1f}/10"
                        drug_info['avg_rec_score'] = f"{total_rec_score / count:.3f}"
                        drug_info['avg_rec_score_float'] = total_rec_score / count
                        
                        # Sort side effects by percentage
                        drug_info['side_effects'].sort(key=lambda x: x['percentage'], reverse=True)
                        drug_info['side_effects'] = drug_info['side_effects'][:5]  # Top 5
                        
                        common_drugs.append(drug_info)
                
                # Sort common drugs by average rec_score
                common_drugs.sort(key=lambda x: x['avg_rec_score_float'], reverse=True)
        
        # Limit individual recommendations to top 5
        for condition in conditions:
            all_recommendations[condition] = all_recommendations[condition][:5]
        
        return render_template('results.html', 
                             conditions=conditions,
                             all_recommendations=all_recommendations,
                             common_drugs=common_drugs,
                             has_side_effects=has_side_effects)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/compare', methods=['POST'])
def compare():
    """Compare 2-3 drugs side by side"""
    try:
        condition = request.form.get('condition', '').lower()
        drug1 = request.form.get('drug1', '').strip()
        drug2 = request.form.get('drug2', '').strip()
        drug3 = request.form.get('drug3', '').strip()
        
        if not condition or not drug1 or not drug2:
            return jsonify({'error': 'Please provide condition and at least 2 drugs'})
        
        # Collect drug names
        drug_names = [drug1, drug2]
        if drug3:
            drug_names.append(drug3)
        
        # Filter by condition
        condition_drugs = recommendation_data[
            recommendation_data['condition'].str.lower() == condition
        ].copy()
        
        if len(condition_drugs) == 0:
            return jsonify({'error': 'No drugs found for this condition'})
        
        # Find drugs (case-insensitive)
        compared_drugs = []
        for drug_name in drug_names:
            drug_data = condition_drugs[
                condition_drugs['drugName'].str.lower() == drug_name.lower()
            ]
            
            if len(drug_data) == 0:
                return jsonify({'error': f'Drug "{drug_name}" not found for {condition}'})
            
            drug_row = drug_data.iloc[0]
            
            # Extract pros and cons from reviews (simplified)
            pros = [
                f"High effectiveness with {drug_row['positive_ratio']*100:.1f}% positive reviews",
                f"Average rating of {drug_row['avg_rating']:.1f}/10",
                f"Trusted by {int(drug_row['total_votes'])} users"
            ]
            
            cons = [
                f"Based on {int(drug_row['total_votes'])} reviews only" if drug_row['total_votes'] < 100 else "Large sample size available",
                f"{(1-drug_row['positive_ratio'])*100:.1f}% users reported negative experiences"
            ]
            
            compared_drugs.append({
                'name': drug_row['drugName'],
                'avg_rating': f"{drug_row['avg_rating']:.1f}/10",
                'positive_ratio': f"{drug_row['positive_ratio']*100:.1f}%",
                'total_votes': int(drug_row['total_votes']),
                'rec_score': f"{drug_row['rec_score']:.3f}",
                'pros': pros,
                'cons': cons,
                'is_winner': False
            })
        
        # Mark the best drug (highest rec_score)
        if compared_drugs:
            best_idx = max(range(len(compared_drugs)), 
                          key=lambda i: float(compared_drugs[i]['rec_score']))
            compared_drugs[best_idx]['is_winner'] = True
        
        # Prepare data for charts
        drug_names_chart = [d['name'] for d in compared_drugs]
        ratings = [float(d['avg_rating'].split('/')[0]) for d in compared_drugs]
        positive_ratios = [float(d['positive_ratio'].rstrip('%')) for d in compared_drugs]
        total_votes = [d['total_votes'] for d in compared_drugs]
        rec_scores = [float(d['rec_score']) for d in compared_drugs]
        
        return render_template('comparison_results.html',
                             condition=condition.title(),
                             drugs=compared_drugs,
                             drug_names=drug_names_chart,
                             ratings=ratings,
                             positive_ratios=positive_ratios,
                             total_votes=total_votes,
                             rec_scores=rec_scores)
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
