# 🏥 AI Drug Recommendation System

An intelligent drug recommendation system powered by machine learning that predicts drug ratings, provides personalized recommendations, and enables drug comparisons based on patient conditions and reviews.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Machine Learning Models](#machine-learning-models)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Endpoints](#api-endpoints)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Contributing](#contributing)

## 🎯 Overview

This system leverages natural language processing (NLP) and machine learning to analyze drug reviews and provide intelligent recommendations. It helps patients and healthcare professionals make informed decisions by:

- Predicting sentiment (positive/negative) of drug reviews
- Recommending top-rated drugs for specific medical conditions
- Comparing multiple drugs side-by-side
- Analyzing side effects and user experiences

## ✨ Features

### 1. **Sentiment Prediction**
- Analyzes drug reviews using three ML models (Random Forest, Decision Tree, SVM)
- Provides confidence scores for predictions
- Handles negation patterns and strong negative indicators
- Real-time text preprocessing and analysis

### 2. **Drug Recommendation**
- Multi-condition support (up to 3 conditions simultaneously)
- Identifies common drugs effective for multiple conditions
- Ranks drugs based on:
  - Positive review ratio
  - Average rating (out of 10)
  - Total user votes
  - Composite recommendation score

### 3. **Drug Comparison**
- Side-by-side comparison of 2-3 drugs
- Visual charts for easy comparison
- Pros and cons analysis
- Highlights the best-performing drug

### 4. **Side Effects Analysis**
- Displays top 5 side effects for each drug
- Shows percentage of users experiencing each side effect
- Mention count from user reviews

## 🛠️ Technology Stack

### Backend
- **Flask** - Web framework
- **Python 3.8+** - Programming language
- **scikit-learn** - Machine learning library
- **NLTK** - Natural language processing
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing

### Frontend
- **HTML5/CSS3** - Structure and styling
- **JavaScript** - Interactive features
- **Chart.js** - Data visualization (for comparisons)

### ML/NLP Tools
- **TF-IDF Vectorizer** - Text feature extraction
- **Porter Stemmer** - Word stemming
- **Stopwords Removal** - Text preprocessing

## 🤖 Machine Learning Models

### 1. Random Forest Classifier (Primary Model)
- **Accuracy**: ~92%
- **Configuration**: 100 estimators, max_depth=20
- **Use Case**: Main prediction model with probability estimates

### 2. Decision Tree Classifier
- **Accuracy**: ~88%
- **Configuration**: max_depth=20
- **Use Case**: Fast predictions, interpretable results

### 3. Support Vector Machine (SVM)
- **Accuracy**: ~91%
- **Configuration**: LinearSVC with calibration
- **Use Case**: High-dimensional text classification

### Text Preprocessing Pipeline
1. Lowercase conversion
2. Special character removal
3. Tokenization
4. Stopword removal
5. Porter stemming
6. TF-IDF vectorization (5000 features)

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/Ravin2313/ai-drug-recommendation-system.git
cd ai-drug-recommendation-system
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data
```python
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### Step 5: Prepare Dataset
Place the following CSV files in the project root:
- `drugsComTrain_raw.csv`
- `drugsComTest_raw.csv`
- `drug_side_effects.csv` (optional)

### Step 6: Train Models
```bash
python train_models.py
```

This will generate:
- `random_forest_model.pkl`
- `decision_tree_model.pkl`
- `svm_model.pkl`
- `vectorizer.pkl`
- `recommendation_data.csv`

### Step 7: Run the Application
```bash
python app.py
```

Visit `http://localhost:8080` in your browser.

## 🚀 Usage

### Home Page
Navigate through three main features:
1. **Predict** - Analyze drug review sentiment
2. **Recommend** - Get drug recommendations for conditions
3. **Compare** - Compare multiple drugs side-by-side

### Predict Sentiment
1. Go to "Predict" page
2. Enter a drug review text
3. Click "Predict"
4. View predictions from all three models with confidence scores

### Get Recommendations
1. Go to "Recommend" page
2. Select 1-3 medical conditions from dropdowns
3. Click "Get Recommendations"
4. View:
   - Common drugs (effective for all selected conditions)
   - Top 5 drugs for each condition
   - Side effects, ratings, and user votes

### Compare Drugs
1. Go to "Compare" page
2. Select a medical condition
3. Choose 2-3 drugs from the dropdown
4. Click "Compare"
5. View side-by-side comparison with charts

## 📁 Project Structure

```
ai-drug-recommendation-system/
│
├── app.py                          # Main Flask application
├── train_models.py                 # Model training script
├── extract_side_effects.py         # Side effects extraction
├── requirements.txt                # Python dependencies
│
├── templates/                      # HTML templates
│   ├── home.html                   # Landing page
│   ├── predict.html                # Sentiment prediction page
│   ├── recommend.html              # Recommendation page
│   ├── results.html                # Recommendation results
│   ├── compare.html                # Drug comparison page
│   └── comparison_results.html     # Comparison results
│
├── *.pkl                           # Trained models (generated)
├── *.csv                           # Dataset files
│
└── README.md                       # Project documentation
```

## 🔌 API Endpoints

### GET Endpoints
- `/` - Home page
- `/predict_page` - Sentiment prediction page
- `/recommend_page` - Drug recommendation page
- `/compare_page` - Drug comparison page

### POST Endpoints
- `/predict` - Predict review sentiment
  - **Input**: `review_text` (form data)
  - **Output**: JSON with predictions and confidence

- `/recommend` - Get drug recommendations
  - **Input**: `condition1`, `condition2`, `condition3` (form data)
  - **Output**: HTML with recommendations

- `/compare` - Compare drugs
  - **Input**: `condition`, `drug1`, `drug2`, `drug3` (form data)
  - **Output**: HTML with comparison charts

- `/get_drugs_by_condition` - Get drugs for a condition
  - **Input**: `condition` (JSON)
  - **Output**: JSON array of drug names

## 📊 Dataset

### Source
- **drugsComTrain_raw.csv** - Training data
- **drugsComTest_raw.csv** - Testing data

### Features
- `drugName` - Name of the drug
- `condition` - Medical condition treated
- `review` - User review text
- `rating` - User rating (1-10)
- `date` - Review date
- `usefulCount` - Number of users who found review helpful

### Preprocessing
- Sentiment labels: Rating ≥ 6 = Positive, < 6 = Negative
- Text cleaning: Lowercase, remove special chars, stemming
- Feature extraction: TF-IDF with 5000 features

### Recommendation Score Formula
```
rec_score = (positive_ratio × 0.5) + (avg_rating/10 × 0.3) + (normalized_votes × 0.2)
```

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 92.3% | 0.93 | 0.91 | 0.92 |
| Decision Tree | 88.1% | 0.89 | 0.87 | 0.88 |
| SVM | 91.5% | 0.92 | 0.90 | 0.91 |

### Key Insights
- Random Forest performs best overall
- Rule-based negation handling improves accuracy by ~3%
- TF-IDF with 5000 features provides optimal balance

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available for educational purposes.

## 👥 Authors

- **Ravin** - [GitHub Profile](https://github.com/Ravin2313)

## 🙏 Acknowledgments

- Dataset: UCI Machine Learning Repository (Drugs.com reviews)
- Flask framework and scikit-learn library
- NLTK for natural language processing tools

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: This system is for educational and informational purposes only. Always consult healthcare professionals for medical advice.
