# Multi-Source Business Entity Resolution using Machine Learning

## 1. Project overview
This project is a Flask-based application for matching business records across multiple noisy data sources using machine learning and fuzzy similarity. The application can be used locally in VS Code and works with uploaded CSV files from Source 1, Source 2, and Source 3.

## 2. Problem statement
Business identity data often arrives from multiple sources with inconsistent names, addresses, phones, emails, and websites. This project identifies which records from Source 2 and Source 3 refer to the same real-world entity as records in Source 1.

## 3. Features
- CSV upload for Source 1, Source 2, and Source 3
- Intelligent column mapping for business fields
- Field normalization and preprocessing
- Candidate generation with blocking strategies
- Similarity feature extraction with RapidFuzz
- Weak supervision when labels are not available
- Random forest classification with optional XGBoost support
- Results export to CSV
- Unmatched record export
- Analytics dashboard with charts
- Manual review for uncertain matches

## 4. Architecture
The system is modular and organized into separate Python files for preprocessing, candidate generation, feature engineering, model training, evaluation, and entity matching.

## 5. Technologies
- Python 3.11+
- Flask
- Pandas
- NumPy
- Scikit-learn
- RapidFuzz
- Joblib
- Matplotlib
- Plotly
- XGBoost (optional)

## 6. Folder structure
```text
entity_resolution_app/
├── app.py
├── requirements.txt
├── README.md
├── config.py
├── data/
│   ├── source1.csv
│   ├── source2.csv
│   └── source3.csv
├── models/
│   └── entity_resolution_model.pkl
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── similarity.py
│   ├── candidate_generation.py
│   ├── feature_engineering.py
│   ├── model.py
│   ├── entity_matching.py
│   └── evaluation.py
├── templates/
│   ├── index.html
│   ├── upload.html
│   ├── results.html
│   ├── analytics.html
│   └── error.html
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
└── output/
    ├── matches.csv
    └── unmatched.csv
```

## 7. Installation
Open a terminal in VS Code and run the following commands:

```powershell
cd d:\ML_Project\entity_resolution_app
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 8. Virtual environment setup
```powershell
python -m venv venv
venv\Scripts\activate
```

## 9. Running the application
```powershell
python app.py
```
Then open the browser at:

[http://127.0.0.1:5000](http://127.0.0.1:5000/)

## 10. Dataset format
The application expects three CSV files:
- Source 1: reference business list
- Source 2: additional business records
- Source 3: additional business records

Typical columns may include:
- business_name
- name
- company_name
- address
- city
- state
- postal_code
- zip
- phone
- email
- website

The preprocessing layer detects equivalent fields automatically.

## 11. ML methodology
The application trains a machine learning classifier using numerical similarity features such as:
- name_similarity
- address_similarity
- phone_similarity
- email_similarity
- website_similarity
- city_similarity
- postal_similarity

It supports supervised training when a label column is found and weak supervision otherwise.

## 12. Entity matching workflow
1. Read the uploaded CSV files.
2. Normalize and standardize fields.
3. Generate candidate records using blocking.
4. Extract similarity features for each candidate pair.
5. Train or load the model.
6. Predict match probability.
7. Apply thresholds.
8. Save the results and unmatched records.

## 13. Evaluation metrics
If label data is available, the application calculates:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

When no genuine labels are present, the application uses weak labels and clearly labels them as weak supervision.

## 14. Example input
Sample datasets are included in the data folder. They contain exact matches, fuzzy matches, missing values, and unmatched businesses.

## 15. Example output
```csv
source1_id,source2_id,source3_id,overall_probability,match_status
S1_001,S2_101,,0.94,HIGH_CONFIDENCE
S1_002,,,0.21,NO_MATCH
```

## 16. Troubleshooting
- If a CSV is empty, the app will show a friendly error.
- If a column is missing, the code fills the value with nulls instead of crashing.
- If an uploaded file is not CSV, the form blocks it.
- If the model cannot train on weak labels, the app returns a helpful message.

## 17. Future improvements
- Add more advanced blocking strategies
- Move to distributed processing with Spark
- Add active learning and human-in-the-loop review
- Expand feature engineering with geocoding and entity embeddings

## 18. License
This project is intended for local development and experimentation in VS Code.
