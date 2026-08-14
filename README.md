# Phishing Website Detection Using Machine Learning

A machine learning system that detects phishing websites using 
URL-based feature analysis. Built as part of my BSc Cyber Security 
Honours dissertation at Birmingham City University.

## How It Works

The system analyses URLs by extracting numerical features that 
distinguish phishing sites from legitimate ones — such as URL length, 
presence of HTTPS, number of special characters, and subdomain depth.
These features are passed through a logistic regression classifier 
that outputs a prediction with a confidence score.

## Models

### Prototype Model (train_model.py)
Built from scratch using manual feature engineering. Extracts 6 
lexical features from any raw URL and classifies it as phishing 
or legitimate. Demonstrates an end-to-end pipeline from a raw 
URL string to a real-time prediction.

**Features extracted:**
- URL length
- Number of dots
- HTTPS presence (1 or 0)
- Number of hyphens
- Number of special characters (@, ?, =, &)
- Number of digits

**Results:** 100% accuracy on prototype dataset of 16 labelled URLs.

### Production Model (train_real.py)
Trained on the Kaggle Web Page Phishing Detection Dataset — 
11,430 real-world URLs with 87 pre-extracted features. 
StandardScaler normalisation applied after train/test split 
to prevent data leakage.

**Results:** 96% accuracy | 95% precision | 96% recall

## Run With Docker

No Python installation required.

### Build the container

### Predict a URL

## Tech Stack
- Python
- scikit-learn
- pandas
- NumPy
- Docker

## Dataset
Kaggle Web Page Phishing Detection Dataset — 11,430 URLs, 
87 features, balanced 50/50 phishing and legitimate.