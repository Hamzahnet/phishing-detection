# Phishing Detection — Machine Learning Project

A machine learning project for detecting phishing websites through URL feature extraction and classification. Built as part of my BSc Cyber Security dissertation at Birmingham City University.

## What it does

Analyses URLs and extracts numerical features to classify them as phishing or legitimate. The feature extraction pipeline feeds into a logistic regression classifier trained on labelled URL datasets.

## Features extracted from URLs

- URL length
- Number of dots (subdomain indicators)
- HTTPS check (encrypted vs unencrypted)
- Number of hyphens
- Number of special characters (@, ?, =, &)
- Number of digits

## Files

- `Day2.py` — URL feature extractor, normalisation pipeline, labelled dataset builder
- `Neuron.py` — Single neuron implementation from scratch using NumPy
- `create_dataset.py` — Creates a labelled CSV dataset of phishing and legitimate URLs
- `train_model.py` — Trains a logistic regression classifier, evaluates accuracy and predicts new URLs
- `urls.csv` — Labelled dataset of 16 URLs (8 legitimate, 8 phishing)

## Results so far

- Model accuracy: 100% on test set
- Predicts phishing URLs with 99.97% confidence
- Next step: train on real dataset from PhishTank or UCI ML Repository

## Run with Docker

No Python installation required.

### Build the container
docker build -t phishing-detector .

### Predict a URL
docker run phishing-detector python predict.py "https://www.google.com"
docker run phishing-detector python predict.py "http://suspicious-site.com/login?user=admin"

### Output
URL: https://www.google.com
Prediction: Legitimate
Confidence: 99.68%

## Dissertation

**Title:** Phishing Website Detection Using Machine Learning
**Supervisor:** Dr Hamza Mutaher
**Institution:** Birmingham City University


## Tech stack

Python · NumPy · scikit-learn · Git





