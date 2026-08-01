# Phishing Detection — Machine Learning Project

A machine learning project for detecting phishing websites through URL feature extraction and classification. Built as part of my BSc Cyber Security dissertation at Birmingham City University.

## What it does

Analyses URLs and extracts numerical features to classify them as phishing or legitimate. The feature extraction pipeline feeds into a machine learning classifier trained on labelled URL datasets.

## Features extracted from URLs

- URL length
- Number of dots (subdomain indicators)
- HTTPS check (encrypted vs unencrypted)
- Number of hyphens
- Number of special characters (@, ?, =, &)
- Number of digits

## Current components

- `Day2.py` — URL feature extractor, normalisation pipeline, labelled dataset builder
- `Neuron.py` — Single neuron implementation from scratch using NumPy

## Run with Docker

No Python installation required.

### Build the container
docker build -t phishing-detector .

### Predict a URL
docker run phishing-detector python predict.py "https://www.google.com"
docker run phishing-detector python predict.py "http://suspicious-site.com/login?user=admin"

### Output
URL: https://www.google.com
Prediction: ✅ Legitimate
Confidence: 99.68%

## Dissertation

**Title:** Phishing Website Detection Using Machine Learning  
**Supervisor:** Dr Hamza Mutaher  
**Institution:** Birmingham City University  
**Module:** CMP6200 — Individual Honours Project

## Tech stack

Python · NumPy · scikit-learn (coming soon) · Git
