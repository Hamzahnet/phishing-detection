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

## Dissertation

**Title:** Phishing Website Detection Using Machine Learning  
**Supervisor:** Dr Hamza Mutaher  
**Institution:** Birmingham City University  
**Module:** CMP6200 — Individual Honours Project

## Tech stack

Python · NumPy · scikit-learn (coming soon) · Git
