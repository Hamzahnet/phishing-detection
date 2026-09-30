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

## Model Comparison (compare_models.py)
| Model               | Accuracy | Precision | Recall | F1  |
| Logistic Regression | 96%      | 96%       | 95%    | 96% |
| Random Forest       | 97%      | 97%       | 96%    | 97% |

Random Forest outperforms Logistic Regression across all metrics.
The modest gap suggests Logistic Regression remains the viable interpretable baseline
whilst Random Forest captures non-linier patterns more.

## Adversarial Robustness Testing (adversarial_attack.py)

To evaluate real-world robustness beyond standard accuracy metrics, the 
production Random Forest model was stress-tested against adversarial 
perturbations — simulating how an attacker might modify phishing URL 
characteristics to evade detection.

**Perturbations tested on phishing URLs:**

| Perturbation | URLs Evaded |
|---|---|
| Add HTTPS token | 1 |
| Remove one dot (subdomain reduction) | 4 |
| Remove one hyphen | 0 |
| Combined HTTPS + dot removal | 4 |

**Key finding:** The model's classification is disproportionately 
sensitive to subdomain structure (`nb_dots`), which functions as the 
dominant evasion vector. Superficial changes such as HTTPS presence or 
hyphen count had minimal individual impact, and combining perturbations 
did not compound evasion beyond the dot-reduction effect alone.

This finding informs the next phase of the project: retraining with 
adversarial examples to harden the model against subdomain-based evasion, 
followed by deployment as a real-time Chrome browser extension.

## Dissertation
- **Title:** Phishing Website Detection Using Machine Learning
- **Supervisor:** Dr Hamza Mutaher
- **Institution:** Birmingham City University
- **Module:** CMP6200 — Individual Honours Project
