import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


#importing features extractor from Day 2
def extract_features(url):
    length = len(url)
    dot = url.count('.')
    httpcheck = 1 if url.startswith('https') else 0
    special = sum(url.count(c) for c in ['@', '?', '=', '&'])
    numdig = sum(1 for c in url if c.isdigit())
    hyphen = url.count('-')
    return [length, dot, httpcheck, special, numdig, hyphen]

#load dataset
df = pd.read_csv('urls.csv')

#extracting features from all URLS
X = np.array([extract_features(url) for url in df['url']])
Y = np.array(df['label'])

#split into training test sets 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#train logistic regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

#evaluation
y_pred  = model.predict(X_test)
print(f"Accuracy: {accuracy_score(Y_test, y_pred):.2f}")
print(classification_report(Y_test , y_pred, target_names=['legitimate', 'phishing']))

#test on a new  url
test_url = "http://paypal-verify.suspicious.com/login?user=admin"
features = np.array([extract_features(test_url)])
prediction = model.predict(features)
probability = model.predict_proba(features)
print(f"\nTest URL: {test_url}")
print(f"Prediction: {'Phishing' if prediction[0] == 1 else 'Legitimate'}")
print(f"Confidence: {max(probability[0]):.2%}")