import numpy as np
import joblib
import sys

def extract_features(url):
    length = len(url)
    dot = url.count('.')
    httpcheck = 1 if url.startswith('https') else 0
    special = sum(url.count(c) for c in ['@', '?', '=', '&'])
    numdig = sum(1 for c in url if c.isdigit())
    hyphen = url.count('-')
    return [length, dot, httpcheck, special, numdig, hyphen]


#loading saved model
model = joblib.load('model.pkl')


#get urls from command line arguments
url = sys.argv[1] if len(sys.argv) > 1 else "http://example.com"

features = np.array([extract_features(url)])
prediction = model.predict(features)
probability = model.predict_proba(features)

print(f"URL: {url}")
print(f"Prediction: {'Phishing' if prediction[0] == 1 else 'Legitimate'}")
print(f"Confidence: {max(probability[0]):.2%}")