import pandas as pd
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

#loading the kaggle dataset
df = pd.read_csv('dataset_phishing.csv')

#seperate features and labels
X = df.drop(columns=['url', 'status'])
y = df['status'].map({'legitimate': 0, 'phishing': 1})

#spliting
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#training
model = LogisticRegression(max_iter=5000)
model.fit(X_train,y_train)

#eval
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred, target_names=['legitimate', 'phishing']))

#saving model
joblib.dump(model, 'model_real.pkl')
print("Real model saved as model_real.pkl")
