import pandas as pd 
import joblib 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('dataset_phishing.csv')

x = df.drop(columns=['url', 'status'])
y = df['status'].map({'legitimate': 0, 'phishing': 1})

#first step split first ensures test rows cant be used temporarily
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)
print(f"Train rows (study pile): {len(x_train)}")
print(f"Test rows (not accessible): {len(x_test)}")

#second step make twins from the train phishin rows only
twins = x_train[y_train == 1].copy()
twins['nb_dots'] = twins['nb_dots'].apply(lambda x: max(0, x - 1))
print(f"Twins created (train phishing rows, dot removed): {len(twins)}")

#step 3 stack the twins onto the train pile then retrain model
x_train_aug = pd.concat([x_train, twins], ignore_index=True)
y_train_aug = pd.concat([y_train, pd.Series([1] * len(twins))], ignore_index=True)
print(f"Train rows after adding twins: {len(x_train_aug)}")

rf_hardened = RandomForestClassifier(n_estimators=100, random_state=42)
rf_hardened.fit(x_train_aug, y_train_aug)
print("Hardened model trained")


# fourth step attacking the test phishing rows on both models
rf_original = joblib.load('model_rf.pkl')

x_test_phish = x_test[y_test == 1]
print(f"Test phishing rows: {len(x_test_phish)}")

def count_evasion(model, x_phish, dots_removed=1):
    before = model.predict(x_phish)
    attacked = x_phish.copy()
    attacked['nb_dots'] = attacked['nb_dots'].apply(lambda x: max(0, x - dots_removed))
    after = model.predict(attacked)
    return (( before == 1) & (after == 0)).sum()
for d in [1,2,3]:
    print(f"Dots removed: {d} - Original: {count_evasion(rf_original, x_test_phish, d)} | Hardened: {count_evasion(rf_hardened, x_test_phish, d)}")



#fifth step clean performance on the full test set on both models
for name, model in [("Original", rf_original), ("Hardened", rf_hardened)]:
    pred = model.predict(x_test)
    print(f"\n{name} model on clean test set")
    print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
    print(classification_report(y_test, pred, target_names=['legitimate', 'phishing']))


