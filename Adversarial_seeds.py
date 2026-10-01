import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('dataset_phishing.csv')
x = df.drop(columns=['url', 'status'])
y = df['status'].map({'legitimate': 0, 'phishing': 1})

def count_evasion(model, x_phish, dots_removed=1):
    before = model.predict(x_phish)
    attacked = x_phish.copy()
    attacked['nb_dots'] = attacked['nb_dots'].apply(lambda v: max(0, v - dots_removed))
    after = model.predict(attacked)
    return ((before == 1) & (after == 0)).sum()

def run_experiment(seed):
    #split first
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)

    #original model trained fresh on this seed's train rows
    rf_original = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_original.fit(x_train, y_train)

    #twins from train phishing rows only, then retrain
    twins = x_train[y_train == 1].copy()
    twins['nb_dots'] = twins['nb_dots'].apply(lambda v: max(0, v - 1))
    x_aug = pd.concat([x_train, twins], ignore_index=True)
    y_aug = pd.concat([y_train, pd.Series([1] * len(twins))], ignore_index=True)
    rf_hardened = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_hardened.fit(x_aug, y_aug)

    #the test phishing rows only
    x_test_phish = x_test[y_test == 1]
    print(f"\nSeed {seed}")
    for d in [1, 2, 3]:
        print(f"  Dots removed {d} | Original: {count_evasion(rf_original, x_test_phish, d)} | Hardened: {count_evasion(rf_hardened, x_test_phish, d)}")
    print(f"  Clean accuracy | Original: {accuracy_score(y_test, rf_original.predict(x_test)):.4f} | Hardened: {accuracy_score(y_test, rf_hardened.predict(x_test)):.4f}")

for seed in [1, 7, 42, 99, 123]:
    run_experiment(seed)