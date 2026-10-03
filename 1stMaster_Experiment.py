import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# setting changable for future datasets
DATASET_PATH = 'dataset_phishing.csv'
DROP_COLUMNS = ['url', 'status']
LABEL_COLUMNS = 'status'
LABLE_MAPS = {'legitimate': 0, 'phishing': 1}
TEST_SIZE = 0.2
MAIN_SEED = 42
SEEDS = [1,7,42,99,123]

def load_data():
    df = pd.read_csv(DATASET_PATH)
    x = df.drop(columns=DROP_COLUMNS)
    y = df[LABEL_COLUMNS].map(LABLE_MAPS)
    return(x,y)

def split_data(x,y,seed):
    return train_test_split(x,y, test_size=TEST_SIZE, random_state=seed)

def compare_models(x_train, x_test, y_train, y_test):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    lr = LogisticRegression(max_iter=5000)
    lr.fit(x_train_scaled, y_train)

    rf = RandomForestClassifier(n_estimators=100, random_state=MAIN_SEED)
    rf.fit(x_train, y_train)

    nn = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=MAIN_SEED)
    nn.fit(x_train_scaled, y_train)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (name, model, xt) in zip(axes, [("Logistic Regression", lr, x_test_scaled),
                                            ("Random Forest", rf, x_test),
                                            ("Neural Network", nn, x_test_scaled)]):
        pred = model.predict(xt)
        print(f"\n=== {name} ===")
        print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
        print(classification_report(y_test, pred, target_names=['legitimate', 'phishing']))

        cm = confusion_matrix(y_test, pred)
        tn, fp, fn, tp = cm.ravel()
        print(f"False positives (safe flagged): {fp} | False negatives (phishing missed): {fn}")

        ConfusionMatrixDisplay(cm, display_labels=['legitimate', 'phishing']).plot(ax=ax)
        ax.set_title(name)

    plt.tight_layout()
    plt.savefig('confusion_matrices_master.png')
    plt.close()
    print("\nConfusion matrices saved as confusion_matrices_master.png")
    return rf

def count_evasion(model, x_phish, dots_removed =1):
    before = model.predict(x_phish)
    attacked = x_phish.copy()
    attacked['nb_dots'] = attacked['nb_dots'].apply(lambda v: max(0, v - dots_removed))
    after = model.predict(attacked)
    return ((before == 1) & (after == 0)).sum()

def harden_model(x_train, y_train, seed=MAIN_SEED):
    twins = x_train[y_train == 1].copy()
    twins['nb_dots'] = twins['nb_dots'].apply(lambda v: max(0, v - 1))

    x_aug = pd.concat([x_train, twins], ignore_index=True)
    y_aug = pd.concat([y_train, pd.Series([1] * len(twins))], ignore_index=True)

    rf_hardened = RandomForestClassifier(n_estimators=100, random_state=seed)
    rf_hardened.fit(x_aug, y_aug)

    print(f"Twins created: {len(twins)} | Train rows after twins: {len(x_aug)}")
    return rf_hardened

def run_experiment(x, y, seed):
    x_tr, x_te, y_tr, y_te = split_data(x, y, seed)

    rf_orig = RandomForestClassifier(n_estimators=100, random_state=MAIN_SEED)
    rf_orig.fit(x_tr, y_tr)
    rf_hard = harden_model(x_tr, y_tr)

    x_te_phish = x_te[y_te == 1]
    row = {'seed': seed}
    for d in [1, 2, 3]:
        row[f'orig_{d}dot'] = count_evasion(rf_orig, x_te_phish, d)
        row[f'hard_{d}dot'] = count_evasion(rf_hard, x_te_phish, d)
    row['acc_orig'] = round(accuracy_score(y_te, rf_orig.predict(x_te)), 4)
    row['acc_hard'] = round(accuracy_score(y_te, rf_hard.predict(x_te)), 4)
    return row


x,y = load_data()
x_train, x_test, y_train, y_test = split_data(x, y, MAIN_SEED)
print(f"Train rows: {len(x_train)}  /  Test rows: {len(x_test)}")
rf_original = compare_models(x_train, x_test, y_train, y_test)
rf_hardened = harden_model(x_train, y_train)
x_test_phish = x_test[y_test == 1]
print(f"\nTest phishing rows: {len(x_test_phish)}")

for d in [1, 2, 3]:
    print(f"Dots removed {d} | Original: {count_evasion(rf_original, x_test_phish, d)} | Hardened: {count_evasion(rf_hardened, x_test_phish, d)}")


print("\n=== FIVE-SEED CHECK ===")
results = pd.DataFrame([run_experiment(x, y, s) for s in SEEDS])
print(results.to_string(index=False))

print("\nTotal evasions across 5 seeds:")
print(results[['orig_1dot', 'hard_1dot', 'orig_2dot', 'hard_2dot', 'orig_3dot', 'hard_3dot']].sum())

results.to_csv('results_seeds.csv', index=False)