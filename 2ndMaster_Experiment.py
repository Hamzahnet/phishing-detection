import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,f1_score, confusion_matrix)

DATASET_PATH     = 'dataset_phishing.csv'
DROP_COLUMNS = ['url', 'status']
LABEL_COLUMN = 'status'
LABEL_MAP = {'legitimate': 0, 'phishing': 1}
TEST_SIZE = 0.2
MAIN_SEED = 42
SEEDS = [1, 7, 42, 99, 123]


MODELS = {
    "Logistic Regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
    "Decision Tree": lambda: DecisionTreeClassifier(random_state=MAIN_SEED),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=100, random_state=MAIN_SEED),
    "Gradient Boosting": lambda: GradientBoostingClassifier(random_state=MAIN_SEED),
    "Neural Network": lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=MAIN_SEED)),
}

def load_data():
    df = pd.read_csv(DATASET_PATH)
    x = df.drop(columns=DROP_COLUMNS)
    y = df[LABEL_COLUMN].map(LABEL_MAP)
    return x, y

def split_data(x, y, seed):
    return train_test_split(x, y, test_size=TEST_SIZE, random_state=seed)

def count_evasion(model, x_phish, dots_removed= 1):
    before = model.predict(x_phish)
    attacked = x_phish.copy()
    attacked['nb_dots'] = attacked['nb_dots'].apply(lambda v: max(0, v - dots_removed))
    after = model.predict(attacked)
    return ((before == 1) & (after == 0)).sum()

def harden_any(build_model, x_train, y_train):
    twins = x_train[y_train == 1].copy()
    twins['nb_dots'] = twins['nb_dots'].apply(lambda v: max(0, v-1))
    x_aug = pd.concat([x_train, twins], ignore_index=True)
    y_aug = pd.concat([y_train, pd.Series([1] * len(twins))], ignore_index=True)
    model = build_model()
    model.fit(x_aug, y_aug)
    return model

def test_one_model(name, build_model, x_tr, x_te, y_tr, y_te):
    original = build_model()
    original.fit(x_tr, y_tr)
    hardened = harden_any(build_model, x_tr, y_tr)

    x_te_phish = x_te[y_te == 1]
    row = {'model': name}

    pred = original.predict(x_te)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    row['accuracy'] = round(accuracy_score(y_te, pred), 4)
    row['precision'] = round(precision_score(y_te, pred), 4)
    row['recall'] = round(recall_score(y_te, pred), 4)
    row['f1'] = round(f1_score(y_te, pred), 4)
    row['FP'] = fp
    row['FN'] = fn

    hpred = hardened.predict(x_te)
    htn, hfp, hfn, htp = confusion_matrix(y_te, hpred).ravel()
    row['acc_hardened'] = round(accuracy_score(y_te, hpred), 4)
    row['FP_hard'] = hfp
    row['FN_hard'] = hfn
    row['recall_hard'] = round(recall_score(y_te, hpred), 4)
    row['precision_hard'] = round(precision_score(y_te, hpred), 4)

    for d in [1, 2, 3]:
        row[f'orig_{d}dot'] = count_evasion(original, x_te_phish, d)
        row[f'hard_{d}dot'] = count_evasion(hardened, x_te_phish, d)
    return row

x, y = load_data()

def run_all_seeds():
    all_rows = []
    for seed in SEEDS:
        x_tr, x_te, y_tr, y_te = split_data(x, y, seed)
        for name, build in MODELS.items():
            print(f"Seed {seed} | {name}")
            row = test_one_model(name, build, x_tr, x_te, y_tr, y_te)
            row['seed'] = seed
            all_rows.append(row)
    return pd.DataFrame(all_rows)



results_all = run_all_seeds()
results_all.to_csv('results_all_models_5seeds.csv', index=False)

evasion_cols = [c for c in results_all.columns if 'dot' in c]
print("\nTotal evasions across 5 seeds:")
print(results_all.groupby('model')[evasion_cols].sum())

print("\nMean metrics across 5 seeds (original vs hardened):")
print(results_all.groupby('model')[['accuracy', 'acc_hardened', 'recall', 'recall_hard', 'FP', 'FP_hard', 'FN', 'FN_hard']].mean().round(3))

print("\nSpread (std) of accuracy across 5 seeds:")
print(results_all.groupby('model')[['accuracy', 'acc_hardened']].std().round(4))