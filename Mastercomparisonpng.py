import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results_all_models_5seeds.csv')
g = df.groupby('model')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: accuracy
acc = g[['accuracy', 'acc_hardened']].mean()
acc.columns = ['Original', 'Hardened']
acc.plot(kind='bar', ax=axes[0, 0], ylim=(0.90, 0.98))
axes[0, 0].set_title('Mean clean accuracy (5 seeds)')
axes[0, 0].set_ylabel('Accuracy')

# Panel 2: false positives
fp = g[['FP', 'FP_hard']].mean()
fp.columns = ['Original', 'Hardened']
fp.plot(kind='bar', ax=axes[0, 1])
axes[0, 1].set_title('Mean false positives (safe sites wrongly flagged)')
axes[0, 1].set_ylabel('Errors per split')

# Panel 3: false negatives
fn = g[['FN', 'FN_hard']].mean()
fn.columns = ['Original', 'Hardened']
fn.plot(kind='bar', ax=axes[1, 0])
axes[1, 0].set_title('Mean false negatives (phishing missed)')
axes[1, 0].set_ylabel('Errors per split')

# Panel 4: evasions
ev = g[['orig_3dot', 'hard_3dot']].sum()
ev.columns = ['Original', 'Hardened']
ev.plot(kind='bar', ax=axes[1, 1])
axes[1, 1].set_title('Total evasions, 3 dots removed (5 seeds)')
axes[1, 1].set_ylabel('URLs that evaded')

for ax in axes.flat:
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('model_comparison.png')
plt.close()
print("Saved model_comparison.png")