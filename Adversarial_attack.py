import pandas as pd
import numpy  as np
import joblib 

#loading trained randomm forest model
model = joblib.load('model_rf.pkl')

#dataset loading
df = pd.read_csv('dataset_phishing.csv')

#only phishing
phishing_df = df[df['status'] == 'phishing'].copy()

#features and labels
X = phishing_df.drop(columns=['url', 'status'])

#looping the prediction
original_prediction = model.predict(X) # array of phishing URLS

X_modified = X.copy()
X_modified['https_token'] = 1   #MAKES A COPY OF THE DATA NEVER CHANGE THE ORIGINAL
#SETTING HTTPS FEATURE TO 1 SIMULATING AN ATTACKER USING HTTPS

new_predictions = model.predict(X_modified)
evaded = (original_prediction ==1) & (new_predictions == 0)
print(f"Number of phishing URLs that evaded detection: {evaded.sum()}")
#this line checks the original predition that was phishing and the new predcition against the changes  to see the difference
'''line 1-27 focuses on https modification from attackers
out of all datasets 1 managed to evade detection implying
tha tthe model doesnt rely heavily on https precence below will be subdomains'''



X_modified2 = X.copy()
X_modified2['nb_dots'] = X_modified2['nb_dots'].apply(lambda x: max(0, x - 1))
new_predictions2 = model.predict(X_modified2)
evaded2 = (original_prediction ==1) & (new_predictions2 == 0)
print(f"Number of  phishing URLs evaded detection (nb_dots -1): {evaded2.sum()}")
'''there where 4 evated URLs implying that removing
a subdomain level fools the model more than adding HTTPS
the dot count is weighed more heavy than https precence
i am stress testing my model more legitimate urls use fewer
subdomains or dots meanign this model is suseptible to an attacker
using fewer subdomains below this ill include a hyphen level reduction'''


X_modified3 = X.copy()
X_modified3['nb_hyphens'] = X_modified3['nb_hyphens'].apply(lambda x: max(0, x - 1))
new_predictions3 = model.predict(X_modified3)
evaded3 = (original_prediction == 1) & (new_predictions3 == 0)
print(f"Number of phishing URLs that evaded detection (nb_hyphen -1): {evaded3.sum()}")
'''the removal of a hyphen feature level had no effect
meaning that it cares hevily on pattern recognition
to differentiate legitimacy with phishing below i will attempt
to include a mix of changes as an attacker would not just change single features'''


X_combined = X.copy()
X_combined['https_token'] = 1
X_combined['nb_dots'] = X_combined['nb_dots'].apply(lambda x: max(0, x - 1))
new_predictions_combined = model.predict(X_combined)
evaded_combined = (original_prediction == 1) & (new_predictions_combined == 0)
print(f"Number of phishing URLs that evaded detection (combined HTTPS + nb_dots -1): {evaded_combined.sum()}")

'''now i will provide a mitigation to the perturbed nb_dot -1
the mitigation will allow  the model to adapt to sub domain changes
'''
#Adverserial Retraining 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

#load dataset again for retraining
full_df = pd.read_csv('dataset_phishing.csv')
x_full = full_df.drop(columns=['url', 'status'])
y_full = full_df['status'].map({'legitimate': 0, 'phishing': 1})

#generating adversarial examples: phishing rows with nb_dots reduced by 1
#still phishing urls just pertubed structurally
adversarial_x = x_full[y_full == 1].copy()
adversarial_x['nb_dots'] = adversarial_x['nb_dots'].apply(lambda x: max(0, x - 1))
adversarial_y = pd.Series([1] * len(adversarial_x))
'''the retraining helps the model understand lower subdomains 
can still  be phishing teachign the model to recognise low subdomains count
now i will combine the original data with adverarial examples to create a mix
and allow the originall data and the augmented data to be processed in theory making my rF model smarter'''
X_augnemted = pd.concat([x_full, adversarial_x], ignore_index=True)
y_augmented = pd.concat([y_full, adversarial_y], ignore_index=True)
print(f"Original dataset size: {len(x_full)}")
print(f"Augmented dataset size: {len(X_augnemted)}")
'''the above is me adding to the  dataset that goes through the model
now i will retrain the random forest model and create a hardened copy of it to see the difference
adversarial training makes on a model'''
#split thee augmented dataset
X_train_aug, X_test_aug, y_train_aug, y_test_aug = train_test_split(X_augnemted, y_augmented, test_size=0.2, random_state=42)

#training the new hardened RF model
rf_hardened = RandomForestClassifier(n_estimators=100, random_state=42)
rf_hardened.fit(X_train_aug, y_train_aug)

#evaluate on the test set
y_pred_hardened = rf_hardened.predict(X_test_aug)
print(f"\nHardened model accuracy: {accuracy_score(y_test_aug, y_pred_hardened):.2f}")
print(classification_report(y_test_aug, y_pred_hardened, target_names=['legitimate', 'phishing']))

#saving the hardened model
joblib.dump(rf_hardened, 'model_rf_hardened.pkl')
print("Hardened model saved as model_rf_hardened.pkl")
'''excellent results ive seen a big increase in performance now lets see if 
the hardened model can resist the evation better  than the original model
'''
#loading phishing again for testing
phishing_test = df[df['status'] == 'phishing'].copy()
x_phishing_test = phishing_test.drop(columns=['url', 'status'])

#original results using hardened model
original_prediction_hardened = rf_hardened.predict(x_phishing_test)

#Applying the same nb_dots  -1 attack
x_attack_hardened = x_phishing_test.copy()
x_attack_hardened['nb_dots'] = x_attack_hardened['nb_dots'].apply(lambda x: max(0, x -1))
new_predictions_hardened = rf_hardened.predict(x_attack_hardened)

evaded_hardened = (original_prediction_hardened ==1) & (new_predictions_hardened == 0)
print(f"\n==Hardened model - nb_dots -1 attack==")
print(f"Number of phishing URLs that evaded detection (hardened model): {evaded_hardened.sum()}")
print(f"compared to original model: 4 evaded")