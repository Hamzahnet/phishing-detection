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



