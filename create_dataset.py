import pandas as pd

# Creating a labelled dataset of URLs
data = {
    'url': [
        'https://www.google.com',
        'https://www.amazon.co.uk',
        'https://www.github.com',
        'https://www.linkedin.com',
        'https://www.bbc.co.uk',
        'https://www.microsoft.com',
        'https://www.apple.com',
        'https://www.youtube.com',
        'http://paypa1-secure.login.verify.com/account',
        'http://amazon-security-alert.tk/login?user=admin&token=abc',
        'http://apple-id-locked.click/restore?id=123456',
        'http://secure-bankofengland.suspicious.com/verify',
        'http://microsoft-alert.ru/fix?session=abc&user=admin',
        'http://halifax-bank-secure.phishing.net/login',
        'http://verify-paypal.suspicious-domain.com/account?user=test',
        'http://192.168.1.1/login?user=admin&pass=1234',
    ],
    'label': [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
}

df = pd.DataFrame(data)
df.to_csv('urls.csv', index=False)
print("Dataset created successfully")
print(df)