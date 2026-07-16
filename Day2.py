
import numpy as np

def extract_features(url):
    length = len(url)
    dot = url.count('.')
    httpcheck = 1 if url.startswith('https') else 0
    special = sum(url.count(c) for c in ['@', '?', '=', '&'])
    numdig = sum(1 for c in url if c.isdigit())
    hyphen = url.count('-')

    return hyphen, length, dot, httpcheck, special, numdig

def normailise(features):
    array = np.array([features])
    normailised = (array - array.min()) / (array.max() - array.min())
    return normailised
        

def main():
    url = "http://paypa1-secure.login.verify@192.168.1.1/account?user=admin&token=abc123"
    features = extract_features(url)
    normailised = normailise(features)
    print(f"url: {url}")
    print(f"features: {features}")
    print(f"normalised: {normailised}")

main()
