# url featres extractor for fishing detection dissert
def extract_features(url):
    length = len(url)
    dot = url.count('.')
    httpcheck = 1 if url.startswith('https') else 0
    special = sum(url.count(c) for c in ['@', '?', '=', '&'])
    numdig = sum(1 for c in url if c.isdigit())
    hyphen = url.count('-')

    return hyphen, length, dot, httpcheck, special, numdig

def main():
    url = "http://paypa1-secure.login.verify@192.168.1.1/account?user=admin&token=abc123"
    features = extract_features(url)
    print(f"url: {url}")
    print(f"features: {features}")

main()