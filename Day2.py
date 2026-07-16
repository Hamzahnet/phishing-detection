import numpy as np                                       

def extract_features(url):
    length = len(url)                                                     #provides the length of the URL
    dot = url.count('.')                                                 # number of dots or subdomains used
    httpcheck = 1 if url.startswith('https') else 0                        # to check if the url is using encrypted communes
    special = sum(url.count(c) for c in ['@', '?', '=', '&'])                       #checking url for special charecter
    numdig = sum(1 for c in url if c.isdigit())                                #see if url contains numerical digits
    hyphen = url.count('-')                                                  #hyphens
       
    return hyphen, length, dot, httpcheck, special, numdig                    #returning all values to be used in main

def normailise(features):
    array = np.array([features])                                                #sorting features in an array format
    normailised = (array - array.min()) / (array.max() - array.min())            #normalisin the features and using normalisation calculation
    return normailised                                                    
        
def build_dataset():
    urls = {
        "https ://google.com" : 0,
        "https://paypa1-secure.login.com": 1,
        "https://www.amazon.co.uk" : 0,
        "https://www.github.com" : 0,                                                 #dictionary to store legitimate and phishing urls
        "http://amazon-security-alert.tk/login?user=admin&token=abc123" : 1,
        "http://apple-id-locked.click/restore?id=123456&token=xyz" : 1,
        "http://secure-bankofengland.suspicious-domain.com/verify" : 1,
        "https://www.bbc.co.uk/news" : 0,
        "https://www.linkedin.com/feed" : 0,
    }

    for url, label in urls.items():                                                  #for each url and their label 
        feature = extract_features(url)                                        #runing function against the urls
        print(f"URL: {url} | Label: {label} | Features: {feature}")                #printing it in a format 

    return feature



def main():
    url = "http://paypa1-secure.login.verify@192.168.1.1/account?user=admin&token=abc123"     #assigned a url
    features = extract_features(url)                                                          #veriable to call back function
    normailised = normailise(features)                                                        #another call back function
    print(f"url: {url}")                                                                       #printing url
    print(f"features: {features}")                                                             #printing the urls features
    print(f"normalised: {normailised}")                                                         #urls features shown as normalised
    build_dataset()                                      

main()
