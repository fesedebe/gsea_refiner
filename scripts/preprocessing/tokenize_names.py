import re
from typing import List

STOPWORDS = {"and", "of", "the", "in", "for", "with", "on", "to", "a"}

def tokenize_name(name: str, stopwords=STOPWORDS) -> List[str]:
    #Tokenize a single gene set name into individual words - split on whitespace and remove stopwords.
    tokens = re.split(r'\s+', name) 
    tokens = [word for word in tokens if word not in stopwords]
    return tokens

def tokenize_corpus(names: List[str], stopwords=STOPWORDS) -> List[List[str]]:
    return [tokenize_name(name, stopwords) for name in names]

if __name__ == "__main__":
    cleaned_names = [
        "m phase",
        "g2m checkpoint",
        "atp hydrolysis activity",
        "chromosome segregation",
        "chromosomal region"
    ]
    tokenized = tokenize_corpus(cleaned_names)
    
    print("Tokenized Pathway Names:")
    print(tokenized)
