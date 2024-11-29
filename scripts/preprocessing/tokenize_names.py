import re
from typing import List

STOPWORDS = {"and", "of", "the", "in", "for", "with", "on", "to", "a"}

def tokenize_name(name: str, stopwords=STOPWORDS) -> List[str]:
    """
    Tokenize a single gene set name into individual words, removing stopwords.
    
    Args:
        name (str): A cleaned pathway name.
        stopwords (set): A set of stopwords to exclude.
        
    Returns:
        List[str]: Tokenized words from the pathway name.
    """
    tokens = re.split(r'\s+', name) # Split on whitespace
    tokens = [word for word in tokens if word not in stopwords]
    return tokens

def tokenize_corpus(names: List[str], stopwords=STOPWORDS) -> List[List[str]]:
    """
    Tokenize and clean a list of pathway names.
    
    Args:
        names (List[str]): A list of cleaned pathway names.
        stopwords (set): A set of stopwords to exclude.
        
    Returns:
        List[List[str]]: Tokenized pathway names.
    """
    return [tokenize_name(name, stopwords) for name in names]

if __name__ == "__main__":
    #Example usage
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
