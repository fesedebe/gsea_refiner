import pytest
from gsea_refiner.preprocess import clean_gene_set_name, tokenize_name, tokenize_corpus

def test_clean_gene_set_names():
    raw_names = ["GO_CELL_CYCLE", "REACTOME_ATP_HYDROLYSIS"]
    expected_output = ["cell cycle", "atp hydrolysis"]
    cleaned_names = [clean_gene_set_name(name) for name in raw_names]

    assert cleaned_names == expected_output

def test_tokenize_name():
    name = "cell cycle and checkpoint"
    assert tokenize_name(name) == ["cell", "cycle", "checkpoint"]
    assert tokenize_name(name, stopwords={"cell", "checkpoint", "and"}) == ["cycle"]
    assert tokenize_name(name, stopwords=None) == ["cell", "cycle", "and", "checkpoint"]

def test_tokenize_corpus():
    input_names = ["regulation of cell cycle", "g2m checkpoint"]
    expected_output = [["regulation", "cell", "cycle"], ["g2m", "checkpoint"]]
    
    assert tokenize_corpus(input_names) == expected_output