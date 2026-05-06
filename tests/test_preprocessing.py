from gsea_refiner.preprocessing.clean import clean_gene_set_name
from gsea_refiner.preprocessing.tokenize import tokenize_corpus, tokenize_name


def test_clean_strips_prefix_and_lowercases():
    assert clean_gene_set_name("GO_CELL_CYCLE") == "cell cycle"
    assert clean_gene_set_name("REACTOME_ATP_HYDROLYSIS") == "atp hydrolysis"


def test_clean_handles_multiple_prefix_segments():
    assert clean_gene_set_name("GOBP_IMMUNE_RESPONSE") == "immune response"
    assert clean_gene_set_name("KEGG_MAPK_SIGNALING_PATHWAY") == "mapk signaling pathway"


def test_clean_handles_no_prefix():
    assert clean_gene_set_name("something") == "something"


def test_tokenize_name_removes_stopwords():
    assert tokenize_name("cell cycle and checkpoint") == ["cell", "cycle", "checkpoint"]


def test_tokenize_name_custom_stopwords():
    assert tokenize_name("cell cycle", stopwords={"cell"}) == ["cycle"]


def test_tokenize_name_no_stopwords():
    assert tokenize_name("regulation of apoptosis", stopwords=None) == [
        "regulation",
        "of",
        "apoptosis",
    ]


def test_tokenize_corpus_returns_list_of_lists():
    names = ["regulation of cell cycle", "g2m checkpoint"]
    result = tokenize_corpus(names)
    assert result == [["regulation", "cell", "cycle"], ["g2m", "checkpoint"]]
