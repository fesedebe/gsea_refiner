import re


def clean_gene_set_name(name: str) -> str:
    name = name.lower()
    name = re.sub(r'^[^_]*_', '', name)
    name = name.replace('_', ' ').strip()
    return name
