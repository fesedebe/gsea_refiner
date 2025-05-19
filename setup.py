from setuptools import setup, find_packages

setup(
    name="gsea_refiner",
    version="1.0",
    description="Transformer-based categorization and interpretation of GSEA pathway enrichment results",
    author="Favour Esedebe",
    packages=find_packages(),
    install_requires=[line.strip() for line in open("requirements.txt")],
    include_package_data=True,
)