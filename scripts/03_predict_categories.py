from gsea_refiner.classification.predict import predict_categories

def main():
    predict_categories(
        input_file="data/input/new_gsea_results.txt",
        model_dir="data/models/biobert_finetuned",
        output_file="data/output/gsea_predictions.csv"
    )

if __name__ == "__main__":
    main()
