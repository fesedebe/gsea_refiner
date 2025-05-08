from gsea_refiner.categorization.fine_tune_transformer import fine_tune_biobert

def main():
    input_csv = "data/training/labeled_pathways.csv"
    model_out_dir = "data/models/biobert_finetuned"

    fine_tune_biobert(
        input_csv=input_csv,
        model_out_dir=model_out_dir,
        num_train_epochs=5,
        batch_size=16,
        lr=2e-5
    )

if __name__ == "__main__":
    main()
