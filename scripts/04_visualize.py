from gsea_refiner.visualization.plot import load_and_merge_data, plot_classification_results


def run_classification_visualization(
    classification_file,
    nes_file,
    pval_col="padj",
    signedlogpval_col="signedlogp",
    nes_col="NES",
    interactive=False,
    pval_threshold=None,
    nes_threshold=None,
    category_filter=None,
    pathway_col="tokenized_pathway",
    category_col="predicted_category",
    palette="Blues",
    save_path=None,
    top_n=10,
    pathway_fontsize=14,
):
    df = load_and_merge_data(classification_file, nes_file)
    plot_classification_results(
        df,
        pval_col=pval_col,
        signedlogpval_col=signedlogpval_col,
        nes_col=nes_col,
        interactive=interactive,
        pval_threshold=pval_threshold,
        nes_threshold=nes_threshold,
        category_filter=category_filter,
        pathway_col=pathway_col,
        category_col=category_col,
        palette=palette,
        save_path=save_path,
        top_n=top_n,
        pathway_fontsize=pathway_fontsize,
    )


if __name__ == "__main__":
    classification_file = "data/output/classification_results.csv"
    nes_file = "data/intermediate/filtered_pathways.csv"
    save_path = "data/output/nes_classification_plot.png"

    run_classification_visualization(
        classification_file,
        nes_file,
        interactive=False,
        pval_col="padj",
        signedlogpval_col="signedlogp",
        nes_col="NES",
        category_filter="Cell Cycle",
        save_path=save_path,
    )
