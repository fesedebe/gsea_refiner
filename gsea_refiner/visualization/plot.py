import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pdb

def load_and_merge_data(classification_file: str, nes_file: str, pathway_col="tokenized_pathway") -> pd.DataFrame:
    class_df = pd.read_csv(classification_file)
    nes_df = pd.read_csv(nes_file)
    merged_df = pd.merge(class_df, nes_df, on=pathway_col)
    return merged_df

def plot_classification_results(df, pval_col="padj", signedlogpval_col=None, nes_col="NES", interactive=False, 
                                pval_threshold=None, nes_threshold=None, category_filter="DNA Repair", 
                                pathway_col="tokenized_pathway", category_col="predicted_category", 
                                palette="Blues", save_path=None, top_n=10, pathway_fontsize=14) -> None:
    """Generates an NES bar plot grouped by category, colored by p-value or signed log p-value."""
    
    # Ensure NES and signedlogp are treated as numeric
    df[nes_col] = pd.to_numeric(df[nes_col], errors='coerce')
    if signedlogpval_col:
        df[signedlogpval_col] = pd.to_numeric(df[signedlogpval_col], errors='coerce')
    df[pval_col] = pd.to_numeric(df[pval_col], errors='coerce')
    
    # Apply category filter
    df = df[df[category_col] == category_filter]
    
    # Apply threshold filters, if specified
    if pval_threshold is not None and signedlogpval_col is None:
        df = df[df[pval_col] <= pval_threshold]
    if nes_threshold is not None:
        df = df[abs(df[nes_col]) >= nes_threshold]
    
    # Sort data and filter top N pathways
    df = df.sort_values(by=[category_col, nes_col], ascending=[True, False])
    df = df.head(top_n)
    
    # Determine coloring variable
    color_col = signedlogpval_col if signedlogpval_col else pval_col
    ascending_order = False if signedlogpval_col else True  # Higher is more significant for signedlogp
    df["color_scale"] = df[color_col].rank(method='min', ascending=ascending_order)
    
    significance_label = "(signedlogpvalue)" if signedlogpval_col else "(padj)"

    if not df.empty:
        if interactive:
            fig = px.bar(
                df, y=pathway_col, x=nes_col, color="color_scale", 
                facet_col=category_col, text=nes_col,
                color_continuous_scale=palette,
                title=f"NES Bar Plot for {category_filter}",
                orientation='h'
            )
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.show()
        else:
            plt.figure(figsize=(10, 10))
            ax = sns.barplot(data=df, y=pathway_col, x=nes_col, hue=color_col, palette=palette, orient="h")
            ax.set_ylabel("Pathway", fontsize=pathway_fontsize)
            ax.set_xlabel("NES")
            ax.set_title(f"NES Bar Plot for {category_filter}")
            ax.legend(title=f"Significance Scale {significance_label}")
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=pathway_fontsize)
            plt.tight_layout()
            
            if save_path:
                print(f"Saving plot to {save_path}")
                plt.savefig(save_path, dpi=300)
            plt.show()
    else:
        raise ValueError("No data available after filtering. Adjust thresholds or check data integrity.")

def run_classification_visualization(classification_file, nes_file, pval_col="padj", signedlogpval_col="signedlogp", nes_col="NES", interactive=False, 
                                     pval_threshold=None, nes_threshold=None, category_filter=None, 
                                     pathway_col="tokenized_pathway", category_col="predicted_category", 
                                     palette="Blues", save_path=None, top_n=10, pathway_fontsize=14):
    df = load_and_merge_data(classification_file, nes_file)
    plot_classification_results(df, pval_col, signedlogpval_col, nes_col, interactive, 
                                pval_threshold, nes_threshold, category_filter, 
                                pathway_col, category_col, palette, save_path, top_n, pathway_fontsize)
    
if __name__ == "__main__":
    classification_file = "data/output/classification_results.csv"
    nes_file = "data/intermediate/filtered_pathways.csv"
    save_path = "results/figures/cellcycle_classification_plot.png"
    
    run_classification_visualization(
        classification_file, 
        nes_file, 
        interactive=False, 
        pval_col="padj", 
        signedlogpval_col="signedlogp", 
        nes_col="NES",
        save_path=save_path,
        category_filter="Cell Cycle",
        top_n=10
    )