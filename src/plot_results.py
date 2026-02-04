
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set matplotlib style for publication quality
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 16

def plot_n_vs_sharpe(summary_csv_path, output_dir):
    """
    Plots Universe Size (N) vs Mean Sharpe Ratio.
    """
    if not os.path.exists(summary_csv_path):
        print(f"Warning: Summary CSV not found at {summary_csv_path}")
        return

    df = pd.read_csv(summary_csv_path)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(
        data=df, 
        x='N', 
        y='Mean Sharpe', 
        hue='Method', 
        style='Method', 
        markers=True, 
        dashes=False, 
        linewidth=2.5, 
        markersize=9
    )
    
    plt.title('Mean Sharpe Ratio vs. Number of Assets (N)', fontsize=16)
    plt.xlabel('Number of Assets (N)')
    plt.ylabel('Annualized Sharpe Ratio')
    plt.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "n_vs_sharpe.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved N vs Sharpe plot to {save_path}")
    plt.close()

def plot_sharpe_boxplot(raw_csv_path, output_dir):
    """
    Plots boxplot of Sharpe Ratios for each method across different N.
    """
    if not os.path.exists(raw_csv_path):
        print(f"Warning: Raw CSV not found at {raw_csv_path}")
        return

    df = pd.read_csv(raw_csv_path)
    
    # Rename methods for better plotting readability
    method_rename_map = {
        'Sample Covariance': 'Sample',
        'Market Factor': 'Market',
        'PCA': 'PCA',
        'POET': 'POET',
        'Linear Shrinkage': 'Linear',
        'Nonlinear Shrinkage': 'NonLinear'
    }
    df['Method'] = df['Method'].replace(method_rename_map)
    
    # Create a plot for N=500 explicitly (High Dimensional Case)
    n_target = 500
    df_subset = df[df['N'] == n_target]
    
    if not df_subset.empty:
        plt.figure(figsize=(10, 6))
        
        # Use hue for coloring, and x for separation
        sns.boxplot(
            data=df_subset, 
            x='Method', 
            y='Sharpe Ratio (Ann)', 
            hue='Method', 
            palette='viridis', # Better color palette
            dodge=False,
            legend=True # Ensure legend is on implies hue is used
        )
        
        plt.title(f'Distribution of Sharpe Ratios (N={n_target})', fontsize=16)
        plt.xlabel('Estimation Method')
        plt.ylabel('Annualized Sharpe Ratio')
        # Move legend to outside if needed, or keeping it inside if space permits
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f"sharpe_boxplot_N{n_target}.png")
        plt.savefig(save_path, dpi=300)
        print(f"Saved Boxplot for N={n_target} to {save_path}")
        plt.close()
    
    # FacetGrid for all N
    g = sns.catplot(
        data=df, 
        x='Method', 
        y='Sharpe Ratio (Ann)', 
        col='N', 
        col_wrap=3,
        hue='Method', # Color by method
        kind='box', 
        height=4, 
        aspect=1.2,
        palette='viridis',
        sharey=False,
        dodge=False 
    )
    
    # Improve labels
    g.set_axis_labels("", "Sharpe Ratio (Ann)")
    g.set_titles("N = {col_name}")
    g.set_xticklabels(rotation=45)
    
    # Adjust legend
    if g.legend:
        sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    else:
         # Fallback if legend is not attached automatically
         plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    g.fig.suptitle('Sharpe Ratio Distribution by N', y=1.02, fontsize=16)
    
    save_path_all = os.path.join(output_dir, "sharpe_boxplot_all_N.png")
    plt.savefig(save_path_all, dpi=300)
    print(f"Saved All N Boxplots to {save_path_all}")
    plt.close()

def plot_combined_n_boxplot(raw_csv_path, output_dir):
    """
    Plots a SINGLE combined boxplot: X=N, Y=Sharpe, Hue=Method.
    This allows easy comparison of distributions across N.
    """
    if not os.path.exists(raw_csv_path):
        return

    df = pd.read_csv(raw_csv_path)
    
    # Rename methods
    method_rename_map = {
        'Sample Covariance': 'Sample',
        'Market Factor': 'Market',
        'PCA': 'PCA',
        'POET': 'POET',
        'Linear Shrinkage': 'Linear',
        'Nonlinear Shrinkage': 'NonLinear'
    }
    df['Method'] = df['Method'].replace(method_rename_map)

    plt.figure(figsize=(12, 7))
    sns.boxplot(
        data=df, 
        x='N', 
        y='Sharpe Ratio (Ann)', 
        hue='Method', 
        palette='viridis',
        width=0.7
    )
    
    plt.title('Distribution of Sharpe Ratios across Universe Sizes (N)', fontsize=16)
    plt.xlabel('Number of Assets (N)')
    plt.ylabel('Annualized Sharpe Ratio')
    plt.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "sharpe_boxplot_combined_N.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved Combined N Boxplot to {save_path}")
    plt.close()

if __name__ == "__main__":
    # We use the specific path provided by the user related to Experiment 1 (output30)
    base_dir = r"c:\Users\scarl\Documents\Research\data\output\output30\output3"
    output_dir = r"c:\Users\scarl\Documents\Research\data\output\figures"
    os.makedirs(output_dir, exist_ok=True)
    
    summary_csv = os.path.join(base_dir, "experiment_results_summary.csv")
    raw_csv = os.path.join(base_dir, "experiment_results_raw.csv")

    print(f"Reading data from: {base_dir}")
    
    # 1. Plot N vs Sharpe (Line)
    plot_n_vs_sharpe(summary_csv, output_dir)
    
    # 2. Plot Boxplots (Individual/Facet)
    plot_sharpe_boxplot(raw_csv, output_dir)
    
    # 3. Plot Combined Boxplot (X=N)
    plot_combined_n_boxplot(raw_csv, output_dir)
