import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Keep the classes to ensure the heatmap order remains the same
CLASSES = ['buffalo', 'elephant', 'rhino', 'zebra']

def plot_bar_chart(df):
    plt.figure(figsize=(10, 6), facecolor="#FFFFFF")
    ax = plt.gca()
    ax.set_facecolor("#FFFFFF")
    
    x = np.arange(len(df['model']))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df['AP_5_95'], width, label='AP@[.5:.95]', color='#1f77b4')
    bars2 = ax.bar(x + width/2, df['AP_50'], width, label='AP@.50', color='#ff7f0e')
    
    ax.set_ylabel('Score')
    ax.set_title('Animal Detection Model Comparison (mAP Scores)')
    ax.set_xticks(x)
    ax.set_xticklabels(df['model'])
    ax.legend()
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom')
                        
    os.makedirs('results/figures', exist_ok=True)
    plt.savefig('results/figures/model_comparison_bar.png', dpi=180, bbox_inches='tight')
    plt.close()

def plot_heatmap(per_class_data):
    df = pd.DataFrame(per_class_data)
    if df.empty: return
    
    pivot = df.pivot(index="model", columns="class_name", values="AP@.50")
    pivot = pivot[CLASSES] # Ensure consistent order
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd")
    plt.title("Per-Class AP@.50 Heatmap")
    plt.savefig('results/figures/per_class_heatmap.png', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    print("Loading previously saved metrics...")
    
    # Load the summary metrics directly into a DataFrame
    df_metrics = pd.read_csv('results/metrics_summary.csv')
    
    # Load the per-class metrics and convert to list of dicts to match original function format
    df_per_class = pd.read_csv('results/per_class_ap.csv')
    per_class_data = df_per_class.to_dict(orient='records')
    
    print("Generating Bar Chart...")
    plot_bar_chart(df_metrics)
    
    print("Generating Heatmap...")
    plot_heatmap(per_class_data)
    
    print("========================================================")
    print("🎉 Plotting Complete! Figures saved to results/figures/")
    print("========================================================")