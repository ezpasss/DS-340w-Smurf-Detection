import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CSV_FILE = "justice_league_dataset_final.csv"

def plot_all_metrics():
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print("❌ Data file not found.")
        return

    # Map 0/1 to names
    df['Rank'] = df['label'].map({0: 'Bronze', 1: 'Challenger'})

    # We will plot 3 metrics: Gold, CS, APM
    metrics = ['gold', 'damage_dealt', 'kills']
    titles = ['Gold Difference', 'CS (Farming Skill)', 'APM (Mechanics)']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, metric in enumerate(metrics):
        # Group by Rank + Minute
        avg_data = df.groupby(['Rank', 'minute'])[metric].mean().reset_index()
        
        sns.lineplot(
            data=avg_data, 
            x='minute', y=metric, hue='Rank', 
            marker='o', palette=['#cd7f32', '#0000FF'], ax=axes[i]
        )
        
        axes[i].set_title(titles[i], fontsize=14)
        axes[i].set_xlabel('Minute')
        axes[i].set_ylabel('Value')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xticks(range(0, 16, 2))

    plt.tight_layout()
    plt.savefig("all_metrics_comparison.png")
    print("✅ Saved graph to 'all_metrics_comparison.png'")
    plt.show()

if __name__ == "__main__":
    plot_all_metrics()