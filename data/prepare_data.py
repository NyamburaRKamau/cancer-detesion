import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
import os

os.makedirs("data", exist_ok=True)
os.makedirs("models/plots", exist_ok=True)

def prepare_and_visualize():
    print("🔬 Loading Wisconsin Breast Cancer Dataset...")
    data = load_breast_cancer()
    
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    df['diagnosis'] = df['target'].map({1: 'Benign', 0: 'Malignant'})
    
    print(f"✅ Dataset loaded:")
    print(f"   Total samples : {len(df):,}")
    print(f"   Malignant     : {(df['diagnosis']=='Malignant').sum():,}")
    print(f"   Benign        : {(df['diagnosis']=='Benign').sum():,}")
    print(f"   Features      : {len(data.feature_names)}")

    # ── Plot 1: Class distribution ─────────────────────────
    print("\n📊 Generating plots...")
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['#E74C3C', '#2ECC71']
    counts = df['diagnosis'].value_counts()
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_title('Tumor Diagnosis Distribution', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Diagnosis', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(count), ha='center', fontweight='bold', fontsize=13)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('models/plots/class_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ class_distribution.png")

    # ── Plot 2: Correlation heatmap (top 10 features) ──────
    top_features = ['mean radius', 'mean texture', 'mean perimeter',
                    'mean area', 'mean smoothness', 'mean compactness',
                    'mean concavity', 'mean concave points',
                    'mean symmetry', 'mean fractal dimension']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[top_features].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, ax=ax, linewidths=0.5, annot_kws={'size': 8})
    ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('models/plots/correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ correlation_heatmap.png")

    # ── Plot 3: Feature distributions by diagnosis ─────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    key_features = ['mean radius', 'mean texture', 'mean area',
                    'mean concavity', 'mean compactness', 'mean smoothness']
    
    for ax, feat in zip(axes.flat, key_features):
        for diag, color in [('Malignant', '#E74C3C'), ('Benign', '#2ECC71')]:
            subset = df[df['diagnosis'] == diag][feat]
            ax.hist(subset, bins=30, alpha=0.6, color=color, label=diag, edgecolor='white')
        ax.set_title(feat.replace('mean ', '').title(), fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.suptitle('Feature Distributions: Malignant vs Benign', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('models/plots/feature_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ feature_distributions.png")

    # Save data
    df.to_csv("data/cancer_data.csv", index=False)
    print("\n💾 Saved to data/cancer_data.csv")
    return df

if __name__ == "__main__":
    prepare_and_visualize()
