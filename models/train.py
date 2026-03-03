import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
import warnings
warnings.filterwarnings('ignore')

def train():
    print("🔬 Loading cancer dataset...")
    data = load_breast_cancer()
    X    = data.data
    y    = data.target  # 1=Benign, 0=Malignant
    feature_names = data.feature_names

    print(f"   {X.shape[0]} samples × {X.shape[1]} features")

    # Scale
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train models
    models = {
        'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=200, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    }

    best_auc, best_model, best_name = 0, None, ""
    all_results = {}

    print("\n🏋️  Training models...")
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
        cv  = cross_val_score(clf, X_scaled, y, cv=5, scoring='roc_auc').mean()
        print(f"   {name}: ROC-AUC={auc:.4f} | CV={cv:.4f}")
        all_results[name] = {'auc': auc, 'cv': cv}
        if auc > best_auc:
            best_auc, best_model, best_name = auc, clf, name

    print(f"\n🏆 Best: {best_name} (ROC-AUC: {best_auc:.4f})")

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:,1]

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred,
          target_names=['Malignant', 'Benign']))

    # ── Plot 4: ROC Curve ───────────────────────────────────
    print("📊 Generating ROC curve...")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='#E74C3C', lw=2,
            label=f'{best_name} (AUC = {best_auc:.4f})')
    ax.plot([0,1],[0,1], 'k--', lw=1, alpha=0.5, label='Random (AUC = 0.50)')
    ax.fill_between(fpr, tpr, alpha=0.1, color='#E74C3C')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve — Breast Cancer Detection', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('models/plots/roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ roc_curve.png")

    # ── Plot 5: Confusion Matrix ────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=['Malignant','Benign'],
                yticklabels=['Malignant','Benign'],
                ax=ax, linewidths=2)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('models/plots/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ confusion_matrix.png")

    # ── Plot 6: Feature Importance ──────────────────────────
    importance = pd.DataFrame({
        'feature':    feature_names,
        'importance': np.abs(best_model.coef_[0]) if hasattr(best_model, 'coef_') else best_model.feature_importances_

    }).sort_values('importance', ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(importance['feature'], importance['importance'],
                   color='#3498DB', edgecolor='white')
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title('Top 15 Most Important Features\nfor Cancer Detection',
                 fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('models/plots/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ feature_importance.png")

    # ── Plot 7: PCA scatter ─────────────────────────────────
    pca    = PCA(n_components=2, random_state=42)
    X_pca  = pca.fit_transform(X_scaled)
    colors = np.where(y == 0, '#E74C3C', '#2ECC71')
    
    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(X_pca[:,0], X_pca[:,1],
                        c=colors, alpha=0.6, s=40, edgecolors='white', linewidth=0.5)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#E74C3C', label='Malignant'),
                       Patch(facecolor='#2ECC71', label='Benign')]
    ax.legend(handles=legend_elements, fontsize=12)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax.set_title('PCA — Malignant vs Benign Tumor Clusters',
                 fontsize=15, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('models/plots/pca_clusters.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ pca_clusters.png")

    # Save model
    with open("models/classifier.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("models/feature_names.pkl", "wb") as f:
        pickle.dump(list(feature_names), f)

    pd.DataFrame([{
        "model":   best_name,
        "roc_auc": round(best_auc, 4),
        "n_train": len(X_train),
        "n_test":  len(X_test),
    }]).to_csv("models/metrics.csv", index=False)

    print(f"\n✅ All done! ROC-AUC: {best_auc:.4f}")

if __name__ == "__main__":
    train()
