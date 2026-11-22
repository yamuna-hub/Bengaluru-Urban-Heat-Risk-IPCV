"""Train an MLP (feed-forward ANN/DNN) on baseline and texture IPCV features.

Run after urban_heat_ipcv_experiment_VSCODE.py has generated
IPCV_Experiment/training_samples_with_texture.csv and model_comparison.csv.
The split and metrics match the original IPCV comparison script.
"""
from pathlib import Path
import json
import warnings
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'IPCV_Experiment'
CSV = OUT / 'training_samples_with_texture.csv'
COMPARISON = OUT / 'model_comparison.csv'
FEATURE_SETS = {
    'Baseline': ['LST_C', 'NDBI', 'NDVI'],
    'Texture': ['LST_C', 'NDBI', 'NDVI', 'Texture'],
}


def run():
    if not CSV.exists():
        raise FileNotFoundError(f'Missing {CSV}\nRun: python urban_heat_ipcv_experiment_VSCODE.py')
    df = pd.read_csv(CSV)
    needed = set(['Risk_Class', *FEATURE_SETS['Texture']])
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError(f'Missing dataset columns: {sorted(missing)}')
    # The original comparison uses all the aligned rows in this exact CSV.
    if df[list(needed)].isnull().any().any():
        raise ValueError('CSV contains missing values. Re-run the IPCV comparison to create a clean dataset.')
    if not np.isfinite(df[FEATURE_SETS['Texture']].to_numpy(dtype=float)).all():
        raise ValueError('Non-finite features detected in CSV.')
    encoder = LabelEncoder()
    y = encoder.fit_transform(df['Risk_Class'])
    if len(encoder.classes_) < 2 or min(np.bincount(y)) < 4:
        raise ValueError('Insufficient class samples for stratified comparison.')
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=0.25, random_state=42, stratify=y)
    print(f'Dataset: {len(df)} rows | Train: {len(train_idx)} | Test: {len(test_idx)}')
    print('Classes:', list(encoder.classes_))
    metrics = []
    for setting, cols in FEATURE_SETS.items():
        # Standardization is fitted on training rows only (no test leakage).
        # Hidden layers (64,32) make this a two-hidden-layer feed-forward DNN.
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
                alpha=0.001, learning_rate_init=0.001, max_iter=600,
                early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=25, random_state=42,
            )),
        ])
        x = df[cols].to_numpy(dtype=float)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            pipeline.fit(x[train_idx], y[train_idx])
        for w in caught:
            if issubclass(w.category, ConvergenceWarning):
                print(f'WARNING ({setting}): {w.message}')
        pred = pipeline.predict(x[test_idx])
        row = {
            'Features': setting, 'Model': 'MLP (DNN)',
            'Accuracy': accuracy_score(y[test_idx], pred),
            'Precision': precision_score(y[test_idx], pred, average='macro', zero_division=0),
            'Recall': recall_score(y[test_idx], pred, average='macro', zero_division=0),
            'F1 Score': f1_score(y[test_idx], pred, average='macro', zero_division=0),
        }
        metrics.append(row)
        print(f"{setting} MLP accuracy: {row['Accuracy']:.4f} | macro F1: {row['F1 Score']:.4f}")
        report = classification_report(y[test_idx], pred, target_names=list(encoder.classes_), output_dict=True, zero_division=0)
        (OUT / f'mlp_{setting.lower()}_classification_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        joblib.dump({'model': pipeline, 'classes': encoder.classes_.tolist(), 'features': cols}, OUT / f'mlp_{setting.lower()}_model.joblib')
        ConfusionMatrixDisplay(confusion_matrix(y[test_idx], pred), display_labels=encoder.classes_).plot(cmap='Blues', values_format='d')
        plt.title(f'MLP - {setting} Confusion Matrix')
        plt.tight_layout()
        plt.savefig(OUT / f'mlp_{setting.lower()}_confusion_matrix.png', dpi=180)
        plt.close()
        mlp = pipeline.named_steps['mlp']
        fig, ax = plt.subplots(figsize=(7,4))
        ax.plot(mlp.loss_curve_)
        ax.set(xlabel='Epoch', ylabel='Training loss', title=f'MLP - {setting} Training Loss')
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(OUT / f'mlp_{setting.lower()}_loss.png', dpi=180)
        plt.close(fig)
    result = pd.DataFrame(metrics)
    result.to_csv(OUT / 'mlp_dl_results.csv', index=False)
    if COMPARISON.exists():
        prior = pd.read_csv(COMPARISON)
        required = ['Features', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
        if set(required).issubset(prior.columns):
            prior = prior[required]
            combined = pd.concat([prior[prior['Model'] != 'MLP (DNN)'], result], ignore_index=True)
            combined.to_csv(OUT / 'ml_vs_dl_comparison.csv', index=False)
            pivot = combined.pivot(index='Model', columns='Features', values='Accuracy') * 100
            ax = pivot.plot.bar(figsize=(10, 5), rot=15)
            ax.set_ylabel('Accuracy (%)')
            ax.set_title('ML vs MLP Deep Learning: Baseline vs Texture')
            ax.set_ylim(0, 100)
            plt.tight_layout()
            plt.savefig(OUT / 'ml_vs_dl_accuracy_comparison.png', dpi=200)
            plt.close()
            print('Combined ML/DL comparison saved:', OUT / 'ml_vs_dl_comparison.csv')
        else:
            print('Existing model_comparison.csv columns differ. MLP results saved separately.')
    else:
        print('Existing ML comparison CSV not found; saved MLP results separately.')
    print('\n', result.round(4).to_string(index=False))
    print('\nNOTE: Risk labels come from the heat-risk formula, not independent field observations.')
    print('DONE. Results saved in:', OUT)


if __name__ == '__main__':
    run()
