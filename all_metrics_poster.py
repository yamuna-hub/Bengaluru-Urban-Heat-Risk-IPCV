
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

folder = Path(__file__).resolve().parent / "IPCV_Experiment"
df = pd.read_csv(folder / "ml_vs_dl_comparison.csv")

metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
order = ["XGBoost", "Random Forest", "Decision Tree", "SVM", "MLP (DNN)"]

df = df.set_index(["Model", "Features"])

rows = []
for model in order:
    for features in ["Baseline", "Texture"]:
        result = df.loc[(model, features)]
        rows.append([
            model,
            features,
            *[f"{float(result[m])*100:.2f}%" for m in metrics]
        ])

columns = [
    "Model", "Features",
    "Accuracy", "Precision", "Recall", "F1-score"
]

fig, ax = plt.subplots(figsize=(14, 6))
ax.axis("off")
ax.set_title(
    "ML vs Deep Learning - Performance Comparison",
    fontsize=18, fontweight="bold", pad=25
)

table = ax.table(
    cellText=rows,
    colLabels=columns,
    cellLoc="center",
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)

for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor("#17365D")
        cell.set_text_props(color="white", weight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#EDF3FA")

plt.tight_layout()
output = folder / "All_ML_DL_Metrics_Poster.png"
plt.savefig(output, dpi=300, bbox_inches="tight")
plt.show()

print("Saved:", output)
