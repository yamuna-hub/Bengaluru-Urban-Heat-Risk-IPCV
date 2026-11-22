
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load actual MLP results
folder = Path(__file__).resolve().parent / "IPCV_Experiment"
df = pd.read_csv(folder / "mlp_dl_results.csv")

print("Deep Learning Results:")
print(df)

# Baseline and Texture results
baseline = df[df["Features"] == "Baseline"].iloc[0]
texture = df[df["Features"] == "Texture"].iloc[0]

metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

baseline_values = [baseline[m] * 100 for m in metrics]
texture_values = [texture[m] * 100 for m in metrics]

# Create comparison graph
x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(10, 6))

bars1 = plt.bar(
    x - width/2,
    baseline_values,
    width,
    label="Baseline MLP",
    color="royalblue"
)

bars2 = plt.bar(
    x + width/2,
    texture_values,
    width,
    label="Texture-Enhanced MLP",
    color="orange"
)

plt.xticks(x, metrics)
plt.ylabel("Performance (%)")
plt.xlabel("Evaluation Metrics")
plt.title("Deep Learning MLP Performance Comparison")
plt.ylim(95, 100)
plt.legend()
plt.grid(axis="y", alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.05,
            f"{height:.2f}%",
            ha="center",
            fontsize=9
        )

plt.tight_layout()

# Save graph
output = folder / "DL_MLP_Comparison.png"
plt.savefig(output, dpi=300, bbox_inches="tight")

print("Graph saved:", output)

plt.show()
