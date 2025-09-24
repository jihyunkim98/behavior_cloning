import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("predictions.csv")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df["VC.Steer.Ang"].dropna(), bins=100, alpha=0.6, label="gt_steer")
if "pred_steering_angle" in df:
    axes[0].hist(df["pred_steering_angle"].dropna(), bins=100, alpha=0.6, label="pred_steer")
axes[0].set_title("Steering distribution"); axes[0].legend()

axes[1].hist(df["VC.Gas"].dropna().clip(0,1), bins=100, alpha=0.6, label="gt_gas")
if "pred_gas" in df:
    axes[1].hist(df["pred_gas"].dropna().clip(0,1), bins=100, alpha=0.6, label="pred_gas")
axes[1].set_title("Gas distribution"); axes[1].legend()

plt.tight_layout(); plt.show()