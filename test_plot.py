import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

predictions_path = r"C:\Users\BM\Desktop\behavior_cloning\predictions.csv"
sim_log_path = r"C:\Users\BM\Desktop\behavior_cloning\sim_log.csv"

# Load only needed columns for speed and clarity
pred_cols = ["Time", "pred_gas", "pred_steering_angle"]
sim_cols = ["Time", "CmdGas", "CmdSteer", "ReadGas", "ReadSteer"]

pred = pd.read_csv(predictions_path, usecols=pred_cols)
sim = pd.read_csv(sim_log_path, usecols=sim_cols)

# Ensure sorted by time
pred = pred.sort_values("Time").reset_index(drop=True)
sim = sim.sort_values("Time").reset_index(drop=True)

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

# Gas comparison
axes[0].plot(pred["Time"], pred["pred_gas"], label="Pred Gas", color="#1f77b4", linewidth=1.2)
axes[0].plot(sim["Time"], sim["CmdGas"], label="CmdGas", color="#ff7f0e", linewidth=1.0, alpha=0.9)
axes[0].plot(sim["Time"], sim["ReadGas"], label="ReadGas", color="#2ca02c", linewidth=1.0, alpha=0.9)
axes[0].set_title("Gas: Prediction vs Command vs Read")
axes[0].set_ylabel("Gas")
axes[0].grid(True, linestyle=":", alpha=0.5)
axes[0].legend(loc="best")

# Steering comparison
axes[1].plot(pred["Time"], pred["pred_steering_angle"], label="Pred Steering", color="#1f77b4", linewidth=1.2)
axes[1].plot(sim["Time"], sim["CmdSteer"], label="CmdSteer", color="#ff7f0e", linewidth=1.0, alpha=0.9)
axes[1].plot(sim["Time"], sim["ReadSteer"], label="ReadSteer", color="#2ca02c", linewidth=1.0, alpha=0.9)
axes[1].set_title("Steering: Prediction vs Command vs Read")
axes[1].set_xlabel("Time [s]")
axes[1].set_ylabel("Steering Angle")
axes[1].grid(True, linestyle=":", alpha=0.5)
axes[1].legend(loc="best")

plt.tight_layout()
plt.savefig("predictions.png", dpi=150)