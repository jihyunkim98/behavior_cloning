import argparse
import os
from typing import List

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from model import MLP
from data_loader import Default_Cols, Input, Output


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on a CSV and export predictions")
    parser.add_argument("--csv", type=str, required=True, help="Path to source CSV for inference")
    parser.add_argument("--best_model", type=str, default="bc_model.pth", help="Checkpoint path")
    parser.add_argument("--out_csv", type=str, default="predictions.csv", help="Output CSV path")
    parser.add_argument("--plot", action="store_true", help="Save a PNG plot comparing GT vs predictions")
    parser.add_argument("--plot_png", type=str, default="predictions.png", help="Plot output path")
    parser.add_argument("--export-openloop", action="store_true", help="Also export a minimal Open Loop file for CarMaker")
    parser.add_argument("--openloop_csv", type=str, default="openloop.dat", help="Open Loop table path (space-separated)")
    return parser.parse_args()


def build_input_matrix(df: pd.DataFrame, col_map: dict, input_keys: List[str]) -> np.ndarray:
    required_columns = [col_map[k] for k in input_keys]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    X = df[required_columns].astype(np.float32).values
    return X


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isfile(args.best_model):
        raise FileNotFoundError(f"Checkpoint not found: {args.best_model}")
    best_model = torch.load(args.best_model, map_location="cpu")

    net = MLP(input_dim=len(Input), output_dim=len(Output)).to(device)
    net.load_state_dict(best_model["model"])
    net.eval()

    if not os.path.isfile(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")
    df = pd.read_csv(args.csv)

    X = build_input_matrix(df, Default_Cols, Input)
    valid_mask = ~np.isnan(X).any(axis=1)
    X_valid = X[valid_mask]

    mean = best_model["mean"].to(device)
    std = best_model["std"].to(device)

    with torch.no_grad():
        xb = torch.from_numpy(X_valid).to(device)
        xb = (xb - mean) / std
        preds = net(xb).cpu().numpy()

    # Ensure parent directory exists for outputs
    out_dir = os.path.dirname(args.out_csv) or "."
    os.makedirs(out_dir, exist_ok=True)
    if args.plot_png:
        os.makedirs(os.path.dirname(args.plot_png) or ".", exist_ok=True)
    if args.export_openloop and args.openloop_csv:
        os.makedirs(os.path.dirname(args.openloop_csv) or ".", exist_ok=True)

    out = df.copy()
    for key in Output:
        out[f"pred_{key}"] = np.nan
    out.loc[valid_mask, [f"pred_{k}" for k in Output]] = preds
    out.to_csv(args.out_csv, index=False)
    print(f"Saved predictions to {args.out_csv}")

    # Export minimal Open Loop table
    if args.export_openloop:
        time_col = Default_Cols.get("time", "Time")
        if time_col not in out.columns:
            raise ValueError(f"Time column '{time_col}' not found in source CSV")

        # Prefer predicted columns if present; fall back to GT columns
        pred_cols = {k: f"pred_{k}" for k in Output}
        value_series = {}
        for k in Output:
            pred_col = pred_cols[k]
            gt_col = Default_Cols[k]
            if pred_col in out.columns and out[pred_col].notna().any():
                value_series[gt_col] = out[pred_col]
            elif gt_col in out.columns:
                value_series[gt_col] = out[gt_col]
            else:
                raise ValueError(f"Neither predicted '{pred_col}' nor ground-truth '{gt_col}' found in CSV")

        openloop_df = pd.DataFrame({
            "Time": out[time_col],
            **value_series,
        })
        # Space-separated ASCII is robust for CarMaker OpenLoop
        openloop_df.to_csv(args.openloop_csv, index=False, sep=" ")
        print(f"Saved Open Loop table to {args.openloop_csv}")

    if args.plot:
        fig, axes = plt.subplots(len(Output), 1, figsize=(10, 2.5 * len(Output)), sharex=True)
        if len(Output) == 1:
            axes = [axes]
        time_col = Default_Cols.get("time")
        t = out[time_col] if (time_col and time_col in out.columns) else pd.RangeIndex(len(out))
        for ax, key in zip(axes, Output):
            gt_col = Default_Cols[key]
            if gt_col in out.columns:
                ax.plot(t, out[gt_col], label=f"gt_{key}", alpha=0.6)
            ax.plot(t, out[f"pred_{key}"], label=f"pred_{key}")
            ax.set_ylabel(key)
            ax.legend(loc="best")
        axes[-1].set_xlabel(time_col if time_col else "index")
        plt.tight_layout()
        plt.savefig(args.plot_png, dpi=150)
        plt.close(fig)
        print(f"Saved plot to {args.plot_png}")


if __name__ == "__main__":
    main()


"""
python .\test.py --csv .\DriveData\BC_Track1\BC_G29_Test_142621.csv --best_model .\bc_model.pth --out_csv .\predictions.csv --plot --plot_png .\predictions.png --export-openloop --openloop_csv .\openloop.dat
"""

"""
python test.py --csv DriveData/BC_Track1/BC_G29_Test_142621.csv --best_model bc_model.pth --export-openloop --plot
"""