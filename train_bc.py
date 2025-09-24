import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import MLP
from data_loader import make_train_val_loaders, Standardizer, Input, Output


def parse_args():
    parser = argparse.ArgumentParser(description="Train behavior cloning policy")
    parser.add_argument("--train", type=str, default="train.txt", help="Path to train file list")
    parser.add_argument("--val", type=str, default="val.txt", help="Path to val file list")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--best_model", type=str, default="bc_model.pth")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    print(f"Starting training")
    args = parse_args()
    set_seed(args.seed)

    dataset_train, dataset_validation, dataloader_train, dataloader_validation = make_train_val_loaders(
        args.train, args.val, batch_size=args.batch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = MLP(input_dim=len(Input), output_dim=len(Output)).to(device)

    # Standardize by train statistics only
    std = Standardizer()
    std.fit(dataset_train.X.to(device))

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr)

    best_val_rmse = float("inf")

    for epoch in range(1, args.epochs + 1):
        network.train()
        epoch_loss_sum = 0.0
        num_train_samples = 0

        for xb, yb in dataloader_train:
            xb = std.transform(xb.to(device))
            yb = yb.to(device)

            pred = network(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss_sum += loss.item() * xb.size(0)
            num_train_samples += xb.size(0)

        train_mse = epoch_loss_sum / max(1, num_train_samples)

        # Validation
        network.eval()
        sum_squared_error = 0.0
        count = 0
        with torch.no_grad():
            for xb, yb in dataloader_validation:
                xb = std.transform(xb.to(device))
                yb = yb.to(device)
                pr = network(xb)
                sum_squared_error += ((pr - yb) ** 2).sum().item()
                count += yb.numel()
        val_rmse = (sum_squared_error / max(1, count)) ** 0.5

        print(f"[{epoch:03d}] train_mse={train_mse:.6f}  val_rmse={val_rmse:.4f}")

        if val_rmse < best_val_rmse - 1e-6:
            best_val_rmse = val_rmse
            torch.save(
                {
                    "model": network.state_dict(),
                    "mean": std.mean.cpu(),
                    "std": std.std.cpu(),
                    "inputs": Input,
                    "outputs": Output,
                },
                args.best_model,
            )

    # 테스트 평가는 test_bc.py에서 별도로 수행


if __name__ == "__main__":
    main()


"""
python train_bc.py --train train.txt --val val.txt --epochs 20 --batch_size 512 --lr 1e-3 --best_model bc_model.pth
"""