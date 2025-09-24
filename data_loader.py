import json
import numpy as np
import pandas as pd
import torch

from dataclasses import dataclass
from typing import List, Dict, Optional
from torch.utils.data import Dataset, DataLoader

# CarMaker csv 데이터 column 매핑
Default_Cols : Dict[str, str] = {
    "time": "Time",
    "vx": "Car.vx",
    "vy": "Car.vy",
    "ax": "Car.ax",
    "ay": "Car.ay",
    "yaw": "Car.Yaw",
    "yaw_rate": "Car.YawRate",
    "gas": "VC.Gas",
    "steering_angle": "VC.Steer.Ang",
}

Input : List[str] = ["vx", "vy", "ax", "ay", "yaw", "yaw_rate"]
Output : List[str] = ["gas", "steering_angle"]

# csv 파일 리스트 읽기
def load_file_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

class BCDataset(Dataset):
    """
    여러 CSV를 이어붙여 (state, action) 데이터셋
    """

    def __init__(self,files : List[str], cols : Dict[str, str] = Default_Cols, input : List[str] = Input, output : List[str] = Output, drop_nan : bool = True):
        self.files = files
        self.cols = cols
        self.input_keys = input
        self.output_keys = output

        feature_arrays: List[np.ndarray] = []
        target_arrays: List[np.ndarray] = []

        for file_path in files:
            df = pd.read_csv(file_path)

            required_columns = [cols[key] for key in self.input_keys + self.output_keys]
            missing = [column for column in required_columns if column not in df.columns]
            if missing:
                raise ValueError(f"{file_path}에 누락 컬럼: {missing}")

            features = df[[cols[key] for key in self.input_keys]].astype(np.float32).values
            targets = df[[cols[key] for key in self.output_keys]].astype(np.float32).values

            if drop_nan:
                valid = ~np.isnan(features).any(axis=1) & ~np.isnan(targets).any(axis=1)
                features, targets = features[valid], targets[valid]

            feature_arrays.append(features)
            target_arrays.append(targets)

        self.X = torch.from_numpy(np.concatenate(feature_arrays, axis=0))
        self.Y = torch.from_numpy(np.concatenate(target_arrays, axis=0))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]


@dataclass
class Standardizer:
    mean: Optional[torch.Tensor] = None
    std: Optional[torch.Tensor] = None

    def fit(self, X: torch.Tensor):
        self.mean = X.mean(0)
        self.std = X.std(0).clamp_min(1e-6)

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("Standardizer is not fitted.")
        return (X - self.mean) / self.std

    def save(self, path: str):
        payload = {"mean": self.mean.cpu().tolist(), "std": self.std.cpu().tolist()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "Standardizer":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        scaler = cls()
        scaler.mean = torch.tensor(payload["mean"], dtype=torch.float32)
        scaler.std = torch.tensor(payload["std"], dtype=torch.float32)
        return scaler


def make_loaders(
    train_list: str,
    val_list: str,
    test_list: str,
    batch_size: int = 512,
    num_workers: int = 0,
    cols: Dict[str, str] = Default_Cols,
    input: List[str] = Input,
    output: List[str] = Output,
):
    train_files = load_file_list(train_list)
    val_files = load_file_list(val_list)
    test_files = load_file_list(test_list)

    train_dataset = BCDataset(train_files, cols, input, output)
    val_dataset = BCDataset(val_files, cols, input, output)
    test_dataset = BCDataset(test_files, cols, input, output)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def make_train_val_loaders(
    train_list: str,
    val_list: str,
    batch_size: int = 512,
    num_workers: int = 0,
    cols: Dict[str, str] = Default_Cols,
    input: List[str] = Input,
    output: List[str] = Output,
):
    """
    Train/Val 전용 데이터셋/로더 생성 유틸. 테스트 분리용.
    """
    train_files = load_file_list(train_list)
    val_files = load_file_list(val_list)

    train_dataset = BCDataset(train_files, cols, input, output)
    val_dataset = BCDataset(val_files, cols, input, output)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dataset, val_dataset, train_loader, val_loader




