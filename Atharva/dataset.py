import transformers
import torch
from commons import TASKS
import pandas as pd


class MultiTaskDataset:
    """A PyTorch style Dataset for Multi-Task setting."""
    def __init__(self, csv_path, backbone_name, max_len):
        self.data = self.read_df(path=csv_path)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(backbone_name)
        self.max_len = max_len

    @staticmethod
    def read_df(path):
        df = pd.read_csv(path)
        df = df.fillna(0)
        df[TASKS.HUMOR_CONTROVERSY.value] = df[TASKS.HUMOR_CONTROVERSY.value].astype("int")
        return df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inps = self.tokenizer(
            self.data.iloc[idx]["text"],
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )
        inps = {k: v.squeeze(0) for k, v in inps.items()}
        label = {
            TASKS.IS_HUMOR.value: torch.tensor(self.data.iloc[idx][TASKS.IS_HUMOR.value]),
            TASKS.HUMOR_CONTROVERSY.value: torch.tensor(self.data.iloc[idx][TASKS.HUMOR_CONTROVERSY.value]),
            TASKS.HUMOR_RATING.value: torch.tensor(self.data.iloc[idx][TASKS.HUMOR_RATING.value], dtype=torch.float),
        }

        return inps, label
