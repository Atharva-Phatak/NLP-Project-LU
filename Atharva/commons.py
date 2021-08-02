"""This file contains dataclasses which hold parameters and data paths"""
from enum import Enum
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class DataPaths:
    """Paths to Dataset."""
    train_df_path: str =  "datasets/train.csv"
    val_df_path: str = "datasets/dev.csv"
    gold_test_df_path: str = "datasets/test.csv"


@dataclass
class MultiTaskParams:
    """Dataclass holding parameters for multi-task setting."""
    backbone_name: str = "roberta-base"
    batch_size: int = 32
    max_len: int = 256
    num_epochs: int = 5
    seed: int = 42
    inference_batch_size: int = 64


@dataclass
class SingleTaskParams:
    """Dataclass holding parameters for single-task."""
    backbone_name: str = "prajjwal1/bert-small"
    max_len: int = 256
    batch_size: int = 16
    num_epochs: int = 5
    seed: int = 42
    inference_batch_size: int = 16


class TASKS(Enum):
    IS_HUMOR = "is_humor"
    HUMOR_CONTROVERSY = "humor_controversy"
    HUMOR_RATING = "humor_rating"
    OFFENSE_RATING = "offense_rating"


def multitaskloss(op, y):
    """Function for calculating loss in multitask setting."""
    ids = y[TASKS.IS_HUMOR.value] == 1
    is_humor_loss = F.cross_entropy(op[TASKS.IS_HUMOR.value], y[TASKS.IS_HUMOR.value].long())
    humor_controvery_loss = F.cross_entropy(
        op[TASKS.HUMOR_CONTROVERSY.value][ids], y[TASKS.HUMOR_CONTROVERSY.value][ids].long()
    )
    humor_rating_loss = F.mse_loss(op[TASKS.HUMOR_RATING.value][ids], y[TASKS.HUMOR_RATING.value][ids].view(-1, 1))

    return (is_humor_loss + humor_controvery_loss + humor_rating_loss).float()
