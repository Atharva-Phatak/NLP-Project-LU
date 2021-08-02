import transformers
import torch.nn as nn
import torch
from commons import TASKS


class HydraNetwork(nn.Module):
    """Architecture used for models in Multi-Task-Setting."""

    def __init__(self, model_path):
        super(HydraNetwork, self).__init__()

        self.model = transformers.AutoModel.from_pretrained(model_path, return_dict=False)
        in_features = self.model.pooler.dense.out_features
        self.is_humor_head = torch.nn.Linear(in_features=in_features, out_features=2)
        self.humor_controversy_head = torch.nn.Linear(in_features=in_features, out_features=2)
        self.humor_rating_head = torch.nn.Linear(in_features=in_features, out_features=1)

    def forward(self, x):
        _, o_2 = self.model(**x)
        return {
            TASKS.IS_HUMOR.value: self.is_humor_head(o_2),
            TASKS.HUMOR_CONTROVERSY.value: self.humor_controversy_head(o_2),
            TASKS.HUMOR_RATING.value: self.humor_rating_head(o_2),
        }


class BackBoneModel(torch.nn.Module):
    """Architecture used by models in Single-Task-Setting."""

    def __init__(self, out_features, model_path):
        super(BackBoneModel, self).__init__()
        self.model = transformers.AutoModel.from_pretrained(model_path, return_dict=False)
        in_features = self.model.pooler.dense.out_features
        self.linear = torch.nn.Linear(in_features, out_features)
        # self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x):

        _, o_2 = self.model(**x)
        op = self.linear(o_2)
        return op
