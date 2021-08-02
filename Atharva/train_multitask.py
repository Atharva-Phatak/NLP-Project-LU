from commons import MultiTaskParams, multitaskloss, DataPaths, TASKS
from mtm_metrics import SklearnF1, SklearnMSE
from dataset import MultiTaskDataset
from multitask_trainer import MultiTaskTrainer
from torch.utils.data import DataLoader
from torchflare.experiments import ModelConfig
from archs import HydraNetwork
import torch

# Creating multi-task dataset for training and validation
train_ds = MultiTaskDataset(
    csv_path=DataPaths.train_df_path, backbone_name=MultiTaskParams.backbone_name, max_len=MultiTaskParams.max_len
)
val_ds = MultiTaskDataset(
    csv_path=DataPaths.val_df_path, backbone_name=MultiTaskParams.backbone_name, max_len=MultiTaskParams.max_len
)

# Creating training and validation dataloaders
train_dl = DataLoader(train_ds, batch_size=MultiTaskParams.batch_size, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=MultiTaskParams.batch_size, shuffle=False)

# Configuration for experiment as required by TorchFlare
config = ModelConfig(
    nn_module=HydraNetwork,
    module_params={"model_path": MultiTaskParams.backbone_name},
    optimizer="Adam",
    optimizer_params={"lr": 3e-5},
    criterion=multitaskloss,
)

# List of metrics we are going to use.
metric_list = [
    SklearnF1(target_name=TASKS.IS_HUMOR.value),
    SklearnF1(target_name=TASKS.HUMOR_CONTROVERSY.value),
    SklearnMSE(target_name=TASKS.HUMOR_RATING.value),
]

if __name__ == "__main__":
    multitask_exp = MultiTaskTrainer(
        num_epochs=MultiTaskParams.num_epochs, seed=MultiTaskParams.seed, fp16=True, device="cuda"
    )
    multitask_exp.compile_experiment(model_config=config, callbacks=None, metrics=metric_list)
    multitask_exp.fit_loader(train_dl, val_dl)
    torch.save(multitask_exp.state.model.state_dict(), "roberta--multitask.bin")
