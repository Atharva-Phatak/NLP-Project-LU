from torchflare.experiments import ModelConfig, Experiment
from stm_metrics import SklearnF1
from torchflare.datasets import TextDataloader
import torchflare.callbacks as cbs
from commons import SingleTaskParams, DataPaths
from archs import BackBoneModel
import transformers
import pandas as pd

tokenizer = transformers.AutoTokenizer.from_pretrained(SingleTaskParams.backbone_name)

# Reading training and validation data
train_df = pd.read_csv(DataPaths.train_df_path)
valid_df = pd.read_csv(DataPaths.val_df_path)

# Creating training and validation dataloaders
train_dl = TextDataloader.from_df(
    df=train_df,
    input_col="text",
    label_cols="is_humor",
    tokenizer=tokenizer,
    max_len=SingleTaskParams.max_len,
).get_loader(batch_size=SingleTaskParams.batch_size, shuffle=True)

valid_dl = TextDataloader.from_df(
    df=valid_df,
    input_col="text",
    label_cols="is_humor",
    tokenizer=tokenizer,
    max_len=SingleTaskParams.max_len,
).get_loader(batch_size=SingleTaskParams.batch_size, shuffle=False)

# Creating model-config for the experiment
config = ModelConfig(
    nn_module=BackBoneModel,
    module_params={"out_features": 2, "model_path": SingleTaskParams.backbone_name},
    optimizer="AdamW",
    optimizer_params={"lr": 2e-5, "weight_decay": 1e-3},
    criterion="cross_entropy",
)

# Defining callbacks
callbacks = [
    cbs.ModelCheckpoint(
        file_name="./bert-small.bin",
        save_dir="./",
        mode="max",
        monitor="val_f1_score",
    ),
    cbs.CosineAnnealingWarmRestarts(T_0=1),
]
# Defining metric list.
metric_list = [SklearnF1()]

if __name__ == "__main__":
    bert_exp = Experiment(num_epochs=SingleTaskParams.num_epochs, seed=SingleTaskParams.seed, fp16=True, device="cuda")
    bert_exp.compile_experiment(model_config=config, callbacks=callbacks, metrics=metric_list)
    bert_exp.fit_loader(train_dl, valid_dl)
