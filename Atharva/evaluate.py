import argparse
import torch
from torchflare.experiments import to_device
import sklearn.metrics as skm
from commons import TASKS, MultiTaskParams, SingleTaskParams, DataPaths
from archs import BackBoneModel, HydraNetwork
from torchflare.datasets import TextDataloader
import transformers
import pandas as pd
from dataset import MultiTaskDataset


def init_nets(task):
    """Init the models."""
    if task == "stm":
        model = BackBoneModel(out_features=2, model_path=SingleTaskParams.backbone_name)
    elif task == "mtm":
        model = HydraNetwork(model_path=MultiTaskParams.backbone_name)
    return model


def load_weights(model, path):
    """Load trained weights fo the model."""
    ckpt = torch.load(path)
    if "model_state_dict" in ckpt.keys():
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    return model


def init_dl(task):
    """Create dataloaders according to task."""
    test_df = pd.read_csv(DataPaths.gold_test_df_path)
    if task == "stm":
        tokenizer = transformers.AutoTokenizer.from_pretrained(SingleTaskParams.backbone_name)

        test_dl = TextDataloader.from_df(
            df=test_df,
            input_col="text",
            label_cols=None,
            tokenizer=tokenizer,
            max_len=256,
        ).get_loader(batch_size=16, shuffle=False)

    elif task == "mtm":
        test_ds = MultiTaskDataset(
            csv_path=DataPaths.gold_test_df_path, backbone_name=MultiTaskParams.backbone_name, max_len=256
        )
        test_dl = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False)
    return test_dl, test_df


def single_task_inference(model, test_dl):
    """Run single task inference."""
    preds = []
    for batch in test_dl:
        batch = to_device(batch, device="cuda")
        with torch.no_grad():
            op = model(batch)
        preds.extend(torch.argmax(op, dim=1).cpu().numpy())
    return preds


def multi_task_inference(model, test_dl):
    """Run multitask inference."""
    is_humor = []
    humor_contro = []
    humor_rating = []
    for batch in test_dl:
        batch = to_device(batch[0], device="cuda")
        with torch.no_grad():
            op = model(batch)
        is_humor.extend(torch.argmax(op[TASKS.IS_HUMOR.value], dim=1).cpu().numpy())
        humor_contro.extend(torch.argmax(op[TASKS.HUMOR_CONTROVERSY.value], dim=1).cpu().numpy())
        humor_rating.extend(op[TASKS.HUMOR_RATING.value].cpu().numpy())
    return is_humor, humor_contro, humor_rating


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="mtm", help="The task to run inference.")
    parser.add_argument("--model_path", type=str, default=None, help="path to saved weights of trained model(.bin file)")
    args = parser.parse_args()
    model = init_nets(task=args.task)
    test_df, test_dl = init_dl(task=args.task)
    model = load_weights(model=model, path=args.model_path)

    if args.task == "mtm":
        is_humor, humor_contro, humor_rating = multi_task_inference(model=model, test_dl=test_dl)
        print(f"Task 1A-Gold-Test(F1-Score) : {skm.f1_score(test_df.loc[:, 'is_humor'].values, is_humor)}")
        print(
            f"Task 1B-Gold-Test(RMSE) : {skm.mean_squared_error(test_df.loc[:, 'humor_rating'].values, humor_rating, squared=False)}"
        )
        print(f"Task 1C-Gold-Test(F1-Score) : {skm.f1_score(test_df.loc[:, 'humor_controversy'].values, humor_contro)}")
    elif args.task == "stm":
        preds = single_task_inference(model=model, test_dl=test_dl)
        print(f"Task 1A-Gold-Test(F1-Score) : {skm.f1_score(test_df.is_humor.values, preds)}")
