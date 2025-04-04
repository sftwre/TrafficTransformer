from pathlib import Path
import pandas as pd
from sklearn.model_selection import KFold
from utils import get_annotations, parallel_preprocess_dataset
from dataset import DashcamDataset
from transforms import basic_transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import TrafficTransformer
from loss import TrafficLoss
import argparse


def train(model, dataloader, loss_fn, optimizer, device="cpu"):

    # set model in training mode
    model.train()

    batch_loss = 0

    for batch in dataloader:
        optimizer.zero_grad()

        frames = batch["frames"].to(device)
        labels = batch["label"].unsqueeze(1).to(device)
        alert_time = batch["alert_time"].unsqueeze(1).to(device)

        pred_scores, pred_alerts = model(frames)

        inputs = {"pred_scores": pred_scores, "pred_alerts": pred_alerts}
        targets = {"label": labels, "alert_time": alert_time}

        loss = loss_fn(inputs=inputs, targets=targets)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_loss += loss.cpu().item()

    epoch_loss = batch_loss / len(dataloader)
    return epoch_loss


@torch.no_grad()
def eval(model, dataloader, loss_fn, device="cpu"):

    model.eval()

    batch_loss = 0

    for batch in dataloader:
        frames = batch["frames"].to(device)
        labels = batch["label"].unsqueeze(1).to(device)
        alert_time = batch["alert_time"].unsqueeze(1).to(device)
        pred_scores, pred_alerts = model(frames)

        inputs = {"pred_scores": pred_scores, "pred_alerts": pred_alerts}
        targets = {"label": labels, "alert_time": alert_time}
        loss = loss_fn(inputs=inputs, targets=targets)
        batch_loss += loss.cpu().item()

    eval_loss = batch_loss / len(dataloader)
    return eval_loss


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train a Transformer model to predict crash events and event time."
    )
    parser.add_argument(
        "--num_epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-3,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--k", type=int, default=2, help="Number of cross-validation folds"
    )
    parser.add_argument(
        "--image_size", type=int, default=224, help="Input image dimension for training"
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=16,
        help="Number of image frames to sample from videos",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=4,
        help="Number of processes to use for data loading",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of train/val samples in each batch",
    )
    args = parser.parse_args()

    data_base_dir = Path("./nexar-collision-prediction")
    df_train = pd.read_csv(data_base_dir / "train.csv")

    df_train["time_of_event"] = df_train["time_of_event"].fillna(0)
    df_train["time_of_alert"] = df_train["time_of_alert"].fillna(0)


annotations = get_annotations(df_train)
    annotations = get_annotations(df_train)

    video_dir = data_base_dir / "train"
    video_ids = list(annotations.keys())
    processed_data = parallel_preprocess_dataset(
        video_dir,
        video_ids,
        num_frames=args.n_frames,
        image_size=args.image_size,
        num_workers=args.n_workers,
    )
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_fn = TrafficLoss()


    # Set up k-fold validation
    kf = KFold(n_splits=args.k, shuffle=True, random_state=42)


    for fold, (train_idx, val_idx) in enumerate(kf.split(df_train)):

        # model init
        model = TrafficTransformer(image_size=args.image_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        train_data = df_train.iloc[train_idx]
        val_data = df_train.iloc[val_idx]

        train_annons = dict()
        train_frames = dict()

        for id in train_data.id.tolist():
            id = f"{id:05d}"
            train_annons[id] = annotations[id]
            train_frames[id] = processed_data[id]

        val_annons = dict()
        val_frames = dict()

        for id in val_data.id.tolist():
            id = f"{id:05d}"
            val_annons[id] = annotations[id]
            val_frames[id] = processed_data[id]

        train_dataset = DashcamDataset(
            processed_data=train_frames,
            annotations=train_annons,
            transform=basic_transforms,
        )
        val_dataset = DashcamDataset(
            processed_data=val_frames,
            annotations=val_annons,
            transform=basic_transforms,
        )

        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=True
        )

    train_loss = train(model, train_dataloader, loss_fn, optimizer, device=device)
    val_loss = eval(model, val_dataloader, loss_fn, device=device)

        # train model on K-1 folds
        for epoch in range(args.num_epochs):

            train_loss = train(
                model,
                train_dataloader,
                loss_fn,
                optimizer,
                device=device,
            )
            val_loss = eval(model, val_dataloader, loss_fn, device=device)
