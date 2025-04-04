from pathlib import Path
import pandas as pd
from sklearn.model_selection import KFold
from utils import get_annotations, parallel_preprocess_dataset
    save_model,
from dataset import DashcamDataset
from transforms import basic_transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import TrafficTransformer
from loss import TrafficLoss
import argparse
from torch.utils.tensorboard import SummaryWriter
import logging
import time


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

    parser.add_argument(
        "--tb_logging",
        action="store_true",
        default=False,
        help="Flag to enable TensorBoard logging",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Loading training data...")
    data_base_dir = Path("./nexar-collision-prediction")
    df_train = pd.read_csv(data_base_dir / "train.csv")

    df_train["time_of_event"] = df_train["time_of_event"].fillna(0)
    df_train["time_of_alert"] = df_train["time_of_alert"].fillna(0)

    logger.info("Extracting annotations and key frames from video data...")
    start_time = time.time()
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
    elapsed_time = (time.time() - start_time) / 60
    logger.info(f"Processed data in {elapsed_time:.2f} minutes.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_fn = TrafficLoss()

    if args.tb_logging:
        writer = SummaryWriter()

    tb_tag = "{}, Fold[{}]"

    # Set up k-fold validation
    kf = KFold(n_splits=args.k, shuffle=True, random_state=42)

    logger.info(f"Performing cross-validation with {args.k} folds...")

    for fold, (train_idx, val_idx) in enumerate(kf.split(df_train)):

        # model init
        model = TrafficTransformer(image_size=args.image_size).to(device)
        logger.info(f"Initialized TrafficTransormer and moved to {device}")
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

        best_loss = float("inf")

        state_dict = {
            "model": model.state_dict(),
            "epoch": -1,
        }

        fold_str = f"{fold+1}/{args.k}"
        train_tag = tb_tag.format(
            "Train loss",
            fold_str,
        )

        val_tag = tb_tag.format(
            "Val loss",
            fold_str,
        )

        # train model on K-1 folds
        for epoch in range(args.num_epochs):

            start_time = time.time()
            logger.info(f"Training epoch [{epoch+1}/{args.num_epochs}]...")
            train_loss = train(
                model,
                train_dataloader,
                loss_fn,
                optimizer,
                device=device,
            )

            elapsed_time = (time.time() - start_time) / 60
            logger.info(f"Training completed in {elapsed_time:.2f} minutes")

            start_time = time.time()
            logger.info(f"Validating model on hold-out set...")
            val_loss = eval(model, val_dataloader, loss_fn, device=device)

            elapsed_time = (time.time() - start_time) / 60
            logger.info(f"Validation completed in {elapsed_time:.2f} minutes")

            if val_loss < best_loss:
                best_loss = val_loss
                state_dict["model"] = model.state_dict()
                state_dict["epoch"] = epoch + 1
                logger.info(f"Best validation loss updated to: {best_loss:.3f}")

            log = (
                tb_tag.format("", fold_str)
                + f"-> train loss: {train_loss:.3f}, val_loss: {val_loss:.3f}"
            )

            logger.info(log)

            if args.tb_logging:
                writer.add_scalar(train_tag, train_loss, epoch)
                writer.add_scalar(val_tag, val_loss, epoch)

        save_model(state_dict, f"best_model_fold_{fold+1}.pth")
