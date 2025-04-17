# pytorch imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# stl imports
import time
import logging
import argparse
from pathlib import Path

# local imports
from dataset import DashcamDataset
from transforms import basic_transforms
from model import TrafficTransformer
from loss import TrafficLoss
from utils import save_model, get_lr_scheduler, get_model_device
from preprocessing import parallel_preprocess_dataset, get_annotations

# aux imports
import pandas as pd
from sklearn.model_selection import KFold

global_step = 0

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train(
    model,
    dataloader,
    criterion,
    optimizer,
    lr_scheduler=None,
    eval_iter=150,
    **kwargs,
):

    # set model in training mode
    model.train()

    device = get_model_device(model)
    iter_loss = batch_loss = 0

    # set optional args
    clip_grad = kwargs.get("clip_grad", False)
    max_norm = kwargs.get("max_norm", 1.0)
    grad_flow = kwargs.get("grad_flow", False)

    for batch in dataloader:

        frames = batch["frames"].to(device)
        labels = batch["label"].unsqueeze(1).to(device)
        alert_time = batch["alert_time"].unsqueeze(1).to(device)

        # update iteration step
        global global_step
        global_step += 1

        pred_scores, pred_alerts = model(frames)

        # log layer outputs if it's enabled
        if model.writer and model.layer_outputs:
            model.writer.add_scalar(
                "backbone", model.layer_outputs["backbone"][-1], global_step
            )
            model.writer.add_scalar(
                "proj_layer", model.layer_outputs["proj_layer"][-1], global_step
            )
            model.writer.add_scalar(
                "fusion_layer", model.layer_outputs["fusion_layer"][-1], global_step
            )
            model.writer.add_scalar(
                "classifier", model.layer_outputs["classifier"][-1], global_step
            )
            model.writer.add_scalar(
                "regressor", model.layer_outputs["regressor"][-1], global_step
            )

        inputs = {"pred_scores": pred_scores, "pred_alerts": pred_alerts}
        targets = {"label": labels, "alert_time": alert_time}

        # compute loss and grads
        loss = criterion(inputs=inputs, targets=targets)
        loss.backward()

        # clip gradients if enabled
        if clip_grad:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

        # plot gradient flow through network
        if grad_flow and global_step % eval_iter == 0:
            model.plot_grad_flow(global_step)

        optimizer.step()
        optimizer.zero_grad()

        batch_loss += loss.cpu().item()
        iter_loss += loss.cpu().item()

        # step through lr_scheduler
        if lr_scheduler:

            match lr_scheduler.__class__.__name__:
                case "ReduceLROnPlateau":
                    if global_step % eval_iter == 0:
                        # reduce learning rate if loss plateaus
                        lr_scheduler.step(iter_loss / eval_iter)
                        iter_loss = 0

                case "OneCycleLR":
                    lr_scheduler.step()

        # delete reference count to reduce memory usage
        del frames, labels, alert_time, pred_scores, pred_alerts, loss

        if device.type == "cuda":
            torch.cuda.empty_cache()

    epoch_loss = batch_loss / len(dataloader)

    return epoch_loss


@torch.no_grad()
def validate(model, dataloader, loss_fn, device="cpu"):

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


def init_model(**kwargs) -> nn.Module:
    """
    Initializes the TrafficTransformer model with untrained weights on the specified device.

    Args:
        image_size (int): Input image dimension for training.
        writer (SummaryWriter): TensorBoard writer for logging.
        device (torch.device): Device to load the model on
        checkpoint_path (str): Path to pre-trained weights.

    Returns:
        nn.Module: Initialized TrafficTransformer model.
    """
    # init model
    model = TrafficTransformer(
        image_size=kwargs["image_size"],
        writer=kwargs["writer"],
        layer_outputs=kwargs["layer_outputs"],
    )

    logger.info(f"Initialized TrafficTransformer")

    # load model checkpoint if provided
    if kwargs["checkpoint_path"]:
        logger.info(f"Loading model checkpoint from -> {kwargs['checkpoint_path']}")
        state_dict = torch.load(kwargs["checkpoint_path"])
        model.load_state_dict(state_dict["model"])
        logger.info("Model checkpoint loaded successfully")

    model.to(kwargs["device"])
    logger.info(f"Model moved to {kwargs['device']}")
    return model


def train_driver(
    model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, **kwargs
) -> dict:
    """
    Driver function to train the model over a set number of epochs.

    Args:
        model: The model to be trained.
        train_dataloader: DataLoader for the training dataset.
        val_dataloader: DataLoader for the validation dataset.
        criterion: Loss function.
        optimizer: Optimizer for the model.
        num_epochs: Number of epochs to train the model.
        kwargs: Additional arguments such as device, lr_scheduler, etc.
    Returns:
        dict: State dictionary containing optimal parameters on the validation set.
    """

    best_loss = float("inf")

    state_dict = {
        "model": model.state_dict(),
        "epoch": -1,
    }

    lr_scheduler = kwargs.get("lr_scheduler", None)

    for epoch in range(num_epochs):
        start_time = time.time()
        logger.info(f"Training epoch [{epoch+1}/{num_epochs}]...")
        train_loss = train(
            model,
            train_dataloader,
            criterion,
            optimizer,
            lr_scheduler=lr_scheduler,
            clip_grad=kwargs["clip_grad"],
            max_norm=kwargs["max_norm"],
            grad_flow=kwargs["grad_flow"],
        )

        elapsed_time = (time.time() - start_time) / 60
        logger.info(f"Training completed in {elapsed_time:.2f} minutes")

        start_time = time.time()
        logger.info(f"Validating model on hold-out set...")
        val_loss = validate(model, val_dataloader, criterion, device=device)

        elapsed_time = (time.time() - start_time) / 60
        logger.info(f"Validation completed in {elapsed_time:.2f} minutes")

        if val_loss < best_loss:
            best_loss = val_loss
            state_dict["model"] = model.state_dict()
            state_dict["epoch"] = epoch + 1
            logger.info(f"Best validation loss updated to: {best_loss:.3f}")

        log = f"train loss: {train_loss:.3f}, val_loss: {val_loss:.3f}"
        logger.info(log)

        if model.writer:
            train_tag = kwargs.get("train_tag", "Train loss")
            val_tag = kwargs.get("val_tag", "Val loss")
            lr_tag = kwargs.get("lr_tag", "LR Scheduler")

            model.writer.add_scalar(train_tag, train_loss, epoch)
            model.writer.add_scalar(val_tag, val_loss, epoch)

            if lr_scheduler:
                model.writer.add_scalar(lr_tag, lr_scheduler.get_last_lr()[0], epoch)

    return state_dict


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
        "--filename",
        type=str,
        default="traffic_transformer.pth",
        help="Filename to save the trained model",
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
        help="Path to model checkpoint for loading",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="sample.csv",
        help="Filename with training data",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="plateau",
        choices=["", "plateau", "one_cycle"],
        help="Type of learning rate scheduler to use",
    )
    parser.add_argument(
        "--k", type=int, default=0, help="Number of cross-validation folds"
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
        default=4,
        help="Number of train/val samples in each batch",
    )

    parser.add_argument(
        "--tb_logging",
        action="store_true",
        default=False,
        help="Flag to enable TensorBoard logging",
    )
    parser.add_argument("--grad_flow", action="store_true", default=False)
    parser.add_argument("--layer_outputs", action="store_true", default=False)
    parser.add_argument(
        "--clip_grad",
        action="store_true",
        default=False,
        help="Flag to enable gradient clipping",
    )
    parser.add_argument(
        "--max_norm",
        type=float,
        default=1.0,
        help="Maximum norm for gradient clipping",
    )

    args = parser.parse_args()

    logger.info("Loading training data...")
    data_base_dir = Path("./nexar-collision-prediction")

    try:
        df_train = pd.read_csv(data_base_dir / args.input_file)
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        raise

    df_train["time_of_event"] = df_train["time_of_event"].fillna(0)
    df_train["time_of_alert"] = df_train["time_of_alert"].fillna(0)

    logger.info("Extracting annotations and key frames from training data...")
    start_time = time.time()
    annotations = get_annotations(df_train)

    if args.input_file in ["train.csv", "sample.csv"]:
        df_val = pd.read_csv(data_base_dir / "val.csv")
        df_val["time_of_event"] = df_val["time_of_event"].fillna(0)
        df_val["time_of_alert"] = df_val["time_of_alert"].fillna(0)

        annotations.update(get_annotations(df_val))

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
    logger.info(f"Data preprocessing completed in {elapsed_time:.2f} minutes.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    traffic_loss = TrafficLoss()

    writer = None
    if args.tb_logging:
        writer = SummaryWriter()

    torch.manual_seed(1234)

    try:

        # Set up k-fold validation
        if args.k > 0:

            logger.info(f"Performing cross-validation with {args.k} folds...")

            # tensorboard tag to seperate training information by fold
            tb_tag = "{}, Fold {}"
            kf = KFold(n_splits=args.k, shuffle=True, random_state=42)

            for fold, (train_idx, val_idx) in enumerate(kf.split(df_train)):

                model = init_model(
                    image_size=args.image_size,
                    checkpoint_path=args.checkpoint_path,
                    device=device,
                    writer=writer,
                    layer_outputs=args.layer_outputs,
                )

                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

                lr_scheduler = get_lr_scheduler(
                    args.lr_scheduler,
                    optimizer,
                    args.num_epochs,
                    steps_per_epoch=train_idx.shape[0] // args.batch_size,
                )

                if lr_scheduler:
                    scheduler_name = lr_scheduler.__class__.__name__
                    logger.info(
                        f"Training with {scheduler_name} learning rate scheduler"
                    )

                # Use K-1 folds for training and 1 fold for validation
                train_data = df_train.iloc[train_idx]
                val_data = df_train.iloc[val_idx]

                # extract training data by id
                train_annons = dict()
                train_frames = dict()

                for id in train_data.id.tolist():
                    id = f"{id:05d}"
                    train_annons[id] = annotations[id]
                    train_frames[id] = processed_data[id]

                # extract validation data by id
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

                fold_str = f"{fold+1}"

                train_tag = tb_tag.format(
                    "Train loss",
                    fold_str,
                )
                val_tag = tb_tag.format(
                    "Val loss",
                    fold_str,
                )

                lr_tag = tb_tag.format(
                    "LR Scheduler",
                    fold_str,
                )

                logger.info(f"Fold {fold_str} training started...")

                state_dict = train_driver(
                    model,
                    train_dataloader,
                    val_dataloader,
                    traffic_loss,
                    optimizer,
                    args.num_epochs,
                    lr_scheduler=lr_scheduler,
                    clip_grad=args.clip_grad,
                    max_norm=args.max_norm,
                    grad_flow=args.grad_flow,
                    train_tag=train_tag,
                    val_tag=val_tag,
                    lr_tag=lr_tag,
                )

                filename = f"traffic_transformer_fold_{fold_str}.pth"
                save_model(state_dict, filename)

        else:
            logger.info("Initializing model...")
            start_time = time.time()
            model = init_model(
                image_size=args.image_size,
                checkpoint_path=args.checkpoint_path,
                device=device,
                writer=writer,
                layer_outputs=args.layer_outputs,
            )

            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

            lr_scheduler = get_lr_scheduler(
                args.lr_scheduler,
                optimizer,
                args.num_epochs,
                steps_per_epoch=len(processed_data) // args.batch_size,
            )

            if lr_scheduler:
                scheduler_name = lr_scheduler.__class__.__name__
                logger.info(f"Training with {scheduler_name} learning rate scheduler")

            """
            Extract validation data by
            moving validation samples from main storage to validation storage by id
            and then remove from main.
            """
            val_annons = dict()
            val_frames = dict()

            for id in df_val.id.tolist():
                id = f"{id:05d}"
                val_annons[id] = annotations[id]
                val_frames[id] = processed_data[id]

                del annotations[id]
                del processed_data[id]

            train_dataset = DashcamDataset(
                processed_data=processed_data, annotations=annotations, transform=None
            )
            train_dataloader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True
            )

            val_dataset = DashcamDataset(
                processed_data=val_frames, annotations=val_annons, transform=None
            )
            val_dataloader = DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=True
            )

            logger.info(f"Training model for {args.num_epochs} Epochs...")

            state_dict = train_driver(
                model,
                train_dataloader,
                val_dataloader,
                traffic_loss,
                optimizer,
                args.num_epochs,
                lr_scheduler=lr_scheduler,
                clip_grad=args.clip_grad,
                max_norm=args.max_norm,
                grad_flow=args.grad_flow,
            )

            save_model(state_dict, args.filename)

    except Exception as e:
        logger.exception(f"{e}")
    finally:
        if args.tb_logging:
            writer.close()
