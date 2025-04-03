from pathlib import Path
import pandas as pd
from sklearn.model_selection import KFold
from utils import get_annotations, parallel_preprocess_dataset
from dataset import DashcamDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import TrafficTransformer
from loss import TrafficLoss


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


data_base_dir = Path("./nexar-collision-prediction")
df_train = pd.read_csv(data_base_dir / "sample.csv")
df_train["time_of_event"] = df_train["time_of_event"].fillna(0)
df_train["time_of_alert"] = df_train["time_of_alert"].fillna(0)


annotations = get_annotations(df_train)
video_dir = data_base_dir / "train"
video_ids = list(annotations.keys())
processed_data = parallel_preprocess_dataset(
    video_dir, video_ids, num_frames=32, num_workers=4
)

# model init
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = TrafficTransformer(image_size=240).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

loss_fn = TrafficLoss()


# Set up k-fold validation
k = 5  # Number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Example: Iterate through each fold
for fold, (train_idx, val_idx) in enumerate(kf.split(df_train)):
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
        processed_data=train_frames, annotations=train_annons
    )
    val_dataset = DashcamDataset(processed_data=val_frames, annotations=val_annons)

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    train_loss = train(model, train_dataloader, loss_fn, optimizer, device=device)
    val_loss = eval(model, val_dataloader, loss_fn, device=device)

    print(
        f"Fold [{fold+1}/{k}] -> train loss: {train_loss:.3f}, val loss: {val_loss:.3f}"
    )
