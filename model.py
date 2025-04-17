import torch
import torch.nn as nn
from torchvision.models import vit_b_16
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import get_model_device


class TrafficTransformer(nn.Module):

    def __init__(
        self,
        image_size: int = 224,
        output_dim: int = 64,
        writer: SummaryWriter = None,
        layer_outputs: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.hidden_dim = 768
        self.output_dim = output_dim
        self.image_size = image_size
        self.writer = writer
        self.layer_outputs = defaultdict(list)

        self.backbone = vit_b_16(weights="DEFAULT", image_size=self.image_size)
        self.backbone.heads = nn.Sequential()

        self.classifier = nn.Sequential(
            nn.Linear(self.output_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.regressor = nn.Sequential(
            nn.Linear(self.output_dim, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1)
        )

        self.proj_layer = nn.Linear(self.hidden_dim, self.output_dim)
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim), nn.ReLU()
        )

    def forward(self, x):

        batch_size, seq_len, ch, h, w = x.size()
        features = []

        # pass individual frames through ViT
        for t in range(seq_len):
            feats = self.backbone(x[:, t])
            features.append(feats)

        # concat along time dimension
        x = torch.stack(features, dim=1)

        if self.layer_outputs:
            v, i = x.detach().cpu().median(dim=2)
            b, _ = v.median(dim=1)
            self.layer_outputs["backbone"].append(np.median(b.numpy()))

        # project features to lower dimension for classification and regression
        x = self.proj_layer(x)

        if self.layer_outputs:
            v, i = x.detach().cpu().median(dim=2)
            pl, _ = v.median(dim=1)
            self.layer_outputs["proj_layer"].append(np.median(pl.numpy()))

        # get mean representation of sequence
        x = x.mean(dim=1)  # [batch_size, T, output_dim]
        x = self.fusion_layer(x)  # [batch_size, output_dim]

        if self.layer_outputs:
            fl, _ = x.detach().cpu().median(dim=1)
            self.layer_outputs["fusion_layer"].append(np.median(fl.numpy()))

        pred_scores = self.classifier(x)
        pred_alerts = self.regressor(x)

        if self.layer_outputs:
            c = pred_scores.detach().cpu().numpy().flatten()
            self.layer_outputs["classifier"].append(np.median(c))

            p = pred_alerts.detach().cpu().numpy().flatten()
            self.layer_outputs["regressor"].append(np.median(p))

        return pred_scores, pred_alerts

    @torch.no_grad()
    def batch_predict(self, dataloader) -> tuple[list, list]:
        """
        Performs batch predictions to return scores and video IDs.
        Args:
            model: The trained model for inference.
            dataloader: DataLoader for the dataset.
            device: Device to perform inference on (default: "cuda").
        Returns:
            scores: List of predicted scores.
            events: List of predicted event times.
            video_ids: List of video IDs corresponding to the predictions.
        """
        self.eval()

        scores = []
        events = []
        video_ids = []

        device = get_model_device(self)

        for batch in tqdm(dataloader):

            # Transfer data to the device
            frames = batch["frames"].to(device)
            batch_video_ids = batch["video_id"]

            pred_scores, pred_alerts = self.forward(frames)

            scores.extend(pred_scores.cpu().numpy())
            events.extend(pred_alerts.cpu().numpy())
            video_ids.extend(batch_video_ids)

            del frames, pred_scores, pred_alerts
            if device.type == "cuda":
                torch.cuda.empty_cache()

        return scores, events, video_ids

    def plot_grad_flow(self, iter: int):
        """
        Plot gradient flow through the network to Tensorboard.
        Args:
            iter: iteration number
        """

        if self.writer is None:
            raise ValueError("No writer provided to model for Tensorboard logging.")

        ave_grads = []
        layers = []
        for n, p in self.named_parameters():
            if (p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().cpu().numpy().item())

        with plt.ioff():
            plt.plot(ave_grads, alpha=0.3, color="b")
            plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
            plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
            plt.xlim(xmin=0, xmax=len(ave_grads))
            plt.xlabel("Layers")
            plt.ylabel("average gradient")
            plt.title("Gradient flow")
            plt.grid(True)
            self.writer.add_figure("Gradient flow", plt.gcf(), iter)
            plt.clf()
