import torch
import torch.nn as nn
from torchvision.models import vit_b_16


class TrafficTransformer(nn.Module):

    def __init__(self, image_size=160, output_dim=64, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hidden_dim = 768
        self.output_dim = output_dim
        self.image_size = image_size

        self.backbone = vit_b_16(image_size=self.image_size)
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

        # project features to lower dimension for classification and regression
        x = self.proj_layer(x)

        # get mean representation of sequence
        x = x.mean(dim=1)  # [batch_size, T, output_dim]
        x = self.fusion_layer(x)  # [batch_size, output_dim]

        pred_scores = self.classifier(x)
        pred_alerts = self.regressor(x)

        return pred_scores, pred_alerts
