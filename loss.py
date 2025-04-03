import torch
import torch.nn.functional as F
from typing import Dict


class TrafficLoss(object):

    def __init__(self, w_cls: float = 1.0, w_reg: float = 1.0):
        """
        Computes combined classification and regression loss on nexar crash dataset.

        Args:
            w_cls: classification loss weighting
            w_reg: regression loss weighting
        """

        self.w_cls = w_cls
        self.w_reg = w_reg

    def __call__(
        self, inputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Combines classification and regression loss on targets

        Args:
            inputs: dictionary with predictions for crash event and event time
            targets: dictionary with g.t. data

        Returns:
            Total loss
        """

        cls_loss = F.binary_cross_entropy(
            input=inputs["pred_scores"], target=targets["label"]
        )
        reg_loss = F.mse_loss(input=inputs["pred_alerts"], target=targets["alert_time"])

        loss = self.w_cls * cls_loss + self.w_reg * reg_loss
        return loss
