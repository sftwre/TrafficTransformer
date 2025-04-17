# torch imports
import torch
from torch.utils.data import DataLoader

# visualization imports
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# local imports
from utils import generate_submission
from dataset import DashcamDataset
from transforms import val_transforms
from model import TrafficTransformer
from preprocessing import get_annotations, parallel_preprocess_dataset

# stl imports
import sys
import time
from pathlib import Path
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# aux imports
import pandas as pd
import numpy as np


def gen_clf_plots(y_true: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
    """
    Generate classification plots: ROC, PR, and DET curves.
    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted scores.
    Returns:
        plt.Figure: Figure containing the plots.
    """

    with plt.ioff():
        # PR Curve
        precision, recall, _ = metrics.precision_recall_curve(y_true, y_pred)

        AP = metrics.average_precision_score(y_true, y_pred)
        pr_display = metrics.PrecisionRecallDisplay(
            precision=precision, recall=recall, average_precision=AP
        )

        # ROC Curve
        fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
        AUC = metrics.roc_auc_score(y_true, y_pred)
        roc_display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=AUC)

        # DET Curve
        fpr, fnr, _ = metrics.det_curve(y_true, y_pred)
        det_display = metrics.DetCurveDisplay(
            fpr=fpr, fnr=fnr, estimator_name="TrafficTransformer"
        )

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(28, 8))

        ax1.grid(True)
        ax1.set_title("ROC Curve")

        ax2.grid(True)
        ax2.set_title("Precision-Recall Curve")

        ax3.grid(True)
        ax3.set_title("DET Curve")

        roc_display.plot(ax=ax1)
        pr_display.plot(ax=ax2)
        det_display.plot(ax=ax3)

        return fig


def get_eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate evaluation metrics for the model.
    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    # Example values, replace with actual calculations
    r2 = metrics.r2_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    rmse = metrics.root_mean_squared_error(y_true, y_pred)

    return {
        "R^2": r2,
        "MAE": mae,
        "RMSE": rmse,
    }


def gen_reg_plots(y_true: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
    """
    Generate regression plots: Residual plot and Prediction error plot.
    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted scores.
    Returns:
        plt.Figure: Figure containing the plots.
    """

    with plt.ioff():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

        """
        Residual plot
        """
        residuals = y_true - y_pred

        ax1.scatter(y_true, residuals, color="dodgerblue", edgecolor="k", alpha=0.7)
        ax1.axhline(y=0, color="red", linestyle="--")
        t_delta = 2
        xticks = np.arange(0, y_true.max() + t_delta, t_delta)
        ax1.set_xticks(ticks=xticks, labels=xticks, rotation=45)
        ax1.set_xlabel("Actual time of event (seconds)")

        r_delta = 2
        yticks = np.arange(residuals.min(), residuals.max() + r_delta, r_delta)
        ax1.set_yticks(
            ticks=yticks, labels=list(map(lambda x: f"{x:.2f}", yticks.tolist()))
        )
        ax1.set_ylabel("Residual time (seconds)")
        ax1.set_title("Residual Plot")
        ax1.grid(True)

        eval_metrics = get_eval_metrics(y_true, y_pred)
        r2 = eval_metrics["R^2"]
        mae = eval_metrics["MAE"]
        rmse = eval_metrics["RMSE"]

        ax1.plot([], [], label=f"R^2: {r2:.2f}")
        ax1.plot([], [], label=f"MAE: {mae:.2f}")
        ax1.plot([], [], label=f"RMSE: {rmse:.2f}")

        ax1.legend(loc="upper left")

        """
        Prediction error plot
        """
        bin_width = 1
        bins = np.arange(residuals.min(), residuals.max() + bin_width, bin_width)

        counts, bins, _ = ax2.hist(
            residuals,
            bins=bins,
            color="skyblue",
            edgecolor="black",
            weights=np.ones(len(residuals)) / len(residuals),
        )
        ax2.set_xlabel("Residual (seconds)")
        ax2.set_ylabel("Percentage")
        ax2.set_title("Distribution of Residuals")

        # Format the y-axis
        heights = np.array([float(patch.get_height()) for patch in ax2.containers[0]])
        y_delta = 0.02
        yticks = np.arange(0, heights.max() + y_delta, y_delta)
        ylabels = list((yticks / heights.sum()) * 100)
        ylabels = list(map(lambda x: f"{x:.2f}%", ylabels))
        ax2.set_yticks(ticks=yticks, labels=ylabels)
        ax2.tick_params(axis="y", labelrotation=45)

        ax2.yaxis.set_major_formatter(PercentFormatter(1))

        ax2.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()

        return fig


if __name__ == "__main__":

    # Define CLI arguments
    parser = argparse.ArgumentParser(description="TrafficTransformer Evaluation Script")
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for DataLoader"
    )
    parser.add_argument(
        "--image_size", type=int, default=224, help="Image size for preprocessing"
    )
    parser.add_argument(
        "--n_frames", type=int, default=16, help="Number of frames per video"
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=16,
        help="Number of workers for data preprocessing",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Path to the model checkpoint file",
        required=True,
    )
    parser.add_argument(
        "--filename", type=str, default="submission.csv", help="Output filename"
    )

    parser.add_argument(
        "--submit",
        action="store_true",
        default=False,
        help="Flag to indicate whether to submit the results",
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset to evaluate on"
    )
    args = parser.parse_args()

    try:

        # Adjust the rest of the code to use the parsed arguments
        data_path = Path("./nexar-collision-prediction")

        logger.info(f"Loading {args.dataset} ...")
        df_eval = pd.read_csv(data_path / args.dataset)

        # set flag to determine which eval dataset to use
        bool_test = args.dataset == "test.csv"

        logger.info("Extracting annotations and key frames from video data...")
        start_time = time.time()
        eval_annons = get_annotations(df_eval, test=bool_test)

        video_dir = data_path / "test" if bool_test else data_path / "train"
        video_ids = list(eval_annons.keys())
        eval_processed = parallel_preprocess_dataset(
            video_dir,
            video_ids,
            num_frames=args.n_frames,
            num_workers=args.n_workers,
            image_size=args.image_size,
        )
        elapsed_time = (time.time() - start_time) / 60
        logger.info(f"Data preprocessing completed in {elapsed_time:.2f} minutes.")

        eval_dataset = DashcamDataset(eval_processed, eval_annons, val_transforms)
        eval_dataloader = DataLoader(
            eval_dataset, batch_size=args.batch_size, shuffle=False
        )

        logger.info("Initializing model for inference...")

        # load trained model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = TrafficTransformer()

        checkpoint_path = Path(args.checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        logger.info(f"Performing inference on {device.type}...")

        # predict crash events on test set
        scores, events, video_ids = model.batch_predict(eval_dataloader)

        # generate plots
        if not bool_test:

            # fill nan values with 0
            df_eval["time_of_event"] = df_eval["time_of_event"].fillna(0)
            df_eval["time_of_alert"] = df_eval["time_of_alert"].fillna(0)

            y_true_crash = df_eval.target.to_numpy()
            y_pred_crash = np.array(scores).flatten()

            y_true_event = df_eval.time_of_event.to_numpy()
            y_pred_event = np.array(events).flatten()

            logger.info("Generating evaluation plots...")
            plot_dir = Path("./plots") / checkpoint_path.stem
            plot_dir.mkdir(parents=True, exist_ok=True)

            fig = gen_clf_plots(y_true_crash, y_pred_crash)
            fig.savefig(plot_dir / "clf_plots.png")

            fig = gen_reg_plots(y_true_event, y_pred_event)
            fig.savefig(plot_dir / "reg_plots.png")

            logger.info(f"Evaluation plots saved to -> {plot_dir}.")

        # save predictions
        if args.submit:
            logger.info("Generating submission file...")
            output_path = data_path / "submission.csv"
            generate_submission(scores, video_ids, output_path=output_path)

    except Exception as e:
        logger.exception(e)
        sys.exit()
