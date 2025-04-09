# torch imports
import torch
from torch.utils.data import DataLoader

# local imports
from utils import generate_submission
from dataset import DashcamDataset
from transforms import val_transforms
from model import TrafficTransformer
from preprocessing import get_annotations, parallel_preprocess_dataset

# stl imports
import time
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# aux imports
import pandas as pd
from tqdm import tqdm


@torch.no_grad()
def predict(model, dataloader, device="cuda") -> tuple[list, list]:
    """
    Performs batch predictions to return scores and video IDs.
    Args:
        model: The trained model for inference.
        dataloader: DataLoader for the dataset.
        device: Device to perform inference on (default: "cuda").
    Returns:
        predictions: List of predicted scores.
        video_ids: List of video IDs corresponding to the predictions.
    """
    model = model.to(device)
    model.eval()

    predictions = []
    video_ids = []

    start_time = time.time()

    for batch in tqdm(dataloader):

        # Transfer data to the device
        frames = batch["frames"].to(device)
        batch_video_ids = batch["video_id"]

        scores, _ = model(frames)

        predictions.extend(scores.cpu().numpy())
        video_ids.extend(batch_video_ids)

        del frames, scores
        if device == "cuda":
            torch.cuda.empty_cache()

    elapsed_time = (time.time() - start_time) / 60
    logger.info(f"Inference completed in {elapsed_time:.2f} minutes.")

    return predictions, video_ids


if __name__ == "__main__":
    import argparse

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
        default=2,
        help="Number of workers for data preprocessing",
    )

    args = parser.parse_args()

    # Adjust the rest of the code to use the parsed arguments
    data_path = Path("./nexar-collision-prediction")

    logger.info("Loading test dataset...")
    df_test = pd.read_csv(data_path / "test.csv")

    logger.info("Extracting annotations and key frames from video data...")
    start_time = time.time()
    test_annons = get_annotations(df_test, test=True)

    video_dir = data_path / "test"
    video_ids = list(test_annons.keys())
    test_processed = parallel_preprocess_dataset(
        video_dir,
        video_ids,
        num_frames=args.n_frames,
        num_workers=args.n_workers,
        image_size=args.image_size,
    )
    elapsed_time = (time.time() - start_time) / 60
    logger.info(f"Data preprocessing completed in {elapsed_time:.2f} minutes.")

    test_dataset = DashcamDataset(test_processed, test_annons, val_transforms)
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    logger.info("Initializing model for inference...")

    # load trained model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TrafficTransformer()
    checkpoint_path = Path("./models/eval/best_model_fold_1.pth")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    logger.info(f"Performing inference on {device.type}...")

    # predict crash events on test set
    preds, video_ids = predict(model, test_dataloader)

    # save predictions
    output_path = data_path / "submission.csv"
    generate_submission(preds, video_ids, output_path=output_path)
