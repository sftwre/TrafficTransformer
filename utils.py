import os
import time
import cv2
import numpy as np
import pandas as pd
from multiprocessing import Pool
from pathlib import Path
import torch
import torch.nn as nn
from logging import getLogger


def get_annotations(df: pd.DataFrame) -> dict:
    annotations = {}
    for _, row in df.iterrows():
        video_id = f"{int(row['id'].item()):05d}"
        annotations[video_id] = {
            "label": int(row["target"].item()),
            "alert_time": row["time_of_alert"].item(),
            "event_time": row["time_of_event"].item(),
        }
    return annotations


def extract_keyframes(video_path, num_frames=12, target_size=(160, 160)):
    """
    Extracts key frames from the video, focusing on the final part where collisions typically occur.
    Uses exponential distribution to give more weight to frames closer to the end.
    """
    cap = cv2.VideoCapture(video_path)
    logger = getLogger(__name__)

    if not cap.isOpened():
        logger.error(f"Could not open the video: {video_path}")
        return np.zeros((num_frames, target_size[0], target_size[1], 3), dtype=np.uint8)

    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames <= 0:
        logger.error(f"Video without frames: {video_path}")
        cap.release()
        return np.zeros((num_frames, target_size[0], target_size[1], 3), dtype=np.uint8)

    # Calculate video duration in seconds
    duration = total_frames / fps if fps > 0 else 0

    # If the video is short (less than 10 seconds), distribute frames uniformly
    if duration < 10:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        # Concentrate 80% of frames in the last 3 seconds (critical area)
        end_frames = int(num_frames * 0.8)
        start_frames = num_frames - end_frames

        # Calculate the starting index for the last 3 seconds
        last_seconds = 3
        last_frame_count = min(int(fps * last_seconds), total_frames - 1)
        start_idx = max(0, total_frames - last_frame_count)

        # Exponential distribution to give more weight to the last frames
        # This creates indices that are more densely packed toward the end
        end_indices = np.array(
            [
                start_idx + int((total_frames - start_idx - 1) * (i / end_frames) ** 2)
                for i in range(1, end_frames + 1)
            ]
        )

        # Initial frames distributed uniformly for context
        begin_indices = (
            np.linspace(0, start_idx - 1, start_frames, dtype=int)
            if start_idx > 0
            else np.zeros(start_frames, dtype=int)
        )

        # Combine indices
        frame_indices = np.concatenate([begin_indices, end_indices])

    # Extract selected frames
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Use higher resolution and better interpolation
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LANCZOS4)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            frames.append(np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8))

    cap.release()
    return np.array(frames, dtype=np.uint8)


def process_video(args):
    """
    Function to process an individual video.
    """
    video_path, video_id, num_frames, image_size = args
    try:
        # Extract frames with higher resolution
        frames = extract_keyframes(
            video_path, num_frames=num_frames, target_size=(image_size, image_size)
        )

        return video_id, {"frames": frames}

    except Exception as e:
        logger = getLogger(__name__)
        logger.exception(f"Error processing video {video_id}: {str(e)}")
        return video_id, None


def parallel_preprocess_dataset(
    video_dir, video_ids, num_frames=8, image_size=160, num_workers=4
):
    """
    Pre-processes multiple videos in parallel.
    """
    args_list = []
    for video_id in video_ids:
        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        if os.path.exists(video_path):
            args_list.append((video_path, video_id, num_frames, image_size))

    start_time = time.time()

    logger = getLogger(__name__)
    logger.info(
        f"Starting parallel pre-processing of {len(args_list)} videos with {num_workers} workers..."
    )

    processed_data = {}
    with Pool(num_workers) as p:
        results = p.map(process_video, args_list)
        for video_id, data in results:
            if data is not None:
                processed_data[video_id] = data

    logger.info(f"Pre-processing completed in {time.time() - start_time:.2f} seconds.")
    logger.info(f"Processed {len(processed_data)} out of {len(args_list)} videos.")

    return processed_data


def count_trainable_params(model: nn.Module) -> int:
    """
    Counts the number of trainable parameters in a PyTorch model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(state_dict: dict, filename: str):
    """
    Writes model weights to disk.
    Model weights are saved to ./models/<filename>.pth

    Args:
        state_dict: Dictionary with arbitrary model data and training info
        filename: name of model file
    """

    model_dir = Path(f"./models")

    if not model_dir.exists():
        os.mkdir(model_dir)

    torch.save(state_dict, model_dir / filename)

    logger = getLogger(__name__)
    logger.info(f"Model weights saved to -> {model_dir / filename}")
