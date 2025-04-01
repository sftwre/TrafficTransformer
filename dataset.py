from torch.utils.data import Dataset


class DashcamDataset(Dataset):
    def __init__(self, processed_data, annotations, transform=None):
        self.processed_data = processed_data
        self.annotations = annotations
        self.transform = transform
        self.video_ids = list(processed_data.keys())

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        data = self.processed_data[video_id]

        # Get frames and optical flow
        frames = data["frames"]

        # Apply transformations to frames
        if self.transform:
            frames_tensor = self.transform(frames)
        else:
            # Convert to tensor manually
            frames_tensor = (
                torch.from_numpy(frames.transpose(0, 3, 1, 2)).float() / 255.0
            )

        # Load label and alert time
        label = self.annotations[video_id]["label"]
        alert_time = self.annotations[video_id].get("alert_time", 0)

        return {
            "frames": frames_tensor,
            "label": torch.tensor(label).float(),
            "alert_time": torch.tensor(alert_time).float(),
            "video_id": video_id,
        }
