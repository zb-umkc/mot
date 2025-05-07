# External imports
import pandas as pd
import torch
import torchvision.transforms as transforms
from tqdm.auto import tqdm
import cv2

# Local imports
from detector import *
from config import *


def run_inference(seq_name, detection_model, reid_model):
    """
    Run inference on the detection model and re-identification model.
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    detection_model.eval()
    reid_model.eval()

    dataloader, _ = create_detection_datasets(mode="test", seq_name=seq_name)

    pred_boxes = []
    frame = 0
    tracked_objects = {}
    next_id = 1

    for batch_list in tqdm(dataloader):
        for image, _ in batch_list:
            frame += 1
            image = image.to(device)

            with torch.no_grad():
                preds = detection_model(image)

            scores = preds[0]["scores"]
            box_idx = torch.nonzero(scores > det_threshold).squeeze()
            boxes = preds[0]["boxes"][box_idx]
            if boxes.ndim == 1:
                boxes = boxes.unsqueeze(0)

            for box in boxes:
                x1, y1, x2, y2 = box
                x1, y1 = int(x1), int(y1)
                x2, y2 = int(x2), int(y2)

                crop = image[:, :, y1:(y2+1), x1:(x2+1)].squeeze().cpu().numpy()
                if crop.size == 0: # Skip empty crops
                    print("Skipping empty crop")
                    continue

                # Resize and preprocess the crop for the Siamese Network
                crop = cv2.resize(crop.transpose(1,2,0), (224, 224))
                crop = transforms.ToTensor()(crop).unsqueeze(0).to(device)

                # Compute the feature vector using the Siamese Network
                with torch.no_grad():
                    embedding = reid_model.forward_one(crop)

                matched_id = None
                min_distance = reid_threshold  # Initialize as a threshold b/t same and different

                # Compare with existing objects
                for obj_id, obj in tracked_objects.items():
                    if obj["last_seen"] == frame:
                        continue

                    prev_emb = obj["embedding"]
                    distance = torch.norm(embedding - prev_emb, p=2).item()

                    if distance < min_distance:
                        min_distance = distance
                        matched_id = obj_id

                # Assign a new ID
                if matched_id is None:
                    matched_id = next_id
                    next_id += 1

                tracked_objects[matched_id] = {}
                tracked_objects[matched_id]["embedding"] = embedding
                tracked_objects[matched_id]["last_seen"] = frame
                pred_boxes.append([frame, matched_id, x1, y1, x2, y2])

    preds_df = pd.DataFrame(pred_boxes, columns=["frame_id", "obj_id", "x1", "y1", "x2", "y2"])

    return preds_df
