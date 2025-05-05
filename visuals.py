import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from tqdm.auto import tqdm
import cv2
from google.colab.patches import cv2_imshow

from config import *
from detector import *
from reid import *


def demo_augmentation():
    t_raw = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0]==1 else x),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    t_color = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=[0.4,0.4], contrast=[0.4,0.4], saturation=[0.4,0.4], hue=[0.1,0.1]),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0]==1 else x),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    t_gblur = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(15, 15), sigma=8)], p=1.0),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0]==1 else x),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    t_mblur = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda img: motion_blur(img, p=1.0, kernel_size_range=(15, 17))),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0]==1 else x),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    t_gray = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomGrayscale(p=1.0),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0]==1 else x),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    # Create a mini-batch of raw images + augmented images
    ds_raw = MOTDataset(data_dir=data_dir_train, transform=t_raw, mode="train")
    ds_color = MOTDataset(data_dir=data_dir_train, transform=t_color, mode="train")
    ds_gblur = MOTDataset(data_dir=data_dir_train, transform=t_gblur, mode="train")
    ds_mblur = MOTDataset(data_dir=data_dir_train, transform=t_mblur, mode="train")
    ds_gray = MOTDataset(data_dir=data_dir_train, transform=t_gray, mode="train")

    # Grab the same indices from each
    indices = [1000, 2500, 5000]
    imgs_raw = [ds_raw[i][0] for i in indices]
    imgs_color = [ds_color[i][0] for i in indices]
    imgs_gblur = [ds_gblur[i][0] for i in indices]
    imgs_mblur = [ds_mblur[i][0] for i in indices]
    imgs_gray = [ds_gray[i][0] for i in indices]

    # Print shapes for debugging
    print(f"Raw: {[img.shape for img in imgs_raw]}")
    print(f"Color: {[img.shape for img in imgs_color]}")
    print(f"Gaussian Blur: {[img.shape for img in imgs_gblur]}")
    print(f"Motion Blur: {[img.shape for img in imgs_mblur]}")
    print(f"Gray: {[img.shape for img in imgs_gray]}")

    # Create grids directly from the original tensors
    grid_raw = make_grid(imgs_raw, nrow=3, normalize=True)
    grid_color = make_grid(imgs_color, nrow=3, normalize=True)
    grid_gblur = make_grid(imgs_gblur, nrow=3, normalize=True)
    grid_mblur = make_grid(imgs_mblur, nrow=3, normalize=True)
    grid_gray = make_grid(imgs_gray, nrow=3, normalize=True)

    # Display the grids
    plt.figure(figsize=(24, 12))

    # Raw images
    plt.subplot(5, 1, 1)
    plt.title("Raw Images")
    plt.axis('off')
    plt.imshow(grid_raw.permute(1, 2, 0).cpu().numpy())

    # Color jitter
    plt.subplot(5, 1, 2)
    plt.title("Color Jitter")
    plt.axis('off')
    plt.imshow(grid_color.permute(1, 2, 0).cpu().numpy())

    # Gaussian blur
    plt.subplot(5, 1, 3)
    plt.title("Gaussian Blur")
    plt.axis('off')
    plt.imshow(grid_gblur.permute(1, 2, 0).cpu().numpy())

    # Motion blur
    plt.subplot(5, 1, 4)
    plt.title("Motion Blur")
    plt.axis('off')
    plt.imshow(grid_mblur.permute(1, 2, 0).cpu().numpy())

    # Grayscale
    plt.subplot(5, 1, 5)
    plt.title("Grayscale")
    plt.axis('off')
    plt.imshow(grid_gray.permute(1, 2, 0).cpu().numpy())

    plt.tight_layout()
    plt.show()


def demo_reid_pairs(type):
    if type == "pos":
        pos_prob = 1.0
    else:
        pos_prob = 0.0

    siamese_dataset = SiameseDatasetTrain(data_dir=data_dir_train, crop_size=224, pos_prob=pos_prob, max_frame_gap=10)
    siamese_dataloader = DataLoader(siamese_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    test_batch = next(iter(siamese_dataloader))

    print(test_batch[0][0].permute(1, 2, 0).shape)
    print(test_batch[1][0].permute(1, 2, 0).shape)
    print(test_batch[2])

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img1_np = test_batch[0][0].permute(1, 2, 0).cpu().numpy()
    img1 = reverse_normalize(img1_np, mean, std)
    frame1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    cv2_imshow(frame1)

    img2_np = test_batch[1][0].permute(1, 2, 0).cpu().numpy()
    img2 = reverse_normalize(img2_np, mean, std)
    frame2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    cv2_imshow(frame2)



def reverse_normalize(img, mean, std):
    img *= std
    img += mean
    img *= 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def track_objects(dataset, detection_model, siamese_net, output_path, fps=30):
    """
    Track objects on MOT16 dataset and save the output as a video.

    Args:
        dataset: MOTDataset that provides images and ground truth.
        detection_model: The Faster R-CNN model for object detection.
        siamese_net: The Siamese network for object re-identification.
        output_path: Path to save the output video.
        fps: Frames per second for the output video.
    """
    # Initialize video writer
    first_image, _ = dataset[0]  # Get the first image to determine size
    height, width = first_image.shape[1], first_image.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize tracking variables
    tracked_objects = {}  # Dictionary to store tracked objects and their embeddings
    next_id = 1  # Next ID to assign to a new object

    # Process each image in the dataset
    for idx in tqdm(range(len(dataset))):
        image, _ = dataset[idx]  # Get the image (ignore ground truth for tracking)
        image_np = image.permute(1, 2, 0).cpu().numpy()  # Convert tensor to numpy array
        image_np = (image_np * 255).astype(np.uint8)  # Scale to [0, 255]
        frame = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

        # Perform detection
        image_tensor = image.unsqueeze(0).to(device)
        with torch.no_grad():
            detections = detection_model(image_tensor)[0]

        # Extract features for each detected object
        scores = detections['scores']
        boxes = detections['boxes'][scores > 0.7]  # Apply confidence threshold
        for box in boxes:
            x1, y1, x2, y2 = box.int().tolist()
            crop = image_np[y1:y2, x1:x2]
            if crop.size == 0:  # Skip empty crops
                continue

            # Resize and preprocess the crop for the Siamese Network
            crop = cv2.resize(crop, (224, 224))
            crop = transforms.ToTensor()(crop).unsqueeze(0).to(device)

            # Compute the feature vector using the Siamese Network
            with torch.no_grad():
                embedding = siamese_net.forward_one(crop)

            # Compare with previous embeddings (tracking logic)
            matched_id = None
            min_distance = 1.0
            for obj_id, prev_emb in tracked_objects.items():
                distance = torch.norm(embedding - prev_emb, p=2).item()
                if distance < min_distance:
                  min_distance = distance
                  matched_id = obj_id

            if matched_id is not None:
                # Update the tracked object's embedding
                tracked_objects[matched_id] = embedding
            else:
                # Assign a new ID
                tracked_objects[next_id] = embedding
                matched_id = next_id
                next_id += 1

            # Draw bounding box and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {matched_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write frame to output video
        out.write(frame)

    # Release resources
    out.release()
    cv2.destroyAllWindows()


output_path = 'output__tracking_video.mp4'
data_dir_train = f"{base_dir}/MOT16/train"
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = MOTDataset(data_dir=data_dir_train, transform=transform)
track_objects(dataset=train_dataset,
              detection_model=model,
              siamese_net=siamese_net,
              output_path=output_path)

################################################

# Function to draw bounding boxes and save video
def create_video_with_bboxes(dataloader, output_file):
    # Get the first batch to determine image size
    for images, targets in dataloader:
        # The images tensor has shape [batch_size, channels, height, width]
        # We need to move the channels to the last dimension for OpenCV
        image_np = images[0].permute(1, 2, 0).numpy()  # Access the first image in the batch and permute
        height, width, _ = image_np.shape
        break

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, 30, (width, height))

    # Iterate through the dataloader
    for images, targets in tqdm(dataloader):
        image_np = images[0].permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        frame = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Draw bounding boxes
        for box in targets:  # List of boxes for the image
            obj_id, x_min, y_min, x_max, y_max = box
            x_min, y_min = int(x_min), int(y_min)
            x_max, y_max = int(x_max), int(y_max)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {obj_id.item()}", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write frame to video
        out.write(frame)

        # Display the frame
        # Display the frame using cv2_imshow
        #cv2_imshow(frame)  # Replace cv2.imshow with cv2_imshow
        #cv2.imshow('Video with Bounding Boxes', frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

    out.release()
    cv2.destroyAllWindows()


# if __name__ == '__main__':
#     # Example usage
#     base_dir = '/content/drive/MyDrive/data/CS-5567'
#     data_dir_train = f"{base_dir}/MOT16/train"
#     transform = transforms.Compose([
#         transforms.ToTensor()
#     ])

#     train_dataset = MOTDataset(data_dir=f"{base_dir}/MOT16/train", transform=transform)
#     train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

#     output_file = 'mot_output_video.mp4'
#     create_video_with_bboxes(train_dataloader, output_file)

################################################

# Function for displaying one frame with its boxes (manually)

def display_frame(data_dir, seq, frame_id, boxes=None, label_loc="middle"):
    filename = ("000000" + str(frame_id))[-6:]
    img_path = f"{data_dir}/{seq}/img1/{filename}.jpg"
    image = cv2.imread(img_path)

    if boxes is not None:
        boxes = boxes[boxes["frame_id"] == frame_id]
    else:
        try:
            cols = ["frame_id", "obj_id", "x1", "y1", "w", "h", "del1", "del2", "del3"]
            gt_path = os.path.join(data_dir, seq, "gt/gt.txt")
            boxes = pd.read_csv(gt_path, header=None, names=cols)
            boxes = boxes[boxes["frame_id"] == frame_id]
            boxes["x1"] = boxes["x1"].astype(int)
            boxes["y1"] = boxes["y1"].astype(int)
            boxes["x2"] = boxes["x1"] + boxes["w"]
            boxes["y2"] = boxes["y1"] + boxes["h"]
            boxes = boxes[["frame_id", "obj_id", "x1", "y1", "x2", "y2", "w", "h"]]
        except:
            print("No ground truth file found for this sequence")
            return

    # Draw bounding boxes
    for index, box in boxes.iterrows():  # List of predicted boxes
        cv2.rectangle(
            img=image,
            pt1=(box["x1"], box["y1"]),
            pt2=(box["x2"], box["y2"]),
            color=(0, 255, 0),
            thickness=2
        )
        if label_loc == "middle":
          label_x = int((box["x1"] + box["x2"])/2)
          label_y = int((box["y1"] + box["y2"])/2)
        elif label_loc == "top":
          label_x = box["x1"]
          label_y = box["y1"] - 10
        elif label_loc == "bottom":
          label_x = box["x1"]
          label_y = box["y2"] + 10
        cv2.putText(image, f"ID: {box['obj_id']}", (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2_imshow(image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


for i in range(1, 11):
    display_frame(data_dir=data_dir_test, seq="MOT16-01", frame_id=i, boxes=preds, label_loc="bottom")

################################################

def test_one_image(faster_rcnn_model, siamese_net):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    transform = transforms.Compose(
        [transforms.ToTensor()]
    )
    test_dataset = MOTDataset(data_dir=data_dir_test, transform=transform, mode="test")
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

    ckpt_file_name = f"{base_dir}/model_20250401.pth"
    model_dict = torch.load(ckpt_file_name, weights_only=False)
    model.load_state_dict(model_dict["model_state_dict"])
    model.to(device)
    model.eval()

    siamese_ckpt_file_name = f"{base_dir}/siamese_model_20250401.pth"
    siamese_model_dict = torch.load(siamese_ckpt_file_name, weights_only=False)
    siamese_net.load_state_dict(siamese_model_dict["model_state_dict"])
    siamese_net.to(device)
    siamese_net.eval()

    tracked_objects = {}
    next_id = 1
    for batch_list in test_dataloader:
        for images, targets in batch_list:
            images = list(image.to(device) for image in images)

            with torch.no_grad():
                preds = model(images)

            image_np = images[0].permute(1, 2, 0).cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)
            frame = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            # Draw bounding boxes
            threshold = 0.7
            scores = preds[0]["scores"]
            box_idx = torch.nonzero(scores > threshold).squeeze()
            for box in preds[0]["boxes"][box_idx]:  # List of predicted boxes
                x_min, y_min, x_max, y_max = box
                x_min, y_min = int(x_min), int(y_min)
                x_max, y_max = int(x_max), int(y_max)
                #cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                # cv2.putText(frame, f"ID: {obj_id.item()}", (x_min, y_min - 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                crop = image_np[y_min:y_max, x_min:x_max]
                if crop.size == 0:  # Skip empty crops
                    continue

                # Resize and preprocess the crop for the Siamese Network
                crop = cv2.resize(crop, (224, 224))
                crop = transforms.ToTensor()(crop).unsqueeze(0).to(device)

                # Compute the feature vector using the Siamese Network
                with torch.no_grad():
                    embedding = siamese_net.forward_one(crop)

                # Compare with previous embeddings (tracking logic)
                matched_id = None
                min_distance = float('inf')
                max_similarity = -1  # Initialize to a very low similarity value

                for obj_id, prev_embedding in tracked_objects.items():
                    cosine_sim = F.cosine_similarity(embedding, prev_embedding).item()
                    if cosine_sim > max_similarity and cosine_sim > 0.8:  # Cosine similarity threshold
                        max_similarity = cosine_sim
                        matched_id = obj_id

                if matched_id is not None:
                    # Update the tracked object's embedding
                    tracked_objects[matched_id] = embedding
                else:
                    # Assign a new ID
                    tracked_objects[next_id] = embedding
                    matched_id = next_id
                    next_id += 1

                # Draw bounding box and ID
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {matched_id}", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the frame
            cv2_imshow(frame)

            #cv2.destroyAllWindows()
            #cv2_imshow(frame)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            #return preds

            return preds


preds = test_one_image(faster_rcnn_model=model, siamese_net=siamese_net)

