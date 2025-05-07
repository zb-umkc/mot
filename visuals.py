# External imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from tqdm.auto import tqdm
import cv2
from google.colab.patches import cv2_imshow

# Local imports
from config import *
from detector import *
from reid import *


def demo_augmentation():
    """
    Visualize examples of the augmentation process for the MOT16 dataset.
    """
    t_raw = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    t_color = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ColorJitter(
                brightness=[0.4, 0.4],
                contrast=[0.4, 0.4],
                saturation=[0.4, 0.4],
                hue=[0.1, 0.1],
            ),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    t_gblur = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=(15, 15), sigma=8)], p=1.0
            ),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    t_mblur = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(
                lambda img: motion_blur(img, p=1.0, kernel_size_range=(15, 17))
            ),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    t_gray = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomGrayscale(p=1.0),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

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
    plt.axis("off")
    plt.imshow(grid_raw.permute(1, 2, 0).cpu().numpy())

    # Color jitter
    plt.subplot(5, 1, 2)
    plt.title("Color Jitter")
    plt.axis("off")
    plt.imshow(grid_color.permute(1, 2, 0).cpu().numpy())

    # Gaussian blur
    plt.subplot(5, 1, 3)
    plt.title("Gaussian Blur")
    plt.axis("off")
    plt.imshow(grid_gblur.permute(1, 2, 0).cpu().numpy())

    # Motion blur
    plt.subplot(5, 1, 4)
    plt.title("Motion Blur")
    plt.axis("off")
    plt.imshow(grid_mblur.permute(1, 2, 0).cpu().numpy())

    # Grayscale
    plt.subplot(5, 1, 5)
    plt.title("Grayscale")
    plt.axis("off")
    plt.imshow(grid_gray.permute(1, 2, 0).cpu().numpy())

    plt.tight_layout()
    plt.show()


def demo_reid_pairs(type):
    """
    Visualize examples of positive and negative pairs for the Siamese network.
    """
    if type == "pos":
        pos_prob = 1.0
    else:
        pos_prob = 0.0

    siamese_dataset = SiameseDatasetTrain(
        data_dir=data_dir_train, crop_size=224, pos_prob=pos_prob, max_frame_gap=10
    )
    siamese_dataloader = DataLoader(
        siamese_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True
    )
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
    """
    Reverse the normalization of an image for display.
    """
    img *= std
    img += mean
    img *= 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def display_frame(data_dir, seq, frame_id, boxes=None, label_loc="middle"):
    """
    Display a single frame with bounding boxes.
    If boxes is not provided, it will attempt to load the ground truth boxes from the dataset.
    """
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
    for index, box in boxes.iterrows():
        cv2.rectangle(
            img=image,
            pt1=(box["x1"], box["y1"]),
            pt2=(box["x2"], box["y2"]),
            color=(0, 255, 0),
            thickness=2,
        )
        if label_loc == "middle":
            label_x = int((box["x1"] + box["x2"]) / 2)
            label_y = int((box["y1"] + box["y2"]) / 2)
        elif label_loc == "top":
            label_x = box["x1"]
            label_y = box["y1"] - 10
        elif label_loc == "bottom":
            label_x = box["x1"]
            label_y = box["y2"] + 10
        cv2.putText(
            image,
            f"ID: {box['obj_id']}",
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    cv2_imshow(image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def generate_video(data_loader, boxes, output_file, dim, label_loc="middle"):
    """
    Generate a video from the specified data_loader and save it to the specified location.
    The video will include bounding boxes and IDs for the detected objects.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_file, fourcc, 30, dim)
    frame = 0

    # ImageNet stats for reversing normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for batch_list in tqdm(data_loader):
        for image, _ in batch_list:
            frame += 1
            image_np = image.squeeze().permute(1, 2, 0).cpu().numpy()
            image_np = reverse_normalize(image_np, mean=mean, std=std)
            img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            boxes_f = boxes[boxes["frame_id"] == frame]

            # Draw bounding boxes
            for index, box in boxes_f.iterrows():
                if label_loc == "middle":
                    label_x = int((box["x1"] + box["x2"]) / 2)
                    label_y = int((box["y1"] + box["y2"]) / 2)
                elif label_loc == "top":
                    label_x = box["x1"]
                    label_y = box["y1"] - 10
                elif label_loc == "bottom":
                    label_x = box["x1"]
                    label_y = box["y2"] + 10

                cv2.rectangle(
                    img=img,
                    pt1=(box["x1"], box["y1"]),
                    pt2=(box["x2"], box["y2"]),
                    color=(0, 255, 0),
                    thickness=2,
                )
                cv2.putText(
                    img,
                    f"ID: {box['obj_id']}",
                    (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            out.write(img)

    out.release()
    cv2.destroyAllWindows()
