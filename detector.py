# Import necessary packages
import numpy as np
import os
import random
from datetime import date
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    fasterrcnn_resnet50_fpn_v2,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm.auto import tqdm
import cv2

from config import *


def load_backbone():
    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
        trainable_backbone_layers=1)
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# Load pre-trained detection model from Google Drive
def load_trained_detection_model(filename):
    model = fasterrcnn_resnet50_fpn()
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.load_state_dict(torch.load(filename, weights_only=False)["model_state_dict"])
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    return model


class MOTDataset(Dataset):
    def __init__(self, data_dir, transform=None, mode="train", seq_name=None):
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.seq_name = seq_name

        # # Saving, just in case
        # val_transform = transforms.Compose([
        #     transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        #     transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        #     transforms.ToTensor(),
        #     transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0]==1 else x)
        # ])

        # For sake of time, treating train/val equally, then no transform for test
        if mode == "train":
            # Color, blur, grayscale, tensor, blur, shape, normalize
            if self.transform is None:
              self.transform = transforms.Compose([
                  transforms.ToPILImage(),
                  transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                  transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2))], p=0.5),
                  transforms.RandomGrayscale(p=0.05),
                  transforms.ToTensor(),
                  transforms.Lambda(lambda img: motion_blur(img, p=0.2)),
                  transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0]==1 else x),
                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]),
              ])
            self.ground_truth = self._load_ground_truth()
        else:
            assert self.seq_name is not None, "Sequence name must be provided for inference mode"

            # Tensor, shape, normalize
            if self.transform is None:
              self.transform = transforms.Compose([
                  transforms.ToTensor(),
                  transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0]==1 else x),
                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]),
              ])
            self.ground_truth = {}

        self.imgs = self._load_images()

    def _load_images(self):
        imgs = []
        for seq_name in os.listdir(self.data_dir):
            img_dir = os.path.join(self.data_dir, seq_name, "img1")
            if os.path.isdir(img_dir):
                img_list = [f"{seq_name}/img1/{img_name}" for img_name in os.listdir(img_dir)]
                imgs.extend(img_list)

        return sorted(imgs)

    def _load_ground_truth(self):
        gt = {}

        # Filter to specified seq if using inference mode
        if self.mode == "train":
            dirs = os.listdir(self.data_dir)
        else:
            dirs = [self.seq_name]

        for seq_name in dirs:
            if seq_name not in gt:
                gt[seq_name] = {}
            gt_path = os.path.join(self.data_dir, seq_name, "gt/gt.txt")

            with open(gt_path, 'r') as f:
                for line in f:
                    frame_id, obj_id, x1, y1, w, h, _, _, _ = line.strip().split(',')
                    x2 = int(x1) + int(w)
                    y2 = int(y1) + int(h)
                    frame_id = int(frame_id)
                    if frame_id not in gt[seq_name]:
                        gt[seq_name][frame_id] = []
                    gt[seq_name][frame_id].append([int(obj_id), float(x1), float(y1), float(x2), float(y2)])
        return gt

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = f"{self.data_dir}/{self.imgs[idx]}"
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # image = image.permute(1, 2, 0)
        seq_name = img_path.split("/")[-3]
        frame_id = int(img_path.split("/")[-1].split('.')[0])

        if self.mode == "train":
            ground_truth_data = self.ground_truth.get(seq_name, []).get(frame_id, [])
        else:
            ground_truth_data = []

        image = self.transform(image)

        # Handling the shape of the image (Just incase to prevent any errors)
        if len(image.shape) == 2:  # Grayscale
          image = image.unsqueeze(0).repeat(3, 1, 1)
        elif image.shape[0] > 3:
          image = image[:3, :, :]

        #image_tensor = image.permute(2,0,1).float() / 255.0

        return image, ground_truth_data


def custom_collate_fn(batch):
    images = [item[0] for item in batch]
    gt = [item[1] for item in batch]
    # Object ID is item[0], but 1 in labels represents the class "person"
    gt_dicts = [
        {"boxes": torch.Tensor([item[1:] for item in img]).type(torch.float32),
         "labels": torch.Tensor([1 for item in img]).type(torch.int64)}
        for img in gt]

    grouped_batch = {}
    for i in range(len(batch)):
      size = images[i].shape
      if size not in grouped_batch:
        grouped_batch[size] = {
            "images": [],
            "gt": []
        }
      grouped_batch[size]["images"].append(images[i])
      grouped_batch[size]["gt"].append(gt_dicts[i])

    final_batch = []
    for size, data in grouped_batch.items():
      images = torch.stack(data["images"])
      gt_dicts = data["gt"]
      final_batch.append((images, gt_dicts))

    return final_batch


def random_split(dataset, val_size=0.2):
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    val_num = int(len(dataset) * val_size)
    train_indices = sorted(indices[:-val_num])
    val_indices = sorted(indices[-val_num:])

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    return train_dataset, val_dataset


def motion_blur(image, p=0.2, kernel_size_range=(3, 9)):
    """Add motion blur to simulate fast movement"""
    if random.random() > p:
        return image

    # Create motion blur kernel
    kernel_size = random.choice(range(kernel_size_range[0], kernel_size_range[1], 2))
    kernel = torch.zeros((kernel_size, kernel_size))

    # Random direction
    direction = random.uniform(0, 1)
    if direction < 0.25:  # Horizontal
        kernel[kernel_size // 2, :] = 1.0
    elif direction < 0.5:  # Vertical
        kernel[:, kernel_size // 2] = 1.0
    elif direction < 0.75:  # Diagonal \
        for i in range(kernel_size):
            kernel[i, i] = 1.0
    else:  # Diagonal /
        for i in range(kernel_size):
            kernel[i, kernel_size - 1 - i] = 1.0

    kernel = kernel / kernel.sum()

    # Apply convolution to each channel separately
    blurred = image.clone()
    for c in range(image.shape[0]):
        channel = image[c].unsqueeze(0).unsqueeze(0)
        channel_kernel = kernel.unsqueeze(0).unsqueeze(0)
        blurred[c] = torch.nn.functional.conv2d(
            channel,
            channel_kernel.float(),
            padding=kernel_size//2
        ).squeeze()

    return blurred


def create_datasets(mode, seq_name=None):
    if mode == "train":
        detection_dataset_train = MOTDataset(data_dir=data_dir_train, mode="train")
        detection_dataset_train, detection_dataset_val = random_split(detection_dataset_train, val_size=0.2)

        dataloader1 = DataLoader(
            detection_dataset_train,
            batch_size=16,
            shuffle=True,
            collate_fn=custom_collate_fn,
            num_workers=min(4, os.cpu_count()),
            pin_memory=True
        )
        dataloader2 = DataLoader(
            detection_dataset_val,
            batch_size=16,
            shuffle=True,
            collate_fn=custom_collate_fn,
            num_workers=min(4, os.cpu_count()),
            pin_memory=True
        )
    elif mode == "test":
        detection_dataset_test = MOTDataset(data_dir=data_dir_test, mode="test", seq_name=seq_name)
        dataloader1 = DataLoader(
            detection_dataset_test,
            batch_size=1,
            shuffle=False,
            collate_fn=custom_collate_fn,
            num_workers=min(4, os.cpu_count()),
            pin_memory=True
        )
        dataloader2 = None
    else:
        raise ValueError("Invalid mode. Use 'train' or 'test'.")
    
    return dataloader1, dataloader2


def train_one_epoch(model, optimizer, data_loader, device, warmup):
    model.train()

    lr_warmup = None
    if warmup:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    # batch_num = 0
    for batch_list in tqdm(data_loader):
        for images, targets in batch_list:
            # print(f"Batch {batch_num}")
            images = list(image.to(device, non_blocking=True) for image in images)
            targets = [{k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            losses = model(images, targets)
            tot_loss = sum(loss for loss in losses.values())

            optimizer.zero_grad()
            tot_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if lr_warmup is not None:
                lr_warmup.step()

            lr = optimizer.param_groups[0]["lr"]
            # batch_num += 1

    losses["total"] = tot_loss

    return losses, lr


def evaluate(model, val_dataloader, device):
    loss_list = []

    for batch_list in tqdm(val_dataloader):
        for images, targets in batch_list:
            images = list(image.to(device, non_blocking=True) for image in images)
            targets = [{k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            with torch.no_grad():
              losses = model(images, targets)

            tot_loss = sum(loss for loss in losses.values())

    loss_list.append(tot_loss.item())
    avg_loss = np.mean(loss_list)

    return avg_loss


def print_results(epoch, lr, train_losses, val_loss):
    print(f"Epoch {epoch} | Learning Rate: {lr}")
    print(f"-- Train Loss: {train_losses['total']}")
    print(f"---- Classifier: {train_losses['loss_classifier']}")
    print(f"---- Box Reg: {train_losses['loss_box_reg']}")
    print(f"---- Objectness: {train_losses['loss_objectness']}")
    print(f"---- RPN Box Reg: {train_losses['loss_rpn_box_reg']}")
    print(f"-- Val Loss: {val_loss}")


def train_detection_model():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = load_backbone()
    model.to(device)

    train_dataloader, val_dataloader = create_datasets(mode="train")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )
    num_epochs = 4

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        warmup = False
        # warmup = True if epoch == 0 else False
        train_losses, lr = train_one_epoch(model, optimizer, train_dataloader, device, warmup)
        val_loss = evaluate(model, val_dataloader, device)

        # update the learning rate
        lr_scheduler.step()

        print_results(epoch, lr, train_losses, val_loss)

    print("\nTraining Completed")

    torch.save({
    'epoch': epoch+1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': train_losses,
    'val_loss': val_loss
    }, save_filename_det)

    print(f"Model Saved to: {save_filename_det}")
