import pandas as pd
import os
import random
from datetime import date
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
from tqdm.auto import tqdm
import cv2

from config import *


def load_trained_reid_model(filename):
  model = SiameseNetwork()

  model.load_state_dict(torch.load(filename, weights_only=False)["model_state_dict"])
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  model.to(device)

  return model


class SiameseNetwork(nn.Module):
  def __init__(self):
      super(SiameseNetwork, self).__init__()
      self.backbone = models.resnet18(pretrained=True)
      self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
      self.fc = nn.Sequential(
          nn.Linear(512, 512),
          nn.BatchNorm1d(512),
          nn.ReLU(),
          nn.Linear(512, 256),
          nn.BatchNorm1d(256)
      )

  def forward_one(self, x):
      x = self.backbone(x)
      x = x.view(x.size(0), -1)  # Flatten
      return self.fc(x)

  def forward(self, input1, input2):
      output1 = self.forward_one(input1)
      output2 = self.forward_one(input2)
      return output1, output2


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, reg_weight=0.01):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.reg_weight = reg_weight

    def forward(self, output1, output2, label):
        # Euclidean distance (L2 norm by default)
        distance = F.pairwise_distance(output1, output2)

        # Contrastive loss
        loss = label * distance.pow(2) + \
               (1 - label) * F.relu(self.margin - distance).pow(2)

        reg_loss = output1.norm(dim=1).mean() + output2.norm(dim=1).mean()

        return loss.mean() + self.reg_weight * reg_loss


class SiameseDatasetTrain(Dataset):
    def __init__(self, data_dir, crop_size=256, pos_prob=0.5, max_frame_gap=10):
      self.data_dir = data_dir
      self.crop_size = crop_size
      self.pos_prob = pos_prob
      self.max_frame_gap = max_frame_gap
      self.img_res = {
          "MOT16-01": (1920, 1080),
          "MOT16-02": (1920, 1080),
          "MOT16-03": (1920, 1080),
          "MOT16-04": (1920, 1080),
          "MOT16-05": (640, 480),
          "MOT16-06": (640, 480),
          "MOT16-07": (1920, 1080),
          "MOT16-08": (1920, 1080),
          "MOT16-09": (1920, 1080),
          "MOT16-10": (1920, 1080),
          "MOT16-11": (1920, 1080),
          "MOT16-12": (1920, 1080),
          "MOT16-13": (1920, 1080),
          "MOT16-14": (1920, 1080)
      }
      self.ground_truth = self._load_ground_truth()

      self.transform = transforms.Compose([
          transforms.ToPILImage(),
          transforms.Resize(256), # Resize to 256x256 first
          transforms.CenterCrop(crop_size),
          transforms.ToTensor(),
          transforms.Normalize(
              mean=[0.485, 0.456, 0.406], # ImageNet stats
              std=[0.229, 0.224, 0.225]
          )
      ])

    # Load all gt files and concatenate into single object (Pandas DF)
    def _load_ground_truth(self):
        seqs = os.listdir(self.data_dir)
        cols = ["frame_id", "obj_id", "x1", "y1", "w", "h", "del1", "del2", "del3"]
        gt_list = []
        for seq in seqs:
            seq_name = [s for s in self.img_res.keys() if s in seq][0]
            img_res = self.img_res[seq_name]
            gt_path = os.path.join(self.data_dir, seq, "gt/gt.txt")
            gt_new = pd.read_csv(gt_path, header=None, names=cols)
            gt_new["seq"] = seq
            gt_new["x2"] = gt_new["x1"] + gt_new["w"]
            gt_new["y2"] = gt_new["y1"] + gt_new["h"]
            gt_new[["x1", "x2"]] = gt_new[["x1", "x2"]].clip(0, img_res[0])
            gt_new[["y1", "y2"]] = gt_new[["y1", "y2"]].clip(0, img_res[1])
            gt_new["w"] = gt_new["x2"] - gt_new["x1"]
            gt_new["h"] = gt_new["y2"] - gt_new["y1"]
            gt_list.append(gt_new)

        gt = pd.concat(gt_list, ignore_index=True)
        gt = gt[["seq", "frame_id", "obj_id", "x1", "y1", "x2", "y2", "w", "h"]]

        return gt

    def _load_image(self, seq, frame_id, x1, y1, x2, y2):

        filename = ("000000" + str(frame_id))[-6:]
        img_path = f"{self.data_dir}/{seq}/img1/{filename}.jpg"
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image[y1:(y2+1), x1:(x2+1)]

        return image

    def _get_random_positive(self, seq, frame_id, obj_id):
        # Pos Example = Same seq_name, different frame_id, same obj_id
        elig_boxes = self.ground_truth[
            (self.ground_truth["seq"] == seq) &
            (self.ground_truth["obj_id"] == obj_id) &
            (abs(self.ground_truth["frame_id"] - frame_id).between(1, self.max_frame_gap))
        ]
        pos_idx = random.randint(0, len(elig_boxes)-1)
        pos_box = elig_boxes.iloc[pos_idx]
        pos_img = self._load_image(
            pos_box["seq"],
            pos_box["frame_id"],
            pos_box["x1"],
            pos_box["y1"],
            pos_box["x2"],
            pos_box["y2"]
        )
        return pos_img

    def _get_random_negative(self, seq, frame_id, obj_id):
        # Neg Example = Same seq_name, different frame_id, different obj_id
        elig_boxes = self.ground_truth[
            (self.ground_truth["seq"] == seq) &
            (self.ground_truth["obj_id"] != obj_id) &
            (abs(self.ground_truth["frame_id"] - frame_id).between(1, self.max_frame_gap))
        ]
        neg_idx = random.randint(0, len(elig_boxes)-1)
        neg_box = elig_boxes.iloc[neg_idx]
        neg_img = self._load_image(
            neg_box["seq"],
            neg_box["frame_id"],
            neg_box["x1"],
            neg_box["y1"],
            neg_box["x2"],
            neg_box["y2"]
        )
        return neg_img

    def __len__(self):
      return len(self.ground_truth)

    def __getitem__(self, idx):
      box1 = self.ground_truth.iloc[idx]
      img1 = self._load_image(box1["seq"], box1["frame_id"],
                              box1["x1"], box1["y1"], box1["x2"], box1["y2"])

      if random.random() < self.pos_prob:
          # Get positive example
          img2 = self._get_random_positive(box1["seq"], box1["frame_id"], box1["obj_id"])
          label = 1
      else:
          # Get negative example
          img2 = self._get_random_negative(box1["seq"], box1["frame_id"], box1["obj_id"])
          label = 0

      # Apply transforms and return
      img1 = self.transform(img1)
      img2 = self.transform(img2)

      return img1, img2, torch.tensor(label, dtype=torch.float32)


# class SiameseDatasetTest(Dataset):
#     def __init__(self, data_dir, seq, preds, crop_size=256):
#       self.data_dir = data_dir
#       self.seq = seq
#       self.preds = preds
#       self.crop_size = crop_size

#       self.transform = transforms.Compose([
#           transforms.ToPILImage(),
#           transforms.Resize(256), # Resize to 256x256 first
#           transforms.CenterCrop(crop_size),
#           transforms.ToTensor(),
#           transforms.Normalize(
#               mean=[0.485, 0.456, 0.406], # ImageNet stats
#               std=[0.229, 0.224, 0.225]
#           )
#       ])

#       self.images = self._load_images(self.seq, self.preds)

#     def _load_images(self, seq, preds):
#         obj_list = []
#         for frame_id in tqdm(preds["frame_id"].unique()):
#             filename = ("000000" + str(frame_id))[-6:]
#             img_path = f"{self.data_dir}/{seq}/img1/{filename}.jpg"
#             image = cv2.imread(img_path)
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#             for idx, box in preds[preds["frame_id"] == frame_id].iterrows():
#                 # box = preds[(preds["frame_id"] == frame_id) & (preds["obj_id"] == obj_id)].iloc[0]
#                 x1, y1, x2, y2 = box[["x1", "y1", "x2", "y2"]]
#                 obj_image = image[y1:(y2+1), x1:(x2+1)]
#                 obj_list.append(obj_image)

#         return obj_list

#     def __len__(self):
#       return len(self.preds)

#     def __getitem__(self, idx):
#       print("Getting Item")
#       img = self.images.iloc[idx]
#       img = self.transform(img)

#       return img



def create_reid_dataset():
    siamese_dataset = SiameseDatasetTrain(data_dir=data_dir_train, crop_size=256, pos_prob=0.5, max_frame_gap=10)
    siamese_dataloader = DataLoader(siamese_dataset, batch_size=64, shuffle=True, num_workers=min(4, os.cpu_count()), pin_memory=True)

    return siamese_dataloader


def train_one_epoch(model, optimizer, dataloader, device, criterion, epoch):
    model.train()
    running_loss = 0.0
    for img1, img2, label in tqdm(dataloader):
        img1, img2, label = img1.to(device, non_blocking=True), img2.to(device, non_blocking=True), label.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward pass
        output1, output2 = model(img1, img2)

        # Compute loss
        loss = criterion(output1, output2, label)

        # Backward pass and optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()

    training_loss = running_loss / len(dataloader)

    return training_loss



def train_reid_model():
    # Initialize the Siamese Network and loss function
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = SiameseNetwork().to(device)

    dataloader = create_reid_dataset()

    criterion = ContrastiveLoss()
    optimizer = optim.Adam([
        {'params': model.backbone.parameters(), 'lr': 1e-4},
        {'params': model.fc.parameters(), 'lr': 5e-4}
    ])

    # Training loop
    for epoch in range(num_epochs_reid):
        training_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            dataloader=dataloader,
            device=device,
            criterion=criterion,
            epoch=epoch
        )
        print(f"Epoch [{epoch+1}/{num_epochs_reid}], Loss: {training_loss}")

    print("\nTraining Completed")

    torch.save({
    'epoch': epoch+1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': training_loss
    }, save_filename_reid)

    print(f"Model Saved to: {save_filename_reid}")
