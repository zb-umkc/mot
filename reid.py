# External imports
import pandas as pd
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
from tqdm.auto import tqdm
import cv2

# Local imports
from config import *


def load_trained_reid_model(filename):
    """
    Load a trained Siamese Network model for object re-identification.
    """
    model = SiameseNetwork()

    model.load_state_dict(torch.load(filename, weights_only=False)["model_state_dict"])
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    return model


class SiameseNetwork(nn.Module):
  """
  A Siamese Network for object re-identification.

  This network uses a ResNet-18 backbone for feature extraction and
  a fully connected layer for embedding generation (length 256).
  """
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
        """
        Forward pass through the backbone and linear layers for a single input.
        """
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

  def forward(self, input1, input2):
        """
        Forward pass through forward_one() for two inputs.
        """
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function for Siamese Network.
    
    This loss function encourages the network to minimize the distance
    between similar pairs and maximize the distance between dissimilar pairs.
    """
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

        # Regularization term
        # Encourages the network to learn a more compact representation
        reg_loss = output1.norm(dim=1).mean() + output2.norm(dim=1).mean()

        return loss.mean() + self.reg_weight * reg_loss


class SiameseDatasetTrain(Dataset):
    """
    A dataset class for training the Siamese Network.
    
    This class loads pairs of cropped images from the MOT16 dataset.
    """
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

    def _load_ground_truth(self):
        """
        Load MOT16 ground truth data and concatenate into a single DataFrame.
        
        Each row contains the sequence name, frame ID, object ID, bounding box coordinates,
        and width/height of the bounding box.
        """
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
        """
        Load an image from the specified sequence and frame ID and crop it to the bounding box.
        """
        filename = ("000000" + str(frame_id))[-6:]
        img_path = f"{self.data_dir}/{seq}/img1/{filename}.jpg"
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image[y1:(y2+1), x1:(x2+1)]

        return image

    def _get_random_positive(self, seq, frame_id, obj_id):
        """
        Get a random positive example (same object, different frame).
        """
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
        """
        Get a random negative example (different object, different frame).
        """
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
        """
        Return the number of samples in the dataset.
        """
        return len(self.ground_truth)

    def __getitem__(self, idx):
        """
        Get a labeled sample from the dataset.

        This includes loading the cropped image, loading a positive or
        negative example, and transforming both.
        """
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


def random_split(dataset, val_size=0.2):
    """
    Randomly split a dataset into training and validation sets.

    NOTE:
    We mistakenly used an 80/20 split on the bounding boxes, which resulted
    in the same objects appearing in both the training and validation sets (data leakage).

    The correct approach is to split the object IDs into training and validation sets,
    then obtain the bounding boxes for each object ID.
    """
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    val_num = int(len(dataset) * val_size)
    train_indices = sorted(indices[:-val_num])
    val_indices = sorted(indices[-val_num:])

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    return train_dataset, val_dataset


def create_reid_dataset(val_size=0.2, train_batch_size=64, val_batch_size=64):
    """
    Create the training and validation datasets for the Siamese Network.
    """
    siamese_dataset = SiameseDatasetTrain(data_dir=data_dir_train, crop_size=256, pos_prob=0.5, max_frame_gap=10)
    
    train_dataset, val_dataset = random_split(siamese_dataset, val_size=val_size)
    
    train_siamese_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=min(4, os.cpu_count()), pin_memory=True
    )
    val_siamese_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=min(4, os.cpu_count()), pin_memory=True
    )

    return train_siamese_dataloader, val_siamese_dataloader


def train_one_epoch(model, optimizer, dataloader, device, criterion, epoch):
    """
    Train the model for one epoch.
    """
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

def evaluate(model, val_loader, criterion, device):
    """
    Evaluate the model on the validation set.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    with torch.no_grad():
        for img1, img2, label in tqdm(val_loader):
            img1 = img1.to(device, non_blocking=True)
            img2 = img2.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            
            # Forward pass
            output1, output2 = model(img1, img2)
            
            # Compute loss
            loss = criterion(output1, output2, label)
            
            # Accumulate loss
            total_loss += loss.item() * img1.size(0)  # Multiply by batch size
            total_samples += img1.size(0)

            # Calculate other metrics
            distance = torch.norm(output1 - output2, p=2).item()
            pred = distance < reid_threshold
            tp += int((pred == 1) & (label == 1))
            fp += int((pred == 1) & (label == 0))
            tn += int((pred == 0) & (label == 0))
            fn += int((pred == 0) & (label == 1))
              
    avg_loss = total_loss / total_samples
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = (tp) / (tp + fp)
    recall = (tp) / (tp + fn)
  
    return avg_loss, accuracy, precision, recall

def train_reid_model():
    """
    Train the Siamese Network for object re-identification.
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = SiameseNetwork().to(device)
    train_dataloader, val_dataloader = create_reid_dataset(val_batch_size=1)
    criterion = ContrastiveLoss()
    optimizer = optim.Adam([
        {'params': model.backbone.parameters(), 'lr': 1e-4},
        {'params': model.fc.parameters(), 'lr': 5e-4}
    ])

    for epoch in range(num_epochs_reid):
        training_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            dataloader=train_dataloader,
            device=device,
            criterion=criterion,
            epoch=epoch
        )
        l, a, p, r = evaluate(model, val_dataloader, criterion, device)
        print(f"Epoch [{epoch+1}/{num_epochs_reid}]")
        print(f"-- Training Loss: {training_loss}")
        print(f"-- Val Loss: {l}")
        print(f"-- Val Accuracy: {a}")
        print(f"-- Val Precision: {p}")
        print(f"-- Val Recall: {r}")

    print("\nTraining Completed")

    torch.save({
    'epoch': epoch+1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': training_loss,
    'val_loss': l,
    'val_acc': a,
    'val_prec': p,
    'val_rec': r
    }, save_filename_reid)

    print(f"Model Saved to: {save_filename_reid}")
