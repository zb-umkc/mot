from datetime import date
import torch
import numpy as np

# Set PyTorch seed
myseed = 314
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

# General filepaths
base_dir = '/content/drive/MyDrive/data/CS-5567'
data_dir_train = f"{base_dir}/MOT16/train"
data_dir_test = f"{base_dir}/MOT16/test"
today = date.today().strftime("%Y%m%d")

# Detection model configurations
ckpt_filename_det = f"{base_dir}/detection_model_20250501.pth"
save_filename_det = f"{base_dir}/detection_model_{today}.pth"
num_epochs_det = 4
det_threshold = 0.7

# Re-ID model configurations
ckpt_filename_reid = f"{base_dir}/reid_model_20250425.pth"
save_filename_reid = f"{base_dir}/reid_model_{today}.pth"
num_epochs_reid = 4
reid_threshold = 0.7

# Image resolutions
img_res = {
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
