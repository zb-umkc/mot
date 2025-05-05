from datetime import date


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
