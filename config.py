"""
Configuration constants for the dental X-ray pose estimation thesis project.
"""

# Detection thresholds
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.4

# Image size used by the model
IMG_SIZE = 512

# OKS category thresholds
OKS_LOW = 0.7
OKS_HIGH = 0.85

# Base path to the dataset and model files on Google Drive
BASE_PATH = "/content/drive/MyDrive/Colab Notebooks/Keypoint_detection.v10-512px-adaptive.yolov8"

# Model weights path
MODEL_PATH = f"{BASE_PATH}/model_02062025.pt"
