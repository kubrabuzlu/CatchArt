import os
import wandb
import configparser
import torch
from torchvision import transforms
from pathlib import Path
from distutils.util import strtobool

from dataloader import prepare_dataloaders
from model import create_painter_model
from engine import train
from evaluate import evaluate_and_log
from dotenv import load_dotenv
from set_seed import set_seed

# ------------- LOAD ENVIRONMENT -------------
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# -------------------- CONFIGURATION -------------------- #
# Load config from config.ini
config = configparser.ConfigParser()
config.read("config.ini")

# ------------- W&B Setup -------------
if config["WANDB"]["WANDB_MODE"] == "offline":
    os.environ["WANDB_MODE"] = "offline"
else:
    os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
wandb.login(relogin=True)
wandb_project = config["WANDB"]["PROJECT_NAME"]
wandb_name = config["WANDB"]["RUN_NAME"]


# -------------------- GENERAL SETTINGS -------------------- #
model_name = config["MODEL_PARAMETERS"]["MODEL_NAME"]
PRETRAINED = bool(strtobool(config["MODEL"]["PRETRAINED"]))
FINE_TUNE = bool(strtobool(config["MODEL"]["FINE_TUNE"]))
num_epochs = int(config["MODEL_PARAMETERS"]["NUM_EPOCHS"])
batch_size = int(config["IMG_PARAMETERS"]["BATCH_SIZE"])
learning_rate = float(config["MODEL_PARAMETERS"]["LEARNING_RATE"])
img_size = (int(config["IMG_PARAMETERS"]["IMG_WIDTH"]), int(config["IMG_PARAMETERS"]["IMG_HEIGHT"]))
model_save_path = Path(config["OUTPUT"]["MODEL_SAVE_PATH"])
early_stopping_patience = int(config["MODEL_PARAMETERS"]["EARLY_STOPPING_PATIENCE"])
schedular_patience = int(config["MODEL_PARAMETERS"]["SCHEDULAR_PATIENCE"])

# -------------------- SET SEED -------------------- #
seed = int(config["RANDOM"]["RANDOM_SEED"])
set_seed(seed)

# -------------------- DEVICE -------------------- #
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------- DATA -------------------- #
data_dir = Path(config["DATA"]["DATA_DIR"])
valid_split = float(config["DATA"]["DATA_VALID_SPLIT"])

# Image transforms
data_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor()
])

# Load data
train_dataloader, test_dataloader, class_names = prepare_dataloaders(data_path=data_dir,
                                                                     batch_size=batch_size,
                                                                     img_size=img_size,
                                                                     valid_split=valid_split,
                                                                     seed=seed)

# -------------------- MODEL -------------------- #
model = create_painter_model(model_name=model_name,
                                      num_classes=len(class_names)).to(device)

# -------------------- OPTIMIZER & LOSS -------------------- #
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# -------------------- WANDB INIT -------------------- #
wandb.init(project=wandb_project, name=wandb_name, config={
    "epochs": num_epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "img_size": img_size,
    "model": "resnet50",
})

# -------------------- TRAINING -------------------- #

train(model=model,
      train_dataloader=train_dataloader,
      test_dataloader=test_dataloader,
      optimizer=optimizer,
      loss_fn=loss_fn,
      epochs=num_epochs,
      device=device,
      checkpoint_path=model_save_path,
      early_stopping_patience=early_stopping_patience,
      scheduler_patience=schedular_patience)

# -------------------- EVALUATION -------------------- #
evaluate_and_log(model=model,
                 dataloader=test_dataloader,
                 class_names=class_names,
                 device=device,
                 model_save_path=model_save_path)

wandb.finish()
