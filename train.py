"""
Script for training a ResNet18 or I3D to classify a pulmonary nodule as benign or malignant.
"""
from models.model_2d import ResNet18, ResNet50
from models.model_convnexttiny import ConvNextTiny

from models.model_3d import I3D
from models.ensemble_model import EnsembleWrapper
from models.SwinTiny import SwinTiny
from models.SwinBase import SwinBase
from models.DeitSmall import DeiTSmall
from models.DeitBase import DeiTBase
from models.model_mobilenet_v3_large import MobileNetLarge
from models.model_convnexttiny import ConvNextTiny
from models.model_convnexttinyv2 import ConvNextTinyV2
from models.model_medicalnets import MedicalNetModel
from dataloader import get_data_loader
import logging
import numpy as np
import torch
import sklearn.metrics as metrics
from tqdm import tqdm
import warnings
import random
import pandas as pd 
from experiment_config import config
from datetime import datetime
import argparse
import timm  # Add this at the top with your other imports
import math
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, roc_curve, auc
import json
from pathlib import Path
torch.backends.cudnn.benchmark = True
DEVICE = config.DEVICE 


logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s][%(asctime)s] %(message)s",
    datefmt="%I:%M:%S",
)

def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
def convert_paths_to_str(obj):
    if isinstance(obj, dict):
        return {k: convert_paths_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_paths_to_str(item) for item in obj]
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return obj
def make_weights_for_balanced_classes(labels):
    """Making sampling weights for the data samples
    :returns: sampling weights for dealing with class imbalance problem

    """
    n_samples = len(labels)
    unique, cnts = np.unique(labels, return_counts=True)
    cnt_dict = dict(zip(unique, cnts))

    weights = []
    for label in labels:
        weights.append(n_samples / float(cnt_dict[label]))
    return weights

def load_data(train_csv_path, valid_csv_path):
    train_df = pd.read_csv(train_csv_path)
    valid_df = pd.read_csv(valid_csv_path)

    def log_counts(df, name):
        pos = int(df.label.sum())
        total = len(df)
        logging.info(f"{name} - Positive: {pos}, Negative: {total - pos}")

    log_counts(train_df, "Train")
    log_counts(valid_df, "Validation")
    return train_df, valid_df
def create_dataloaders(train_df, valid_df):
    weights = torch.DoubleTensor(make_weights_for_balanced_classes(train_df.label.values))
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(train_df))

    train_loader = get_data_loader(
        config.DATADIR, train_df, config.MODE, sampler=sampler,
        workers=config.NUM_WORKERS, batch_size=config.BATCH_SIZE,
        rotations=config.ROTATION, translations=config.TRANSLATION,
        size_mm=config.SIZE_MM, size_px=config.SIZE_PX
    )

    valid_loader = get_data_loader(
        config.DATADIR, valid_df, config.MODE,
        workers=config.NUM_WORKERS, batch_size=config.BATCH_SIZE,
        rotations=None, translations=None,
        size_mm=config.SIZE_MM, size_px=config.SIZE_PX
    )

    return train_loader, valid_loader
    
def initialize_model():
    if config.MODE == "2D":
        return ResNet50()
    elif config.MODE == "3D":
        return I3D(num_classes=1, input_channels=3, pre_trained=True, freeze_bn=True)
    elif config.MODE == "vit":
        model = DeiTSmall(pretrained=True).to(device)
    elif config.MODE == "swintiny":
        model = SwinTiny(pretrained=True).to(device)
    elif config.MODE == "mobilenetv3L":
        model = MobileNetLarge(pretrained=True).to(device)
    elif config.MODE == "convnexttiny":
        model = ConvNextTiny(pretrained=True).to(device)
    elif config.MODE == "convnexttinyv2":
        model = ConvNextTinyV2(pretrained=True).to(device)
    elif config.MODE == "MedicalNetResnet10":
        model = MedicalNetModel(model_depth=10, pretrained=True, pretrained_path = "models/pretrained_resnets/resnet_10.pth").to(device)
    elif config.MODE == "MedicalNetResnet18":
        model = MedicalNetModel(model_depth=18, pretrained=True, pretrained_path = "models/pretrained_resnets/resnet_18.pth").to(device)
    elif config.MODE == "MedicalNetResnet34":
        model = MedicalNetModel(model_depth=34, pretrained=True, pretrained_path = "models/pretrained_resnets/resnet_34.pth").to(device)
    elif config.MODE == "MedicalNetResnet50":
        model = MedicalNetModel(model_depth=50, pretrained=True, pretrained_path = "models/pretrained_resnets/resnet_50.pth").to(device)
    elif config.MODE == "MedicalNetResnet101":
        model = MedicalNetModel(model_depth=101, pretrained=True, pretrained_path = "models/pretrained_resnets/resnet_101.pth").to(device)
    elif config.MODE == "MedicalNetResnet152":
        model = MedicalNetModel(model_depth=152, pretrained=True, pretrained_path = "models/pretrained_resnets/resnet_152.pth").to(device)
    elif config.MODE == "MedicalNetResnet200":
        model = MedicalNetModel(model_depth=200, pretrained=True, pretrained_path = "models/pretrained_resnets/resnet_200.pth").to(device) 
    elif config.MODE == "ensemble": # Different handling for model ensembles
        if config.MODEL =="swin_tiny":
            base = SwinTiny(pretrained=True, num_classes=1)
        elif config.MODEL =="swin_base":
            base = SwinBase(pretrained=True, num_classes=1)
        elif config.MODEL =="deit_small":
            base = DeiTSmall( num_classes=1, pretrained=True)
        elif config.MODEL == "deit_base":
            base = DeiTBase(num_classes=1, pretrained=True)
        elif config.MODEL =="resnet50":
            base = ResNet50()
        return EnsembleWrapper(base, num_images=config.NUM_IMAGES, num_outputs=1)
    else:
        raise ValueError(f"Invalid mode: {config.MODE}")

    return model

def lr_schedule(optimizer):
    def lr_lambda(epoch):
        if epoch < 5:
            return (epoch + 1) / 5
        return 0.5 * (1 + math.cos(math.pi * (epoch - 5) / (config.EPOCHS - 5)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)  

def evaluate_model(model, loader, criterion):
    model.eval()
    total_loss, steps = 0, 0
    device = DEVICE
    y_pred, y_true = torch.tensor([], device=device), torch.tensor([], device=device)

    with torch.no_grad():
        for batch in loader:
            x, y = batch["image"].to(device), batch["label"].float().to(device)
            out = model(x).squeeze()
            loss = criterion(out, y.squeeze())
            total_loss += loss.item()
            steps += 1
            y_pred = torch.cat([y_pred, out.view(-1)])
            y_true = torch.cat([y_true, y.view(-1)])

    avg_loss = total_loss / steps
    y_pred_sigmoid = torch.sigmoid(y_pred).cpu().numpy()
    y_true = y_true.cpu().numpy()
    binary_preds = (y_pred_sigmoid >= 0.5).astype(int)

    cm = confusion_matrix(y_true, binary_preds)
    TP, FP, FN, TN = cm[1, 1], cm[0, 1], cm[1, 0], cm[0, 0]
    fpr, tpr, _ = roc_curve(y_true, y_pred_sigmoid)
    auc_score = auc(fpr, tpr)

    return avg_loss, auc_score, (TP, FP, FN, TN)

def train_one_epoch(model, loader, criterion, optimizer, scheduler, use_sched = False):
    model.train()
    total_loss, steps = 0, 0
    for batch in tqdm(loader, desc="Training"):
        x, y = batch["image"].to(DEVICE), batch["label"].float().to(DEVICE)
        optimizer.zero_grad()
        out = model(x).squeeze()
        loss = criterion(out, y.squeeze())
        loss.backward()
        optimizer.step()
        if use_sched:
           scheduler.step()
        total_loss += loss.item()
        steps += 1
    return total_loss / steps

    
def save_best_model(model, epoch, auc_score, save_path, metadata):
    torch.save(model.state_dict(), save_path / "best_metric_model.pth")
    metadata.update({"best_auc": auc_score, "epoch": epoch})
    with open(save_path / "config.json", "w") as f:
        json.dump(convert_paths_to_str(metadata), f, indent=4)

def train(train_csv_path, valid_csv_path, exp_save_root):
    set_seeds(config.SEED)
    train_df, valid_df = load_data(train_csv_path, valid_csv_path)
    train_loader, valid_loader = create_dataloaders(train_df, valid_df)

    model = initialize_model()
    model = model.to(DEVICE)

    # Enable multi-GPU usage if available
    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
        
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = lr_schedule(optimizer)

    history = {
        "train_loss": [], # "train_eval_loss": [], "train_auc": [],
        "val_loss": [], "val_auc": [],
       # "train_tp": [], "train_fp": [], "train_fn": [], "train_tn": [],
        "tp": [], "fp": [], "fn": [], "tn": []
    }

    best_auc, patience_counter, best_epoch = -1, 0, 0

    for epoch in range(config.EPOCHS):
        if patience_counter > config.PATIENCE:
            logging.info(f"Early stopping after {patience_counter} epochs of no improvement.")
            break

        logging.info(f"\nEpoch {epoch + 1}/{config.EPOCHS}")

        #train one epoch
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, use_sched = config.USE_SCHED)

        # Evaluate on training set
       # train_eval_loss, train_auc, (TP_train, FP_train, FN_train, TN_train) = evaluate_model(model, train_loader, criterion)
        
        # Evaluate on validation set
        val_loss, val_auc, (TP, FP, FN, TN) = evaluate_model(model, valid_loader, criterion)

        # Logging
       # history["train_eval_loss"].append(train_eval_loss)
       # history["train_auc"].append(train_auc)
       # history["train_tp"].append(TP_train)
       # history["train_fp"].append(FP_train)
       # history["train_fn"].append(FN_train)
       # history["train_tn"].append(TN_train)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["tp"].append(TP)
        history["fp"].append(FP)
        history["fn"].append(FN)
        history["tn"].append(TN)
        
        logging.info(
            f"Epoch {epoch+1}: Train Loss = {train_loss:.4f} "
        #    f"TrainEvalLoss={train_eval_loss:.4f}, TrainLoss={train_loss:.4f}, TrainAUC={train_auc:.4f}, "
         #   f"TrainTP={TP_train}, TrainFP={FP_train}, TrainFN={FN_train}, TrainTN={TN_train} | "
            f"ValLoss={val_loss:.4f}, ValAUC={val_auc:.4f}, "
            f"ValTP={TP}, ValFP={FP}, ValFN={FN}, ValTN={TN}"
            )

        # Save best model
        if val_auc > best_auc:
            best_auc, best_epoch, patience_counter = val_auc, epoch + 1, 0
            metadata = {
                "train_csv": str(train_csv_path),
                "valid_csv": str(valid_csv_path),
                "config": config.to_dict(),
            }
            save_best_model(model, best_epoch, best_auc, exp_save_root, metadata)
        else:
            patience_counter += 1

        # Save history
        pd.DataFrame(history).to_csv(exp_save_root / "epoch_losses.csv", index=False)

    logging.info(f"Training complete. Best AUC: {best_auc:.4f} at epoch {best_epoch}")

    
def test(train_csv_path, valid_csv_path):
    logging.info("Running test mode: 1 epoch on small data subset")
    set_seeds(config.SEED)

    # Load and subsample data
    train_df = pd.read_csv(train_csv_path).sample(n=32, random_state=42)
    valid_df = pd.read_csv(valid_csv_path).sample(n=32, random_state=42)

    train_loader, valid_loader = create_dataloaders(train_df, valid_df)

    model = initialize_model().to(DEVICE)
    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = lr_schedule(optimizer)

    logging.info("Starting 1 epoch of training...")
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler)
    train_eval_loss, train_auc, _ = evaluate_model(model, train_loader, criterion)
    val_loss, val_auc, _ = evaluate_model(model, valid_loader, criterion)

    logging.info(
        f"Test run results — TrainLoss={train_loss:.4f}, TrainAUC={train_auc:.4f}, "
        f"ValLoss={val_loss:.4f}, ValAUC={val_auc:.4f}"
    )
    
    
if __name__ == "__main__":
    experiment_name = f"{config.EXPERIMENT_NAME}-{config.MODE}-{datetime.today().strftime('%Y%m%d')}-{config.VERSION}"

    exp_save_root = config.EXPERIMENT_DIR / experiment_name
    exp_save_root.mkdir(parents=True, exist_ok=True)

    # start training run
    train(
        train_csv_path=config.CSV_DIR_TRAIN,
        valid_csv_path=config.CSV_DIR_VALID,
        exp_save_root=exp_save_root,
        )


