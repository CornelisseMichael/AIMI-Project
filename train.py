"""
Script for training a ResNet18 or I3D to classify a pulmonary nodule as benign or malignant.
"""
from models.model_2d import ResNet18
from models.model_3d import I3D
from models.model_vit import DeiTSmall
from models.model_mobilenet_v3_large import MobileNetLarge
from models.model_convnexttiny import ConvNextTiny
from models.model_convnexttinyv2 import ConvNextTinyV2
from models.model_medicalnets import MedicalNetModel
#from models.model_vit import DeiTSmall
from dataloader import get_data_loader
import logging
import numpy as np
import torch
import sklearn.metrics as metrics
from tqdm import tqdm
import warnings
import random
import pandas
from experiment_config import config
from datetime import datetime
import argparse
import timm 
import math
from sklearn.metrics import confusion_matrix

import json
from pathlib import Path
torch.backends.cudnn.benchmark = True

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s][%(asctime)s] %(message)s",
    datefmt="%I:%M:%S",
)
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


def train(
    train_csv_path,
    valid_csv_path,
    exp_save_root,
    ):
    """
    Train a ResNet18 or an I3D model
    """
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    logging.info(f"Training with {train_csv_path}")
    logging.info(f"Validating with {valid_csv_path}")

    train_df = pandas.read_csv(train_csv_path)
    valid_df = pandas.read_csv(valid_csv_path)

    print()

    logging.info(
        f"Number of malignant training samples: {train_df.label.sum()}"
    )
    logging.info(
        f"Number of benign training samples: {len(train_df) - train_df.label.sum()}"
    )
    print()
    logging.info(
        f"Number of malignant validation samples: {valid_df.label.sum()}"
    )
    logging.info(
        f"Number of benign validation samples: {len(valid_df) - valid_df.label.sum()}"
    )

    # create a training data loader
    weights = make_weights_for_balanced_classes(train_df.label.values)
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(train_df))

    train_loader = get_data_loader(
        config.DATADIR,
        train_df,
        mode=config.MODE,
        sampler=sampler,
        workers=config.NUM_WORKERS,
        batch_size=config.BATCH_SIZE,
        rotations=config.ROTATION,
        translations=config.TRANSLATION,
        size_mm=config.SIZE_MM,
        size_px=config.SIZE_PX,
    )

    valid_loader = get_data_loader(
        config.DATADIR,
        valid_df,
        mode=config.MODE,
        workers=config.NUM_WORKERS,
        batch_size=config.BATCH_SIZE,
        rotations=None,
        translations=None,
        size_mm=config.SIZE_MM,
        size_px=config.SIZE_PX,
    )

    device = torch.device("cuda:0")

    if config.MODE == "2D":
        model = ResNet18().to(device)
    elif config.MODE == "3D":
        model = I3D(
            num_classes=1,
            input_channels=3,
            pre_trained=True,
            freeze_bn=True,
        ).to(device)
    elif config.MODE == "vit":
        model = DeiTSmall(pretrained=True).to(device)
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
    

    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # scheduler with 5-epoch warmup and cosine decay:
    def lr_lambda(epoch):
        if epoch < 5:
            return float(epoch + 1) / 5
        return 0.5 * (1 + math.cos(math.pi * (epoch - 5) / (config.EPOCHS - 5)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


    # start a typical PyTorch training
    best_metric = -1
    best_metric_epoch = -1
    epochs = config.EPOCHS
    patience = config.PATIENCE
    counter = 0

    epoch_train_loss = []
    epoch_val_loss = []
    epoch_val_auc = []
    
    epoch_val_tp = []
    epoch_val_fp = []
    epoch_val_fn = []
    epoch_val_tn = []
    for epoch in range(epochs):

        if counter > patience:
            logging.info(f"Model not improving for {patience} epochs")
            break

        logging.info("-" * 10)
        logging.info("epoch {}/{}".format(epoch + 1, epochs))

        # train

        model.train()

        epoch_loss = 0
        step = 0

        for batch_data in tqdm(train_loader):
            step += 1
            inputs, labels = batch_data["image"], batch_data["label"]
            labels = labels.float().to(device)
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs.squeeze(), labels.squeeze())
            loss.backward()
            optimizer.step()

            #scheduler for thevit
            scheduler.step()
            epoch_loss += loss.item()
            epoch_len = len(train_df) // train_loader.batch_size
            if step % 100 == 0:
                logging.info(
                    "{}/{}, train_loss: {:.4f}".format(step, epoch_len, loss.item())
                )
        epoch_loss /= step
        logging.info(
            "epoch {} average train loss: {:.4f}".format(epoch + 1, epoch_loss)
        )

        epoch_train_loss.append(epoch_loss)

        # validate

        model.eval()

        epoch_loss = 0
        step = 0

        with torch.no_grad():

            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y = torch.tensor([], dtype=torch.float32, device=device)
            for val_data in valid_loader:
                step += 1
                val_images, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                val_images = val_images.to(device)
                val_labels = val_labels.float().to(device)
                outputs = model(val_images)
                loss = loss_function(outputs.squeeze(), val_labels.squeeze())
                epoch_loss += loss.item()
                y_pred = torch.cat([y_pred, outputs], dim=0)
                y = torch.cat([y, val_labels], dim=0)

                epoch_len = len(valid_df) // valid_loader.batch_size

            epoch_loss /= step
            logging.info(
                "epoch {} average valid loss: {:.4f}".format(epoch + 1, epoch_loss)
            )

            epoch_val_loss.append(epoch_loss)

            y_pred = torch.sigmoid(y_pred.reshape(-1)).data.cpu().numpy().reshape(-1)
            y = y.data.cpu().numpy().reshape(-1)

            # Convert the predicted probabilities to binary labels (if needed)
            y_pred_binary = (y_pred >= 0.5).astype(int)  # Convert to binary labels using 0.5 threshold
            
            # Compute confusion matrix
            cm = confusion_matrix(y, y_pred_binary)
            
            # Extract True Positives, False Positives, False Negatives, True Negatives
            TP = cm[1, 1]  # True positives: correct positive predictions
            FP = cm[0, 1]  # False positives: negatives predicted as positives
            FN = cm[1, 0]  # False negatives: positives predicted as negatives
            TN = cm[0, 0]  # True negatives: correct negative predictions

            fpr, tpr, _ = metrics.roc_curve(y, y_pred)
            auc_metric = metrics.auc(fpr, tpr)

            epoch_val_auc.append(auc_metric)
            epoch_val_tp.append(TP)
            epoch_val_fp.append(FP)
            epoch_val_fn.append(FN)
            epoch_val_tn.append(TN)

            epoch_losses = {
                "epoch_train_loss":epoch_train_loss,
                "epoch_val_loss":epoch_val_loss,
                "epoch_val_auc":epoch_val_auc,
                "epoch_val_tp":epoch_val_tp,
                "epoch_val_fp":epoch_val_fp,
                "epoch_val_fn":epoch_val_fn,
                "epoch_val_tn":epoch_val_tn,
                
            }
            epoch_losses_df = pandas.DataFrame(epoch_losses)
            csv_file_path = exp_save_root / "epoch_losses.csv" 
            epoch_losses_df.to_csv(csv_file_path, index=False)

            if auc_metric > best_metric:

                counter = 0
                best_metric = auc_metric
                best_metric_epoch = epoch + 1

                torch.save(
                    model.state_dict(),
                    exp_save_root / "best_metric_model.pth",
                )
                metadata = {
                    "train_csv": train_csv_path,
                    "valid_csv": valid_csv_path,
                    "config": config.to_dict(),
                    "best_auc": best_metric,
                    "epoch": best_metric_epoch,
                }
                #np.save(
               #     exp_save_root / "config.npy",
               #     metadata,
               # )

                # Define the file path where you want to save the JSON

                # Define the file path where you want to save the JSON
                metadata_clean = convert_paths_to_str(metadata)
                json_file_path = exp_save_root / "config.json"
                
                # Save metadata to a JSON file
                with open(json_file_path, 'w') as f:
                    json.dump(metadata_clean, f, indent=4)

                logging.info("saved new best metric model")

            logging.info(
                "current epoch: {} current AUC: {:.4f} best AUC: {:.4f} at epoch {}".format(
                    epoch + 1, auc_metric, best_metric, best_metric_epoch
                )
            )
        counter += 1

    logging.info(
        "train completed, best_metric: {:.4f} at epoch: {}".format(
            best_metric, best_metric_epoch
        )
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
