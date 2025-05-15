# dataloader.py
Change the following lines according to the model. I.e., if you are using 2D model, add it to the if statement:

Line 255:
```python
if self.mode == "2D" or self.mode =="vit":
    output_shape = (1, self.size_px, self.size_px)
else:
    output_shape = (self.size_px, self.size_px, self.size_px)
```

Line 389:
```python
if mode == "2D":
    # replicate the channel dimension
    patch = np.repeat(patch, 3, axis=0)
else:
    patch = np.expand_dims(patch, axis=0)

return patch
```

For example, with the vit, we change the above to:

Line 255:
```python
if self.mode == "2D" or self.mode =="vit":
    output_shape = (1, self.size_px, self.size_px)
else:
    output_shape = (self.size_px, self.size_px, self.size_px)
```

Line 389:
```python
if mode == "2D" or mode== "vit":
    # replicate the channel dimension
    patch = np.repeat(patch, 3, axis=0)
else:
    patch = np.expand_dims(patch, axis=0)

return patch # patch.shape == (3, 64, 64)
```

# do_build.sh

Change the following in line 7:
```python
DOCKER_IMAGE_TAG="luna25-baseline-open-development-phase"
```

To something that reflects the name of your model, such as:
Change the following in line 7:
```python
DOCKER_IMAGE_TAG="luna25-vit-open-development-phase"
```


# do_save.sh
Similarly to the previous shell, change the `DOCKER_IMAGE_TAG` in line 9 to the newly chosen name.


# do_test_run.sh
Again, change the name of `DOCKER_IMAGE_TAG` in line 7 to the newly chosen name.


# experiment_config.py
This file requires many modifications.

IF RUNNING ON CLUSTER, ensure you have the following:
```python
self.WORKDIR = Path("/vol/csedu-nobackup/course/IMC037_aimi/group01/luna25-vit") # Set this to your actual workpath, such as "/vol/csedu-nobackup/course/IMC037_aimi/group01/3DCNN". If running locally, you can use something like Path("C:/Users/myuser/university/AIMI/luna25-vit").
self.DATADIR = Path("/vol/csedu-nobackup/course/IMC037_aimi/group01/data/nodule_blocks/luna25_nodule_blocks") # If running locally, you can use Path("C:/Users/myuser/university/AIMI/luna25/dataset_csv")
self.CSV_DIR = Path("/vol/csedu-nobackup/course/IMC037_aimi/group01/data/data_csv") # Same thing here
self.CSV_DIR_TRAIN = ... # Again, ensure the path is correct, you can use the same csv for both
self.CSV_DIR_VALID = ... # Again, ensure the path is correct, you can use the same csv for both

self.EXPERIMENT_NAME = ... # Give it a representative name of the model you're running
self.MODE = "3DCNN" # For example, if you run with 3D CNN you use "3DCNN", and you ensure this is also an option in train.py and processor.py


self.SEED           = 2025 # Leave as is
self.NUM_WORKERS    = 8 # Can lower if it's too much

# since ViT-base/DeiT expects 224×224 inputs:
self.SIZE_PX        = 224  # If you expect 64x64, use 64, or if you expect 128x128, use 128, etc...

# physical size no longer needed for ViT patching, but should you need it, leave as is
self.SIZE_MM      = 50  

# grayscale, so for x,y,z all with a single channel, we have 3 channels in total.
# The swin_tiny_patch4_window7_224 ViT model expects this as an argument.
self.IN_CHANS       = 3  

# smaller batch to fit GPU memory
self.BATCH_SIZE     = 16  

self.ROTATION       = ((-20, 20), (-20,20),(-20,20))  # Leave as is
self.TRANSLATION    = True  # Leave as is

# Num epochs and early stopping (patience), change as desired
self.EPOCHS         = 50  
self.PATIENCE       = 10  

# The ViT has internal patch size (16×16), however, this is the patch size of the nodule.
self.PATCH_SIZE     = [64, 128, 128]

# finetuning LR & regularization, change as desired
self.LEARNING_RATE  = 3e-5  
self.WEIGHT_DECAY   = 1e-2  

self.VERSION = 1 # If you want, modify the version if you make changes over a previously tested model

# At the end of the train function we do some json logging to track the model behavior during training, 
# so then we need to convert the config to a dictionary. 
def to_dict(self):
"""Converts the Configuration object to a dictionary."""
return {k: v for k, v in self.__dict__.items()}

```

# inference.py
IF RUNNING LOCALLY, change the following to reflect your actual paths:
```py
INPUT_PATH = Path("/input") 
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("/opt/app/resources")
```

If you want to load your own model, i.e. you made the container by uploading to github and letting the website create it,
then you can upload the model as a `.tar.gz`, and then access it with lines 182-184, which reads a resource file.

In lines 253-254, we have to change the mode and model name to reflect the current model:
```python
mode = "vit" #"2D"
model_name = "LUNA25-vit-vit-20250427"
```


# processor.py
Line 34, change the `self.size_px` to the same value as in your config

IMPORTANT!:
Add your model to the if statements in lines 44, and 96.

Add the following function:
```python
def add_prefix_to_state_dict(self, state_dict, prefix="model."):
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        if not k.startswith(prefix):
            new_key = prefix + k
        else:
            new_key = k  # already prefixed
        new_state_dict[new_key] = v
    return new_state_dict
```

and after `ckpt = torch.load`, but before `model.load_state_dict`, add:
```python
# Handle if it contains a full checkpoint dict
if "state_dict" in ckpt:
    ckpt = ckpt["state_dict"]

# Add "model." prefix to keys
ckpt = self.add_prefix_to_state_dict(ckpt, prefix="model.")
```



# train.py
Many changes/additions are necessary here. For the full picture, see the train.py file.

First, add this function:
```python
def convert_paths_to_str(obj):

    if isinstance(obj, dict):
        return {k: convert_paths_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_paths_to_str(item) for item in obj]
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return obj
```

Next, add your mdel to the if statements before the `loss_function` definition. For example:
```python
elif config.MODE == "vit":
    model = timm.create_model(
            "deit_small_patch16_224",        pretrained=True,
        num_classes=1,           # Binary classification
        in_chans=config.IN_CHANS             #greyscale
    ).to(device)
```

After the loss function, if you want to use a LR schedular, you can, for example, add the folowing:
```python
# scheduler with 5-epoch warmup and cosine decay:
def lr_lambda(epoch):
    if epoch < 5:
        return float(epoch + 1) / 5
    return 0.5 * (1 + math.cos(math.pi * (epoch - 5) / (config.EPOCHS - 5)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```
And then after the `optimizer.step()` line in the epoch for loop, add:
```python
scheduler.step()
```

To track model performance during training, add the following outside of the epoch for loop:
```python
epoch_train_loss = []
epoch_val_loss = []
epoch_val_auc = []
epoch_val_tp = []
epoch_val_fp = []
epoch_val_fn = []
epoch_val_tn = []
```
After `logging.info("epoch {} average train loss: {:.4f}".format(epoch + 1, epoch_loss))`, add:
```python
epoch_train_loss.append(epoch_loss)
```
Then after `logging.info("epoch {} average valid loss: {:.4f}".format(epoch + 1, epoch_loss))` add:
```python
epoch_val_loss.append(epoch_loss)
```
then after `y = y.data.cpu().numpy().reshape(-1)` add:
```py
# Convert the predicted probabilities to binary labels (if needed)
y_pred_binary = (y_pred >= 0.5).astype(int)  # Convert to binary labels using 0.5 threshold

# Compute confusion matrix
cm = confusion_matrix(y, y_pred_binary)

# Extract True Positives, False Positives, False Negatives, True Negatives
TP = cm[1, 1]  # True positives: correct positive predictions
FP = cm[0, 1]  # False positives: negatives predicted as positives
FN = cm[1, 0]  # False negatives: positives predicted as negatives
TN = cm[0, 0]  # True negatives: correct negative predictions
```
and after `auc_metric = metrics.auc(fpr, tpr)` add:
```py
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
```
In the metadata dictionary, rename `config` to `config.to_dict()`
then remove:
```py
np.save(
    exp_save_root / "config.npy",
    metadata,
)
```
and replace with:
```py
# Define the file path where you want to save the JSON
metadata_clean = convert_paths_to_str(metadata)
json_file_path = exp_save_root / "config.json"

# Save metadata to a JSON file
with open(json_file_path, 'w') as f:
    json.dump(metadata_clean, f, indent=4)
```

Finally, modify the `experiment_name` in the main loop to:
```py
experiment_name = f"{config.EXPERIMENT_NAME}-{config.MODE}-{datetime.today().strftime('%Y%m%d')}-{config.VERSION}"
```


# Models Folder
This folder should contain a `.py` file with the class of your model, and any files that are necessary for running the model.
I.e., if you need some huggingface library, instead of using `!pip install x`, you can use `!pip download x`, and move all the files into this folder.
The files will be of .whl format.

Here is an example of a model class. Note that for training, for huggingface models (and any models in general), you likely should set pretrained to true,
in order to not train the model from scratch. For inference, this should be turned off, as the weights should be loaded that you saved during training.:
```py
import torch
import torch.nn as nn
import timm  # assuming timm is already installed from .whl

class DeiTSmall(nn.Module):
    def __init__(self, num_classes=1, pretrained=False):  # Set pretrained=False for offline use, i.e. when loading the model in for inference
        super(DeiTSmall, self).__init__()
        self.model = timm.create_model('deit_small_patch16_224', pretrained=False)

        # Replace the classification head
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Example test
if __name__ == "__main__":
    image = torch.randn(4, 3, 224, 224)  # Required input size for DeiT
    model = DeiTSmall()
    output = model(image)
    print(output.shape)
```

# Slurm submission
Finally, to submit training on cluster via slurm, ensure that your `.slurm` file has the following structure:
```shell
#!/bin/bash
#SBATCH --account=cseduimc037
#SBATCH --partition=csedu
#SBATCH --qos=csedu-small
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --output=myjob-%j.out
#SBATCH --error=myjob-%j.err

#virtual environment
source /vol/csedu-nobackup/course/IMC037_aimi/group01/baseline/bin/activate

cd /vol/csedu-nobackup/course/IMC037_aimi/group01/YOUR_DIRECTORY


python train.py
```
then run with `sbatch ./your_file.slurm`
