# Explanation
This repo contains our contribution to the LUNA25 challenge. This README will outline our additions to the baseline provided by the challenge.
In the event of any questions/issues that are unrelated to our contributions, we would like to refer you to the README of the provided baseline, which is available in this repo under the name README_LUNA25, or at the original repo: https://github.com/DIAGNijmegen/luna25-baseline-public.
Nevertheless, should you have issues or questions regarding our code, feel free to create an issue, and we will get back to you ASAP.
As for how you can make changes to the code to add new models, see the ``modifications.md`` file in the `documentation` folder. This contains a guide as to how to make various changes to the baseline code, along with examples and explanations of why they are needed.

# Model Additions
We have added several 2D and 3D models to the baseline code. These can simply be selected by adding the correct name to the model name variable in the `experiment_config.py` file.
These include Swin, DeiT, MedicalNet (3D Resnets), and ConvNeXt.
For the MedicalNet pretrained weights, we refer you to the original repository, which contains links to download them from Google Drive and Tencent Weiyun: https://github.com/Tencent/MedicalNet.

# Training and logging
We have added more thorough metrics for analysis during training, for validating on a validation set, as well as validation on the training set itself, in order to investigate if the model is overfitting.
However, validating on the training dataset roughly doubles the training time, as a result, the metrics for validating on the training set are currently commented out in `train.py`.

# How to run
To run the code, you must implement your model in the `models` folder, followed by adding the models to `if` statements in `dataloader.py, train.py,` and `processor.py` in order to initialize them and ensure the correct input shapes are given.
Then, the model can be trained using `python3 train.py`. 
You can then run inference on the model with `python3 inference.py`
Naturally, you must adapt the paths for the data, save locations, and other directories according to your file structure. This can be done in `experiment_config.py, inference.py, processor.py`. For any additional details, refer to `README_LINA25.md`
