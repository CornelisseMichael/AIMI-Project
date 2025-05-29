# Explanation
This repo contains our contribution to the LUNA25 challenge. This README will outline our additions to the baseline provided by the challenge.
In the event of any questions/issues that are unrelated to our contributions, we would like to refer you to the README of the provided baseline, which is available in this repo under the name README_LUNA25, or at the original repo: https://github.com/DIAGNijmegen/luna25-baseline-public.
Nevertheless, should you have issues or questions regarding our code, feel free to create an issue, and we will get back to you ASAP.
As for how you can make changes to the code to add new models, see the ``modifications.md`` file in the `documentation` folder. This contains an extensive guide as to the various changes needed with examples, and why they are needed.

# Model Additions
We have added several 2D and 3D models to the baseline code. These can simply be selected by adding the correct name to the model name variable in the `experiment_config.py` file.
These include Swin, DeiT, MedicalNet (3D Resnets), and ConvNeXt.
For the MedicalNet pretrained weights, we refer you to the original repository, which contains links to download them from Google Drive and Tencent Weiyun: https://github.com/Tencent/MedicalNet.

# Training and logging
We have added more thorough metrics for analysis during training, for validating on a validation set, as well as validation on the training set itself, in order to investigate if the model is overfitting.
However, validating on the training dataset roughly doubles the training time, as a result, the metrics for validating on the training set are currently commented out in `train.py`.

