# 📦 LUNA25 Baseline Algorithm
Thank you for participating in the [LUNA25 Challenge](https://luna25.grand-challenge.org/).

In LUNA25, we want to use artificial intelligence for lung nodule malignancy risk estimation on low-dose chest CT scans. For this, we have prepared two baseline models (2D and 3D model) that can help you get started. 

The development of your algorithms should be performed using your local GPU or a cloud platform (such as AWS or Azure), while algorithm evaluation will be performed exclusively on the [Grand-Challenge](https://grand-challenge.org/) platform.

## 🗂️ Content
This baseline algorithm provides a framework for training and testing models. While it includes basic scripts, we encourage you to extend and customize them to develop alternative or improved methods.

Important Files:
- 🦾 `train.py`: A script for training the baseline algorithm on local data.
- 🦿 `inference.py`: A script for testing the trained algorithm using a specified configuration.
- 🧮 `Dockerfile`: A file to build a Docker container for deployment on Grand-Challenge. For help on setting up Docker with GPU support you can check the documentation on [Grand-Challenge](https://grand-challenge.org/documentation/setting-up-wsl-with-gpu-support-for-windows-11/) or [Docker](https://docs.docker.com/engine/install/ubuntu/) for additional information.

## ⚙️ Setting up the Environment
To set up the required environment for the baseline algorithm:
1. **Create an environment and esure Python is Installed**: Install Python 3.9 or higher:
    ```bash
    conda create -n luna25-baseline python==3.9
    ```
2. **Install Dependencies**:
    - Run the following command to install the dependencies listed in `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```
3. **Verify Installation**:
    - Test the installation by running:
    ```bash
    python --version
    pip list
    ```
    Ensure all required packages are listed and no errors are reported.

## 🚀 Performing a Training Run
1. **Set up training configurations**

Open `experiment_config.py` to edit your training configurations. Key parameters include:

- `self.MODE`: Set this to 2D or 3D depending on the desired baseline model.
- `self.EXPERIMENT_NAME`: Specify the name of your experiment (e.g. LUNA25-baseline).
- `self.CSV_DIR_TRAIN`: the path to the training csv file
- `self.DATADIR`: the path where the images are stored


2. **Training the Model**

To train the model using the `train.py` script:
```bash
python train.py
```
This script uses the settings from experiment_config.py to initialize and train the model.

## 🧪 Testing the Trained Algorithm
1. **Configure the inference script**

Open the `inference.py` script and configure:
- `INPUT_PATH`: Path to the input data (CT, nodule locations and clinical information). Keep as `Path("/input")` for Grand-Challenge.
- `RESOUCE_PATH`: Path to resources (e.g., pretrained models weights) in the container. Defaults to `/results` directory (see Dockerfile)
- `OUTPUT_PATH`: Path to store the output in your local directory. Keep as `Path("/output")` for Grand-Challenge.
- **Inputs for the `run()` function**:
    - `mode`: Match this to the mode used during training (2D or 3D).
    - `model_name`: Specify the experiment_name matching the training configuration (corresponding to experiment_name directory that contains the model weights in `/results`).

2. **Updating the Docker Image Tag**

In `do_test_run.sh`, update the Docker image tag as needed:
```bash
DOCKER_IMAGE_TAG="luna25-baseline-3d-algorithm-open-development-phase"
```


3. **Running the Test Script**

To test the trained model for running inference run: 
```bash
./do_test_run.sh
``` 

This script performs the following:
- Uses Docker to execute the `inference.py` script.
- Mounts necessary input and output directories.
- Adjusts the Docker image tag (if updated) before running.

## 🐳 Building the Docker Image
To build the Docker container required for submission to Grand-Challenge run:
```bash
./do_save.sh
```
This will output a *.tar.gz file, which can be uploaded to Grand-Challenge.
More information on testing and deploying your container can be found [here](https://grand-challenge.org/documentation/test-and-deploy-your-container/).

## 🛠️ Extending the Baseline
While this baseline provides a starting point, participants are encouraged to:

- Implement advanced AI models.
- Explore alternative data preprocessing and augmentation techniques.
- perform Ensemble Learning
- train models using entire or larger CT scan inputs

For questions, refer to the [LUNA25 Challenge Page](https://luna25.grand-challenge.org/).

Good luck!