from pathlib import Path


class Configuration(object):
    def __init__(self) -> None:

        # Working directory
        self.WORKDIR = Path("C:/Users/chari/m-data-science/ai-medical-imaging/luna25-vit")#Path("/vol/csedu-nobackup/course/IMC037_aimi/group01/luna25-vit")
        self.RESOURCES = self.WORKDIR / "resources"
        # Starting weights for the I3D model
        self.MODEL_RGB_I3D = (
            self.RESOURCES / "model_rgb.pth"
        )
        
        # Data parameters
        # Path to the nodule blocks folder provided for the LUNA25 training data. 
        self.DATADIR = Path("/vol/csedu-nobackup/course/IMC037_aimi/group01/data/nodule_blocks/luna25_nodule_blocks") #Path("V:/projects/luna25/NLST/nodule_blocks")
        # Path to the folder containing the CSVs for training and validation.
        self.CSV_DIR = Path("/vol/csedu-nobackup/course/IMC037_aimi/group01/data/data_csv") #Path("V:/projects/luna25/dataset_csv")
        # We provide an NLST dataset CSV, but participants are responsible for splitting the data into training and validation sets.
        self.CSV_DIR_TRAIN = self.CSV_DIR / "train.csv" # Path to the training CSV
        self.CSV_DIR_VALID = self.CSV_DIR / "val.csv" # Path to the validation CSV

        # Results will be saved in the /results/ directory, inside a subfolder named according to the specified EXPERIMENT_NAME and MODE.
        self.EXPERIMENT_DIR = self.WORKDIR / "results"
        if not self.EXPERIMENT_DIR.exists():
            self.EXPERIMENT_DIR.mkdir(parents=True)
            
        self.EXPERIMENT_NAME = "LUNA25-vit"
        self.MODE = "vit" # 2D or 3D

        # Training parameters
      #  self.SEED = 2025
       # self.NUM_WORKERS = 8
        #self.SIZE_MM = 50
       # self.SIZE_PX = 64
       # self.BATCH_SIZE = 32
     #   self.ROTATION = ((-20, 20), (-20, 20), (-20, 20))
      #  self.TRANSLATION = True
       # self.EPOCHS = 10
      #  self.PATIENCE = 20
      #  self.PATCH_SIZE = [64, 128, 128]
      #  self.LEARNING_RATE = 1e-4
      #  self.WEIGHT_DECAY = 5e-4

	    # Training parameters
        self.SEED           = 2025
        self.NUM_WORKERS    = 8
        
        # since ViT-base/DeiT expects 224×224 inputs:
        self.SIZE_PX        = 224  
        # physical size no longer needed for ViT patching
        self.SIZE_MM      = 50  
        
        # grayscale
        self.IN_CHANS       = 3  
        
        # smaller batch to fit GPU memory
        self.BATCH_SIZE     = 16  
        
        self.ROTATION       = ((-20, 20), (-20,20),(-20,20))  
        self.TRANSLATION    = True  
        
        # fine‐tune longer, with early stopping
        self.EPOCHS         = 50  
        self.PATIENCE       = 10  
        
        # ViT internal patch size (16×16)
        self.PATCH_SIZE     = [64, 128, 128]
        
        # finetuning LR & regularization for transformers
        self.LEARNING_RATE  = 3e-5  
        self.WEIGHT_DECAY   = 1e-2  

        self.VERSION = 1
        
    def to_dict(self):
        """Converts the Configuration object to a dictionary."""
        return {k: v for k, v in self.__dict__.items()}

config = Configuration()
