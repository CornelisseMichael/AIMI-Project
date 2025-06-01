from pathlib import Path


class Configuration(object):
    def __init__(self) -> None:

	# Version for starting training with k-fold, here we train fold 3
        self.VERSION = 3

        # local directories
        self.WORKDIR = Path("/content/luna25-ensemble2d-adam")
        self.DATADIR = Path("/content/luna25_images")
        self.CSV_DIR = Path("/content/luna25-ensemble2d-adam/data")

    	# self.WORKDIR = Path("C:/Users/chari/m-data-science/ai-medical-imaging/luna25-ensemble")
     	# self.DATADIR = Path("C:/Users/chari/m-data-science/ai-medical-imaging/data/luna25_nodule_blocks")
     	# self.CSV_DIR = Path("C:/Users/chari/m-data-science/ai-medical-imaging/data")
        
        
        # cluster directories
        # self.WORKDIR = Path("/vol/csedu-nobackup/course/IMC037_aimi/group01/luna25-ensemble2d-adam")
        # self.DATADIR = Path("/vol/csedu-nobackup/course/IMC037_aimi/group01/data/nodule_blocks/luna25_nodule_blocks") 
        # self.CSV_DIR = Path("/vol/csedu-nobackup/course/IMC037_aimi/group01/luna25-ensemble2d-adam/data")

        # Docker creation directories
        # self.WORKDIR = Path("F:/AIMI/luna25-ensemble2d-adam") # FOR DOCKER CREATION
        # self.DATADIR = Path("F:/AIMI/luna25_images") # FOR DOCKER CREATION
        # self.CSV_DIR = Path("F:/AIMI/luna25-ensemble2d-adam/data") # FOR DOCKER CREATION
        
        
        
        self.RESOURCES = self.WORKDIR / "resources"
        # Starting weights for the I3D model
        self.MODEL_RGB_I3D = (
            self.RESOURCES / "model_rgb.pth"
        )

        # We provide an NLST dataset CSV, but participants are responsible for splitting the data into training and validation sets.
        self.CSV_DIR_TRAIN = self.CSV_DIR / f"train{self.VERSION}.csv" # Path to the training CSV
        self.CSV_DIR_VALID = self.CSV_DIR / f"val{self.VERSION}.csv" # Path to the validation CSV

        # Results will be saved in the /results/ directory, inside a subfolder named according to the specified EXPERIMENT_NAME and MODE.
        self.EXPERIMENT_DIR = self.WORKDIR / "ensemble_models/smaller_weight_decay"
        if not self.EXPERIMENT_DIR.exists():
            self.EXPERIMENT_DIR.mkdir(parents=True)
            
        self.EXPERIMENT_NAME = f"LUNA25-deit_small_model_{self.VERSION}"
        self.MODE = "ensemble_deit_3d"

	    # Training parameters
        self.SEED           = 2025
        self.NUM_WORKERS    = 8
        
        # since ViT-base/DeiT expects 224×224 inputs:
        self.SIZE_PX        = 224  
        
        self.SIZE_MM      = 50  
        
        # rgb
        self.IN_CHANS       = 3  
	    
        self.BATCH_SIZE     = 32
        
        self.ROTATION       = ((-20, 20), (-20,20),(-20,20))  
        self.TRANSLATION    = True  
        
        # fine‐tune longer, with early stopping
        self.EPOCHS         = 60  
        self.PATIENCE       = 10  
        self.PATCH_SIZE     = [64, 128, 128]
        
        # finetuning LR & regularization for transformers
        self.LEARNING_RATE  = 1e-6 
        self.WEIGHT_DECAY   = 1e-3  

        self.NUM_IMAGES = 9 

        self.DEVICE = "cuda"#cuda:0
        self.MODEL = "deit_small"
        self.USE_SCHED = True
        
    def to_dict(self):
        """Converts the Configuration object to a dictionary."""
        return {k: v for k, v in self.__dict__.items()}

config = Configuration()
