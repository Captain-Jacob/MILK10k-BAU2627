import os
import torch

class Config:
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Paths
    TRAIN_METADATA_CSV = "MILK10k_Training_Metadata.csv"
    TEST_METADATA_CSV = "MILK10k_Test_Metadata.csv"
    TRAIN_GROUND_TRUTH_CSV = "MILK10k_Training_GroundTruth.csv"
    
    TRAIN_ZIP = "MILK10k_Training_Input.zip"
    TEST_ZIP = "MILK10k_Test_Input.zip"
    
    TRAIN_IMG_DIR = "MILK10k_Training_Input"
    TEST_IMG_DIR = "MILK10k_Test_Input"
    
    OUTPUT_DIR = "checkpoints"
    RESUME_CHECKPOINT = "checkpoints/latest_resume_checkpoint.pth"
    SUBMISSION_PATH = "submission.csv"
    
    # Optimization & Architecture Settings
    BACKBONE = "efficientnet_b0"
    IMAGE_SIZE = 224
    NUM_CLASSES = 11
    
    # Optimized for RTX 5070 12GB + CPU parallel loading
    BATCH_SIZE = 32         # Doubled for high GPU Tensor Core throughput
    NUM_WORKERS = 4        # Parallel CPU image decoding processes
    EPOCHS = 20
    LR = 3e-4
    WEIGHT_DECAY = 1e-4
    NUM_FOLDS = 5

    CLASSES = [
        "AKIEC", "BCC", "BEN_OTH", "BKL", "DF", 
        "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"
    ]
    
    MONET_KEYWORDS = ["monet", "ulcer", "hair", "vascul", "erythem", "pigment", "gel", "mark"]

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)