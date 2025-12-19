import torch
import os

class Config:
    DATA_PATH = "Course_work/EuroSAT"
    MODELS_PATH = "Course_work/trained_models"
    
    NUM_EPOCHS = 3
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    IMG_SIZE = (64, 64)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    CLASSES = [
        'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 
        'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 
        'River', 'SeaLake'
    ]
    NUM_CLASSES = len(CLASSES)

    TORCH_HOME = os.path.join(os.path.dirname(__file__), 'torch_models')
    os.makedirs(TORCH_HOME, exist_ok=True)
    os.environ['TORCH_HOME'] = TORCH_HOME