import transformers
import torch
import pathlib
import numpy as np
import os
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2


def fix_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def gen_args(config=None):
    w_config={}
    for name ,values in vars(config).items():
        if name.isupper():
            w_config[name] = values  
    return w_config

EXPERIMENT_NAME = "boostingb4_seresnext_all"#"train_crop_voilut_1536_seresnext"
SEED = fix_all_seeds(2023)



# RAW_DATA

CATEGORY_AUX_TARGETS =  ['site_id', 'view', 'implant', 'machine_id', 'age']
TARGET = 'cancer'
ALL_FEAT =  CATEGORY_AUX_TARGETS + [TARGET]
IMG_PATH = "input_data/train_crop_voilut_2048"

# DATALODER 

IMG_SIZE = (2048,1024) # #(1536, 768)#(1024, 512)# (2048,1024)#(1536, 768)
TRAIN_BATCH_SIZE = 2
VALID_BATCH_SIZE =  4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 15
NUM_CLASSES =  1
NUM_WORKERS = 1#min(TRAIN_BATCH_SIZE,4) #1
N_FOLDS = 5
SAMPLE_POS_WEIGHT = 11
MODEL_POS_WEIGHT = 2
LABEL_SMOOTHING = 0
LOCAL_BINARY_PATTERN = False




# MODEL PARAM

# MODEL_BACKBONE = "efficientnet-b2"
MODEL_BACKBONE = "seresnext50_32x4d" #"efficientnet_b4" #'seresnext50_32x4d'##'seresnext50_32x4d'#"efficientnetv2_l"
PATH = str(pathlib.Path().absolute())+''.join('/output/'+EXPERIMENT_NAME)
pathlib.Path(PATH).mkdir(parents=True, exist_ok=True)

PATH_PRETRAINED = 'output/train_crop_voilut_1536_check'
LOAD_PRETRAINED = False
LOAD_PRETRAINED_PATH = {
                  'fold-0': f'{PATH_PRETRAINED}/checkpoint-fold-0_ddp_ep_1_model.bin',
                  'fold-1': f'{PATH_PRETRAINED}/checkpoint-fold-1_ddp_ep_0_model.bin',
                  'fold-2': f'{PATH_PRETRAINED}/checkpoint-fold-2_ddp_ep_2_model.bin',
                  'fold-3': f'{PATH_PRETRAINED}/checkpoint-fold-3_ddp_ep_0_model.bin',
                  'fold-4': f'{PATH_PRETRAINED}/checkpoint-fold-4_ddp_ep_0_model.bin',
                    }



INFERENCE_IMG_PATH = "external_data/train_crop_voilut_1536"
# OPTIM

LEARNING_RATE =  1e-3
ONE_CYCLE_MAX_LR =  1e-3
LR_FINAL_DIV =  10000.0
LR_DIV = 1.0
POS_WEIGHT = 0.9
ONE_CYCLE = True
ONE_CYCLE_PCT_START = 0.1
WARMUP_RATIO = 1

ADAMW = False
ADAMW_DECAY = 0.024


DROPOUT = 0.2
AUX_LOSS_WEIGHT = 94
TTA = False
IN_CHANNELS = 3


# TRAINING

DISTRIBUTED = True
MAX_PARALLEL = 12
FP16 = False
EARLY_STOPPING = 2


# AUGMENTATIONS

TRAIN_TRANSFORM = A.Compose([
                            A.Resize(IMG_SIZE[0],IMG_SIZE[1]),
                            A.CLAHE (clip_limit=4.0, tile_grid_size=(8, 8), p=0.2) ,
                            A.GridDistortion(p=0.3),
                            # A.VerticalFlip(p=0.3),
                            A.HorizontalFlip(p=0.3),
                            # A.LongestMaxSize(max_size=args.img_size, interpolation=1, p=1.0),
                            # A.CenterCrop(height=1024, width=1024, p=1.0),
                            # A.RandomBrightnessContrast(p=0.2, brightness_limit=0.2, contrast_limit=0.2),
                            
                            A.CoarseDropout(always_apply=False, p=0.3, min_holes=8, max_holes=16, min_height=10, min_width=10, max_height=30, max_width=20),
                            # A.ElasticTransform(always_apply=False, p=0.3, alpha=1.0, sigma=50.0, alpha_affine=50.0, interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None, approximate=False),
                            # A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                            # A.Normalize((0.485, 0.485, 0.485), (0.229, 0.229, 0.229)),
                            ToTensorV2()
                            ])

VAL_TRANSFORM = A.Compose([
                            A.Resize(IMG_SIZE[0],IMG_SIZE[1]),
                            # A.Normalize((0.485, 0.485, 0.485), (0.229, 0.229, 0.229)),
                            # A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                            ToTensorV2()
                        ])

TARGET_TRANSFORM = A.Compose([
                            A.Resize(IMG_SIZE[0],IMG_SIZE[1]),
                            # A.CLAHE (clip_limit=4.0, tile_grid_size=(8, 8), p=0.2) ,
                            A.GridDistortion(p=0.5),
                            # A.VerticalFlip(p=0.3),                           
                            A.CoarseDropout(always_apply=False, p=0.6, min_holes=8, max_holes=16, min_height=10, min_width=10, max_height=30, max_width=20),
                            ToTensorV2()
                            ])
# CHECKPOINT

MODEL_PATH = "model.bin"
DEBUG_SAMPLES = False
