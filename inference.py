import config
import dataset
import engine
import utils
import pandas as pd
import numpy as np
import joblib
from apex import amp
import os
import sys
import warnings
import logging
import time
import gc 
import sys

from model import RSNAModel
from sklearn import model_selection
from sklearn import metrics
from sklearn import preprocessing
from sklearn.utils import class_weight
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from clearml import Task

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter


warnings.filterwarnings('ignore')
logging.getLogger("clearml").setLevel(logging.ERROR)

# task = Task.init(project_name = 'Users/test_user', task_name = 'mod_check')
# # task = Task.init(project_name = 'Users/suraj', task_name = '1536_cropped43_1',auto_connect_streams=False)
# task.connect(config.gen_args(config))
# task.execute_remotely(queue_name="xlarge")


def run(fold, rank=None, world_size=None):
    
    torch.cuda.empty_cache()
    # CLEARML
    writer = SummaryWriter('DDP_experiments')
    ddp = rank
    # # setup mp_model and devices for this process
    # dev0 = (rank * 2) % world_size
    # dev1 = (rank * 2 + 1) % world_size

    if rank is None:
        rank =  torch.device('cuda')
        ddp = rank

    train_folds = pd.read_csv('external_data/additional_train.csv')
    train_folds['dump_path'] = train_folds.apply(lambda row : ''.join(config.INFERENCE_IMG_PATH+'/'+str(row['patient_id'])+'/'+str(row['image_id'])+'.png'), axis=1)


    train_set["weight"] = 1
    train_set.loc[train_set.cancer == 1, "weight"] = config.SAMPLE_POS_WEIGHT#len(train_set.loc[train_set.cancer == 0]) / len(train_set.loc[train_set.cancer == 1])
    train_set = train_set.sample(frac=1).reset_index(drop=True)
    valid_set.reset_index(drop=True, inplace=True)

    del train_folds
    print(f'Training data : {train_set.shape} , Validation data :{valid_set.shape}')

    if config.DEBUG_SAMPLES:
        if config.DISTRIBUTED and rank ==0:
            utils.display_sample_image(valid_set, mode=f"valid-fold:{fold}")


    if config.DISTRIBUTED:

            val_dataloader = dataset.RSNADataloader(valid_set, 'dump_path', weights=None,transforms=config.VAL_TRANSFORM, test=True).fetch(
                batch_size=config.VALID_BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=False, drop_last=False,
                ddp=[rank, world_size],validation=True)
        else:

            val_dataloader = dataset.RSNADataloader(valid_set, 'dump_path', weights=None, transforms=config.VAL_TRANSFORM, test=True).fetch(
                batch_size=config.VALID_BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=False, drop_last=False)
        
            
    for fold in range(len(config.LOAD_PRETRAINED_PATH)):
    
    
        # initialize CNN, cost, and optimizer
        model = RSNAModel(output_size = 1,n_cols=len(config.CATEGORY_AUX_TARGETS)).to(rank)
        model_parallel = DataParallel(model,device_ids=[rank])
        if config.LOAD_PRETRAINED:
            print(' loaded model path>>>',config.LOAD_PRETRAINED_PATH[f'fold-{fold}'])
            model_parallel.load_state_dict(torch.load(config.LOAD_PRETRAINED_PATH[f'fold-{fold}'],map_location=torch.device(f'cuda:{fold}')))
            print("Pretrained model loaded")


        if config.DISTRIBUTED:

            model_parallel = DDP(model_parallel, device_ids=[rank], output_device=rank)


        test_loss, pred, true_val, label_ids = engine.evaluate_fn(
            val_dataloader, model_parallel, rank, writer, fold, flatten_size=len(valid_set),inference=True)


        print(len(pred),"PRED SHAPE",len(true_val))
        print(f"| Train Loss = {train_loss} | Valid Loss = {test_loss}")
        print(f"| ROC-AUC Score = {metrics.roc_auc_score(true_val.flatten(), pred.flatten())}")
        print(f"| Normal F1 Score = {metrics.f1_score(true_val.flatten(), np.round(pred.flatten()), average='weighted')}")
        print(f"| Recall Score = {metrics.recall_score(true_val.flatten(), np.round(pred.flatten()))}") 
        print(f"| Precision Score = {metrics.precision_score(true_val.flatten(), np.round(pred.flatten()))}")

        f1_score = engine.optimal_f1(true_val, pred)
        pf1 = engine.pfbeta(true_val.flatten(), np.round(pred.flatten()))
        print(f"| Optimal F1 Score = {f1_score}")
        print(f"| PF Beta Score @ 0.5 = {pf1}")


        print(len(pred),"SANITY CHECK",len(true_val))

        del model_parallel, test_loss, pred, true_val, label_ids
        gc.collect()
        torch.cuda.empty_cache()

                

def ddp_worker_setup(rank: int):
    
    print('worker:', rank)
    world_size = torch.cuda.device_count()
    if world_size > 0:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    else:
        world_size = min(os.cpu_count(), config.MAX_PARALLEL)
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    arg = int(sys.argv[1])
    print('-' * 50)
    print(f'FOLD: {arg}')
    print('-' * 50)
    run(arg, rank, world_size)
    dist.destroy_process_group()
    torch.cuda.empty_cache()
    gc.collect()
    # dist.destroy_process_group(group='local')
    print("Refreshing the system for another fold =>>> sleep for 20 sec")
        
        
    

def main():
    
    torch.cuda.empty_cache()
    
    if config.DISTRIBUTED:
        world_size = torch.cuda.device_count()
        print('Distributed training:', config.DISTRIBUTED, "world_Size", world_size)
        if world_size > 0:
            world_size = min(os.cpu_count(), config.MAX_PARALLEL)
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = '10002'
            mp.spawn(ddp_worker_setup, nprocs=world_size)
    else:
        print("NOT RUNNING DDP")
        run(fold=1)


if __name__ == "__main__":
    main()