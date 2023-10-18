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
import atexit
import signal

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
    
    
    # CLEARML
    writer = SummaryWriter('DDP_experiments')
    ddp = rank
    # # setup mp_model and devices for this process
    # dev0 = (rank * 2) % world_size
    # dev1 = (rank * 2 + 1) % world_size

    if rank is None:
        rank =  torch.device('cuda')
        ddp = rank

    # train_folds = pd.read_csv('input_data/kfold_train.csv')
    train_folds = pd.read_csv('input_data/master_kfold_data.csv')
    train_folds['dump_path'] = train_folds.apply(lambda row : ''.join(config.IMG_PATH+'/'+str(row['patient_id'])+'/'+str(row['image_id'])+'.png'), axis=1)

    # train_folds['dump_path'] = train_folds.apply(lambda row : ''.join(config.IMG_PATH+'/'+str(row['machine_id'])+'/'+str(row['patient_id'])+'/'+str(row['image_id'])+'.png'), axis=1)

    # test- training
    # train_folds = train_folds[train_folds['kfold']!= 4] ## holdout set 
    train_set, valid_set = train_folds[train_folds['kfold']
                                       != fold], train_folds[train_folds['kfold'] == fold]


    train_set["weight"] = 1
    train_set.loc[train_set.cancer == 1, "weight"] = config.SAMPLE_POS_WEIGHT#len(train_set.loc[train_set.cancer == 0]) / len(train_set.loc[train_set.cancer == 1])
    train_set = train_set.sample(frac=1).reset_index(drop=True)
    valid_set.reset_index(drop=True, inplace=True)

    del train_folds
    print(f'Training data : {train_set.shape} , Validation data :{valid_set.shape}')

    if config.DEBUG_SAMPLES:
        if config.DISTRIBUTED and rank ==0:
            utils.display_sample_image(train_set, mode=f"train-fold:{fold}")
            utils.display_sample_image(valid_set, mode=f"valid-fold:{fold}")


    # calculate the pos_weight 
    pos_weight = config.MODEL_POS_WEIGHT#len(train_set[train_set[config.TARGET]==0])/len(train_set[train_set[config.TARGET]==1])

    # initialize CNN, cost, and optimizer
    model = RSNAModel(output_size = 1,n_cols=len(config.CATEGORY_AUX_TARGETS),pos_weight=pos_weight).to(rank)


    if config.DISTRIBUTED:
        # model_parallel_parts = engine.split_model_parallel(model)
        # model_parallel = nn.Sequential(*model_parallel_parts)
        # model_parallel = DataParallel(model,device_ids=[rank])
        if config.LOAD_PRETRAINED:
            print(' loaded model path>>>',config.LOAD_PRETRAINED_PATH[f'fold-{fold}'])
            model_parallel.load_state_dict(torch.load(config.LOAD_PRETRAINED_PATH[f'fold-{fold}'],map_location=torch.device(f'cuda:{fold}')))
            print("Pretrained model loaded")
        # model.load_state_dict(torch.load(model_filename,map_location=torch.device('cuda')))

        # model_parallel = DDP(model_parallel, device_ids=[rank], output_device=rank,find_unused_parameters=True)
        model_parallel = DDP(model, device_ids=[rank], output_device=rank,find_unused_parameters=True)
        # model_parallel = DDP(model_parallel, device_ids=[rank], output_device=rank)


    if config.DISTRIBUTED:

        train_dataloader = dataset.RSNADataloader(train_set, 'dump_path', weights=train_set["weight"].tolist(), transforms=config.TRAIN_TRANSFORM, test=False).fetch(
            batch_size=config.TRAIN_BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=True, drop_last=False,
            ddp=[rank, world_size])

        val_dataloader = dataset.RSNADataloader(valid_set, 'dump_path', weights=None,transforms=config.VAL_TRANSFORM, test=False).fetch(
            batch_size=config.VALID_BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=False, drop_last=False,
            ddp=[rank, world_size],validation=True)

    else:
        train_dataloader = dataset.RSNADataloader(train_set, 'dump_path',weights=None, transforms=config.TRAIN_TRANSFORM, test=False).fetch(
            batch_size=config.TRAIN_BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=True, drop_last=False)

        val_dataloader = dataset.RSNADataloader(valid_set, 'dump_path', weights=None, transforms=config.VAL_TRANSFORM, test=False).fetch(
            batch_size=config.VALID_BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=False, drop_last=False)

    # Optimizers
    optimizer = torch.optim.AdamW(model_parallel.parameters(), lr=config.LEARNING_RATE, weight_decay=config.ADAMW_DECAY)
    # scheduler

    num_train_steps = int(len(train_set)//(world_size * config.TRAIN_BATCH_SIZE))

    if config.WARMUP_RATIO > 0:
        num_warmup_steps = int(config.WARMUP_RATIO * num_train_steps)
    else:
        num_warmup_steps = 0
    print(
        f"Total Training Steps: {num_train_steps}, Total Warmup Steps: {num_warmup_steps}")
    print("Len of Train Loader: ",len(train_dataloader) ," Len of Val Loader: ",len(val_dataloader))

    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=3,
    #                                             num_training_steps=num_warmup_steps)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.99)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.ONE_CYCLE_MAX_LR, epochs=config.EPOCHS,
                                                        steps_per_epoch=len(train_set)//(world_size * config.TRAIN_BATCH_SIZE),
                                                        pct_start=config.ONE_CYCLE_PCT_START)

    # scheduler =torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=config.ONE_CYCLE_MAX_LR,
    #     epochs=config.EPOCHS,
    #     steps_per_epoch=int(len(train_dataloader) / config.TRAIN_BATCH_SIZE * config.EPOCHS ),
    #     pct_start=0.1,
    #     anneal_strategy="cos",
    #     div_factor=config.LR_DIV,
    #     final_div_factor=config.LR_FINAL_DIV,
    # )

    # mixed precision training with NVIDIA Apex
    if config.FP16:
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.FP16_OPT_LEVEL)

    # Early stopping

    early_stopping = engine.EarlyStopping(
        patience=config.EARLY_STOPPING, verbose=True, model_name=f"checkpoint-fold-{fold}", ddp= ddp)

    # Training
    for epoch in range(config.EPOCHS):
        # train_loss = 0.9
        # optimizer.step()
        # scheduler.step()
        train_loss = engine.train_fn(
            train_dataloader, model_parallel, optimizer, rank, scheduler, writer, epoch)

        test_loss, pred, true_val, label_ids = engine.evaluate_fn(
            val_dataloader, model_parallel, rank, writer, epoch,flatten_size=len(valid_set))

        if config.DISTRIBUTED and torch.distributed.get_rank() == 0:

            print(f"EPOCH : {epoch + 1}/{config.EPOCHS}")
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
            print(f"| LEARNING RATE = {scheduler.get_lr()}")


            writer.add_scalars('Loss', {
                    "Train": train_loss,
                    "Valid": test_loss }, epoch)

            writer.add_scalars('Metrics', {
                    "ROC-AUC Score": metrics.roc_auc_score(true_val.flatten(),pred.flatten()),
                    "Normal F1 Score": metrics.f1_score(true_val.flatten(), np.round(pred.flatten()), average='weighted'),
                    "Optimal F1 Score": f1_score[0],
                    "Recall Score": metrics.recall_score(true_val.flatten(), np.round(pred.flatten())), 
                    "Precision Score" : metrics.precision_score(true_val.flatten(), np.round(pred.flatten())),
                    }, epoch)

            writer.add_scalars('LR', {
                    "LR": scheduler.get_lr().pop(),
                    "Last LR": scheduler.get_last_lr().pop(),
                    }, epoch)

            print(len(pred),"SANITY CHECK",len(true_val))
            early_stopping(val_loss=test_loss, model=model_parallel, epoch=epoch, preds=pred.flatten(), true_val=true_val.flatten(), label_ids= label_ids.flatten())
            if early_stopping.early_stop:
                print("Early stopping")
                del model, model_parallel, test_loss, pred, true_val, label_ids, scheduler, optimizer, train_dataloader, val_dataloader
                del train_set, valid_set, writer
                gc.collect()
                break
                
                

def cleanup():
    dist.destroy_process_group()
    torch.cuda.empty_cache()
    gc.collect()

def ddp_worker_setup(rank: int):
    print('worker:', rank)
    # Generate a unique process ID
    # pid = os.getpid()
    # unique_id = f"{rank}_{pid}"
    # print(f"Process with rank {rank} and PID {pid} has unique ID {unique_id}")

    world_size = config.MAX_PARALLEL
    if world_size > 0:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        # torch.multiprocessing.set_sharing_strategy('file_system')
        
    else:
        world_size = min(os.cpu_count(), config.MAX_PARALLEL)
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    arg = int(sys.argv[1])
    print('-' * 50)
    print(f'FOLD: {arg}')
    print('-' * 50)
    # Call barrier to synchronize all processes
    run(arg, rank, world_size)
    print("process done now cleaning")
    # dist.destroy_process_group(group='local')
    atexit.register(cleanup) 
    print("Refreshing the system for another fold =>>> sleep for 20 sec")
    # print(f"Killed : Process with rank {rank} and PID {pid} has unique ID {unique_id}")
    # make sure all child processes have properly exited
    dist.barrier()
    os.kill(os.getpid(), signal.SIGTERM)
    print("All Distributed process killed")
    return
    
    
    
def main():
    # pid = os.getpid()
    world_size = config.MAX_PARALLEL
    if config.DISTRIBUTED:
        print('Distributed training:', config.DISTRIBUTED, "world_Size", world_size)
        if world_size > 0:
            
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = '10004'
            # mp.spawn(ddp_worker_setup, nprocs=world_size)
            with mp.spawn(ddp_worker_setup, nprocs=world_size, join=True) as processes:
                for process in processes:
                    process.terminate()
            
    else:
        print("NOT RUNNING DDP")
        for fold in range(0, 1): #5
            print('-' * 50)
            print(f'FOLD: {fold}')
            print('-' * 50)
            run(fold) 
    
    # dist.barrier()
    # dist.destroy_process_group()
    # os.kill(pid, signal.SIGKILL)
    print("Refreshing DONE!!!")

if __name__ == "__main__":
    main()