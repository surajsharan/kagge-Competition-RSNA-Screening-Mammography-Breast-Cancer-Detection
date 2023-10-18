import subprocess
import sys
import time

children = []
#Step 1: Launch all the children asynchronously
for i in range(10):
    #For testing, launch a subshell that will sleep various times
    popen = subprocess.Popen(["/bin/sh", "-c", "sleep %s" % (i + 8)])
    children.append(popen)
    print ("launched subprocess PID %s" % popen.pid)

#reverse the list just to prove we wait on children in the order they finish,
#not necessarily the order they start
children.reverse()
#Step 2: loop until all children are terminated
while children:
    #Step 3: poll all active children in order
    children[:] = [child for child in children if child.poll() is None]
    print ("Still running: %s" % [popen.pid for popen in children])
    time.sleep(1)

print ("All children terminated")


class RSNAModel(torch.nn.Module):
    def __init__(self, output_size, n_cols,pos_weight=None):
        super().__init__()
        self.no_columns, self.output_size = n_cols, output_size
        self.features = efficientnet_b4(pretrained=True)
        self.dropout = nn.Dropout(0.1)        
        self.classification = nn.Linear(1792,1)
         
    def forward(self, image, meta=None,target=None,image_id=None):   

        x = image
        features = self.features.forward_features(x.float())
        x = F.adaptive_avg_pool2d(features,1)
        x = torch.flatten(x,1,3)
        x = self.dropout(x)
        # CLASSIF
        out = self.classification(x)
        return out
    
import torch
import torch.nn as nn

def loss_fn(self, output, target, pos_weight):
    # Define the smoothing factor
    eps = 0.1

    # Define the class weights
    class_weights = torch.tensor([1, pos_weight], dtype=torch.float)
    class_weights = class_weights / class_weights.sum()

    # Define the target distribution
    num_classes = 2 # binary classification
    target_dist = torch.ones(target.shape[0], num_classes) * eps / num_classes
    target_dist[torch.arange(target.shape[0]), target.long()] = 1 - eps + eps / num_classes
    target_dist = target_dist * class_weights.unsqueeze(0)

    # Compute the loss
    return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)(output, target_dist)



def cleanup():
    dist.destroy_process_group()
    torch.cuda.empty_cache()
    gc.collect()

def ddp_worker_setup(rank: int):
    
    print('worker:', rank)
    world_size = 8
    if world_size > 0:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        torch.multiprocessing.set_sharing_strategy('file_system')
    
    arg = int(sys.argv[1])
    print('-' * 50)
    print(f'FOLD: {arg}')
    print('-' * 50)
    run(arg, rank, world_size)
    print("process done now cleaning")
    atexit.register(cleanup) 
    print("Refreshing the system for another fold =>>> sleep for 20 sec")

    
    
    
def main():
    world_size = 8
    print('Distributed training:', config.DISTRIBUTED, "world_Size", world_size)
    if world_size > 0:

        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '10004'
        mp.spawn(ddp_worker_setup, nprocs=world_size)


    print("All Distributed process killed")
    print("Refreshing DONE!!!")

if __name__ == "__main__":
    main()