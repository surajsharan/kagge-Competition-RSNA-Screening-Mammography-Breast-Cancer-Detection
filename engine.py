import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import config
from datetime import datetime
from sklearn import metrics
import matplotlib.pyplot as plt
from apex import amp

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=5, verbose=False,model_name=None,ddp=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.model_name = model_name
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf # np.Inf: save on loss(lower the better)|-np.Inf:for save on acc(max the better)
        self.ddp = ddp 

    def __call__(self, val_loss, model, epoch, preds, true_val, label_ids):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, preds, true_val, label_ids)
        elif score < self.best_score: # <: save on loss(lower the better)| >:save on acc(max the better)
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, preds, true_val, label_ids)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch, preds, true_val, label_ids):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if self.ddp is None: 
            np.save(''.join(
                config.PATH + "/" + self.model_name+"_ep_"+str(epoch)+"_"+"predictions.npy"), preds)
            np.save(''.join(
                config.PATH + "/" + self.model_name+"_"+"true_val.npy"), true_val)
            ####
            np.save(''.join(
                config.PATH + "/" + self.model_name+"_"+"label_ids.npy"), label_ids)
            ###
            torch.save(model.state_dict(), ''.join(
                config.PATH + "/" + self.model_name+"_ep_"+str(epoch)+"_"+config.MODEL_PATH))
        else:
            if self.ddp is not None and torch.distributed.get_rank()==0:
                
                np.save(''.join(
                config.PATH + "/" + self.model_name+"_ddp_ep_"+str(epoch)+"_"+"predictions.npy"), preds)
                
                np.save(''.join(
                config.PATH + "/" + self.model_name+"_"+"true_val.npy"), true_val)
                ####
                np.save(''.join(
                    config.PATH + "/" + self.model_name+"_"+"label_ids.npy"), label_ids)
                ###
                
                torch.save(model.module.state_dict(), ''.join(
                    config.PATH + "/" + self.model_name+"_ddp_ep_"+str(epoch)+"_"+config.MODEL_PATH))
                
        self.val_loss_min = val_loss


def split_model_parallel(model):
    parts = [
        nn.Sequential(*list(model.features.children())[:5]),
        nn.Sequential(*list(model.features.children())[5:7]),
        nn.Sequential(*list(model.features.children())[7:]),
        nn.Sequential(model.dropout, model.classification)
    ]
    return parts
        
def train_fn(dataloader, model, optimizer, device, scheduler, summary_writer, epoch):  
    model.train()
    total_loss = 0
    for data in tqdm(dataloader, total=len(dataloader)):
        for k, v in data.items():
            data[k] = v.to(device)
        optimizer.zero_grad()
        output, loss = model(**data)
        loss = loss.clone()
        torch.distributed.all_reduce(loss)
        loss /= torch.distributed.get_world_size()
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if torch.distributed.get_rank() == 0:
            summary_writer.add_scalars('TRAINING', { "Train Loss": loss.item()}, epoch)

    return total_loss / len(dataloader)


def pfbeta(labels, predictions, beta=1.):
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(labels)):
        prediction = min(max(predictions[idx], 0), 1)
        if (labels[idx]):
            y_true_count += 1
            ctp += prediction
        else:
            cfp += prediction

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / max(y_true_count, 1)  # avoid / 0
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0

def optimal_f1(labels, predictions):
    labels = labels.flatten()
    thres = np.linspace(0, 1, 101)
    f1s = [pfbeta(labels, predictions > thr) for thr in thres]
    idx = np.argmax(f1s)
    return f1s[idx], thres[idx]


def evaluate_fn(dataloader, model, device, summary_writer, epoch, flatten_size, inference=False):
    model.eval()
    loss_val_total = 0
    counter = 0
    if torch.distributed.get_rank() == 0:
        predictions, true_vals = [], []
        label_ids = []
    with torch.no_grad():

        if inference:
            with amp.autocast(enabled=True):
                for data in tqdm(dataloader, total=len(dataloader)):
                    for k, v in data.items():
                        data[k] = v.to(device)
                    output, _ = model(**data)

                    output = output.detach().cpu().numpy()
                    predictions.append(output)
                predictions = np.concatenate(predictions, axis=0)

        else:
            for data in tqdm(dataloader, total=len(dataloader)):
                counter += 1
                for k, v in data.items():
                    data[k] = v.to(device)
                output, loss = model(**data)
                
                loss = loss.clone()
                torch.distributed.all_reduce(loss)
                loss /= torch.distributed.get_world_size()
                
                loss_val_total += loss.item()
                target = data['target']
                label_id = data['image_id']
                all_out = torch.stack([output.flatten(),target,label_id],dim=1)
                
                
                
                # Calculate the number of padding samples
                num_padding = config.VALID_BATCH_SIZE - len(data['target'])
                if num_padding !=0:
                    target_padding = torch.zeros(num_padding, dtype=torch.int64, device=device)
                    target = torch.cat([data['target'], target_padding], dim=0)
                    
                    padding = torch.zeros(num_padding, 1, dtype=torch.float32, device=device)
                    output = torch.cat([output, padding], dim=0)
                    
                    padding_id = torch.zeros(num_padding, dtype=torch.int64, device=device)
                    label_id = torch.cat([label_id, padding_id], dim=0)
                    
                    all_out = torch.stack([output.flatten(),target,label_id],dim=1)
                    
   
                
                tensor_out_all = torch.zeros(config.VALID_BATCH_SIZE  * torch.distributed.get_world_size(), 3, dtype=torch.float32, device=device)
                # tensor_out = torch.zeros(config.VALID_BATCH_SIZE  * torch.distributed.get_world_size(), 1, dtype=torch.float32, device=device)
                # label_out = torch.zeros(config.VALID_BATCH_SIZE  * torch.distributed.get_world_size(), dtype=torch.int64, device=device)
                # label_id_out = torch.zeros(config.VALID_BATCH_SIZE  * torch.distributed.get_world_size(), dtype=torch.int64, device=device)
                            
                torch.distributed.all_gather_into_tensor(tensor_out_all, all_out)
                # torch.distributed.all_gather_into_tensor(tensor_out, output)
                # torch.distributed.all_gather_into_tensor(label_out, target)
                # torch.distributed.all_gather_into_tensor(label_id_out, label_id)
                #todo one all gather
                

                
                if torch.distributed.get_rank() == 0:
                     
                    output = tensor_out_all[:,0].detach().cpu() #torch.sigmoid(tensor_out).detach().cpu()
                    labels = tensor_out_all[:,1].detach().cpu()
                    label_id_out = tensor_out_all[:,2].detach().cpu()
                    
                    
                    # output = tensor_out.detach().cpu() #torch.sigmoid(tensor_out).detach().cpu()
                    # labels = label_out.detach().cpu()
                    # label_id_out = label_id_out.detach().cpu()
                    
                    if num_padding !=0:
                        output = output.reshape(-1)
                        output = output[torch.nonzero(output)]
                        non_zero_idx = torch.nonzero(label_id_out)
                        label_id_out = label_id_out[non_zero_idx]
                        labels = labels[non_zero_idx]
                        
                    
                    predictions.append(output.reshape(-1))
                    true_vals.append(labels.reshape(-1))
                    label_ids.append(label_id_out.reshape(-1))
                    # summary_writer.add_scalars('VALIDATION', {
                    #                                         "Valid Loss": loss.item(),
                    #                                         "Val Accuracy": metrics.accuracy_score(labels.numpy().flatten(),
                    #                                                                               np.argmax(output.numpy(), axis=1).flatten()),
                    #                                         }, epoch)
                
                if counter % 22 == 0:
                    if torch.distributed.get_rank() == 0:
                        bs, num_channels, height, width = data['image'].shape
                        for ix  in range(bs):
                            # get the i-th image from the batch
                            image_id = data['image_id'][ix]
                            image = data['image'][ix].detach().cpu().permute(1,2,0).numpy()   
                            label = "cancer" if data['target'][ix].detach().cpu().numpy() == 1 else "non-cancer"
                            prediction = "cancer" if np.round(output[ix].numpy().flatten()) == 1 else "non-cancer"
                            color = "green" if label==prediction  else "red"
                            plt.imshow(image, cmap="gray")
                            plt.title(f"Samples/ep_{epoch} - {image_id} - {ix:04d}")

                            plt.figtext(0.5, 0.005, f"Pred: {prediction},Actual: {label}", ha="center", fontsize=14, bbox={"facecolor":color, "alpha":0.5, "pad":5}) 
                            plt.show()

            if torch.distributed.get_rank() == 0:
                
                
                # print("FINaL PREDICTION SHAPE",len(predictions),len(true_vals))
                predictions = np.concatenate(predictions, axis=0)
                true_vals = np.concatenate(true_vals, axis=0)
                label_ids = np.concatenate(label_ids, axis=0)
                predictions = predictions[:flatten_size]
                true_vals = true_vals[:flatten_size]
                label_ids = label_ids[:flatten_size]
                
                print('Sum of true_vals',sum(true_vals))
                # print("AFTER PREDICTION SHAPE",len(predictions),len(true_vals))
                return loss_val_total/ len(dataloader), predictions , true_vals , label_ids
            
            else:
                
                return loss_val_total/ len(dataloader), [], [], []
