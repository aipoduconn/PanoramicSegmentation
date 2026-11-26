import os
# Repository root (two levels above this file), independent of the working directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + os.sep
import sys
sys.path.append(ROOT_DIR) # Better way of doing: export PYTHONPATH="${PYTHONPATH}:/path/to/your/project/"

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import hydra
from omegaconf import DictConfig, OmegaConf
import transformers
from tqdm import tqdm
import time
from torch import nn
from torch.utils.data import DataLoader
from copy import deepcopy
import numpy as np
from PIL import Image
from torchvision import transforms
import pandas as pd
import torch
from accelerate import Accelerator
import time
import mlflow
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights,deeplabv3_resnet50,DeepLabV3_ResNet50_Weights
from src.model import models
from accelerate.logging import get_logger
from torchmetrics import JaccardIndex, Accuracy, F1Score
from torchmetrics.segmentation import MeanIoU
from src.data.utils import IDX2CAT
logger = get_logger(__name__,"INFO")

TRACKING_SERVER_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
CONFIG_PATH = os.path.join(ROOT_DIR, "config/")
TRAIN_DATA_PATH = "data/metadata/train_data.csv"
CONFIG_FILE = "train-config.yaml"

def get_img_trans(trans_img_height=256,trans_img_width=256):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((trans_img_height, trans_img_width)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
def get_mask_trans(trans_img_height=256,trans_img_width=256):
    return transforms.Compose([
        transforms.PILToTensor(),
        transforms.Resize((trans_img_height, trans_img_width))
        
    ])

def split_by_radiograph(data, val_size, random_state=None):
    """Split into train/validation sets by original radiograph.

    Augmented rows keep the `set_name` and `img_id` of the radiograph they were
    generated from, so each radiograph and all of its augmented copies end up in
    the same subset and no augmented copy leaks across the split.
    """
    key = data["set_name"].astype(str) + "/" + data["img_id"].astype(str)
    originals = key[data["type"] == "original"].drop_duplicates()
    _, val_keys = train_test_split(originals, test_size=val_size, random_state=random_state)
    is_val = key.isin(set(val_keys))
    return data[~is_val], data[is_val]


def get_optim(optim_name, model, lr, **optim_kwargs):
    if optim_name == 'AdamW':
        return torch.optim.AdamW(model.parameters(), lr=lr, **optim_kwargs)


def get_lr_scheduler(scheduler_name, optim, tmax, **sched_kwargs):
    if scheduler_name == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=tmax, **sched_kwargs)
    elif scheduler_name == 'cosine_schedule_with_warmup':
        return transformers.get_cosine_schedule_with_warmup(optim, num_training_steps=tmax, **sched_kwargs)
    elif scheduler_name == 'ReduceLROnPlateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optim,**sched_kwargs) #factor,patience


def get_loss(loss_name, **loss_kwargs):
    if loss_name == 'DoubleCrossEntropyLoss':
        return DoubleCrossEntropyLoss(**loss_kwargs)


def __kwargs(val):
    if val == "empty":
        return {}
    else:
        return val 

def calc_val_metrics(model,val_loader,accelerator,loss_fn,num_classes):
    model.eval()
    accuracy = Accuracy(task="multiclass", num_classes=num_classes, ignore_index=0)
    f1 = F1Score(task="multiclass", num_classes=num_classes, ignore_index=0)
    
    iou = MeanIoU(num_classes=num_classes,per_class=True)
    accuracy = Accuracy(task="multiclass", num_classes=num_classes,average=None)
    f1 = F1Score(task="multiclass", num_classes=num_classes,average=None)
    
    # F1Score
    loss=0
    num_batches = 0
    for batch in val_loader:
        imgs = batch['img']
        masks = batch['mask']
        
        with torch.no_grad():
            pred = model(imgs)
        
        loss+= loss_fn(pred,masks).item()  
         
        pred=pred['out']
        pred,masks= accelerator.gather_for_metrics((pred,masks))
        
        # Compute metrics incrementally
        iou(pred.detach().cpu().argmax(1),masks.detach().cpu())
        accuracy(pred.detach().cpu().argmax(1),masks.detach().cpu())
        f1(pred.detach().cpu().argmax(1),masks.detach().cpu())
        num_batches += 1
    
    metrics_df = pd.DataFrame({
            "iou":iou.compute(),
            "accuracy":accuracy.compute(),
            "f1": f1.compute()
        },index=[IDX2CAT[i] for i in range(num_classes)])
        
    val_metrics = {
        "iou_bg":metrics_df["iou"].mean(),
        "accuracy_bg":metrics_df["accuracy"].mean(),
        "f1_bg":metrics_df["f1"].mean(),
        "iou_no_bg":metrics_df.drop(IDX2CAT[0])["iou"].mean(),
        "accuracy_no_bg":metrics_df.drop(IDX2CAT[0])["accuracy"].mean(),
        "f1_no_bg":metrics_df["f1"].drop(IDX2CAT[0]).mean(),
        'loss': loss / num_batches
    }
    
    return val_metrics
        

class DoubleCrossEntropyLoss:
    def __init__(self,num_classes,bg_weight=0.01,weight_aux=0.5):
        self.device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.weight = torch.Tensor([0.1]*num_classes)
        self.weight[0]=bg_weight
        self.weight = self.weight.to(device)
        self.weight_aux=weight_aux
        self.loss = torch.nn.CrossEntropyLoss(weight=self.weight)
        
    def __call__(self,inputs, target):
        losses = {}
        for name, x in inputs.items():
            losses[name] = self.loss(x, target)

        if len(losses) == 1:
            return losses["out"]

        return losses["out"] + self.weight_aux * losses["aux"] 
               


class PanImgDataset(torch.utils.data.Dataset):
    def __init__(self,data, root_dir,trans_img_height=256,trans_img_width=256):
        self.data = data
        self.root_dir=root_dir
        self.trans_img_height=trans_img_height
        self.trans_img_width=trans_img_width

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        img = Image.open(os.path.join(self.root_dir,row['img_rel_path'])).convert('RGB')
        img = get_img_trans(self.trans_img_height,self.trans_img_width)(img)
        
        mask = Image.open(os.path.join(self.root_dir,row['mask_rel_path']))
        mask = get_mask_trans(self.trans_img_height,self.trans_img_width)(mask).long()[0]
        
        img_mask ={
            'img':img,
            'mask':mask
        }
        return img_mask

    def __len__(self):
        return len(self.data)

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name=CONFIG_FILE.split(".")[0])
def train(cfg):
    accelerator = Accelerator(gradient_accumulation_steps=cfg.params.grad_acc_step)
    #Initializing tracker
    if accelerator.is_main_process:
        mlflow.set_tracking_uri(TRACKING_SERVER_URI)
        mlflow.set_experiment(cfg.params.exp_name)
        mlflow.log_params(cfg.params)
        mlflow.log_param("img_size",(cfg.params.img_size.height,cfg.params.img_size.width))
    
    #Save the new params passed through cli
    if cfg.s:
        cfg.s=False
        OmegaConf.save(config=cfg, f=os.path.join(CONFIG_PATH, CONFIG_FILE))
    
    #Load data
    logger.info("Loading Data...")
    train_metadata_path = os.path.join(ROOT_DIR,TRAIN_DATA_PATH)
    data = pd.read_csv(train_metadata_path)
    if not cfg.params.data.get("use_augmented", True):
        data = data.query("type=='original'")
    if cfg.params.data.sets!="all":
        sets = [val.strip() for val in cfg.params.data.sets.split(",")]
        data = data.query("set_name in @sets")
        logger.info("Sets being trained:"+",".join(data['set_name'].drop_duplicates().tolist()))
    train_data,val_data = split_by_radiograph(data,cfg.params.val_size)
    logger.info(f"Train: {len(train_data)} images, validation: {len(val_data)} images")
    
    
    #Preparing data
    train_dataset = PanImgDataset(train_data,ROOT_DIR,cfg.params.img_size.height,cfg.params.img_size.width)
    val_dataset = PanImgDataset(val_data,ROOT_DIR,cfg.params.img_size.height,cfg.params.img_size.width)
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.params.batch_size.train, shuffle=True, num_workers=cfg.params.train_loader_worker_num)
    val_dataloader = DataLoader(
        val_dataset, batch_size=cfg.params.batch_size.val, shuffle=False, num_workers=cfg.params.val_loader_worker_num)
    
    if accelerator.is_main_process:
        mlflow.log_param('num_train_example',len(train_dataset))
        mlflow.log_param('num_val_example',len(val_dataset))
    logger.info("Data Loaded")
    
    logger.info("Loading model...")
    #Load model
    model = models.PanSegModel(cfg.params.model.name,cfg.params.model.num_seg_class)
    logger.info("Model loaded.")
    
    
    #Optimizer
    optimizer = get_optim(cfg.params.optim.name, model,
                    cfg.params.learning_rate, **__kwargs(cfg.params.optim.kwargs))
    
    #LR Scheduler
    tmax = cfg.params.epochs*(len(train_dataset)//cfg.params.batch_size.train + 1)
    lr_scheduler = get_lr_scheduler(
        cfg.params.lr_scheduler.name, optimizer, tmax, **__kwargs(cfg.params.lr_scheduler.kwargs))
    
    #Wrapping objects in accelerator
    model,optimizer,lr_scheduler,train_dataloader,val_dataloader = accelerator.prepare(model,optimizer,lr_scheduler,train_dataloader,val_dataloader)
    
    #Loss
    loss_kwargs = __kwargs(cfg.params.loss.kwargs)
    criterion = get_loss(cfg.params.loss.name,num_classes=cfg.params.model.num_seg_class, **
                            __kwargs(loss_kwargs))
    
    #Training loop
    logger.info("Training starts...")
    for epoch in range(cfg.params.epochs):
        train_loss_sum = 0
        num_batch = 0
        model.train()
        pbar = tqdm(train_dataloader, disable=not accelerator.is_local_main_process)
        train_iou = MeanIoU(num_classes=cfg.params.model.num_seg_class)
        for batch in pbar:
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                #Getting batch data for passing to model
                imgs = batch['img']
                masks = batch['mask']
                
                #Passing data to model for training
                output = model(imgs)
                
                #Calculating loss
                train_loss = criterion(output, masks)
                pred_masks = output['out'].detach().cpu().argmax(axis=1)
                true_masks = masks.detach().cpu()
                
                
                iou = train_iou(pred_masks,true_masks)
                
                #Getting gradients and updating weights
                accelerator.backward(train_loss)
                optimizer.step()
                if cfg.params.lr_scheduler.name!='ReduceLROnPlateau':
                    lr_scheduler.step()
                
                train_loss_sum += train_loss.item()
                num_batch += 1
                train_iou_score = train_iou.compute()
                pbar.set_description("Epoch: %s, Train loss: %f, Cumulative train IOU: %f,train IOU: %f"  % (epoch, train_loss_sum/num_batch,train_iou_score,iou))
        
        if accelerator.is_main_process:
            val_metrics = calc_val_metrics(model,val_dataloader,accelerator,criterion,cfg.params.model.num_seg_class)
            logger.info(val_metrics)
            mlflow.log_metric('train_loss',train_loss_sum/num_batch,epoch)
            mlflow.log_metric('train_iou',train_iou_score,epoch)
            mlflow.log_metrics({'val_'+key:val for key,val in val_metrics.items()},epoch)
        
        if epoch%cfg.params.model.save_epoch_interval==0:
            accelerator.wait_for_everyone()
            model_path = os.path.join(ROOT_DIR,'models',f'deep_lab_state_dict_epoch_{epoch}')
            unwrapped_model = accelerator.unwrap_model(model)
            if accelerator.is_main_process:
                if cfg.params.model.log:
                    mlflow.pytorch.log_state_dict(deepcopy(unwrapped_model.state_dict()),f'models/model_epoch_{epoch}')
                else:
                    torch.save(deepcopy(unwrapped_model.state_dict()), model_path)
     
        if cfg.params.lr_scheduler.name =='ReduceLROnPlateau':
            lr_scheduler.step(val_metrics['iou'])
                
    logger.info("Training finished!!!")        
    
    
if __name__ == "__main__":
    train()
    
    
    
    
