# Pytorch template

# source notebooks: 
# [1] https://www.kaggle.com/yasufuminakama/ranzcr-resnet200d-3-stage-training-sub


COMP = "ranzcr-clip-catheter-line-classification"
FOLDS = "ranzcr-exp12-step3-fold0"
WEIGHTS = "resnet200d-pretrained-weight"

## Inference notebook ----------------------------------------------


# ====================================================
# Directory settings
# ====================================================
import os

OUTPUT_DIR = './'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

TRAIN_PATH = f'../input/{COMP}/train'



# ====================================================
# CFG
# ====================================================
class CFG:
    debug=False
    #device='TPU' # ['TPU', 'GPU']
    #nprocs=1 # [1, 8]
    #print_freq=100
    num_workers=4
    model_name='resnet200d_320'
    size=512
    #scheduler='CosineAnnealingLR' # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']
    #epochs=4
    #T_max=4 # CosineAnnealingLR
    #lr=5e-4 # 1e-4
    #min_lr=1e-6
    batch_size=128 # 16
    #weight_decay=1e-6
    #gradient_accumulation_steps=1
    #max_grad_norm=1000
    seed=416
    target_size=11
    target_cols=['ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal',
                 'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal', 
                 'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal',
                 'Swan Ganz Catheter Present']
    n_fold=5
    trn_fold=[0] # [0, 1, 2, 3, 4]
    #train=True
    




# ====================================================
# Library
# ====================================================
import sys
import os
import ast
import copy
import math
import time
import random
import shutil
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter

import scipy as sp
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.utils import check_random_state
#from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from tqdm.auto import tqdm
from functools import partial

from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import torchvision.models as models
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau

from torch.cuda.amp import autocast, GradScaler

import warnings 
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# ====================================================
# Utils
# ====================================================
def get_score(y_true, y_pred):
    scores = []
    for i in range(y_true.shape[1]):
        score = roc_auc_score(y_true[:,i], y_pred[:,i])
        scores.append(score)
    avg_score = np.mean(scores)
    return avg_score, scores


def get_result(result_df):
    preds = result_df[[f'pred_{c}' for c in CFG.target_cols]].values
    labels = result_df[CFG.target_cols].values
    score, scores = get_score(labels, preds)
    LOGGER.info(f'Score: {score:<.4f}  Scores: {np.round(scores, decimals=4)}')


@contextmanager
def timer(name):
    t0 = time.time()
    LOGGER.info(f'[{name}] start')
    yield
    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s.')


def init_logger(log_file=OUTPUT_DIR+'train.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

LOGGER = init_logger()


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_torch(seed=CFG.seed)




# ====================================================
# Data Loading
# ====================================================
#train = pd.read_csv(f'../input/{COMP}/train.csv')
#folds = pd.read_csv(f'../input/{FOLDS}/folds.csv')
oof_df = pd.read_csv(f'../input/{FOLDS}/oof_df.csv')
for fold in CFG.trn_fold:
    fold_oof_df = oof_df[oof_df['fold']==fold].reset_index(drop=True)
    LOGGER.info(f"========== fold: {fold} result ==========")
    get_result(fold_oof_df)

if CFG.debug:
    test = pd.read_csv(f'../input/{COMP}/sample_submission.csv', nrows=10)
else:
    test = pd.read_csv(f'../input/{COMP}/sample_submission.csv')



# ====================================================
# Dataset
# ====================================================
class TestDataset(Dataset):
    def __init__(self, 
                 df
                 #df_annotations, 
                 #annot_size=50, 
                 ):
        self.df = df
        #self.df_annotations = df_annotations
        #self.annot_size = annot_size
        self.file_names = df['StudyInstanceUID'].values
        #self.labels = df[CFG.target_cols].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = f'{TRAIN_PATH}/{file_name}.jpg'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #query_string = f"StudyInstanceUID == '{file_name}'"
        #df = self.df_annotations.query(query_string)
        # for i, row in df.iterrows():
        #     label = row["label"]
        #     data = np.array(ast.literal_eval(row["data"]))
        #     for d in data:
        #         image[d[1]-self.annot_size//2:d[1]+self.annot_size//2,
        #               d[0]-self.annot_size//2:d[0]+self.annot_size//2,
        #               :] = COLOR_MAP[label]
        # label = torch.tensor(self.labels[idx]).float()
        return image#, label




# ====================================================
# MODEL
# ====================================================
class CustomResNet200D(nn.Module):
    def __init__(self, model_name='resnet200d_320', pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False)
        if pretrained:
            pretrained_path = f'../input/{WEIGHTS}/resnet200d_ra2-bdba9bf9.pth'
            self.model.load_state_dict(torch.load(pretrained_path))
            print(f'load {model_name} pretrained model')
        n_features = self.model.fc.in_features
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_features, CFG.target_size)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.fc(pooled_features)
        #return features, pooled_features, output
        return output



# ====================================================
# Helper functions
# ====================================================
def inference(models, test_loader, device):
    # batch_time = AverageMeter()
    # data_time = AverageMeter()
    # losses = AverageMeter()
    # scores = AverageMeter()
    # # switch to evaluation mode
    # model.eval()
    # trues = []
    # preds = []
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))
    probs = []
    # start = end = time.time()
    #for step, (images, labels) in enumerate(valid_loader):
    for i, (images) in tk0:
        # measure data loading time
        #data_time.update(time.time() - end)
        images = images.to(device)
        #labels = labels.to(device)
        #batch_size = labels.size(0)
        avg_pres = []
        for model in models:
            with torch.no_grad():
                #_, _, y_preds = model(images)
                y_preds1 = model(images)
                y_preds2 = model(images.flip(-1))
            y_preds = (y_preds1.sigmoid().to('cpu').numpy() + y_preds2.sigmoid().to('cpu').numpy()) / 2
            avg_preds.append(y_preds)
        avg_preds = np.mean(avg_preds, axis=0)
        probs.append(avg_preds)
        # loss = criterion(y_preds, labels)
        # losses.update(loss.item(), batch_size)
        # # record accuracy
        # trues.append(labels.to('cpu').numpy())
        # preds.append(y_preds.sigmoid().to('cpu').numpy())
        # if CFG.gradient_accumulation_steps > 1:
        #     loss = loss / CFG.gradient_accumulation_steps
        # # measure elapsed time
        # batch_time.update(time.time() - end)
        # end = time.time()
        # if CFG.device == 'GPU':
        #     if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
        #         print('EVAL: [{0}/{1}] '
        #               'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
        #               'Elapsed {remain:s} '
        #               'Loss: {loss.val:.4f}({loss.avg:.4f}) '
        #               .format(
        #                step, len(valid_loader), batch_time=batch_time,
        #                data_time=data_time, loss=losses,
        #                remain=timeSince(start, float(step+1)/len(valid_loader)),
        #                ))
        # elif CFG.device == 'TPU':
        #     if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
        #         xm.master_print('EVAL: [{0}/{1}] '
        #                         'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
        #                         'Elapsed {remain:s} '
        #                         'Loss: {loss.val:.4f}({loss.avg:.4f}) '
        #                         .format(
        #                         step, len(valid_loader), batch_time=batch_time,
        #                         data_time=data_time, loss=losses,
        #                         remain=timeSince(start, float(step+1)/len(valid_loader)),
        #                         ))
    #trues = np.concatenate(trues)
    #predictions = np.concatenate(preds)
    probs = np.concatenate(probs)
    #return losses.avg, predictions, trues
    return probs






# # ====================================================
# # Train loop
# # ====================================================
# def train_loop(folds, fold):

#     if CFG.device == 'GPU':
#         LOGGER.info(f"========== fold: {fold} training ==========")
#     elif CFG.device == 'TPU':
#         if CFG.nprocs == 1:
#             LOGGER.info(f"========== fold: {fold} training ==========")
#         elif CFG.nprocs == 8:
#             xm.master_print(f"========== fold: {fold} training ==========")

#     # ====================================================
#     # loader
#     # ====================================================
#     trn_idx = folds[folds['fold'] != fold].index
#     val_idx = folds[folds['fold'] == fold].index

#     train_folds = folds.loc[trn_idx].reset_index(drop=True)
#     valid_folds = folds.loc[val_idx].reset_index(drop=True)
    
#     train_folds = train_folds[train_folds['StudyInstanceUID'].isin(train_annotations['StudyInstanceUID'].unique())].reset_index(drop=True)
#     valid_folds = valid_folds[valid_folds['StudyInstanceUID'].isin(train_annotations['StudyInstanceUID'].unique())].reset_index(drop=True)
    
#     valid_labels = valid_folds[CFG.target_cols].values

#     train_dataset = TrainDataset(train_folds, train_annotations,
#                                  transform=get_transforms(data='train'))
#     valid_dataset = TrainDataset(valid_folds, train_annotations,
#                                  transform=get_transforms(data='valid'))

#     if CFG.device == 'GPU':
#         train_loader = DataLoader(train_dataset, 
#                                   batch_size=CFG.batch_size, 
#                                   shuffle=True, 
#                                   num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
#         valid_loader = DataLoader(valid_dataset, 
#                                   batch_size=CFG.batch_size * 2, 
#                                   shuffle=False, 
#                                   num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
#     elif CFG.device == 'TPU':
#         train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
#                                                                         num_replicas=xm.xrt_world_size(),
#                                                                         rank=xm.get_ordinal(),
#                                                                         shuffle=True)
#         train_loader = torch.utils.data.DataLoader(train_dataset,
#                                                    batch_size=CFG.batch_size,
#                                                    sampler=train_sampler,
#                                                    drop_last=True,
#                                                    num_workers=CFG.num_workers)
#         valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset,
#                                                                         num_replicas=xm.xrt_world_size(),
#                                                                         rank=xm.get_ordinal(),
#                                                                         shuffle=False)
#         valid_loader = torch.utils.data.DataLoader(valid_dataset,
#                                                    batch_size=CFG.batch_size * 2,
#                                                    sampler=valid_sampler,
#                                                    drop_last=False,
#                                                    num_workers=CFG.num_workers)

#     # ====================================================
#     # scheduler 
#     # ====================================================
#     def get_scheduler(optimizer):
#         if CFG.scheduler=='ReduceLROnPlateau':
#             scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=CFG.factor, patience=CFG.patience, verbose=True, eps=CFG.eps)
#         elif CFG.scheduler=='CosineAnnealingLR':
#             scheduler = CosineAnnealingLR(optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr, last_epoch=-1)
#         elif CFG.scheduler=='CosineAnnealingWarmRestarts':
#             scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0, T_mult=1, eta_min=CFG.min_lr, last_epoch=-1)
#         return scheduler

#     # ====================================================
#     # model & optimizer
#     # ====================================================
#     if CFG.device == 'TPU':
#         device = xm.xla_device()
#     elif CFG.device == 'GPU':
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     model = CustomResNet200D(CFG.model_name, pretrained=True)
#     model.to(device)

#     optimizer = Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay, amsgrad=False)
#     scheduler = get_scheduler(optimizer)

#     # ====================================================
#     # loop
#     # ====================================================
#     criterion = nn.BCEWithLogitsLoss()

#     best_score = 0.
#     best_loss = np.inf
    
#     for epoch in range(CFG.epochs):
        
#         start_time = time.time()
        
#         # train
#         if CFG.device == 'TPU':
#             if CFG.nprocs == 1:
#                 avg_loss = train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device)
#             elif CFG.nprocs == 8:
#                 para_train_loader = pl.ParallelLoader(train_loader, [device])
#                 avg_loss = train_fn(para_train_loader.per_device_loader(device), model, criterion, optimizer, epoch, scheduler, device)
#         elif CFG.device == 'GPU':
#             avg_loss = train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device)
                
#         # eval
#         if CFG.device == 'TPU':
#             if CFG.nprocs == 1:
#                 avg_val_loss, preds, _ = valid_fn(valid_loader, model, criterion, device)
#             elif CFG.nprocs == 8:
#                 para_valid_loader = pl.ParallelLoader(valid_loader, [device])
#                 avg_val_loss, preds, valid_labels = valid_fn(para_valid_loader.per_device_loader(device), model, criterion, device)
#                 preds = idist.all_gather(torch.tensor(preds)).to('cpu').numpy()
#                 valid_labels = idist.all_gather(torch.tensor(valid_labels)).to('cpu').numpy()
#         elif CFG.device == 'GPU':
#             avg_val_loss, preds, _ = valid_fn(valid_loader, model, criterion, device)
        
#         if isinstance(scheduler, ReduceLROnPlateau):
#             scheduler.step(avg_val_loss)
#         elif isinstance(scheduler, CosineAnnealingLR):
#             scheduler.step()
#         elif isinstance(scheduler, CosineAnnealingWarmRestarts):
#             scheduler.step()

#         # scoring
#         score, scores = get_score(valid_labels, preds)

#         elapsed = time.time() - start_time

#         if CFG.device == 'GPU':
#             LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
#             LOGGER.info(f'Epoch {epoch+1} - Score: {score:.4f}  Scores: {np.round(scores, decimals=4)}')
#         elif CFG.device == 'TPU':
#             if CFG.nprocs == 1:
#                 LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
#                 LOGGER.info(f'Epoch {epoch+1} - Score: {score:.4f}  Scores: {np.round(scores, decimals=4)}')
#             elif CFG.nprocs == 8:
#                 xm.master_print(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
#                 xm.master_print(f'Epoch {epoch+1} - Score: {score:.4f}  Scores: {np.round(scores, decimals=4)}')
        
#         if score > best_score:
#             best_score = score
#             if CFG.device == 'GPU':
#                 LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
#                 torch.save({'model': model.state_dict(), 
#                             'preds': preds},
#                            OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best_score.pth')
#             elif CFG.device == 'TPU':
#                 if CFG.nprocs == 1:
#                     LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
#                 elif CFG.nprocs == 8:
#                     xm.master_print(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
#                 xm.save({'model': model, 
#                          'preds': preds}, 
#                         OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best_score.pth')
        
#         if avg_val_loss < best_loss:
#             best_loss = avg_val_loss
#             if CFG.device == 'GPU':
#                 LOGGER.info(f'Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model')
#                 torch.save({'model': model.state_dict(), 
#                             'preds': preds},
#                            OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best_loss.pth')
#             elif CFG.device == 'TPU':
#                 if CFG.nprocs == 1:
#                     LOGGER.info(f'Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model')
#                 elif CFG.nprocs == 8:
#                     xm.master_print(f'Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model')
#                 xm.save({'model': model, 
#                          'preds': preds}, 
#                         OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best_loss.pth')
    
#     if CFG.nprocs != 8:
#         check_point = torch.load(OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best_score.pth')
#         for c in [f'pred_{c}' for c in CFG.target_cols]:
#             valid_folds[c] = np.nan
#         valid_folds[[f'pred_{c}' for c in CFG.target_cols]] = check_point['preds']

#     return valid_folds








# # ====================================================
# # main
# # ====================================================
# def main():

#     """
#     Prepare: 1.train  2.folds
#     """

#     def get_result(result_df):
#         preds = result_df[[f'pred_{c}' for c in CFG.target_cols]].values
#         labels = result_df[CFG.target_cols].values
#         score, scores = get_score(labels, preds)
#         LOGGER.info(f'Score: {score:<.4f}  Scores: {np.round(scores, decimals=4)}')
    
#     if CFG.train:
#         # train 
#         oof_df = pd.DataFrame()
#         for fold in range(CFG.n_fold):
#             if fold in CFG.trn_fold:
#                 _oof_df = train_loop(folds, fold)
#                 oof_df = pd.concat([oof_df, _oof_df])
#                 if CFG.nprocs != 8:
#                     LOGGER.info(f"========== fold: {fold} result ==========")
#                     get_result(_oof_df)
                    
#         if CFG.nprocs != 8:
#             # CV result
#             LOGGER.info(f"========== CV ==========")
#             get_result(oof_df)
#             # save result
#             oof_df.to_csv(OUTPUT_DIR+'oof_df.csv', index=False)






# if __name__ == '__main__':
#     if CFG.device == 'TPU':
#         def _mp_fn(rank, flags):
#             torch.set_default_tensor_type('torch.FloatTensor')
#             a = main()
#         FLAGS = {}
#         xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=CFG.nprocs, start_method='fork')
#     elif CFG.device == 'GPU':
#         main()




# # save as cpu if on TPU
# if CFG.device == 'TPU':
#     for fold in range(CFG.n_fold):
#         if fold in CFG.trn_fold:
#             # best score
#             state = torch.load(OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best_score.pth')
#             torch.save({'model': state['model'].to('cpu').state_dict(), 
#                         'preds': state['preds']}, 
#                         OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best_score_cpu.pth')
#             # best loss
#             state = torch.load(OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best_loss.pth')
#             torch.save({'model': state['model'].to('cpu').state_dict(), 
#                         'preds': state['preds']}, 
#                         OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best_loss_cpu.pth')


# ====================================================
# inference
# ====================================================
model = CustomResNet200D(CFG.model_name, pretrained=False)
model_path = f'../input/{FOLDS}/resnet200d_320_fold0_best_loss.pth'
model.load_state_dict(torch.load(model_path)['model'])
model.eval()
models = [model.to(device)]

test_dataset = TestDataset(test)
test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False, 
                         num_workers=CFG.num_workers, pin_memory=True)
predictions = inference(models, test_loader, device)


# submission
test[CFG.target_cols] = predictions
test[['StudyInstanceUID'] + CFG.target_cols].to_csv(OUTPUT_DIR+'submission.csv', index=False)
