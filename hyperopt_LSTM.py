# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 22:07:08 2023

@author: u0138175
"""

####################PACKAGES AND FUNCTIONS#######################
import torch
from torch.utils.data import DataLoader, TensorDataset
import re
import os
import sys
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.metrics import roc_auc_score
import logging

logging.getLogger().setLevel(logging.INFO)
sys.path.append(os.getcwd())
from util.DatasetManager import DatasetManager
from util.settings import global_setting, training_setting
from carla.data.catalog.own_catalog import OwnCatalog
from carla.models.catalog.LSTM_TORCH import LSTMModel, CheckpointSaver

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(22)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)

#####################PARAMETERS###################################
dataset_names = ['production', 'sepsis_cases_1', 'sepsis_cases_2', 'sepsis_cases_4',
                 'bpic2012_accepted', 'bpic2012_rejected', 'bpic2012_cancelled']
dataset_name = 'sepsis_cases_2'
print('own dataset', dataset_name)

dataset = OwnCatalog(dataset_name)
train_ratio = global_setting['train_ratio']
dataset_manager = DatasetManager(dataset_name)

best_LSTMs = global_setting['hyper_models']
dir_path = global_setting['hyper_models'] + dataset_name
dataset_manager.ensure_path(best_LSTMs)
dataset_manager.ensure_path(dir_path)
dir_list = os.listdir(dir_path)

n_splits = global_setting['n_splits']
max_evals = global_setting['max_evals']
train_ratio = global_setting['train_ratio']

# we only use the training dataset for hyperoptimization
data_train = dataset.df_train
target_train = dataset.target_train

# 检查数据形状
print(f"Data shape: {data_train.shape}")
print(f"Target shape: {target_train.shape}")
print(f"Target unique values: {np.unique(target_train)}")

skf = StratifiedKFold(n_splits=n_splits)
skf.get_n_splits(data_train, target_train)

data_cv = []
target_cv = []
for i, (train_index, val_index) in enumerate(skf.split(data_train, target_train)):
    data_cv.append(data_train[val_index])
    target_cv.append(target_train[val_index])

# remove the redundant files
if 'desktop.ini' in dir_list:
    dir_list.remove('desktop.ini')

best_auc = 0.5
best_LSTM_name = ""
best_model = None

for LSTM_name in dir_list:
    print('the current LSTM name', LSTM_name)
    print('the current best AUC is', best_auc)

    try:
        # 解析文件名
        split = LSTM_name.split("_")
        checkpoint = torch.load(dir_path + '/' + LSTM_name, map_location=torch.device('cpu'))

        # 提取参数
        LSTM_lr = float(split[1])
        LSTM_hidden_dim1 = int(split[2])
        LSTM_hidden_dim2 = int(split[3])
        lSTM_size = int(split[4])
        LSTM_dropout = float(split[5])

        # 提取epoch
        epoch_match = re.findall(r'\d+', str(split[6])) if len(split) > 6 else re.findall(r'\d+', LSTM_name)
        LSTM_epoch = int(epoch_match[0]) if epoch_match else 0
        print('epoch', LSTM_epoch)

        # 创建模型
        model = LSTMModel(dataset.vocab_size, LSTM_dropout, [LSTM_hidden_dim1, LSTM_hidden_dim2, lSTM_size])
        model.load_state_dict(checkpoint)
        model.eval()
        model.to(device)

        # Perform stratified k-fold cross-validation
        all_preds = []
        all_targets = []
        auc_list = []

        for i, (train_index, val_index) in enumerate(skf.split(data_train, target_train)):
            val_data = data_train[val_index]
            val_target = target_train[val_index]

            # 确保数据格式正确
            # 如果 val_data 是 DataFrame，可能需要转换
            if hasattr(val_data, 'values'):
                val_data = val_data.values

            # 转换为张量并移动到设备
            val_data_tensor = torch.tensor(val_data, dtype=torch.long).to(device)

            # 预测
            with torch.no_grad():
                pred = model(val_data_tensor, mode='val').squeeze(-1)
                pred = pred.cpu().numpy()

            # 计算AUC
            auc = roc_auc_score(val_target, pred)
            auc_list.append(auc)
            print(f'Fold {i}: AUC = {auc:.4f}')

        average_AUC = np.mean(auc_list)
        print(f'Average AUC: {average_AUC:.4f}')

        if average_AUC > best_auc:
            print(f'New best AUC: {average_AUC:.4f} (previous: {best_auc:.4f})')
            best_auc = average_AUC
            best_model = model
            best_LSTM_name = LSTM_name

    except Exception as e:
        print(f"Error processing {LSTM_name}: {e}")
        continue

print(f'Best model: {best_LSTM_name} with AUC: {best_auc:.4f}')

# 保存最佳模型
if best_model is not None:
    path_data_label = best_LSTMs + '/' + dataset_name + '/'
    if not os.path.exists(os.path.join(path_data_label)):
        os.makedirs(os.path.join(path_data_label))
    best_model_path = os.path.join(path_data_label, best_LSTM_name)
    torch.save(best_model.state_dict(), best_model_path)
    print(f"Best model saved to: {best_model_path}")
else:
    print("No valid model found!")