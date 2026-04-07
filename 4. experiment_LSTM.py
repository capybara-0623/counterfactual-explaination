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

best_LSTMs = global_setting['best_LSTMs']
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

# 检查数据
print(f"Data shape: {data_train.shape}")
print(f"Target shape: {target_train.shape}")
print(f"Number of models found: {len(dir_list)}")

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

# 初始化变量
best_auc = 0.5
best_LSTM_name = ""
best_model = None  # 重要：初始化为 None
valid_models_count = 0

for LSTM_name in dir_list:
    print('\n' + '=' * 50)
    print('Processing:', LSTM_name)
    print('Current best AUC:', best_auc)

    try:
        # 解析文件名
        split = LSTM_name.split("_")
        if len(split) < 6:
            print(f"Skipping {LSTM_name}: filename format incorrect")
            continue

        checkpoint = torch.load(dir_path + '/' + LSTM_name, map_location=torch.device('cpu'))

        LSTM_lr = float(split[1])
        LSTM_hidden_dim1 = int(split[2])
        LSTM_hidden_dim2 = int(split[3])
        lSTM_size = int(split[4])
        LSTM_dropout = float(split[5])

        # 提取epoch
        epoch_match = re.findall(r'\d+', str(split[6])) if len(split) > 6 else re.findall(r'\d+', LSTM_name)
        LSTM_epoch = int(epoch_match[0]) if epoch_match else 0
        print(f'Epoch: {LSTM_epoch}')

        # 创建模型
        model = LSTMModel(dataset.vocab_size, LSTM_dropout, [LSTM_hidden_dim1, LSTM_hidden_dim2, lSTM_size])
        model.load_state_dict(checkpoint)
        model.eval()
        model.to(device)

        # Perform stratified k-fold cross-validation
        auc_list = []

        for i, (train_index, val_index) in enumerate(skf.split(data_train, target_train)):
            try:
                val_data = data_train[val_index]
                val_target = target_train[val_index]

                # 确保数据是张量格式
                if not isinstance(val_data, torch.Tensor):
                    if hasattr(val_data, 'values'):
                        val_data = val_data.values
                    val_data = torch.tensor(val_data, dtype=torch.long).to(device)
                else:
                    val_data = val_data.to(device)

                with torch.no_grad():
                    pred = model(val_data, mode='val')
                    pred = pred.squeeze(-1).cpu().detach().numpy()

                auc = roc_auc_score(val_target, pred)
                auc_list.append(auc)
                print(f'Fold {i}: AUC = {auc:.4f}')

            except Exception as e:
                print(f"Error in fold {i}: {e}")
                continue

        if auc_list:
            average_AUC = np.mean(auc_list)
            print(f'Average AUC: {average_AUC:.4f}')
            valid_models_count += 1

            if average_AUC > best_auc:
                print(f'✓ New best model! AUC: {average_AUC:.4f} (was: {best_auc:.4f})')
                best_auc = average_AUC
                best_model = model
                best_LSTM_name = LSTM_name
        else:
            print(f"No valid AUC scores for {LSTM_name}")

    except Exception as e:
        print(f"Error processing {LSTM_name}: {e}")
        continue

# 保存最佳模型
print('\n' + '=' * 50)
print(f"Total valid models processed: {valid_models_count}")

if best_model is not None:
    print(f'Best model: {best_LSTM_name}')
    print(f'Best AUC: {best_auc:.4f}')

    path_data_label = best_LSTMs + '/' + dataset_name + '/'
    if not os.path.exists(os.path.join(path_data_label)):
        os.makedirs(os.path.join(path_data_label))

    best_model_path = os.path.join(path_data_label, best_LSTM_name)
    torch.save(best_model.state_dict(), best_model_path)
    print(f'Model saved to: {best_model_path}')
else:
    print("ERROR: No best model found!")
    print("Possible reasons:")
    print("1. No model files in the directory")
    print("2. All models failed during evaluation")
    print("3. AUC scores were all below 0.5")
    print(f"Directory checked: {dir_path}")
    print(f"Files found: {dir_list}")