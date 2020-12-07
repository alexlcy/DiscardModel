import os
import copy
import ast
import time
import json
import random
import glob
import numpy as np
from functools import partial
from collections import Counter
from tqdm import tqdm_notebook as tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
#from torchsummary import summary


class ICJAIDataset(Dataset):
    def __init__(self, data_path, history_len, data_ratio=1.):
        with open(data_path, 'r') as f:
            self.all_data_str = f.readlines()
        if data_ratio != 1.:
            random.shuffle(self.all_data_str)
            data_size = int(len(self.all_data_str)*data_ratio)
            self.all_data_str = self.all_data_str[:data_size]
        self.hist_len = history_len
        self._init_mj2id()
        self._init_cls_ratios()

    def __getitem__(self, idx):
        data = json.loads(self.all_data_str[idx])
        x, y = self.preprocess(data)
        return x, y

    def __len__(self):
        return len(self.all_data_str)

    def _init_mj2id(self):
        w_dict = {'W' + str(i+1): i for i in range(9)} #万
        b_dict = {'B' + str(i+1): i+9 for i in range(9)} #饼
        t_dict = {'T' + str(i+1): i+18 for i in range(9)} #条
        f_dict = {'F' + str(i+1): i+27 for i in range(4)} #风 东南西北
        j_dict = {'J' + str(i+1): i+31 for i in range(3)} #（剑牌）中发白
        self.mj2id = {**w_dict, **b_dict,**t_dict,**f_dict,**j_dict}

    def _init_cls_ratios(self):
        cls_ratios = {}
        for i, line in enumerate(self.all_data_str):
            turn_data = json.loads(line)
            label = self.mj2id[turn_data[0]['label']]
            cls_ratios[label] = cls_ratios.get(label, 0) + 1
        # cls_ratios = {k: v/len(self.all_data_str) for k, v in cls_ratios.items()}
        self.cls_ratios = cls_ratios

    def tiles2mat(self, mj_list):
        '''
        Args:
        - mj_list (list): list of mahjongs (e.g. ['B1', 'B3', 'B9', 'T1', 'T4'])

        Returns:
        - repr (torch.tensor, float32): shape (4, 34) 
        '''
        repr = torch.zeros(4, 34, dtype=torch.float32)
        count = Counter(mj_list)
        for i in count:
            index = self.mj2id[i]
            nums = count[i]
            for j in range(nums):
                repr[j, index] = 1
        return repr

    def tile2vec(self, tile):
        repr = torch.zeros(1, 34, dtype=torch.float32)
        if tile != '':
            index = self.mj2id[tile]
            repr[:, index] = 1
        return repr

    def preprocess(self, data):
        '''
        Args:
        - data (list): include 5 data from latest to oldest (current state, 4 history states)

        Returns:
        - x (torch.tensor, float32): shape [2+(self.hist_len+1)*17, 34, 1]  (C, H, W)
        - y (torch.tensor, int64): shape [1]
        '''
        x = torch.zeros(2+(self.hist_len+1)*17, 34, dtype=torch.float32)
        y = torch.tensor(self.mj2id[data[0]['label']], dtype=torch.int64)
        for hist_i, hist_data in enumerate(data):
            player = str(hist_data['turn_player'])

            # Own wind (1, 34) & round wind (1, 34)
            if hist_i == 0:
                own_wind = self.tile2vec(hist_data['own_wind'])
                round_wind = self.tile2vec(hist_data['round_wind'])
                x[0, :] = own_wind
                x[1, :] = round_wind

            # Own hand (4, 34)
            own_hand = self.tiles2mat(hist_data['hand'])

            # Own last discard (1, 34)
            own_last_discard = self.tile2vec(hist_data['last_discard'][player])

            # Others' last discard (4, 34)
            others_last_discard = []
            for player_id, last_discard in hist_data['last_discard'].items():
                if player_id != player and last_discard != '':
                    others_last_discard.append(last_discard)
            others_last_discard = self.tiles2mat(others_last_discard)
            
            # Own discard (4, 34)
            own_discard = self.tiles2mat(hist_data['discard'][player])

            # Others' discard (4, 34)
            others_discard = []
            for player_id, discard in hist_data['discard'].items():
                if player_id != player:
                    others_discard += discard
            others_discard = self.tiles2mat(others_discard)

            hist_x = torch.cat([own_hand, own_last_discard, others_last_discard, own_discard, others_discard], dim=0)
            x[2+hist_i*hist_x.shape[0]:2+(hist_i+1)*hist_x.shape[0], :] = hist_x
        return x.unsqueeze(-1), y