from torch.utils.data import IterableDataset
from torch.utils.data import Dataset, DataLoader
import torch
import json
from collections import Counter

class ICJAIIterableDataset(IterableDataset):

    def __init__(self, filename, history_len, data_ratio=1.):
        self.filename = filename
        self.hist_len = history_len
        self._init_mj2id()

    def __iter__(self):
        #Create an iterator
        file_itr = open(self.filename)

        return file_itr
    
    def line_mapper(self, line):
        #Splits the line into text and label and applies preprocessing to the text
        X, y = self.preprocess(line)

        return X, y

    def __iter__(self):
        #Create an iterator
        file_itr = open(self.filename)

        #Map each element using the line_mapper
        mapped_itr = map(self.line_mapper, file_itr)
        
        return mapped_itr
    
    
    def _init_mj2id(self):
        w_dict = {'W' + str(i+1): i for i in range(9)} #万
        b_dict = {'B' + str(i+1): i+9 for i in range(9)} #饼
        t_dict = {'T' + str(i+1): i+18 for i in range(9)} #条
        f_dict = {'F' + str(i+1): i+27 for i in range(4)} #风 东南西北
        j_dict = {'J' + str(i+1): i+31 for i in range(3)} #（剑牌）中发白
        self.mj2id = {**w_dict, **b_dict,**t_dict,**f_dict,**j_dict}
        
        
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
            if not i.startswith("H"):
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
        data = json.loads(data)
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

# dataset = CustomIterableDataset('processed_data/discard_tile_train.nosync.txt', history_len=4)
# dataloader = DataLoader(dataset,batch_size = 64)

# for X, y in dataloader:
#     print(X.shape) # 64
#     print(y.shape) # (64,)
#     break

