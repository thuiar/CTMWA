import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from transformers import ViTModel, BertModel

import numpy as np
class BaseDataset(Dataset):
    def __init__(self, dataset='tum_emo', data_type='train', index=1):
        self.dataset = dataset
        self.index = index
        self.data_type = data_type
        if dataset == 'tum_emo' or 'twitter' in dataset:
            with open(f"data/{dataset}/{data_type}_1.pkl",'rb') as f:
                self.content = pickle.load(f)
        else:
            with open(f"data/{dataset}/{data_type}_{index}.pkl",'rb') as f:
                self.content = pickle.load(f)

        self.texts = np.array(self.content['text']).astype(np.float32)
        self.texts = np.squeeze(self.texts)
        self.images = np.array(self.content['image']).astype(np.float32)
        self.ids = np.array(self.content['id'])

        if self.data_type == 'query':
            self.t_labels = np.array(self.content['t_label'])
            self.v_labels = np.array(self.content['v_label'])
            self.labels = np.array(self.content['m_label'])
        else:
            self.labels = np.array(self.content['label'])

    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample_item = {
            'text': self.texts[idx],
            'image': self.images[idx],
            'label': self.labels[idx],
            'id': self.ids[idx]
        }
        if self.data_type == 'query':
            sample_item['t_label']=self.t_labels[idx]
            sample_item['v_label']=self.v_labels[idx]
        return sample_item