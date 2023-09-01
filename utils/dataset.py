from torch.utils.data import Dataset
import pickle
import numpy as np

class BaseDataset(Dataset):
    def __init__(self, dataset='tum_emo', data_type='train', index=1):
        self.dataset = dataset
        self.data_type = data_type
        with open(f"data/{dataset}/{data_type}_{index}.pkl",'rb') as f:
            self.content = pickle.load(f)
        self.texts = np.array(self.content['text']).astype(np.float32)
        self.texts = np.squeeze(self.texts)
        self.images = np.array(self.content['image']).astype(np.float32)
        self.ids = np.array(self.content['id'])
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
        return sample_item