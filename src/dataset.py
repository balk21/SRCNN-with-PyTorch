import h5py
import numpy as np
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            #(N, 33, 33)
            lr = f['lr'][idx] / 255.
            hr = f['hr'][idx] / 255.
            
            # HR label: 33x33
            # Model Output: 21x21
            # => crop the remaining 6 pixels from each edge to get 21x21
            crop = 6
            hr_cropped = hr[crop:-crop, crop:-crop]
            
            return np.expand_dims(lr, 0), np.expand_dims(hr_cropped, 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])

class EvalDataset(Dataset):
    def __init__(self, h5_file):
        super(EvalDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][str(idx)][:, :] / 255., 0), \
                   np.expand_dims(f['hr'][str(idx)][:, :] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])