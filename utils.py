import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import timm
import os

img_height, img_width = 224, 224

def wrap_model(model):
    if hasattr(model, 'default_cfg'):
        mean, std = model.default_cfg['mean'], model.default_cfg['std']
    else:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return torch.nn.Sequential(transforms.Normalize(mean, std), model)

class EnsembleModel(torch.nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = models
        self.device = next(models[0].parameters()).device

    def forward(self, x):
        return torch.stack([m(x) for m in self.models]).mean(0)

class AdvDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir, targeted=False):
        self.targeted, self.data_dir = targeted, os.path.join(input_dir, 'images')
        df = pd.read_csv(os.path.join(input_dir, 'labels.csv'))
        self.f2l = {r['filename']: ([r['label'], r['targeted_label']] if targeted else r['label']) for _, r in df.iterrows()}
        self.filenames = list(self.f2l.keys())

    def __len__(self): return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img = Image.open(os.path.join(self.data_dir, fname)).convert('RGB').resize((224, 224))
        img = torch.from_numpy(np.array(img).astype(np.float32)/255).permute(2, 0, 1)
        return img, self.f2l[fname], fname