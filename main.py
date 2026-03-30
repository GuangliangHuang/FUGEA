import os
import torch
import tqdm
from PIL import Image
import numpy as np
from .utils import AdvDataset
from .fugea import FUGEA

INPUT_DIR = "./data"
OUTPUT_DIR = "./results"
SURROGATE_MODELS = ["resnet18", "inception_v3"]
BATCH_SIZE = 1
GPU_ID = "0"

def save_images(output_dir, adversaries, filenames):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    adversaries = (adversaries.detach().permute((0, 2, 3, 1)).cpu().numpy() * 255).astype(np.uint8)
    for i, fname in enumerate(filenames):
        Image.fromarray(adversaries[i]).save(os.path.join(output_dir, fname))

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
    
    dataset = AdvDataset(INPUT_DIR)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    attacker = FUGEA(model_names=SURROGATE_MODELS)

    print(f"FUGEA Attack Started | Models: {SURROGATE_MODELS}")
    for images, labels, filenames in tqdm.tqdm(dataloader):
        perturbations = attacker(images, labels)
        save_images(OUTPUT_DIR, images + perturbations.cpu(), filenames)
    
    print(f"Done. Images saved to {OUTPUT_DIR}")

if __name__ == '__main__':
    main()