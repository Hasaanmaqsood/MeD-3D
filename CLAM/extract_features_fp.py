import time
import os
import argparse
import pdb
from functools import partial

import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from PIL import Image
import h5py
import openslide
from tqdm import tqdm

import numpy as np

from utils.file_utils import save_hdf5
from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from models import get_encoder
from openslide.lowlevel import OpenSlideError  # Import OpenSlideError

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Store skipped and corrupt files
skipped_files = []
corrupt_files = []
already_processed_files = []

def compute_w_loader(output_path, loader, model, verbose = 0):
    """
    args:
        output_path: directory to save computed features (.h5 file)
        model: pytorch model
        verbose: level of feedback
    """
    if verbose > 0:
        print(f'Processing a total of {len(loader)} batches')

    mode = 'w'
    for count, data in enumerate(tqdm(loader)):
        with torch.inference_mode():    
            batch = data['img']
            coords = data['coord'].numpy().astype(np.int32)
            batch = batch.to(device, non_blocking=True)
            
            features = model(batch)
            features = features.cpu().numpy().astype(np.float32)

            asset_dict = {'features': features, 'coords': coords}
            save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
            mode = 'a'
    
    return output_path


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--model_name', type=str, default='resnet50_trunc', choices=['resnet50_trunc', 'uni_v1', 'conch_v1'])
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--target_patch_size', type=int, default=224)
args = parser.parse_args()


if __name__ == '__main__':
    print('Initializing dataset...')
    csv_path = args.csv_path
    if csv_path is None:
        raise NotImplementedError

    bags_dataset = Dataset_All_Bags(csv_path)
    
    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
    dest_files = set(os.listdir(os.path.join(args.feat_dir, 'pt_files')))  # Convert to set for fast lookup

    model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size)
            
    _ = model.eval()
    model = model.to(device)
    total = len(bags_dataset)

    loader_kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" else {}

    for bag_candidate_idx in tqdm(range(total)):
        slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
        bag_name = slide_id + '.h5'
        h5_file_path = os.path.join(args.data_h5_dir, bag_name)
        slide_file_path = os.path.join(args.data_slide_dir, slide_id + args.slide_ext)
        pt_file_path = os.path.join(args.feat_dir, 'pt_files', slide_id + '.pt')

        print('\nProgress: {}/{}'.format(bag_candidate_idx, total))
        print(slide_id)

        # ? Skip if already processed
        if os.path.exists(pt_file_path):
            print(f'? Skipping {slide_id}: Features already extracted')
            already_processed_files.append(slide_id)
            continue 

        # ? Skip if the h5 file is missing
        if not os.path.exists(h5_file_path):
            print(f"?? Skipping {slide_id}: Missing {h5_file_path}")
            skipped_files.append(slide_id)
            continue  

        output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
        time_start = time.time()
        
        # ? Try opening the slide file to make sure it exists
        try:
            wsi = openslide.open_slide(slide_file_path)
        except OpenSlideError as e:
            print(f"?? Skipping {slide_id}: Corrupt WSI file ({slide_file_path}) - OpenSlideError: {e}")
            corrupt_files.append(slide_id)
            continue
        except Exception as e:
            print(f"?? Skipping {slide_id}: Unable to open WSI file ({slide_file_path}) - Error: {e}")
            corrupt_files.append(slide_id)
            continue

        dataset = Whole_Slide_Bag_FP(file_path=h5_file_path, 
                                     wsi=wsi, 
                                     img_transforms=img_transforms)

        try:
            loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)
            output_file_path = compute_w_loader(output_path, loader=loader, model=model, verbose=1)

            time_elapsed = time.time() - time_start
            print(f'\n? Computing features for {output_file_path} took {time_elapsed:.2f} s')

            with h5py.File(output_file_path, "r") as file:
                features = file['features'][:]
                print('Features size: ', features.shape)
                print('Coordinates size: ', file['coords'].shape)

            features = torch.from_numpy(features)
            bag_base, _ = os.path.splitext(bag_name)
            torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base + '.pt'))
        except Exception as e:
            print(f"?? Skipping {slide_id}: Error during feature extraction - {e}")
            corrupt_files.append(slide_id)
            continue

    # ? Print summary of skipped and corrupt files at the end
    print("\n========================================")
    print("?? Summary of Skipped and Corrupt Files")

    if already_processed_files:
        print(f"? Total Already Processed: {len(already_processed_files)}")
        for slide in already_processed_files:
            print(f" - {slide}")

    if skipped_files:
        print(f"?? Total Skipped (Missing H5): {len(skipped_files)}")
        for slide in skipped_files:
            print(f" - {slide}")
    
    if corrupt_files:
        print(f"?? Total Corrupt Files (OpenSlideError): {len(corrupt_files)}")
        for slide in corrupt_files:
            print(f" - {slide}")

    if not skipped_files and not corrupt_files and not already_processed_files:
        print("? No files were skipped. All processed successfully.")
    print("========================================")
