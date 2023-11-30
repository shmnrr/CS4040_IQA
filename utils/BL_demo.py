# ========================================================================
# Perceptual Quality Assessment of Smartphone Photography
# PyTorch Version 1.0 by Hanwei Zhu
# Copyright(c) 2020 Yuming Fang, Hanwei Zhu, Yan Zeng, Kede Ma, and Zhou Wang
# All Rights Reserved.
#
# ----------------------------------------------------------------------
# Permission to use, copy, or modify this software and its documentation
# for educational and research purposes only and without fee is hereby
# granted, provided that this copyright notice and the original authors'
# names appear on all copies and supporting documentation. This program
# shall not be used, rewritten, or adapted as the basis of a commercial
# software or hardware product without first obtaining permission of the
# authors. The authors make no representations about the suitability of
# this software for any purpose. It is provided "as is" without express
# or implied warranty.
# ----------------------------------------------------------------------
# This is an implementation of Baseline (BL) models for blind image
# quality assessment.
# Please refer to the following paper:
#
# Y. Fang et al., "Perceptual Quality Assessment of Smartphone Photography" 
# in IEEE Conference on Computer Vision and Pattern Recognition, 2020
#
# Kindly report any suggestions or corrections to hanwei.zhu@outlook.com
# ========================================================================

import torch
import torch.nn as nn
import torchvision
from Prepare_image import Image_load
from PIL import Image
import os
import time
import glob
# from model_utils import get_model, get_device, get_bl_model

class Baseline(nn.Module):
	def __init__(self):
		super(Baseline, self).__init__()
		self.backbone = torchvision.models.resnet50(pretrained=False)
		fc_feature = self.backbone.fc.in_features
		self.backbone.fc = nn.Linear(fc_feature, 1, bias=True)

	def forward(self, x):
		result = self.backbone(x)
		return result

class Demo(object):
    def __init__(self, device, load_weights=True, checkpoint_dir='./models/BL_release.pt' ):
        self.load_weights = load_weights
        self.checkpoint_dir = checkpoint_dir

        self.prepare_image = Image_load(size=448, stride=112)

        self.model = Baseline()
        self.device = device
        self.model.to(self.device)
        self.model_name = type(self.model).__name__

        if self.load_weights:
            self.initialize()

    def predict_quality(self, image_path):
        start_time_full = time.time() * 1000
        image = self.prepare_image(Image.open(image_path).convert("RGB"))
        image = image.to(self.device)
        start_time_inference = time.time() * 1000
        self.model.eval()
        end_time = time.time() * 1000
        
        score = self.model(image).mean()
        # print(f'Score for {image_path}: {score.item()}')
        return (image_path.split('/')[-1], score.item(), end_time - start_time_full, end_time - start_time_inference, start_time_inference - start_time_full)

    def initialize(self):
        ckpt_path = self.checkpoint_dir
        could_load = self._load_checkpoint(ckpt_path)
        if could_load:
            print('Checkpoint load successfully!')
        else:
            raise IOError('Fail to load the pretrained model')

    def _load_checkpoint(self, ckpt):
        if os.path.isfile(ckpt):
            print("[*] loading checkpoint '{}'".format(ckpt))
            checkpoint = torch.load(ckpt, map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint['state_dict'])
            return True
        else:
            return False
        
    def run_on_n_images(self, n: int=1000, image_dir: str='./SpaqLite/app/src/main/assets/SPAQ/TestImage'):
        print(f"Image directory: {image_dir}")
        print(f"Running on {n} images")
        image_paths = glob.glob(os.path.join(image_dir, '*'))
        results = []
        start_time = time.time()
        for image_path in image_paths[:n]:
            results.append(self.predict_quality(image_path))
        end_time = time.time()
        
        return results, (end_time - start_time) * 1000
        
        
class Demo_Universal(object):
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model_name = type(self.model).__name__
        self.prepare_image = Image_load(size=448, stride=112)
        print(f"Model name: {self.model_name}")

    def predict_quality(self, image_path):
        start_time = time.time() * 1000
        image = self.prepare_image(Image.open(image_path).convert("RGB"))
        image = image.to(self.device)
        start_time_inference = time.time() * 1000
        self.model.eval()
        score = self.model(image).mean()
        end_time = time.time() * 1000
        
        # print(f'Score for {image_path}: {score.item()}')
        filename = os.path.basename(image_path)
        return (filename.replace('.png', '.jpg'), score.item(), end_time - start_time, end_time - start_time_inference, start_time_inference - start_time)

    def run_on_n_images(self, n: int=1000, image_dir: str='./SpaqLite/app/src/main/assets/SPAQ/TestImage'):
        print(f"Image directory: {image_dir}")
        print(f"Running on {n} images")
        image_paths = glob.glob(os.path.join(image_dir, '*'))
        results = []
        start_time_full = time.time()
        for image_path in image_paths[:n]:
            results.append(self.predict_quality(image_path))
        end_time_full = time.time()
        
        return results, (end_time_full - start_time_full) * 1000
    

def main():
    ### Example usage
    device = torch.device('cpu')
    
    ### Baseline model
    t = Demo(device)
    
    ### Universal model
    # model = get_model('topiq_nr-spaq', device, True)
    # t = Demo_Universal(model, device)
    
    print(t.run_on_n_images(10))
    

if __name__ == '__main__':
    main()
