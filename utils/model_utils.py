import torch
import pyiqa
from BL_demo import Baseline
from Prepare_image import Image_load
"""
Available models from pyiqa:
'ahiq', 'brisque', 'ckdn', 'clipiqa', 'clipiqa+', 'clipiqa+_rn50_512', 'clipiqa+_vitL14_512', 'clipscore', 
'cnniqa', 'cw_ssim', 'dbcnn', 'dists', 'entropy', 'fid', 'fsim', 'gmsd', 'hyperiqa', 'ilniqe', 'laion_aes', 
'lpips', 'lpips-vgg', 'mad', 'maniqa', 'maniqa-kadid', 'maniqa-koniq', 'maniqa-pipal', 'ms_ssim', 'musiq', 
'musiq-ava', 'musiq-koniq', 'musiq-paq2piq', 'musiq-spaq', 'nima', 'nima-vgg16-ava', 'niqe', 'nlpd', 'nrqm', 
'paq2piq', 'pi', 'pieapp', 'psnr', 'psnry', 'ssim', 'ssimc', 'stlpips', 'stlpips-vgg', 'topiq_fr', 'topiq_fr-pipal', 
'topiq_iaa', 'topiq_iaa_res50', 'topiq_nr', 'topiq_nr-face', 'topiq_nr-flive', 'topiq_nr-spaq', 'tres', 'tres-flive', 
'tres-koniq', 'uranker', 'vif', 'vsi'
"""

def get_device(device: str = 'cpu'):
    device = torch.device("cuda") if torch.cuda.is_available() and device=='cuda' else torch.device("cpu")
    print(f"Using device: {device}")
    return device

def list_models():
    print("Available models:")
    print(pyiqa.list_models())

def create_metric(metric_name, device, as_loss=False, **kwargs):
    metric = pyiqa.create_metric(metric_name, device=device, as_loss=as_loss, **kwargs)
    print(f"Created {metric_name} metric with {'loss' if as_loss else 'default'} setting")
    return metric

def check_metric(metric):
    print(f"Lower score is better for {metric.metric_name}: {metric.lower_better}")

def evaluate_metric(metric, img_path1, img_path2=None):
    score = metric(img_path1, img_path2) if img_path2 else metric(img_path1)
    print(f"Evaluation score for {metric.metric_name}: {score}")

def get_model(model_name, device, as_loss=False, **kwargs):
    model = create_metric(model_name, device, as_loss, **kwargs)
    return model

def get_bl_model(device):
    bl_spaq = Baseline()
    bl_spaq.to(device)
    checkpoint = torch.load('./models/BL_release.pt', map_location=torch.device('cpu'))
    bl_spaq.load_state_dict(checkpoint['state_dict'])
    print("Loaded BL model")
    return bl_spaq

if __name__ == "__main__":
    # Example usage
    device = get_device()
    list_models()

    # Get models
    topiq_nr = get_model('topiq_nr', device, False)
    topiq_nr_spaq = get_model('topiq_nr-spaq', device, False)

    # Check metrics
    check_metric(topiq_nr)
    check_metric(topiq_nr_spaq)
