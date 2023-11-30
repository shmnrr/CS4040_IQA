import torch
import torchvision.transforms as transforms

from model_utils import get_model, get_device, check_metric, get_bl_model
from PIL import Image
from torch.utils.mobile_optimizer import optimize_for_mobile
from BL_demo import Baseline
from Prepare_image import Image_load

def port_model(model_name: str, device, as_loss=False, example_img_x_path: str='./datasets/SPAQ/TestImage/00002.jpg', **kwargs):
    """Port IQA model to Pytorch Lite format. Saved in /mobile_models folder.

    Args:
        model_name (str): model name
        device: user's computing device
        as_loss (bool, optional): use for training. Defaults to False.
        example_img_x_path (str, optional): example distorted image fed to the model. Defaults to './example_images/00001.jpg'.
    """
    model = get_model(model_name, device, as_loss, **kwargs)
    metric = check_metric(model)
    model.eval()
    
    # Create an instance of Image_load
    # Resize and stride as in https://openaccess.thecvf.com/content_CVPR_2020/papers/Fang_Perceptual_Quality_Assessment_of_Smartphone_Photography_CVPR_2020_paper.pdf
    size = 448
    stride = 112
    prepare_image = Image_load(size=size, stride=stride)
    
    example_image_x = Image.open(example_img_x_path)
    
    # Apply the transformation to the images
    img_tensor_x = prepare_image(example_image_x)
    img_tensor_x.to(device)
    
    # Convert the model to TorchScript format
    traced_script_module = torch.jit.trace(model, (img_tensor_x))
        
    # Optimize the TorchScript model for mobile
    traced_script_module_optimized = optimize_for_mobile(traced_script_module)
    
    # Save the TorchScript model in /mobile_models
    traced_script_module_optimized._save_for_lite_interpreter(f"mobile_models/{model_name}_{size}x{stride}_{device.type}_{'lower' if metric else 'higher'}_NR.ptl")
    
def port_bl_model(model: torch.nn.Module, device, as_loss=False, example_img_x_path: str='./datasets/SPAQ/TestImage/00002.jpg', **kwargs):
    """Port BL SPAQ model to Pytorch Lite format. Saved in /mobile_models folder.

    Args:
        model_name (str): model name
        device: user's computing device
        as_loss (bool, optional): use for training. Defaults to False.
        example_img_x_path (str, optional): example distorted image fed to the model. Defaults to './example_images/00002.jpg'.
    """
    model.eval()
    
    # Apply the transformation to the images
    example_image_x = Image.open(example_img_x_path).convert("RGB")
    
    # Create an instance of Image_load
    # Resize and stride as in https://openaccess.thecvf.com/content_CVPR_2020/papers/Fang_Perceptual_Quality_Assessment_of_Smartphone_Photography_CVPR_2020_paper.pdf
    size = 448
    stride = 112
    prepare_image = Image_load(size=size, stride=stride)
    img_tensor_x = prepare_image(example_image_x)
    
    img_tensor_x.to(device)
    
    # Trace the model
    traced_script_module = torch.jit.trace(model, (img_tensor_x))
        
    # Optimize the TorchScript model for mobile
    traced_script_module_optimized = optimize_for_mobile(traced_script_module)
    
    # Save the TorchScript model in /mobile_models
    traced_script_module_optimized._save_for_lite_interpreter(f"lite_models/bl_spaq_{size}x{stride}_{device.type}_higher_NR.ptl")

if __name__ == '__main__':
    # Example usage
    models = ['ahiq', 'brisque', 'ckdn', 'clipiqa', 'clipiqa+', 'clipiqa+_rn50_512', 'clipiqa+_vitL14_512', 'clipscore', 
        'cnniqa', 'cw_ssim', 'dbcnn', 'dists', 'entropy', 'fid', 'fsim', 'gmsd', 'hyperiqa', 'ilniqe', 'laion_aes', 
        'lpips', 'lpips-vgg', 'mad', 'maniqa', 'maniqa-kadid', 'maniqa-koniq', 'maniqa-pipal', 'ms_ssim', 'musiq', 
        'musiq-ava', 'musiq-koniq', 'musiq-paq2piq', 'musiq-spaq', 'nima', 'nima-vgg16-ava', 'niqe', 'nlpd', 'nrqm', 
        'paq2piq', 'pi', 'pieapp', 'psnr', 'psnry', 'ssim', 'ssimc', 'stlpips', 'stlpips-vgg', 'topiq_fr', 'topiq_fr-pipal', 
        'topiq_iaa', 'topiq_iaa_res50', 'topiq_nr', 'topiq_nr-face', 'topiq_nr-flive', 'topiq_nr-spaq', 'tres', 'tres-flive', 
        'tres-koniq', 'uranker', 'vif', 'vsi']
  
    # for model in models:
    #     try:
    #         port_model(model, get_device('cuda'), False, './example_images/00002.jpg')
    #     except:
    #         print(f"Failed to port {model}")
    
    ### Port BL model ###
    # port_bl_model(get_bl_model(), get_device('cpu'), False, './example_images/00002.jpg')

    ### Port topiq_nr model ###
    port_model('topiq_nr', get_device('cpu'), False, './example_images/00002.jpg')
    
    ### Port topiq_nr-spaq model ###
    # port_model('topiq_nr-spaq', get_device('cpu'), False, './example_images/00002.jpg')
