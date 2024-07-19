'''
    This part of the work is borrowed from MicroSAM
    https://github.com/computational-cell-analytics/micro-sam/blob/83997ff4a471cd2159fda4e26d1445f3be79eb08/micro_sam/prompt_based_segmentation.py
'''

import torch
import torch.nn.functional as F

def compute_box_from_mask(mask: torch.Tensor, new_size = 1024):

    '''
        We need to modify it, since the rough mask size is 64 by 64, and we need to extend it to 256,256
        mask is a 64 by 64 torch.Tensor
    '''


    # Calculate the minimum and maximum coordinates
    min_y, min_x = mask[0].min().item(), mask[1].min().item()
    max_y, max_x = mask[0].max().item(), mask[1].max().item()

    min_y, min_x = mask[0].min(), mask[1].min()
    max_y, max_x = mask[0].max(), mask[1].max()
    box = torch.Tensor([min_y, min_x, max_y + 1, max_x + 1]).to(mask.device)
    return convert_coordinates(box, new_size, mask_size=mask.shape[0])



def convert_coordinates(box:torch.Tensor, new_size = 1024, mask_size = 64):
    '''
        The size of the image is a suqre
        We need to change the size of box to 256*256
    '''
    size_ratio = new_size/mask_size
    new_image_center = new_size/2
    new_box = (box - new_image_center/size_ratio) * size_ratio + new_image_center 
    return new_box

def fake_logits_mask(mask:torch.Tensor, eps=1e-3):
    '''
        Input mask is a 64 * 64 mask
    '''

    def inv_sigmoid(x):
        return torch.log(x / (1 - x))
    
    mask = mask.unsqueeze(0).unsqueeze(0).float()
    mask = F.interpolate(mask, (256, 256), mode='nearest')
    mask = mask.squeeze()
    logits = torch.zeros_like(mask, dtype=torch.float32)
    logits[mask == 1] = 1 - eps
    logits[mask == 0] = eps
    logits = inv_sigmoid(logits)

    # resize to the expected mask shape of SAM (256x256)
    assert logits.ndim == 2

    logits = logits[None]
    assert logits.shape == (1, 256, 256), f"{logits.shape}"
    return logits


def logits_to_alpha(logits: torch.Tensor, lower_bound: float = 0.2, upper_bound: float = 1.0):
    '''
        This will generate high fidelity result from low resolution mask
        This generate result should be feed into CLIP to generate text embeddings
    '''
    # Applying the sigmoid function to normalize logits between 0 and 1
    sigmoid_logits = torch.sigmoid(logits)
    
    # Scaling the sigmoid output to the desired alpha range [lower_bound, upper_bound]
    alpha = lower_bound + (sigmoid_logits * (upper_bound - lower_bound))
    
    return alpha