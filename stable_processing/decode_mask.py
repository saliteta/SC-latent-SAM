import numpy as np
from PIL import Image
from io import BytesIO

import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn.functional as F

def decode_from_downscaled_masks(original_masks: np.ndarray, max_size: int) -> np.ndarray:
    """
    Rescale a batch of downscaled masks from 64x64 to original_size to original size.

    Parameters:
    - original_masks (np.ndarray): A batch of masks with shape (n, 64, 64)

    Returns:
    - np.ndarray: A batch of resized masks with shape (n, 1024, 1024)
    """
    # Initialize an empty list to store the resized masks
    resized_masks = []

    # Iterate through each mask in the batch
    print('interpolate original masks ...')
    for mask in tqdm(original_masks):
        # Convert the numpy array to a PIL image
        image_small = Image.fromarray(mask)

        # Resize using nearest neighbor interpolation
        image_large = image_small.resize((max_size, max_size), Image.NEAREST)

        # Convert back to numpy array and append to the list
        labels_large = np.array(image_large)
        resized_masks.append(labels_large)

    # Stack all resized masks into a single numpy array
    return np.stack(resized_masks)

def remove_padding(large_mask: np.ndarray, original_size: tuple) -> np.ndarray:
    """
    Remove padding from an upscaled 1024x1024 image to its original dimensions.

    Parameters:
    - large_mask (np.ndarray): The upscaled mask of shape (1024, 1024) or (n, 1024, 1024)
    - original_size (tuple): The original dimensions (height, width) of the image

    Returns:
    - np.ndarray: The mask with padding removed, resized back to the original dimensions.
    """
    original_height, original_width = original_size

    # Depending on which dimension was padded, remove the padding
    if original_height == max(original_size):
        # Padding was added to the width
        return large_mask[:, : , :original_width]
    else:
        # Padding was added to the height
        return large_mask[:, :original_height, :]


def overlay_mask_on_image(
        original_image_path: str, 
        mask_array: np.ndarray, 
        original_size: tuple, 
        alpha: float = 0.5):
    """
    Overlay a mask as a heatmap onto an original image without saving the heatmap to disk.

    Parameters:
    - original_image_path (str): Path to the original image.
    - mask_array (np.ndarray): The final mask array of the original size.
    - original_size (tuple): The original dimensions (height, width) of the image.
    - alpha (float): Transparency factor of the heatmap.

    Returns:
    - PIL Image: The original image with the heatmap overlay.
    """
    original_image = Image.open(original_image_path)

    # Create a heatmap figure directly without axes or colorbar
    fig, ax = plt.subplots(figsize=(original_size[1] / 100, original_size[0] / 100), dpi=100)
    ax.axis('off')  # Hide the axes
    ax.imshow(mask_array, cmap='viridis', interpolation='nearest', vmin=0, vmax=30)

    # Save the heatmap to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close(fig)

    # Load the heatmap image from the BytesIO buffer
    heatmap_image = Image.open(buf).convert('RGBA')

    # Resize the heatmap image to match the original image size, if not already matching
    heatmap_image = heatmap_image.resize(original_image.size, Image.LANCZOS)

    # Blend the original image and the heatmap
    original_image = original_image.convert('RGBA')  # Ensure original is in RGBA mode
    blended_image = Image.blend(original_image, heatmap_image, alpha=alpha)
    return blended_image


def logits_to_alpha(logits, lower_bound, upper_bound)->np.ndarray:
    # Applying the sigmoid function to normalize logits between 0 and 1
    sigmoid_logits = torch.sigmoid(logits)
    
    # Scaling the sigmoid output to the desired alpha range [lower_bound, upper_bound]
    alpha = upper_bound - (sigmoid_logits * (upper_bound - lower_bound))
    
    return alpha.cpu().numpy()

def logits_to_mask(logits:torch.Tensor, threshold = 0.5) -> np.ndarray:
    '''
        Basic process stratgy that can smooth the segmentation. 
    '''
    # Create a simple averaging kernel
    kernel_size = 4  # Size of the kernel (3x3)
    kernel = torch.ones((kernel_size, kernel_size)).cuda() / (kernel_size ** 2)
    kernel = kernel.expand((1, 1, kernel_size, kernel_size))  # Expand for conv2d compatibility

    # Ensure that the kernel is a floating point tensor
    kernel = kernel.type(torch.float32)
    
    logits = logits.unsqueeze(0).unsqueeze(0)
    filtered_logits = F.conv2d(logits, kernel, padding=1, stride=1)
    filtered_logits = filtered_logits.squeeze()
    
    binary_tensor = torch.where(filtered_logits < threshold, torch.tensor(0.8), torch.tensor(0.0))
    return binary_tensor.cpu().numpy()

def alpha_value_blending(original_image_shape:np.ndarray, masks:torch.Tensor, higher_bound = 0.9):
    #masks = logits_to_alpha(masks, 0.0, higher_bound)
    masks = logits_to_mask(masks)
    max_size = max(original_image_shape)
    image_small = Image.fromarray(masks)

    # Resize using nearest neighbor interpolation
    image_large = np.transpose(np.asarray(image_small.resize((max_size, max_size), Image.BILINEAR)))

    original_height, original_width = original_image_shape

    # Depending on which dimension was padded, remove the padding
    if original_height == max(original_image_shape):
        # Padding was added to the width
        return image_large[: , :original_width]
    else:
        # Padding was added to the height
        return image_large[:original_height, :]
    
def white_overlay(alpha_mask,   original_image) -> Image.Image:
    width, height = original_image.size
    white_overlay = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    alpha_scaled = (255 * alpha_mask).astype(np.uint8)
    
    alpha_mask = Image.fromarray(np.transpose(alpha_scaled))

    # Create an all-white image with the same dimensions as the original image
    white_overlay = Image.new("RGBA", (width, height), (20, 20, 20, 255))  # White image, fully opaque
    # Update the alpha channel of the white overlay with your alpha mask
    white_overlay.putalpha(alpha_mask)
    
    # Blend the original image with the white overlay
    blended_image = Image.alpha_composite(original_image, white_overlay)
    return blended_image