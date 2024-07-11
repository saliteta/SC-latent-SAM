from stable_processing.decode_mask import decode_from_downscaled_masks, remove_padding, overlay_mask_on_image
from tqdm import tqdm
from PIL import Image
import numpy as np
import os

import argparse

def get_image_height_width(image_location):
    img = Image.open(image_location)
    img = np.asarray(img)
    return img.shape[:2]

def get_source_file(npz_location:str, img_dir:str):
    mask = np.load(npz_location)['arr_0']
    img_list = [os.path.join(img_dir, img) for img in sorted(os.listdir(img_dir)) if img.endswith(('.png', '.jpg', '.jpeg'))]
    
    return mask, img_list

def parser():
    parser = argparse.ArgumentParser("Visualize The Maksed Result", add_help=True)
    parser.add_argument("--image_dir", "-i", type=str, required=True, help="path to image folder")
    parser.add_argument("--mask_location", "-m", type=str, required=True, help="Our mask direction")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs", required=True, help="Output directory")
    args = parser.parse_args()
    return args

def main():
    args = parser()
    
    mask_location = args.mask_location
    image_dir = args.image_dir
    output_dir = args.output_dir
    print(output_dir)
    
    
    mask, img_list = get_source_file(mask_location, image_dir)
    
    shape = get_image_height_width(img_list[0])
    
    
    mask = decode_from_downscaled_masks(mask, max(shape))
    mask = remove_padding(mask, original_size=shape)


    print('overlaing image ...')
    for i, image_location in tqdm(enumerate(img_list), total=len(img_list), desc="Processing items"):
        
        blend_image = overlay_mask_on_image(image_location, mask[i], shape, alpha=0.68)
        
        blend_image.save(os.path.join(output_dir,(image_location.split('/')[-1].split('.')[0]+'.png')))
if __name__ == '__main__':
    main()