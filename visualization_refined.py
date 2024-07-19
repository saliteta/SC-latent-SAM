from stable_processing.decode_mask import white_overlay, alpha_value_blending
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
    mask = np.load(npz_location)
    img_list = [os.path.join(img_dir, img) for img in sorted(os.listdir(img_dir)) if img.endswith(('.png', '.jpg', '.jpeg'))]
    
    return mask, 

def generate_image_group(masks:np.ndarray, original_image:str, stored_location):
    original_image = Image.open((original_image)).convert('RGBA')
    count = 0
    for mask in masks:
        alpha_mask = alpha_value_blending(original_image.size, mask)
        overlay_image = white_overlay(alpha_mask, original_image)
        os.makedirs(stored_location,exist_ok=True)
        overlay_image.save(f'output/{stored_location}/{count}.png')
        count += 1
        overlay_image.close()
    

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
    
    
    masks, img_list = get_source_file(mask_location, image_dir)
    

    print('overlaing image ...')
    for i, image_location in tqdm(enumerate(img_list), total=len(img_list), desc="Processing items"):
        
        generate_image_group(
                            masks[image_location], 
                            os.path.join(image_dir, image_location), 
                            os.path.join(output_dir,image_location)
                            )
if __name__ == '__main__':
    main()