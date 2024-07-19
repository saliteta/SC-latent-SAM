'''
    We use the clustering mask as the input to the decoder
    Therefore the latent we have used in batch should be preserve

    Here is the overall stratgy
    - We follow the clustering feature partsm we get the lattent, and we preserve it in some where
    - We get the masks, and store it in some where, 
    - We use the masks as the input and send it to the decoder model to gether with the features
'''

'''
    We find that SAM cannot use dense mask as input along, we need to first convert it to bounding box
    And then we fake the mask's logits as the input and convert masks to bounding box
'''

from segment_anything.modeling.sam import Sam
from segment_anything import SamPredictor, sam_model_registry


from stable_processing.loader import load_dataset
from stable_processing.fake_masks import compute_box_from_mask, fake_logits_mask
from stable_processing.analysis import cluster_kmeans, apply_pca, overall_label, inter_group_cluster_kmeans
from stable_processing.decode_mask import white_overlay, alpha_value_blending

from stable_processing.analysis import heatmap

import torch
import argparse
from tqdm import tqdm 

import numpy as np
import PIL.Image as Image
import os 
K = 10

class sam_batchify(SamPredictor):
    def __init__(self, sam_model: Sam) -> None:
        super().__init__(sam_model)
    
    def feature_extraction(self, images) -> torch.Tensor:
        
        '''
            Takes in BCHW images in torch CUDA and return encoder features
        '''
        
        return self.model.image_encoder(images)
    
    def mask_finegrainded_mask_generation(self, masks: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        '''
            It will use the following methods
            mask input is a low resolution mask, (b, H, W)
            Where H=W=256 (how to transfer to something like this?)
        '''
        mask_T = masks.T
        boxes = compute_box_from_mask(mask_T).unsqueeze(0)
        mask_T = fake_logits_mask(mask_T).unsqueeze(0)
        features = features.unsqueeze(0)

        # we need to make masks and boxes in the shape of B4, and B1HW

        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points = None, 
            boxes = boxes,
            masks = mask_T
        ) # we do not have poitns and boxes as input, we only have a clustered masks

        image_position_embeddings = self.model.prompt_encoder.pe_layer((64, 64)).cuda().unsqueeze(0) # 
        
        # The input shape should be the following:
        low_res_result_masks, iou_prediction = self.model.mask_decoder(
            image_embeddings=features,  # B, C, 64, 64
            image_pe=image_position_embeddings, # B, C, 64, 64
            sparse_prompt_embeddings=sparse_embeddings, # B, 2, C
            dense_prompt_embeddings=dense_embeddings, # B, C, 64, 64
            multimask_output=False,
        )

        

        return low_res_result_masks, iou_prediction
    

    def point_fine_gradined_mask_generation(self, masks: torch.Tensor, features: torch.Tensor) -> torch.Tensor:

        '''
            It will use the following methods
            mask input is a low resolution mask, (H, W)
            Where H=W=256 (how to transfer to something like this?)
        '''
        
        mask_T = masks.T
        points = torch.nonzero(mask_T == 1)+0.5 # move to the center of the pixel
        points_label = torch.ones(len(points)).unsqueeze(0)
        points = self.transform.apply_coords_torch(points, (64, 64)).unsqueeze(0)

        features = features.unsqueeze(0) 
        # how about encode all the points together?        

        sparse_input = (points, points_label)

        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points = sparse_input, 
            boxes = None,
            masks = None
        ) # we do not have poitns and boxes as input, we only have a clustered masks

        image_position_embeddings = self.model.prompt_encoder.pe_layer((64, 64)).cuda().unsqueeze(0) # 

        # The input shape should be the following:
        low_res_result_masks, iou_prediction = self.model.mask_decoder(
            image_embeddings=features,  # B, C, 64, 64
            image_pe=image_position_embeddings, # B, C, 64, 64
            sparse_prompt_embeddings=sparse_embeddings, # B, 2, C
            dense_prompt_embeddings=dense_embeddings, # B, C, 64, 64
            multimask_output=True,
        )
        ## we might need to merge multiple masks
        return low_res_result_masks, iou_prediction


def parser():
    parser = argparse.ArgumentParser("SAM Encoder Test", add_help=True)
    parser.add_argument("--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h")
    parser.add_argument("--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file")
    parser.add_argument("--image_dir", type=str, required=True, help="path to image file")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs", required=True, help="output directory")
    parser.add_argument("--debugging", type=str, default="False", help="if to save the internal output to image folder")
    parser.add_argument("--device", type=str, default="cuda", help="running on cpu only!, default=False")
    parser.add_argument("--batch_num", type=int, default=4, help="The number of images manipulating at one time, default=4")
    args = parser.parse_args()
    return args



def main():
    args = parser()
    sam_checkpoint = args.sam_checkpoint
    sam_version = args.sam_version
    
    print(f'Sam Checkpoint is :{sam_checkpoint}')
    print(f'Sam version is :{sam_version}')
    
    image_directory = args.image_dir
    output_directory = args.output_dir
    
    print(f'Image Directory is :{image_directory}')
    print(f'Output Directory is :{output_directory}')
    
    device = args.device
    
    debugging = args.debugging
    batch_number = args.batch_num


    loader, image_number = load_dataset(image_directory, batch_size=batch_number, num_workers=2) 
    model = sam_batchify(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))
    features_saver = torch.zeros(size = (image_number, 256, 64, 64))

    batched_labels = torch.zeros(size = (image_number, 64, 64))
    batched_prototype = torch.zeros(size = (len(loader), K, 256))
    
    image_count = 0
    batch_count = 0
    
    with torch.no_grad():
        for images, names in tqdm(loader):
            images = images.to(device).squeeze(1)
            features = model.feature_extraction(images)

            features_saver[image_count:len(images)+image_count] = features.to('cpu')

            features = features.permute(0,2,3,1)       
                 
            down_sample_features = apply_pca(features)
            
            labels = cluster_kmeans(features=down_sample_features)
            
            labels = torch.as_tensor(labels, device='cuda')
            features = torch.as_tensor(features, device='cuda')
            
            prototype = group_prototyping(features, labels)
            
            
            batched_labels[image_count:len(images)+image_count] = labels.cpu()
            batched_prototype[batch_count] = prototype.cpu()
            
            image_count += len(images)
            batch_count += 1
    
    del model
    torch.cuda.empty_cache()
    
    batched_prototype = batched_prototype.contiguous().cuda()
    prototype_clustered_result = inter_group_cluster_kmeans(batched_prototype, n_clusters=30) # (prototype_len)
    prototype_clustered_result = prototype_clustered_result.reshape((len(loader), -1)) # (the number of batches, k)
    
    batched_labels = batched_labels.contiguous().cuda()            # (n//b, b, h, w)
    refined_lable = overall_label(batched_labels, prototype_clustered_result)

    np.savez(f'{output_directory}/saved_labels', refined_lable.cpu().numpy())
    np.savez(f'{output_directory}/saved_features', features_saver.numpy())

    del batched_labels, prototype_clustered_result, batched_prototype

    # We need to use refined labels as a masks and send it to sam mask encoder
    model = sam_batchify(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))
    
    # ############### for debugging ##################
    # features_saver = np.load('/home/planner/xiongbutian/sc_latent_sam/output/saved_features.npz')['arr_0']
    # refined_lable = np.load('/home/planner/xiongbutian/sc_latent_sam/output/saved_labels.npz')['arr_0']
    # features_saver = torch.Tensor(features_saver)
    # refined_lable = torch.Tensor(refined_lable)
# 
    # ############### for debugging ##################

    file_name = sorted(os.listdir(image_directory))
    mask_dict = {}
    
    with torch.no_grad():
        for i, (features, masks, name) in tqdm(enumerate(zip(features_saver, refined_lable, file_name))):
            features = features.cuda()
            masks = masks.cuda()
            
            masks_unique = torch.unique(masks) 
            refined_merged_masks = []
            for label in masks_unique:
                ### would be possible to process it batchify?
                refined_masks, _ = model.point_fine_gradined_mask_generation(masks == label, features)
                refined_masks = refined_masks.squeeze()
                refined_mask_merge, _ = torch.max(refined_masks, dim=0)
                refined_merged_masks.append(refined_mask_merge)
                # we recommand to store
                # original_image = Image.open(os.path.join(image_directory,name)).convert('RGBA')
                # alpha_mask = alpha_value_blending(original_image.size, refined_masks_merge)
                # overlay_image = white_overlay(alpha_mask, original_image)
                # overlay_image.save(f'output/{name_write}_{label}.png')
                # overlay_image.close()
            mask_dict[name]  = torch.stack(refined_merged_masks).cpu().numpy()
            # There is no unified shape since some of the unique label might be a lot, some might not be a unique label
            
        np.savez_compressed(f'{output_directory}/refined_mask.npz', mask_dict)
    


if __name__ == '__main__':
    main()    