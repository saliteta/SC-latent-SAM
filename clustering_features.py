'''
    We first extract and clustering features in batch
    We then clustering prototype
'''
from segment_anything.modeling.sam import Sam
from segment_anything import SamPredictor, sam_model_registry


from loader import load_dataset
from analysis import apply_pca, cluster_kmeans, inter_group_cluster_kmeans, group_prototyping, overall_label

import torch
import random
import argparse
from tqdm import tqdm 

import os
import numpy as np
import matplotlib.pyplot as plt

K = 10
OVERALL_CLUSTER = 30

'''
    [x] We first get each label in a batch, save it as a tensor in cpu
    [x] At the same time, we extract prototype and stack as a tensor table, it is a (images_number//b, k)
    [x]  map the original labels within batch according to the tensor table
'''

### Further optimization suggestion
### Only load SAM model that we need to use

sam_checkpoint = '/home/xiongbutian/workspace/GroundingSAM_Fast/sam_vit_h_4b8939.pth'
sam_version = 'vit_h'

class sam_encoder(SamPredictor):
    def __init__(self, sam_model: Sam) -> None:
        super().__init__(sam_model)
    
    def feature_extraction(self, images) -> torch.Tensor:
        
        '''
            Takes in BCHW images in torch CUDA and return encoder features
        '''
        
        return self.model.image_encoder(images)
        

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
    
    model = sam_encoder(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))
    batched_labels = torch.zeros(size = (image_number, 64, 64))
    batched_prototype = torch.zeros(size = (len(loader), K, 256))
    
    image_count = 0
    batch_count = 0
    
    with torch.no_grad():
        for images, names in tqdm(loader):
            images = images.to(device).squeeze(1)
            features = model.feature_extraction(images)

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
    refined_lable = overall_label(batched_labels, prototype_clustered_result).cpu().numpy()
    

    
    np.savez(f'{output_directory}/saved_labels', refined_lable)
    if debugging == 'True':
        random_int = random.sample(range(0, image_number), 40)
        print('this is the image sequence number we would like to use', random_int)

        for i in range(len(random_int)):

            label = refined_lable[random_int[i]]
            # Create a colormap for the labels
            # Plot the labeled mask
            plt.figure(figsize=(6, 6))
            plt.imshow(label, cmap='viridis', vmin=0, vmax=30)
            plt.colorbar()  # shows the color bar
            plt.savefig(f'debugging/img_{random_int[i]}.jpg')
            plt.close()

if __name__ == '__main__':
    main()    

    #refined_label = np.load('/home/xiongbutian/workspace/Gaussian_Based_Model/3D_langSplat/preprocess/debugging/saved_labels.npz')['arr_0']
    #
    #for i in range(600):
    #    label = refined_label[i]
    #    # Create a colormap for the labels
    #    # Plot the labeled mask
    #    plt.figure(figsize=(6, 6))
    #    plt.imshow(label, cmap='viridis', vmin=0, vmax=30)
    #    plt.colorbar()  # shows the color bar
    #    plt.savefig(f'debugging/img_{i}.jpg') 