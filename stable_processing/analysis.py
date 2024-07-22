from cuml.decomposition import PCA
from cuml.cluster import KMeans
import torch.nn.functional as F

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import cupy as cp
import torch

'''
    It seems like, the best result would be using K-means
    And use PCA to do the down sampling to n_components to 20
'''

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

def visualize_clusters(labels, title, batch_size):
    fig = plt.figure(figsize=(8, 8))
    grid_size = int(batch_size**0.5)
    gs = gridspec.GridSpec(grid_size, grid_size, wspace=0.1, hspace=0.1)
    
    for i in range(batch_size):
        ax = fig.add_subplot(gs[i])
        im = ax.imshow(labels[i], cmap='viridis')
        ax.axis('off')

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Colorbar axis
    fig.colorbar(im, cax=cbar_ax)

    plt.suptitle(title)
    plt.savefig(title + '.jpg')
    plt.close()

def cluster_kmeans(features: np.ndarray, n_clusters=10) -> torch.Tensor:
    '''
        Input would be a (b,h,w,c) tensor
        first resize it to (bhw, c) tensor, and do the clustering
        return (b,h,w,) labels 
    '''
    original_shape = features.shape
    features = features.reshape((-1, features.shape[-1]))
    cupy_features = cp.asarray(features.contiguous())  # Ensure contiguous memory for conversion
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(cupy_features)
    return torch.as_tensor(labels, device=features.device).reshape([original_shape[0], original_shape[1], original_shape[2]])


def inter_group_cluster_kmeans(features: torch.Tensor, n_clusters=20) -> torch.Tensor: 
    '''
        Input would be a (prototype_len, features)
        return (prototype_len) labels 
    '''
    original_shape = features.shape
    cupy_features = cp.asarray(features.contiguous().view(-1, original_shape[-1]))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(cupy_features)
    return torch.as_tensor(labels, device=features.device).squeeze().view(original_shape[0], original_shape[1])

def group_prototyping(features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    '''
    features: (b,h,w,c)
    labels: (b,h,w)
    '''
    # Flatten the features and labels
    features = features.reshape((-1, features.shape[-1]))
    labels = labels.reshape(-1)
    label_index = labels.long()

    # Determine the number of unique labels
    num_labels = label_index.max().item() + 1
    

    # Create a one-hot encoding of the labels
    one_hot_labels = torch.nn.functional.one_hot(label_index, num_classes=num_labels).float()

    # Sum the features for each label
    prototype_sum = torch.matmul(one_hot_labels.t(), features)

    # Count the number of features for each label
    prototype_count = one_hot_labels.sum(dim=0, keepdim=True).t()

    # Compute the prototypes by dividing the sum by the count
    prototype = prototype_sum / prototype_count

    return prototype
    
def apply_pca(features: torch.Tensor, n_components=20) -> torch.Tensor:
    '''
        down sampled features from c -> c`
        Input would be a (b,h,w,c) tensor
        first resize it to (bhw, c) tensor, and do the clustering
        return (b,h,w,c`) features 
    '''
    original_shape = features.shape
    features = features.reshape((-1, features.shape[-1]))
    cupy_features = cp.asarray(features.contiguous())  # Ensure contiguous memory for conversion
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(cupy_features)
    return torch.as_tensor(reduced_features, device=features.device).reshape((original_shape[0], original_shape[1], original_shape[2], -1))


def overall_label(
    batched_labels: torch.Tensor, 
    label_look_up_tables: torch.Tensor,
    batch_number: int = 4
    ) -> torch.Tensor:
    '''
    batched_labels: (n, h, w) 
    label_look_up_tables: (n//b, k) k means the number of labels
    '''
    # Get the shape of batched_labels
    _, h, w = batched_labels.shape
    
    # Initialize the output tensor with the same shape as batched_labels
    new_labels = torch.zeros_like(batched_labels)
    
    
    # Iterate over each batch
    for i in range(len(label_look_up_tables)):
        # Get the current lookup table for the batch
        lookup_table = label_look_up_tables[i]
        if i != (len(label_look_up_tables)-1):
            # Flatten the spatial dimensions of the batch
            flat_labels = batched_labels[i*batch_number:(i+1)*batch_number].view(-1).long()
            mapped_labels = lookup_table[flat_labels]
            new_labels[i*batch_number:(i+1)*batch_number] = mapped_labels.view(-1, h, w)

        else:
            flat_labels = batched_labels[i*batch_number:].view(-1).long()
            # Map the original labels to the new labels using the lookup table
            mapped_labels = lookup_table[flat_labels]
            # Reshape back to the original spatial dimensions and assign to the output tensor
            new_labels[i*batch_number:] = mapped_labels.view(-1, h, w)
            
    return new_labels


def heatmap(data:torch.Tensor, img_location):
    plt.figure(figsize=(8, 8))  # Set the size of the figure (optional)
    data = data.cpu().numpy()
    plt.imshow(data, cmap='hot', interpolation='nearest')  # 'hot' colormap goes from black to red to yellow to white
    plt.colorbar()  # Show color scale
    plt.title('Heatmap of 256x256 Array')
    plt.savefig(img_location)
    plt.close()



if __name__ == '__main__':
    batched_labels = torch.randint(size=(671, 64, 64), low = 0, high = 10)
    label_look_up_tables = torch.randint(size=(617//4+1, 10), low=0, high=30)
    overall_label(
        batched_labels,
        label_look_up_tables,
        4
    )