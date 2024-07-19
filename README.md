# Semantic Consistanct Latent Sam 

- We utilize Hierachical Clustering Method to alleviate inconsistancy problem
- We provide a coarse to fine stratgy to refine the rough mask result we have (64*64)
- We achieve a better result only on quntization

## Problem Definition
- Inconsistancy
![inconsistancy](assets/inconsistancy.png)

## Result for Rough Result
![adjacent_comparison](assets/adjacent_comparison.png)
![global_comparison](assets/global_comparison.pn

## Result for Refined Result
![Refined_result](assets/refined_semantic_result.png)

## Install
After clone 
```
    conda create -n sc-latent-sam python=3.9 -y
    conda install -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge  cudatoolkit=11.8 # if u r in China Mainland
    conda install -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/rapidsai/ cuml # if u r in China Mainland
    conda install -c nvidia::cudatoolkit=11.8 # if you are in a place where there is no saction and firewalls
    conda install rapidsai::cuml # if you are in a place where there is no saction and firewalls
```

conda install -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/rapidsai/ -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge cuml=24.06 python=3.9 cuda-version=11.8

Follow  [SAM](https://github.com/facebookresearch/segment-anything) to install othere related package

Download a image dataset 
```
--Image_Folder
    |- image1
       image2
       image4
        ...
```
## Use Code

Change the following code in run.sh:
```
CUDA_VISIBLE_DEVICES=x python -W ignore clustering_features.py \
    --sam_checkpoint 'sam model path' \ # need to be changed
    --image_dir 'image dir ' \ # need to be changed
    --batch_num 4 \
    --output_dir debugging \
    --debugging True \
    --device cuda 
```

## Visualization 
To visualize our result, one can run the following code in run.sh

```
python visualization.py \
    -i /data/grocery_store/10F/input/ \ # Input image 
    -m output/saved_labels.npz \ # Mask File
    -o output/ # Output folder
```
![Visualization](assets/visualization.png)
