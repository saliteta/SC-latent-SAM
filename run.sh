# CUDA_VISIBLE_DEVICES=5 python -W ignore clustering_features.py \
#     --sam_checkpoint /home/xiongbutian/workspace/Foundation_Models/SAM/sam_vit_h_4b8939.pth \
#     --image_dir /data/grocery_store/10F/images/ \
#     --batch_num 4 \
#     --output_dir output \
#     --debugging False \
#     --device cuda 

python visualization.py \
    -i /data/grocery_store/10F/input/ \
    -m output/saved_labels.npz \
    -o output/ 