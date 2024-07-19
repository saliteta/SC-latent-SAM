CUDA_VISIBLE_DEVICES=2 python -W ignore clustering_features.py \
    --sam_checkpoint /home/planner/xiongbutian/Foundation_Models/SAM/sam_vit_h_4b8939.pth \
    --image_dir /home/planner/xiongbutian/sc_latent_sam/images \
    --batch_num 4 \
    --output_dir output \
    --debugging True \
    --device cuda 

python visualization.py \
    -i /data/grocery_store/10F/input/ \
    -m output/saved_labels.npz \
    -o output/ 