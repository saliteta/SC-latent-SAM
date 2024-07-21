CUDA_VISIBLE_DEVICES=2 python -W ignore fine_mask_segmentation.py \
    --sam_checkpoint /home/planner/xiongbutian/Foundation_Models/SAM/sam_vit_h_4b8939.pth \
    --image_dir /home/planner/xiongbutian/ignores/images \
    --batch_num 4 \
    --output_dir /home/planner/xiongbutian/ignores/output \
    --debugging False \
    --device cuda 

python visualization_refined.py \
    -i /home/planner/xiongbutian/ignores/images \
    -m /home/planner/xiongbutian/ignores/output/refined_mask.npz \
    -o /home/planner/xiongbutian/ignores/output/visualization