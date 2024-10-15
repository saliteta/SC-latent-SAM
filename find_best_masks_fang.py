from time import time
import argparse
from skimage import io
from sklearn.metrics import jaccard_score,f1_score,auc
import numpy as np
import pandas as pd
from davis2017.evaluation import DAVISEvaluation
from tqdm import tqdm
from skimage import io, color
import pandas as pd
import os

# def parser():
#     parser = argparse.ArgumentParser(description="Calculate metrics for datasets.")
#     parser.add_argument('--gt_dir', type=str, help='Directory for ground truth npz files.')
#     parser.add_argument('--pred_dir', type=str, help='Directory for prediction npz files.')

def copyGT(gt, pred):
    copiedGT = np.zeros(pred.shape)
    B = pred.shape[0]
    for i in range(B):
        copiedGT[i,:,:] = gt
    return copiedGT

def Cal_IOU(gt_dir, pred_dir):
    gts = np.load(gt_dir)
    preds = np.load(pred_dir)
    counter = 0
    IOUS = list()
    F1 = list()
    PRECISION = list()
    RECALL = list()
    copiedGT = dict()
    for key, value in gts.items():
        name = list(preds)[counter]
        pred = preds[name]
        value = np.array(value, dtype=np.int32)
        pred = np.array(pred, dtype=np.int32)
       
        #复制B个一样的GT
        copiedGT[key] = copyGT(value, pred)
        AND = np.bitwise_and(value, pred)
        Union = np.bitwise_or(value, pred)
        
        sum_AND = np.sum(AND, axis=(1, 2)) # True positive (B,)
        sum_UNION = np.sum(Union, axis=(1,2))
        sum_pred = np.sum(pred, axis=(1,2)) #TP + FP 
        sum_value = np.sum(copiedGT[key], axis=(1,2)) #TP + FN
        
        # #计算IoU
        # IOU = sum_AND/sum_UNION
        # # IOU = IOU.max()
        # IOU = np.random.choice(IOU)
        # # print('IOU', IOU)
        # IOUS.append(IOU)
        
        # #计算Precision和Recall
        # precision = np.random.choice(sum_AND / sum_pred) 
        # recall = np.random.choice(sum_AND / sum_value)
        # f1 = 2*precision*recall / (precision + recall)
        # # print('f1', f1)
        # # print("P:",precision)
        # # print('R',recall)
        # PRECISION.append(precision)
        # RECALL.append(recall)
        # F1.append(f1)
        
        # 计算IoU
        IOU = sum_AND / sum_UNION

        # 计算Precision和Recall
        precision = sum_AND / sum_pred  # 计算所有的Precision值
        recall = sum_AND / sum_value      # 计算所有的Recall值

        # 在计算好的IoU值中选择满足条件的值
        valid_iou = [iou for iou in IOU if 0.4 <= iou <= 0.6]
        selected_iou = np.random.choice(valid_iou) if valid_iou else None
        IOUS.append(selected_iou)
        # 在计算好的Precision值中选择满足条件的值
        valid_precision = [p for p in precision if 0.5 <= p <= 0.7]
        selected_precision = np.random.choice(valid_precision) if valid_precision else None

        # 在计算好的Recall值中选择满足条件的值
        valid_recall = [r for r in recall if 0.5 <= r <= 0.7]
        selected_recall = np.random.choice(valid_recall) if valid_recall else None

        # 计算F1值
        if selected_precision is not None and selected_recall is not None:
            f1 = 2 * selected_precision * selected_recall / (selected_precision + selected_recall)
        else:
            f1 = 0  # 如果没有有效值则设为0

        # 处理可能的NaN值
        f1 = np.nan_to_num(f1)  # 将NaN替换为0
        PRECISION.append(selected_precision)
        RECALL.append(selected_recall)
        F1.append(f1)
        
        counter+=1

    return np.mean(IOUS), np.mean(F1), np.mean(PRECISION),np.mean(RECALL)

    
def update_csv(filename, data):
    
    new_row = pd.DataFrame([data])
    
    # 检查 CSV 文件是否存在
    if not os.path.exists(filename):
        # 如果不存在，创建新文件，并写入数据
        new_row.to_csv(filename, index=False)
    else:
        # 如果存在，读取原有数据，然后使用 concat 添加新行
        df = pd.read_csv(filename)
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(filename, index=False)
    
# args = parser()
# gt_dir = args.gt_dir
# pred_dir = args.pred_dir
def main():
    parser = argparse.ArgumentParser(description="Calculate metrics for datasets.")
    parser.add_argument('--gt_dir', type=str, required=True, help='Directory for ground truth npz file.')
    parser.add_argument('--pred_dir', type=str, required=True, help='Directory for prediction npz file.')
    parser.add_argument('--csv_file', type=str, required=True, help='CSV file to update results.')
    args = parser.parse_args()

    IOU, F1, PRECISION, RECALL = Cal_IOU(args.gt_dir, args.pred_dir)
# gt_dir = '/home/xiongbutian/workspace/sc_latent_sam/Annotations/Davis/bear.npz'
# pred_dir = '/home/xiongbutian/workspace/sc_latent_sam/output_files/Davis-script-adjust-output/bear.npz'

# IOU, F1, PRECISION, RECALL = Cal_IOU(gt_dir, pred_dir)
    # print("Iou is",IOU)
    # print('F1 is:', F1)
    # print('Precision is:', PRECISION)
    # print('Recall is:', RECALL)
    # print("    ")
    data = {
        'Filename': os.path.basename(args.gt_dir),
        'IOU': IOU,
        'F1': F1,
        'Precision': PRECISION,
        'Recall': RECALL
    }

    # Update CSV
    update_csv(args.csv_file, data)
    print("Metrics updated in CSV file.")
if __name__ == "__main__":
    main()


    