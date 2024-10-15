import sys
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
from davis2017.davis import DAVIS
from davis2017.metrics import db_eval_boundary, db_eval_iou
from davis2017 import utils
from davis2017.results import Results
from scipy.optimize import linear_sum_assignment

from skimage import io
from skimage.measure import label, regionprops
from sklearn.metrics import jaccard_score, f1_score


class DAVISEvaluation(object):
    def __init__(self, davis_root, task, gt_set, sequences='all', codalab=False):
        """
        Class to evaluate DAVIS sequences from a certain set and for a certain task
        :param davis_root: Path to the DAVIS folder that contains JPEGImages, Annotations, etc. folders.
        :param task: Task to compute the evaluation, chose between semi-supervised or unsupervised.
        :param gt_set: Set to compute the evaluation
        :param sequences: Sequences to consider for the evaluation, 'all' to use all the sequences in a set.
        """
        self.davis_root = davis_root
        self.task = task
        self.dataset = DAVIS(root=davis_root, task=task, subset=gt_set, sequences=sequences, codalab=codalab)

    @staticmethod
    def _evaluate_semisupervised(all_gt_masks, all_res_masks, all_void_masks, metric):
        if all_res_masks.shape[0] > all_gt_masks.shape[0]:
            sys.stdout.write("\nIn your PNG files there is an index higher than the number of objects in the sequence!")
            sys.exit()
        elif all_res_masks.shape[0] < all_gt_masks.shape[0]:
            zero_padding = np.zeros((all_gt_masks.shape[0] - all_res_masks.shape[0], *all_res_masks.shape[1:]))
            all_res_masks = np.concatenate([all_res_masks, zero_padding], axis=0)
        j_metrics_res, f_metrics_res = np.zeros(all_gt_masks.shape[:2]), np.zeros(all_gt_masks.shape[:2])
        for ii in range(all_gt_masks.shape[0]):
            if 'J' in metric:
                j_metrics_res[ii, :] = db_eval_iou(all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks)
            if 'F' in metric:
                f_metrics_res[ii, :] = db_eval_boundary(all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks)
        return j_metrics_res, f_metrics_res

    @staticmethod
    def _evaluate_unsupervised(all_gt_masks, all_res_masks, all_void_masks, metric, max_n_proposals=20):
        if all_res_masks.shape[0] > max_n_proposals:
            sys.stdout.write(f"\nIn your PNG files there is an index higher than the maximum number ({max_n_proposals}) of proposals allowed!")
            sys.exit()
        elif all_res_masks.shape[0] < all_gt_masks.shape[0]:
            zero_padding = np.zeros((all_gt_masks.shape[0] - all_res_masks.shape[0], *all_res_masks.shape[1:]))
            all_res_masks = np.concatenate([all_res_masks, zero_padding], axis=0)
        j_metrics_res = np.zeros((all_res_masks.shape[0], all_gt_masks.shape[0], all_gt_masks.shape[1]))
        f_metrics_res = np.zeros((all_res_masks.shape[0], all_gt_masks.shape[0], all_gt_masks.shape[1]))
        for ii in range(all_gt_masks.shape[0]):
            for jj in range(all_res_masks.shape[0]):
                if 'J' in metric:
                    j_metrics_res[jj, ii, :] = db_eval_iou(all_gt_masks[ii, ...], all_res_masks[jj, ...], all_void_masks)
                if 'F' in metric:
                    f_metrics_res[jj, ii, :] = db_eval_boundary(all_gt_masks[ii, ...], all_res_masks[jj, ...], all_void_masks)
        if 'J' in metric and 'F' in metric:
            all_metrics = (np.mean(j_metrics_res, axis=2) + np.mean(f_metrics_res, axis=2)) / 2
        else:
            all_metrics = np.mean(j_metrics_res, axis=2) if 'J' in metric else np.mean(f_metrics_res, axis=2)
        row_ind, col_ind = linear_sum_assignment(-all_metrics)
        return j_metrics_res[row_ind, col_ind, :], f_metrics_res[row_ind, col_ind, :]
    
    # def find_best_mask(gt_mask_path, pred_mask_dir):
    #     gt_mask = io.imread(gt_mask_path) > 0
        
    #     best_mask = None
    #     highest_iou = 0
        
    #     for mask_file in sorted(os.listdir(pred_mask_dir)):
    #         pred_mask_path = os.path.join(pred_mask_dir,mask_file)
    #         pred_mask = io.imread(pred_mask_path) > 0
            
    #         gt_mask_flat = gt_mask.flatten()
    #         pred_mask_flat = pred_mask.flatten()
            
    #         iou = jaccard_score(gt_mask_flat, pred_mask_flat)
            
    #         if iou > highest_iou:
    #             highest_iou = iou
    #             best_mask = pred_mask
            
    #     return best_mask
    
    # def mkdir_best_mask(result_path, gt_mask_path, pred_mask_dir):
    #     dir_name = 'best_masks'
    #     os.mkdir(f"{result_path}/{dir_name}")
    #     for pred_masks in pred_mask_dir:
            
    def find_and_save_best_masks(gt_mask_path, pred_masks_dir, save_dir, img_code):
        """
        Find the best mask in terms of IoU with the ground truth mask and save it.

        Args:
            gt_mask_path (str): Path to the ground truth mask image.
            pred_masks_dir (str): Path to the directory containing predicted masks.
            save_dir (str): Directory to save the best mask.
            img_code (str): Code of the image for naming the output mask.
        """
        # Load GT mask
        gt_mask = io.imread(gt_mask_path) > 0  # Assuming binary mask

        # Initialize variables to store the best mask and highest IoU
        best_mask = None
        highest_iou = 0
        best_mask_file = None

        # Iterate over all predicted masks in the directory
        for mask_file in sorted(os.listdir(pred_masks_dir)):
            pred_mask_path = os.path.join(pred_masks_dir, mask_file)
            pred_mask = io.imread(pred_mask_path) > 0  # Assuming binary mask

            # Flatten masks for IoU calculation
            gt_mask_flat = gt_mask.flatten()
            pred_mask_flat = pred_mask.flatten()

            # Calculate IoU
            iou = jaccard_score(gt_mask_flat, pred_mask_flat)

            # Check if this mask is the best
            if iou > highest_iou:
                highest_iou = iou
                best_mask = pred_mask
                best_mask_file = mask_file

        if best_mask is not None:
            # Save the best mask
            save_path = os.path.join(save_dir, f"{img_code}_best_mask.png")
            io.imsave(save_path, best_mask.astype(np.uint8) * 255)
            print(f"Saved best mask for {img_code} with IoU {highest_iou} at {save_path}")
        else:
            print(f"No valid mask found for {img_code}")

    # def evaluate(self, res_path, metric=('J', 'F'), debug=False):
    #     metric = metric if isinstance(metric, tuple) or isinstance(metric, list) else [metric]
    #     if 'T' in metric:
    #         raise ValueError('Temporal metric not supported!')
    #     if 'J' not in metric and 'F' not in metric:
    #         raise ValueError('Metric possible values are J for IoU or F for Boundary')

    #     # Containers
    #     metrics_res = {}
    #     if 'J' in metric:
    #         metrics_res['J'] = {"M": [], "R": [], "D": [], "M_per_object": {}}
    #     if 'F' in metric:
    #         metrics_res['F'] = {"M": [], "R": [], "D": [], "M_per_object": {}}

    #     # Sweep all sequences
    #     results = Results(root_dir=res_path)
    #     for seq in tqdm(list(self.dataset.get_sequences())):
    #         all_gt_masks, all_void_masks, all_masks_id = self.dataset.get_all_masks(seq, True)
    #         if self.task == 'semi-supervised':
    #             all_gt_masks, all_masks_id = all_gt_masks[:, 1:-1, :, :], all_masks_id[1:-1]
    #         all_res_masks = results.read_masks(seq, all_masks_id)
    #         if self.task == 'unsupervised':
    #             j_metrics_res, f_metrics_res = self._evaluate_unsupervised(all_gt_masks, all_res_masks, all_void_masks, metric)
    #         elif self.task == 'semi-supervised':
    #             j_metrics_res, f_metrics_res = self._evaluate_semisupervised(all_gt_masks, all_res_masks, None, metric)
    #         for ii in range(all_gt_masks.shape[0]):
    #             seq_name = f'{seq}_{ii+1}'
    #             if 'J' in metric:
    #                 [JM, JR, JD] = utils.db_statistics(j_metrics_res[ii])
    #                 metrics_res['J']["M"].append(JM)
    #                 metrics_res['J']["R"].append(JR)
    #                 metrics_res['J']["D"].append(JD)
    #                 metrics_res['J']["M_per_object"][seq_name] = JM
    #             if 'F' in metric:
    #                 [FM, FR, FD] = utils.db_statistics(f_metrics_res[ii])
    #                 metrics_res['F']["M"].append(FM)
    #                 metrics_res['F']["R"].append(FR)
    #                 metrics_res['F']["D"].append(FD)
    #                 metrics_res['F']["M_per_object"][seq_name] = FM

    #         # Show progress
    #         if debug:
    #             sys.stdout.write(seq + '\n')
    #             sys.stdout.flush()
    #     return metrics_res
   
    def evaluate(self, res_path, metric=('J', 'F'), debug=False):
        metric = metric if isinstance(metric, tuple) or isinstance(metric, list) else [metric]
        if 'T' in metric:
            raise ValueError('Temporal metric not supported!')
        if 'J' not in metric and 'F' not in metric:
            raise ValueError('Metric possible values are J for IoU or F for Boundary')

        # Containers
        metrics_res = {}
        if 'J' in metric:
            metrics_res['J'] = {"M": [], "R": [], "D": [], "M_per_object": {}}
        if 'F' in metric:
            metrics_res['F'] = {"M": [], "R": [], "D": [], "M_per_object": {}}

        # Directory to save best masks
        best_mask_dir = os.path.join(res_path,'best_masks')
        os.makedirs(best_mask_dir,exist_ok=True)
        
        # Sweep all sequences
        results = Results(root_dir=res_path)
        for seq in tqdm(list(self.dataset.get_sequences())):
            all_gt_masks, all_void_masks, all_masks_id = self.dataset.get_all_masks(seq, True)
            if self.task == 'semi-supervised':
                all_gt_masks, all_masks_id = all_gt_masks[:, 1:-1, :, :], all_masks_id[1:-1]
            all_res_masks = results.read_masks(seq, all_masks_id)
            if self.task == 'unsupervised':
                j_metrics_res, f_metrics_res = self._evaluate_unsupervised(all_gt_masks, all_res_masks, all_void_masks, metric)
            elif self.task == 'semi-supervised':
                j_metrics_res, f_metrics_res = self._evaluate_semisupervised(all_gt_masks, all_res_masks, None, metric)
            for ii in range(all_gt_masks.shape[0]):
                seq_name = f'{seq}_{ii+1}'
                gt_mask_path = f'/home/xiongbutian/workspace/davis2017-evaluation/DAVIS/Annotations/480p/blackswan'
                pred_masks_dir = os.path.join(res_path, seq, str(ii+1))
                
                self.find_and_save_best_masks(gt_mask_path, pred_masks_dir, best_mask_dir, seq_name)
                
                # exit()
                if 'J' in metric:
                    [JM, JR, JD] = utils.db_statistics(j_metrics_res[ii])
                    metrics_res['J']["M"].append(JM)
                    metrics_res['J']["R"].append(JR)
                    metrics_res['J']["D"].append(JD)
                    metrics_res['J']["M_per_object"][seq_name] = JM
                if 'F' in metric:
                    [FM, FR, FD] = utils.db_statistics(f_metrics_res[ii])
                    metrics_res['F']["M"].append(FM)
                    metrics_res['F']["R"].append(FR)
                    metrics_res['F']["D"].append(FD)
                    metrics_res['F']["M_per_object"][seq_name] = FM

            # Show progress
            if debug:
                sys.stdout.write(seq + '\n')
                sys.stdout.flush()
        return metrics_res
