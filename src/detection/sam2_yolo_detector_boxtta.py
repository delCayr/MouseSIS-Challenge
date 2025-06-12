import numpy as np
import torch
import ultralytics
from sam2.build_sam import build_sam2
from .TTAUtils import TTAUtils
# from transformers import SamModel, SamProcessor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from scipy.optimize import linear_sum_assignment
from ..utils import suppress_stdout_stderr


class SamYoloDetector:
    def __init__(self, yolo_path, device="cuda:0") -> None:
        self.detector = ultralytics.YOLO(yolo_path)
        self.sam2_model = build_sam2(
            "configs/sam2.1/sam2.1_hiera_l.yaml",
            "models/checkpoint_40.pt",
            device=device,
        )

        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
        self.device = device

    
    def run_yolo(self, img):
        with suppress_stdout_stderr():
            result = self.detector(img)[0]
        boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        return boxes_xyxy, scores

    def calculate_box_iou(self, boxA_xyxy, boxB_xyxy): # Standard IoU
        xA = np.maximum(boxA_xyxy[0], boxB_xyxy[0])
        yA = np.maximum(boxA_xyxy[1], boxB_xyxy[1])
        xB = np.minimum(boxA_xyxy[2], boxB_xyxy[2])
        yB = np.minimum(boxA_xyxy[3], boxB_xyxy[3])
        interArea = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)
        boxAArea = (boxA_xyxy[2] - boxA_xyxy[0]) * (boxA_xyxy[3] - boxA_xyxy[1])
        boxBArea = (boxB_xyxy[2] - boxB_xyxy[0]) * (boxB_xyxy[3] - boxB_xyxy[1])
        iou = interArea / (boxAArea + boxBArea - interArea + 1e-7)
        return iou
    
    def run_with_box_tta(self,img,yolo_conf_thresh=0.25, iou_match_thresh=0.3):  
        H_orig, W_orig = img.shape[:2]
        self.tta_utils = TTAUtils(img_shape=(H_orig, W_orig)) # Initialize TTAUtils

        ref_boxes_xyxy, ref_scores = self.run_yolo(img)
        keep_ref = ref_scores >= yolo_conf_thresh
        ref_boxes_xyxy = ref_boxes_xyxy[keep_ref]
        ref_scores = ref_scores[keep_ref]

        if not len(ref_boxes_xyxy):
            return None,None
        
        num_ref_objs=len(ref_boxes_xyxy) #有几个框
        
        #store all recovered boxes corresponding to each ref_box
        all_recovered_boxes_for_ref = [[ref_boxes_xyxy[i]] for i in range(num_ref_objs)] # Start with original
        all_recovered_scores_for_ref = [[ref_scores[i]] for i in range(num_ref_objs)]  # Start with original
        
        tta_configs = [
            {"name": "hflip", "mode": "hflip"},
            {"name": "scale_1.2", "mode": "scale", "params": {"scale_factor": 1.2}},
            {"name": "rotate_3", "mode": "rotate", "params": {"angle": 3}},
            {"name": "rotate_-3", "mode": "rotate", "params": {"angle": -3}},
            {"name": "hflip_scale_1.2", "mode": "hflip_scale", "params": {"scale_factor": 1.2}},
        ]
        
        for config in tta_configs:
            mode_str = config["mode"]
            params = config.get("params", {})
            scale_factor = params.get("scale_factor", 1.0)
            angle = params.get("angle", 0)

            img_aug, (H_aug, W_aug) = self.tta_utils.apply_transform(img, mode_str, 
                                                                     scale_factor=scale_factor, 
                                                                     angle=angle)
            
            aug_boxes_xyxy, aug_scores = self.run_yolo(img_aug)
            keep_aug = aug_scores >= yolo_conf_thresh
            aug_boxes_xyxy = aug_boxes_xyxy[keep_aug]
            aug_scores = aug_scores[keep_aug]

            if not len(aug_boxes_xyxy): continue

            recovered_boxes_xyxy = self.tta_utils.reverse_transform_boxes(
                aug_boxes_xyxy, mode_str, (H_aug, W_aug), 
                scale_factor_applied=scale_factor, angle_applied=angle
            )

            if len(recovered_boxes_xyxy) > 0 and num_ref_objs > 0:
                cost_matrix = np.zeros((num_ref_objs, len(recovered_boxes_xyxy)))
                for i in range(num_ref_objs):
                    for j in range(len(recovered_boxes_xyxy)):
                        iou_val = self.calculate_box_iou(ref_boxes_xyxy[i], recovered_boxes_xyxy[j])
                        cost_matrix[i, j] = -iou_val
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                for r, c in zip(row_ind, col_ind):
                    if -cost_matrix[r, c] >= iou_match_thresh:
                        
                        ref_box_current = ref_boxes_xyxy[r]
                        recovered_box_current = recovered_boxes_xyxy[c]                        
                        # 计算面积
                        ref_area = (ref_box_current[2] - ref_box_current[0]) * (ref_box_current[3] - ref_box_current[1])
                        rec_area = (recovered_box_current[2] - recovered_box_current[0]) * (recovered_box_current[3] - recovered_box_current[1])


                        if rec_area > ref_area * 1.5:#or rec_area < ref_area * 0.4
                            # print(f"TTA Box size outlier: ref_area={ref_area:.0f}, rec_area={rec_area:.0f}. Skipping for fusion with ref_obj {r}.")
                            continue                         
                        all_recovered_boxes_for_ref[r].append(recovered_boxes_xyxy[c])
                        all_recovered_scores_for_ref[r].append(aug_scores[c])

        final_fused_boxes_for_sam = []
        final_fused_box_scores = []
        for i in range(num_ref_objs):
            if not all_recovered_boxes_for_ref[i]: continue
            boxes_to_fuse = np.array(all_recovered_boxes_for_ref[i])
            scores_for_fusion = np.array(all_recovered_scores_for_ref[i])
            if len(boxes_to_fuse) > 0:
                weights = scores_for_fusion / (np.sum(scores_for_fusion) + 1e-7)
                fused_box = np.sum(boxes_to_fuse * weights[:, np.newaxis], axis=0)
                final_fused_boxes_for_sam.append(fused_box.tolist())
                final_fused_box_scores.append(np.max(scores_for_fusion))

        if not final_fused_boxes_for_sam:
            print("No objects after TTA box fusion.")
            return None, None
            
        sam_input_boxes = [final_fused_boxes_for_sam]
        with torch.no_grad():
            self.sam2_predictor.set_image(img)

            masks, iou_scores,logits = self.sam2_predictor.predict(
                point_coords=None, point_labels=None, box=sam_input_boxes)

            
            masks = torch.tensor(masks)
            iou_scores = torch.tensor(iou_scores)
            # iou_scores = outputs.iou_scores.cpu()[0]
            if len(masks.shape) < 4:
                masks = masks.unsqueeze(dim=0)
            num_instances, nb_predictions, height, width = masks.shape  # H,W  720 1280
            if len(iou_scores.shape) < 2:
                iou_scores = iou_scores.unsqueeze(dim=0)  # torch.Size([2, 3])
            max_indices = iou_scores.argmax(dim=1, keepdim=True)  # torch.Size([2, 1])

            # max_indices=max_indices = np.argmax(iou_scores, axis=1).reshape(-1, 1)
            gather_indices = max_indices[..., None, None].expand(
                -1, 1, height, width
            )  # torch.Size([2, 1, 720, 1280])
            selected_masks = torch.gather(
                masks, 1, gather_indices
            ).squeeze(
                1
            )  # masks: torch.Size([2, 3, 720, 1280])  evi2d:torch.Size([1, 3, 720, 1280])

        return selected_masks.cpu().numpy(), final_fused_box_scores