import numpy as np
import torch
import ultralytics
from sam2.build_sam import build_sam2

# from transformers import SamModel, SamProcessor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from scipy.optimize import linear_sum_assignment
from ..utils import suppress_stdout_stderr


class SamYoloDetector:
    def __init__(self, yolo_path, device="cuda:3") -> None:
        self.detector = ultralytics.YOLO(yolo_path)
        # self.sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
        # self.sam_model = SamModel.from_pretrained("/data/hj_data/CVPR/Mouse/MouseSIS-main/models/sam-vit-huge").to(device)
        # self.sam_processor = SamProcessor.from_pretrained("/data/hj_data/CVPR/Mouse/MouseSIS-main/models/sam-vit-huge")

        self.sam2_model = build_sam2(
            "configs/sam2.1/sam2.1_hiera_l.yaml",
            "/data/hj_data/CVPR/Mouse/MouseSIS-0.1.0/pretrained/sam2/sam2.1_hiera_large.pt",
            device=device,
        )

        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
        self.device = device

    def run(self, img):
        with suppress_stdout_stderr():
            result = self.detector(img)[0]

        boxes = result.boxes.xyxy.detach().cpu().numpy()  # x1, y1, x2, y2  (1,4)
        scores = result.boxes.conf.detach().cpu().numpy()  # (1,)

        if not len(boxes):
            return None, None

        boxes_list = [[boxes.tolist()]]
        # inputs = self.sam_processor(img.transpose(2, 0, 1), input_boxes=[boxes_list], return_tensors="pt").to(self.device)

        with torch.no_grad():
            self.sam2_predictor.set_image(img)
            # outputs = self.sam_model(**inputs)

            # masks = self.sam_processor.image_processor.post_process_masks(
            #     outputs.pred_masks.cpu(),
            #     inputs["original_sizes"].cpu(),
            #     inputs["reshaped_input_sizes"].cpu()
            # )[0]
            masks, iou_scores, logits = self.sam2_predictor.predict(
                point_coords=None, point_labels=None, box=[boxes_list]
            )  # numpy array: (2, 3, 720, 1280)
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

        return selected_masks.cpu().numpy(), scores

    def apply_tta(self, img, mode):
        if mode == "none":
            return img
        elif mode == "hflip":
            return np.flip(img, axis=1).copy()
        # elif mode == "vflip":
        #     return np.flip(img, axis=0).copy()
        # elif mode == "hvflip":
        #     return np.flip(np.flip(img, axis=0), axis=1).copy()
        # elif mode == "rot90":
        #     return np.rot90(img, k=1).copy()
        # elif mode == "rot180":
        #     return np.rot90(img, k=2).copy()
        # elif mode == "rot270":
        #     return np.rot90(img, k=3).copy()
        # else:
        #     raise ValueError(f"Unknown TTA mode: {mode}")

    def reverse_tta(self, mask, mode):
        # print(mask.shape)
        if mode == "none":
            return mask
        elif mode == "hflip":
            return np.flip(mask, axis=1).copy()
        # elif mode == "vflip":
        #     return np.flip(mask, axis=0).copy()
        # elif mode == "hvflip":
        #     return np.flip(np.flip(mask, axis=0), axis=1).copy()
        # elif mode == "rot90":
        #     return np.rot90(mask, k=3).copy()
        # elif mode == "rot180":
        #     return np.rot90(mask, k=2).copy()
        # elif mode == "rot270":
        #     return np.rot90(mask, k=1).copy()
        # else:
        #     raise ValueError(f"Unknown  TTA mode:{mode}")

    def extract_mask_embedding(self,img, mask):
        """提取每个mask区域的平均颜色作为特征（简单版本）"""
        if mask.sum() == 0:
            return np.zeros(3)
        masked_pixels = img[mask.astype(bool)]
        return masked_pixels.mean(axis=0)  # RGB 平均值

    def iou(self,mask1,mask2):
        intersection=np.logical_and(mask1,mask2).sum()
        union=np.logical_or(mask1,mask2).sum()
        return intersection/(union+1e-6)
    
    def cosine_similarity(self,a, b):
        norm_a = np.linalg.norm(a) + 1e-6
        norm_b = np.linalg.norm(b) + 1e-6
        return np.dot(a, b) / (norm_a * norm_b)
    
    def run_with_tta(self, img):
        # ref_masks=[ self.reverse_tta(m,'none') for m in self.run(self.apply_tta(img,'none'))[0]]
        detect_masks=self.run(self.apply_tta(img, "none"))[0]
        if detect_masks is None:
            return None,None
        ref_masks = [m for m in self.run(self.apply_tta(img, "none"))[0]]
        num_objs=len(ref_masks)
        
        tta_modes = ["none", "hflip"] #,"rot90","rot180","rot270"
        all_tta_masks_per_obj = [[] for _ in range(num_objs)]  # list of list: [ [m0_none, m0_hflip], [m1_none, m1_hflip], ... ]
        all_tta_scores_per_obj = [[] for _ in range(num_objs)]

        # num_objects = None
        alpha = 0.7 
        for mode in tta_modes:
            aug_img = self.apply_tta(img, mode)
            masks, scores = self.run(aug_img)
            if masks is None:
                continue
            recovered_masks=[self.reverse_tta(m,mode) for m in masks ]
            
            # 提取所有 ref_masks 和 recovered_masks 的 embedding
            ref_embeddings = [self.extract_mask_embedding(img, m) for m in ref_masks]
            rec_embeddings = [self.extract_mask_embedding(img, m) for m in recovered_masks]
            
            #compute iou to match
            cost_matrix=np.zeros((num_objs,len(recovered_masks)))
            
            for i in range(num_objs):
                for j in range(len(recovered_masks)):
                    iou_score = self.iou(ref_masks[i], recovered_masks[j])
                    emb_sim = self.cosine_similarity(ref_embeddings[i], rec_embeddings[j])
                    # cost_matrix[i][j]=-self.iou(ref_masks[i],recovered_masks[j])
                    cost_matrix[i, j] = - (alpha * iou_score + (1 - alpha) * emb_sim)
            
            row_ind,col_ind=linear_sum_assignment(cost_matrix) #在两个集合之间找到元素间最优的匹配关系
            for i,j in zip(row_ind,col_ind):
                if -cost_matrix[i,j]<0.3:
                    continue
                all_tta_masks_per_obj[i].append(recovered_masks[j])
                all_tta_scores_per_obj[i].append(scores[j])

        final_masks = []
        final_scores = []

        for i in range(num_objs):
            if len(all_tta_masks_per_obj[i]) == 0:
                continue
            masks_i = np.stack(all_tta_masks_per_obj[i], axis=0)
            scores_i = np.array(all_tta_scores_per_obj[i])
            weights = scores_i / (np.sum(scores_i) + 1e-6)
            fused_mask = np.sum(masks_i * weights[:, None, None], axis=0)
            fused_mask = (fused_mask >= 0.5).astype(np.uint8)

            final_masks.append(fused_mask)
            final_scores.append(np.max(scores_i))

        return np.stack(final_masks, axis=0), np.array(final_scores)
 # def run_with_tta(self, img):
    #     tta_modes = ["none", "hflip"]  # ", "vflip", "hvflip"
    #     all_masks = []
    #     all_scores = []

    #     for mode in tta_modes:
    #         aug_img = self.apply_tta(img, mode)
    #         masks, scores = self.run(aug_img)
    #         if masks is None:
    #             continue
    #         # masks: N,H,W
    #         for i in range(masks.shape[0]):
    #             recovered_mask = self.reverse_tta(masks[i], mode)  # shape:[H,W]
    #             all_masks.append(recovered_mask)
    #             all_scores.append(scores[i])

    #     if len(all_masks) == 0:
    #         return None, None

    #     all_masks = np.stack(all_masks, axis=0)
    #     all_scores = np.array(all_scores)  # (N,)

    #     weights = all_scores / (np.sum(all_scores) + 1e-6)
    #     fused_mask = np.sum(all_masks * weights[:, None, None], axis=0)
    #     fused_mask = (fused_mask >= 0.5).astype(np.uint8)

    #     return np.expand_dims(fused_mask, axis=0), np.array([np.max(all_scores)])
    
    # def mask_to_bbox(self,mask):
    #     """
    #     将二值 mask 转换为 bounding box (x_min, y_min, x_max, y_max)。

    #     Args:
    #         mask: 二值 numpy 数组，表示 mask。

    #     Returns:
    #         bbox: 包含 bounding box 坐标的列表 [x_min, y_min, x_max, y_max]。 如果 mask 为空，则返回 None。
    #     """
    #     if np.sum(mask) == 0:  # 如果 mask 为空
    #         return None

    #     rows, cols = np.where(mask)
    #     x_min = np.min(cols)
    #     y_min = np.min(rows)
    #     x_max = np.max(cols)
    #     y_max = np.max(rows)
    #     return [x_min, y_min, x_max, y_max]

    # def calculate_eiou(self,mask1, mask2):
    #     """
    #     计算两个 mask 的 EIoU (Efficient Intersection over Union)。

    #     Args:
    #         mask1: 第一个 mask，numpy 数组。
    #         mask2: 第二个 mask，numpy 数组。

    #     Returns:
    #         eiou: EIoU 值。 如果任何一个 mask 为空，则返回 0.0。
    #     """

    #     bbox1 = self.mask_to_bbox(mask1)
    #     bbox2 = self.mask_to_bbox(mask2)

    #     if bbox1 is None or bbox2 is None:
    #         return 0.0

    #     # bbox 格式 [x_min, y_min, x_max, y_max]
    #     x1_min, y1_min, x1_max, y1_max = bbox1
    #     x2_min, y2_min, x2_max, y2_max = bbox2

    #     # 计算交集区域
    #     x_intersection_min = max(x1_min, x2_min)
    #     y_intersection_min = max(y1_min, y2_min)
    #     x_intersection_max = min(x1_max, x2_max)
    #     y_intersection_max = min(y1_max, y2_max)

    #     intersection_width = max(0, x_intersection_max - x_intersection_min)
    #     intersection_height = max(0, y_intersection_max - y_intersection_min)
    #     intersection_area = intersection_width * intersection_height

    #     # 计算 bounding box 的面积
    #     bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    #     bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)

    #     # 计算并集区域
    #     union_area = bbox1_area + bbox2_area - intersection_area

    #     # 计算 IoU
    #     iou = intersection_area / union_area if union_area > 0 else 0.0

    #     # 计算覆盖两个 bounding box 的最小 bounding box
    #     x_enclosing_min = min(x1_min, x2_min)
    #     y_enclosing_min = min(y1_min, y2_min)
    #     x_enclosing_max = max(x1_max, x2_max)
    #     y_enclosing_max = max(y1_max, y2_max)

    #     enclosing_width = x_enclosing_max - x_enclosing_min
    #     enclosing_height = y_enclosing_max - y_enclosing_min

    #     # 计算 EIoU 的组成部分
    #     w1 = x1_max - x1_min
    #     h1 = y1_max - y1_min
    #     w2 = x2_max - x2_min
    #     h2 = y2_max - y2_min

    #     wiou = (w1 - w2)**2 / (enclosing_width**2 + 1e-7)  # 添加一个小的 epsilon 防止除以零
    #     hiou = (h1 - h2)**2 / (enclosing_height**2 + 1e-7)

    #     eiou = iou - 0.5 * (wiou + hiou)

    #     return eiou