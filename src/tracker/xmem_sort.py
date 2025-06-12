"""
A simple Multi-object tracker relying on XMem
heavily inspired by
SORT: A Simple, Online and Realtime Tracker
"""

from __future__ import print_function

import numpy as np
from src.vos import VosPredictor

np.random.seed(0)


def linear_assignment(cost_matrix):
    try:
        import lap

        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment

        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    o = wh / (
        (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
        - wh
    )
    return o


def iou_masks(masks_test, masks_gt):
    """
    Computes IoU between two sets of binary masks.

    Args:
        masks_test (numpy.ndarray): Binary masks of shape [n1, height, width].
        masks_gt (numpy.ndarray): Binary masks of shape [n2, height, width].

    Returns:
        numpy.ndarray: IoU matrix of shape [n1, n2].
    """
    n1, height, width = masks_test.shape
    n2 = masks_gt.shape[0]

    iou_matrix = np.zeros((n1, n2), dtype=np.float32)

    for i in range(n1):
        mask_test = masks_test[i]
        for j in range(n2):
            mask_gt = masks_gt[j]
            intersection = np.sum(
                (mask_test.astype(bool) & mask_gt.astype(bool)).astype(np.float32)
            )
            union = np.sum(
                (mask_test.astype(bool) | mask_gt.astype(bool)).astype(np.float32)
            )

            if union == 0:
                iou = 0.0
            else:
                iou = intersection / union

            iou_matrix[i, j] = iou

    return iou_matrix


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score == None:
        return np.array(
            [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]
        ).reshape((1, 4))
    else:
        return np.array(
            [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]
        ).reshape((1, 5))


def associate_detections_to_trackers(detected_masks, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as binary instance masks)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if len(trackers) == 0:
        return [], np.arange(len(detected_masks)), []

    iou_matrix = iou_masks(detected_masks, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detected_masks):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


# 代表一个正在跟踪的单个目标实例（object instance）的简单追踪器
class XMemSimpleMaskTracker(object):
    """Simple Tracker representing one object instance.
    The state is a binary mask. Predictions are done with a VOS model
    like XMem.
    """

    count = 1  # 是所有 Tracker 的全局计数器，每个新建的 Tracker 会有一个独一无二的 id

    def __init__(self, mask, initial_image, device="cuda:2") -> None:
        self.state = mask  # 当前追踪的掩膜状态
        self.image = initial_image  # 当前参考图

        self.time_since_update = 0  # 距离上次更新过去了几帧
        self.id = XMemSimpleMaskTracker.count  # tracker的唯一编号
        XMemSimpleMaskTracker.count += 1
        self.history = []  # 可以保存历史状态（这里没怎么用）
        self.hits = 0  # 追踪成功累计次数
        self.hit_streak = 0  # 连续追踪成功的次数
        self.age = 0  # tracker存在的总帧数
        self.device = device

    # 当这个目标在当前帧有检测到匹配时
    def update(self, new_mask, new_image):
        self.time_since_update = 0  # 重置 time_since_update = 0
        self.history = []  # 清空 history
        self.hits += 1  # hits 加1（追踪到一次）
        self.hit_streak += 1  # hit_streak 加1（连续命中）
        # 更新当前状态：
        self.state = new_mask
        self.image = new_image

    # 如果当前帧没有检测（即没有直接匹配）
    def predict(self, image):
        predictor = (  # 靠自己的推理模型 VosPredictor 预测出下一个位置
            VosPredictor(  # 用 VOS (Video Object Segmentation)模型 XMem 来做预测
                self.image,  # 传入上一fram的image
                self.state,  # 传入上一frame的mask
                device=self.device,
                model_path="/data/hj_data/CVPR/Mouse/MouseSIS-main/models/XMem-no-sensory.pth",  # 加载 XMem 模型
            )
        )
        pred_mask = predictor.step(image)  # 推理当前帧的掩膜

        self.image = image  # 当前帧的更新
        self.state = pred_mask  # 更新推理得到的mask

        self.age += 1  # 更新tricker的追踪总frame

        if self.time_since_update > 0:  # 如果 time_since_update > 0，意味着中间有断开
            self.hit_streak = 0

        self.time_since_update += 1
        return pred_mask  # 返回新的预测掩膜 pred_mask

    def get_state(self):
        return self.state


class XMemSort(object):
    # 把 iou_threshold 调高一点（比如0.5）， max_age可调大一点（比如2-3）容忍目标短时丢失或变化
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3, device="cuda:2"):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age  # 1 Tracker 最多允许连续 1 帧没匹配到新检测，否则被删掉
        self.min_hits = (
            min_hits  # 3 Tracker 必须连续匹配成功 3 次，才正式认为是“有效的追踪”
        )
        self.iou_threshold = iou_threshold  # 0.3 检测到的 mask 和已有 tracker 的 mask 的 IoU 超过 0.3，才认为是同一个物体
        self.trackers = []  # 当前在追踪的所有目标对象
        self.frame_count = 0  # 记录目前是第几帧
        assert self.max_age > 0
        self.device = device

    # 检测到的masks → 和现有trackers匹配 → 更新老trackers → 新建未匹配的新trackers → 删掉过期trackers → 返回当前活跃的masks+ids
    def update(self, detected_masks, curr_image):
        """Update the tracker by one step.

        Args:
            pred_masks: [n_instances, height, width]
            scores: [n_instances]
            prev_image: [height, width, 3]
            curr_image: [height, width, 3]

        Returns:
            dictionary with:
              masks of active_trackers: [n_active_trackers, height, width],
              list of ids: [n_active_trackers]
        """
        self.frame_count += 1

        trks = []
        active_trackers = {"masks": [], "ids": []}

        for i, trk in enumerate(
            self.trackers
        ):  # 每个 tracker 根据当前帧的图像 curr_image，预测它的当前位置（输出一个 mask）
            trks.append(trk.predict(curr_image))

        trks = np.array(trks)
        assert len(trks) == len(self.trackers)

        # matched: list of index pairs [det_idx, trk_idx]
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            detected_masks, trks, self.iou_threshold
        )

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(detected_masks[m[0]], curr_image)

        # create and initialise new trackers for unmatched detections
        for i in (
            unmatched_dets
        ):  # 对于新的检测（无法匹配到任何旧 tracker），新建一个 tracker 对象
            trk = XMemSimpleMaskTracker(
                detected_masks[i], curr_image, device=self.device
            )
            self.trackers.append(trk)

        i = len(self.trackers)

        for trk in reversed(self.trackers):  # 筛选出活跃的 trackers
            curr_mask = trk.get_state()
            if (
                (trk.time_since_update < 1)
                and (  # 如果 time_since_update < 1（即最近更新过）
                    trk.hit_streak >= self.min_hits
                    or self.frame_count
                    <= self.min_hits  # 并且 hit_streak >= min_hits（连续命中次数足够），或者一开始还在热身阶段
                )
            ):  # 把这个 tracker 当前的 mask 和 id 记到活跃列表中
                active_trackers["masks"].append(curr_mask)
                active_trackers["ids"].append(trk.id)
            i -= 1
            # remove dead tracklet
            if (
                trk.time_since_update > self.max_age
            ):  # 如果 tracker 太久没有更新（超过 max_age 帧），就把它删掉
                self.trackers.pop(i)

        active_trackers["masks"] = np.array(
            active_trackers["masks"]
        )  # 所有活跃目标的 masks和id
        return active_trackers
