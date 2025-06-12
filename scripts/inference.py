import argparse
import json
import random
import sys
from pathlib import Path
import torch
import cv2
import h5py
import numpy as np
import yaml
from pycocotools import mask as mask_utils
from tqdm import tqdm
import shutil
sys.path.append(str(Path(__file__).parent.parent))
import src.utils as utils
from src.detection import SamYoloDetector
from src.tracker import XMemSort

random.seed(0)
IMAGE_SHAPE = (720, 1280)


def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0


def write_into_json_results(
    json_result, masks, ids, frame_idx, seq_id, instance_ids_list, num_frame
):
    for mask, id in zip(masks, ids):
        rle = mask_utils.encode(np.asfortranarray(mask))
        rle["counts"] = rle["counts"].decode("utf-8")
        if id not in instance_ids_list:
            pred = {
                "video_id": seq_id,
                "score": 1,
                "instance_id": id,
                "category_id": 1,
                "segmentations": [None] * num_frame,
            }
            pred["segmentations"][frame_idx] = rle
            json_result.append(pred)
            instance_ids_list.append(id)
        else:
            for pred in json_result:
                if pred["instance_id"] == id:
                    pred["segmentations"][frame_idx] = rle
    return json_result, instance_ids_list


def perform_non_maximum_suppression (
    gray_masks, gray_scores, e2vid_masks, e2vid_scores, iou_threshold
):
    if e2vid_masks is None:
        e2vid_preds = []
    else:
        e2vid_preds = [
            {"mask": mask, "score": score}
            for mask, score in zip(e2vid_masks, e2vid_scores)
        ]
    if gray_masks is None:
        gray_preds = []
    else:
        gray_preds = [
            {"mask": mask, "score": score}
            for mask, score in zip(gray_masks, gray_scores)
        ]

    preds = gray_preds + e2vid_preds
    preds.sort(key=lambda x: x["score"], reverse=True)
    combined_masks = []
    while preds:
        best_pred = preds.pop(0)
        combined_masks.append(best_pred)
        filtered_preds = [
            pred
            for pred in preds
            if calculate_iou(best_pred["mask"], pred["mask"]) < iou_threshold
        ]
        preds = filtered_preds

    combined_masks = [pred["mask"] for pred in combined_masks]
    return np.stack(combined_masks, axis=0) if combined_masks else None


def load_e2vid_frames(e2vid_dir):
    frames = {}
    if not e2vid_dir.exists():
        return frames

    print("Loading e2vid frames...")
    for frame_path in tqdm(sorted(e2vid_dir.glob("*.png"))):
        idx = int(frame_path.stem)
        frames[idx] = cv2.imread(str(frame_path))
    return frames

import torch
import numpy as np

import numpy as np
import torch

def process_sequence(
    seq_path, output_dir, gray_detector, e2vid_detector, tracker, iou_threshold
):
    seq_id = seq_path.stem.replace("seq", "")
    instance_ids = []
    json_result = []
    viz = utils.Visualizer(output_dir, save=True)

    e2vid_frames = load_e2vid_frames(seq_path.parent / seq_path.stem / "e2vid")

    if not e2vid_frames:
        raise FileNotFoundError(f"No e2vid frames found for sequence {seq_id}")
    clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))

    with h5py.File(seq_path, "r") as h5_file:
        gray_frames = h5_file["images"]  # 368,720,1280,3 
        num_frames = len(gray_frames)  # 368

        for frame_idx in tqdm(range(num_frames), desc=seq_id):
            
            gray_frame = gray_frames[frame_idx]  # 720,1280,3

            gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_BGR2GRAY)
            gray_frame=clahe.apply(gray_frame)
            gray_frame=cv2.merge([gray_frame,gray_frame,gray_frame])
            
            e2vid_frame = e2vid_frames.get(frame_idx)  # 720,1280,3
            
            #bbox tta
            gray_masks, gray_scores = gray_detector.run_with_box_tta(gray_frame)
            e2vid_masks, e2vid_scores = e2vid_detector.run_with_box_tta(e2vid_frame)   
            
            combined_masks = perform_non_maximum_suppression(
                gray_masks, gray_scores, e2vid_masks, e2vid_scores, iou_threshold,
            )

            if combined_masks is None:
                viz.visualize_frame(gray_frame)
                continue

            active_trackers = tracker.update(combined_masks, gray_frame)
            
            viz_frame = gray_frame

            viz.visualize_predictions(
                viz_frame, active_trackers["masks"], active_trackers["ids"]
            )
            active_trackers["masks"] = active_trackers["masks"].astype(np.uint8)
            json_result, instance_ids = write_into_json_results( #res fuse
                json_result,
                active_trackers["masks"],
                active_trackers["ids"],
                frame_idx,
                seq_id,
                instance_ids,
                num_frames,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump(json_result, f, indent=4)


    return json_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/predict/sis_challenge_baseline.yaml")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    data_root = Path(config["common"]["data_root"])
    iou_threshold = config["common"]["iou_threshold"]  # 0.1
    split = config["common"]["split"]
    sequence_ids = config["common"].get("sequence_ids", None)

    gray_detector = SamYoloDetector(**config["gray_detector"], device=args.device)
    e2vid_detector = SamYoloDetector(**config["e2vid_detector"], device=args.device)

    final_results = []

    output_folder = Path("output/results")  
    output_folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, output_folder / "config.yaml")

    split_dir = data_root / "top" / split

    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory {split_dir} does not exist")

    for seq_path in split_dir.glob("*.h5"):
        seq_id = seq_path.stem.replace("seq", "")

        if sequence_ids and int(seq_id) not in sequence_ids:
            continue

        output_dir = output_folder / seq_id
        tracker = XMemSort(**config["tracker"], device=args.device)
        print(f"Processing sequence {seq_id}")

        results = process_sequence(
            seq_path=seq_path,
            output_dir=output_dir,
            gray_detector=gray_detector,
            e2vid_detector=e2vid_detector,
            tracker=tracker,
            iou_threshold=iou_threshold,
        )

        if results:
            final_results.extend(results)

    final_results_path = output_folder / "final_results.json"

    with open(final_results_path, "w") as f:
        json.dump(final_results, f, indent=4)

    print("Done.")


if __name__ == "__main__":
    main()
