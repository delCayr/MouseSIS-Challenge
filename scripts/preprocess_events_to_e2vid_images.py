import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import h5py
from evlib.processing.reconstruction import E2Vid

IMAGE_SHAPE = (720, 1280)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/MouseSIS')
    return parser.parse_args()

def process_sequence(seq_path: Path, reconstructor: E2Vid):
    seq_folder = seq_path.parent / seq_path.stem
    output_dir = seq_folder / 'e2vid'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {seq_path.name}")
    with h5py.File(seq_path, 'r') as f:
        events = np.stack([f["y"][0:-1], f["x"][0:-1], f["t"][0:-1], f["p"][0:-1]], axis=-1)
        ev_indices = f['img2event'][:]
        ev_indices = np.concatenate(([0], ev_indices))  # For first frame uses all events before first image timestamp

        for i, (start, end) in enumerate(tqdm(zip(ev_indices[:-1], ev_indices[1:]), total=len(ev_indices)-1)):
            e2vid = reconstructor(events[start:end])
            output_path = output_dir / f"{str(i).zfill(8)}.png"
            cv2.imwrite(str(output_path), e2vid)

def main():
    args = parse_args()
    data_root = Path(args.data_root)
    source_dir = data_root / "top"

    for split_dir in source_dir.iterdir():
        if not split_dir.is_dir():
            continue

        for seq_path in split_dir.glob('*.h5'):
            reconstructor = E2Vid(image_shape=IMAGE_SHAPE, use_gpu=True)
            process_sequence(seq_path, reconstructor)

if __name__ == '__main__':
    main()
