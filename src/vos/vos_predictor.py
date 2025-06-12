import numpy as np
import torch
from absl import logging
from torchvision import transforms

from .XMem.dataset.range_transform import im_normalization
from .XMem.inference.inference_core import InferenceCore
from .XMem.model.network import XMem

# Copied from XMem, mostly unused
CONFIG = config_dict = {
    "model": "saves/XMem.pth",
    "d16_path": "../DAVIS/2016",
    "d17_path": "../DAVIS/2017",
    "y18_path": "../YouTube2018",
    "y19_path": "../YouTube",
    "lv_path": "../long_video_set",
    "generic_path": "../event_mice_top_12",
    "dataset": "G",
    "split": "val",
    "output": "../output/event_mice_12",
    "save_all": True,
    "benchmark": False,
    "disable_long_term": False,
    "max_mid_term_frames": 10,
    "min_mid_term_frames": 5,
    "max_long_term_elements": 10000,
    "num_prototypes": 128, #256
    "top_k": 30,
    "mem_every": 5, #3
    "deep_update_every": -1, #0.2
    "save_scores": False,
    "flip": False,
    "size": -1,
    "enable_long_term": True,
    "enable_long_term_count_usage": False,
}


class VosPredictor:
    def __init__(
        self,
        initial_image,
        initial_mask: np.ndarray,
        model_path="models/XMem-no-sensory.pth",
        device="cuda:2",
    ) -> None:
        """_summary_

        Args:
            initial_image: first frame of the video as numpy array (H, W, 3)
            initial_mask: binary mask {0, 1} of object in first frame as numpy array (H, W).
            model_path: _description_. Defaults to "artifacts/XMem.pth".
        """
        self.device = device
        network = XMem(CONFIG, model_path).to(self.device).eval()
        model_weights = torch.load(model_path)
        network.load_weights(model_weights, init_as_zero_if_needed=True)
        self.processor = InferenceCore(network, config=CONFIG)  # 推理的类

        self.transform = self.im_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                im_normalization,
            ]
        )

        self._init_processor_with_mask(initial_image, initial_mask) #传入上1 frame img and mask

    def _init_processor_with_mask(self, initial_image, initial_mask):
        assert initial_mask is not None

        initial_mask = torch.from_numpy(initial_mask).float().to(self.device)[None]
        labels = range(1, 2)  # only one label (mice) represented by '1'
        self.processor.set_all_labels([1])

        self.step(initial_image, gt_mask=initial_mask, labels=labels, end=False)
        logging.debug("Initialized processor with first mask!")

    def step(
        self, image, gt_mask=None, labels=None, end=False
    ):  # 输入图像预测接下来的frame
        """Predicting the next binary mask with the next frame of the video.

        Args:
            image: the next frame of the video as numpy array (H, W, 3)

        Returns:
            binary prediction mask
        """
        image = self.transform(image).to(self.device)  # (720, 1280, 3)

        prob = self.processor.step(image, gt_mask, labels, end=end)

        out_mask = torch.argmax(prob, dim=0)
        out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)  # torch.Size([720, 1280])

        return out_mask
