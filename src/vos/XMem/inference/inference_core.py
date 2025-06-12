from src.vos.XMem.util.tensor_util import pad_divide_by, unpad

from ..model.aggregate import aggregate
from ..model.network import XMem
from .memory_manager import MemoryManager


class InferenceCore:
    def __init__(self, network: XMem, config):
        self.config = config
        self.network = network
        self.mem_every = config["mem_every"]
        self.deep_update_every = config["deep_update_every"]
        self.enable_long_term = config["enable_long_term"]

        # if deep_update_every < 0, synchronize deep update with memory frame
        self.deep_update_sync = self.deep_update_every < 0

        self.clear_memory()
        self.all_labels = None

    def clear_memory(self):
        self.curr_ti = -1
        self.last_mem_ti = 0
        if not self.deep_update_sync:
            self.last_deep_update_ti = -self.deep_update_every
        self.memory = MemoryManager(config=self.config)

    def update_config(self, config):
        self.mem_every = config["mem_every"]
        self.deep_update_every = config["deep_update_every"]
        self.enable_long_term = config["enable_long_term"]

        # if deep_update_every < 0, synchronize deep update with memory frame
        self.deep_update_sync = self.deep_update_every < 0
        self.memory.update_config(config)

    def set_all_labels(self, all_labels):
        # self.all_labels = [l.item() for l in all_labels]
        self.all_labels = all_labels

    def step(
        self, image, mask=None, valid_labels=None, end=False
    ):  # image:3,720,1280,mask:1, 720, 1280,valid_labels:1(mice),
        # image: 3*H*W
        # mask: num_objects*H*W or None
        self.curr_ti += 1
        image, self.pad = pad_divide_by(image, 16)
        image = image.unsqueeze(0)  # add the batch dimension  1,3,720,1280

        is_mem_frame = (
            (self.curr_ti - self.last_mem_ti >= self.mem_every) or (mask is not None)
        ) and (not end)  # True(1 frame)  False
        need_segment = (self.curr_ti > 0) and (
            (valid_labels is None) or (len(self.all_labels) != len(valid_labels))
        )  # False(貌似第一张图片不需要分割), True
        is_deep_update = (
            (self.deep_update_sync and is_mem_frame)  # synchronized
            or (
                not self.deep_update_sync
                and self.curr_ti - self.last_deep_update_ti >= self.deep_update_every
            )  # no-sync
        ) and (not end)  # True  False
        is_normal_update = (not self.deep_update_sync or not is_deep_update) and (
            not end
        )  # False  True
        # key:(1,64,45,80);None;selection:(1,64,45,80)
        key, shrinkage, selection, f16, f8, f4 = self.network.encode_key(
            image,
            need_ek=(self.enable_long_term or need_segment),  # True or False
            need_sk=is_mem_frame,
        )  # True
        multi_scale_features = (
            f16,
            f8,
            f4,
        )  # f16: torch.Size([1, 1024, 45, 80]);f8:torch.Size([1, 512, 90, 160]);f4:torch.Size([1, 256, 180, 320])

        # segment the current frame is needed
        if need_segment:  # False，True
            memory_readout = self.memory.match_memory(key, selection).unsqueeze(
                0
            )  # torch.Size([1, 1, 512, 45, 80])
            hidden, _, pred_prob_with_bg = self.network.segment(
                multi_scale_features,
                memory_readout,
                self.memory.get_hidden(),
                h_out=is_normal_update,
                strip_bg=False,
            )  # torch.Size([1, 2, 720, 1280])
            # remove batch dim
            pred_prob_with_bg = pred_prob_with_bg[0]  # torch.Size([2, 720, 1280])
            pred_prob_no_bg = pred_prob_with_bg[1:]  # torch.Size([1, 720, 1280])
            if is_normal_update:
                self.memory.set_hidden(hidden)  # torch.Size([1, 1, 64, 45, 80])
        else:
            pred_prob_no_bg = pred_prob_with_bg = None

        # use the input mask if any
        if mask is not None:  # 第1个frame时有mask  上 一 frame mask  None
            mask, _ = pad_divide_by(mask, 16)  # 1,720,1280

            if pred_prob_no_bg is not None:
                # if we have a predicted mask, we work on it
                # make pred_prob_no_bg consistent with the input mask
                mask_regions = mask.sum(0) > 0.5
                pred_prob_no_bg[:, mask_regions] = 0
                # shift by 1 because mask/pred_prob_no_bg do not contain background
                mask = mask.type_as(pred_prob_no_bg)
                if valid_labels is not None:
                    shift_by_one_non_labels = [
                        i
                        for i in range(pred_prob_no_bg.shape[0])
                        if (i + 1) not in valid_labels
                    ]
                    # non-labelled objects are copied from the predicted mask
                    mask[shift_by_one_non_labels] = pred_prob_no_bg[
                        shift_by_one_non_labels
                    ]
            pred_prob_with_bg = aggregate(mask, dim=0)

            # also create new hidden states
            self.memory.create_hidden_state(
                len(self.all_labels), key
            )  # key:torch.Size([1, 64, 45, 80])

        # save as memory if needed
        if is_mem_frame:  # 第1frame会进入，False
            value, hidden = self.network.encode_value(
                image,
                f16,
                self.memory.get_hidden(),
                pred_prob_with_bg[1:].unsqueeze(0),
                is_deep_update=is_deep_update,
            )  # value:torch.Size([1, 1, 512, 45, 80])  hidden:torch.Size([1, 1, 64, 45, 80])
            self.memory.add_memory(
                key,
                shrinkage,
                value,
                self.all_labels,
                selection=selection if self.enable_long_term else None,
            )
            self.last_mem_ti = self.curr_ti

            if is_deep_update:
                self.memory.set_hidden(hidden)
                self.last_deep_update_ti = self.curr_ti

        return unpad(pred_prob_with_bg, self.pad)
