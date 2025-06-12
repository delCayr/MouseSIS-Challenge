import ultralytics
from transformers import SamModel, SamProcessor
import torch

from ..utils import suppress_stdout_stderr


class SamYoloDetector:
    def __init__(self, yolo_path, device='cuda:0') -> None:
        self.detector = ultralytics.YOLO(yolo_path)
        # self.sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
        self.sam_model = SamModel.from_pretrained("/data/hj_data/CVPR/Mouse/MouseSIS-main/models/sam-vit-huge").to(device)
        self.sam_processor = SamProcessor.from_pretrained("/data/hj_data/CVPR/Mouse/MouseSIS-main/models/sam-vit-huge")
        self.device = device

    def run(self, img): #720 1280 3
        with suppress_stdout_stderr():
            result = self.detector(img)[0]
        
        boxes = result.boxes.xyxy.detach().cpu().numpy()  # x1, y1, x2, y2  (2,4)  e2vid: (1,4)
        scores = result.boxes.conf.detach().cpu().numpy()#array([    0.89284,     0.81984], dtype=float32) (2,)  e2vid:  0.8145509 (1,)
        
        if not len(boxes):
            return None, None
            
        boxes_list = [[boxes.tolist()]] #3 dim list
        inputs = self.sam_processor(img.transpose(2, 0, 1), input_boxes=[boxes_list], return_tensors="pt").to(self.device) # inputs.data['pixel_values']:torch.Size([1, 3, 1024, 1024])
        
        with torch.no_grad():
            outputs = self.sam_model(**inputs) #outputs.pred_masks.shape  torch.Size([1, 2, 3, 256, 256])
            masks = self.sam_processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(), #  e2vid: torch.Size([1, 1, 3, 256, 256])
                inputs["original_sizes"].cpu(), #tensor([[ 720, 1280]], device='cuda:7')  H,W
                inputs["reshaped_input_sizes"].cpu() #tensor([[ 576, 1024]], device='cuda:7')
            )[0] #torch.Size([2, 3, 720, 1280])
            
            iou_scores = outputs.iou_scores.cpu()[0] #tensor([[[0.9680, 0.9761, 0.9753], [0.9360, 0.9482, 0.9483]]], device='cuda:7')   evi2d: tensor([[0.6240, 0.7980, 0.8395]]) torch.Size([1, 3])
            num_instances, nb_predictions, height, width = masks.shape #H,W  720 1280  evi2d:torch.Size([1, 3, 720, 1280])
            max_indices = iou_scores.argmax(dim=1, keepdim=True) #tensor([[1],[2]]) 
            gather_indices = max_indices[..., None, None].expand(-1, 1, height, width) #shape: torch.Size([2, 1, 720, 1280])
            selected_masks = torch.gather(masks, 1, gather_indices).squeeze(1)  # shape: torch.Size([2, 720, 1280])
            
        return selected_masks.cpu().numpy(), scores