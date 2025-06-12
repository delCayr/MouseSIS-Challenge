import numpy as np
import cv2
import math

class TTAUtils:
    def __init__(self, img_shape):
        """
        Initializes TTAUtils with the original image shape.
        Args:
            img_shape (tuple): (H_orig, W_orig) of the original image.
        """
        self.H_orig, self.W_orig = img_shape[:2]

    def apply_transform(self, img, mode, scale_factor=1.0, angle=0):
        """
        Applies a specified TTA transformation to an image.
        Args:
            img (np.ndarray): Input image.
            mode (str): Transformation mode (e.g., "none", "hflip", "vflip", "scale", 
                                            "rotate", "hflip_scale", "hflip_rotate").
            scale_factor (float): Factor for scaling.
            angle (float): Angle in degrees for rotation.
        Returns:
            np.ndarray: Transformed image.
            tuple: Shape of the transformed image (H_aug, W_aug).
        """
        img_tta = img.copy()
        H_current, W_current = img_tta.shape[:2]

        # 1. Scaling (if part of the mode or specified)
        # Scaling should ideally happen first if combined with rotation to maintain center
        if "scale" in mode and scale_factor != 1.0:
            new_h, new_w = int(H_current * scale_factor), int(W_current * scale_factor)
            img_tta = cv2.resize(img_tta, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            H_current, W_current = new_h, new_w
        
        # 2. Rotation (if part of the mode or specified)
        if "rotate" in mode and angle != 0:
            center = (W_current // 2, H_current // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Calculate new bounding box for the rotated image to avoid cropping
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_W = int((H_current * sin) + (W_current * cos))
            new_H = int((H_current * cos) + (W_current * sin))
            
            # Adjust rotation matrix to account for translation
            M[0, 2] += (new_W / 2) - center[0]
            M[1, 2] += (new_H / 2) - center[1]
            
            img_tta = cv2.warpAffine(img_tta, M, (new_W, new_H), flags=cv2.INTER_LINEAR, borderValue=(0,0,0)) # Pad with black
            H_current, W_current = new_H, new_W

        # 3. Flips (hflip, vflip)
        if "hflip" in mode:
            img_tta = np.ascontiguousarray(np.flip(img_tta, axis=1))
        if "vflip" in mode:
            img_tta = np.ascontiguousarray(np.flip(img_tta, axis=0))
        
        return img_tta, (H_current, W_current)

    def reverse_transform_boxes(self, boxes_aug_xyxy, mode, augmented_img_shape, 
                                scale_factor_applied=1.0, angle_applied=0):
        """
        Reverses TTA transformations for bounding boxes to original image coordinates.
        Args:
            boxes_aug_xyxy (np.ndarray): (N, 4) boxes in augmented image coords.
            mode (str): The TTA mode that was applied.
            augmented_img_shape (tuple): (H_aug, W_aug) of the augmented image.
            scale_factor_applied (float): The scale factor used during augmentation.
            angle_applied (float): The angle used during augmentation.
        Returns:
            np.ndarray: (N, 4) boxes in original image coordinates.
        """
        boxes_recovered = boxes_aug_xyxy.copy()
        H_aug, W_aug = augmented_img_shape

        # Reverse in the opposite order of application: Flips -> Rotation -> Scaling

        # 1. Reverse Flips
        if "vflip" in mode: # Applied after hflip (if both present in mode string)
            boxes_recovered[:, [1, 3]] = H_aug - boxes_recovered[:, [3, 1]]
        if "hflip" in mode:
            boxes_recovered[:, [0, 2]] = W_aug - boxes_recovered[:, [2, 0]]

        # 2. Reverse Rotation
        if "rotate" in mode and angle_applied != 0:
            # This is the tricky part. When we rotated the image, the augmented image
            # might have changed size to fit the rotated content.
            # The boxes_aug_xyxy are in the coordinate system of this (potentially larger)
            # rotated and padded image. We need to map them back to the pre-rotation,
            # post-scaling coordinate system.

            # Calculate the center of the augmented image (where rotation happened)
            center_aug_x, center_aug_y = W_aug / 2, H_aug / 2
            
            # Convert boxes to N,4,2 points (top-left, top-right, bottom-right, bottom-left)
            # then rotate these points, then find the new axis-aligned bounding box.
            new_boxes = []
            for box in boxes_recovered:
                x1, y1, x2, y2 = box
                points = np.array([
                    [x1, y1], [x2, y1], [x2, y2], [x1, y2]
                ], dtype=np.float32)

                # Translate points to be relative to the rotation center of aug_img
                points[:, 0] -= center_aug_x
                points[:, 1] -= center_aug_y

                # Apply inverse rotation
                # Rotation matrix for -angle
                angle_rad_inv = -np.deg2rad(angle_applied)
                cos_a, sin_a = np.cos(angle_rad_inv), np.sin(angle_rad_inv)
                
                rotated_points_x = points[:, 0] * cos_a - points[:, 1] * sin_a
                rotated_points_y = points[:, 0] * sin_a + points[:, 1] * cos_a
                
                # The size of the image *before* this rotation step (but after scaling)
                # We need to know H_prescale_prerot, W_prescale_prerot
                # If scaling was applied, H_pre_rot = self.H_orig * scale_factor_applied
                # W_pre_rot = self.W_orig * scale_factor_applied
                # If no scaling, H_pre_rot = self.H_orig, W_pre_rot = self.W_orig
                
                H_pre_rot = self.H_orig * scale_factor_applied
                W_pre_rot = self.W_orig * scale_factor_applied
                center_pre_rot_x, center_pre_rot_y = W_pre_rot / 2, H_pre_rot / 2


                # Translate points back using the center of the pre-rotation image
                rotated_points_x += center_pre_rot_x
                rotated_points_y += center_pre_rot_y
                
                # Get axis-aligned bounding box of the rotated points
                min_x, min_y = np.min(rotated_points_x), np.min(rotated_points_y)
                max_x, max_y = np.max(rotated_points_x), np.max(rotated_points_y)
                new_boxes.append([min_x, min_y, max_x, max_y])
            
            boxes_recovered = np.array(new_boxes)
            # After reversing rotation, the coordinates are now in the scaled image space
            # (if scaling was applied), or original image space (if no scaling).
            # The H_aug, W_aug for the next step (scaling reversal) should be H_pre_rot, W_pre_rot.
            H_aug, W_aug = H_pre_rot, W_pre_rot


        # 3. Reverse Scaling
        if "scale" in mode and scale_factor_applied != 1.0:
            # H_aug, W_aug here are the dimensions *before* this scaling step was reversed
            # which is effectively self.H_orig * scale_factor_applied, self.W_orig * scale_factor_applied
            
            # We want to map from this scaled space back to original H_orig, W_orig
            if H_aug == 0 or W_aug == 0: # Should not happen if scale_factor is valid
                return boxes_recovered # Or raise error

            scale_h_inv = self.H_orig / H_aug 
            scale_w_inv = self.W_orig / W_aug
            
            boxes_recovered[:, 0] *= scale_w_inv
            boxes_recovered[:, 1] *= scale_h_inv
            boxes_recovered[:, 2] *= scale_w_inv
            boxes_recovered[:, 3] *= scale_h_inv
        
        # Clip boxes to original image dimensions
        boxes_recovered[:, 0] = np.clip(boxes_recovered[:, 0], 0, self.W_orig)
        boxes_recovered[:, 1] = np.clip(boxes_recovered[:, 1], 0, self.H_orig)
        boxes_recovered[:, 2] = np.clip(boxes_recovered[:, 2], 0, self.W_orig)
        boxes_recovered[:, 3] = np.clip(boxes_recovered[:, 3], 0, self.H_orig)

        return boxes_recovered