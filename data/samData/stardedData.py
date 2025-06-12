import os
import shutil
import re

#分组和重命名处理为Davis格式
# def organize_and_rename_images(src_dir):
#     for filename in os.listdir(src_dir):
#         if not filename.endswith(".png"):
#             continue
        
#         # 匹配 seq 名称和帧编号
#         match = re.match(r'(seq\d{2})_(\d{8})\.png', filename)
#         if not match:
#             continue
        
#         seq_name, frame_num = match.groups()
#         seq_folder = os.path.join(src_dir, seq_name)

#         # 创建目标文件夹（如 seq02）
#         os.makedirs(seq_folder, exist_ok=True)

#         # 重命名规则

#         new_filename = f"{frame_num}.png"

#         # 移动并重命名文件
#         src_path = os.path.join(src_dir, filename)
#         dst_path = os.path.join(seq_folder, new_filename)

#         shutil.move(src_path, dst_path)
#         print(f"Moved: {src_path} -> {dst_path}")

# # 用法
# organize_and_rename_images('/data/hj_data/CVPR/Mouse/MouseSIS-main/data/samData/masks')  # 替换为你的目录路径


# def convert_png_to_jpg(root_dir):
#     for subdir in os.listdir(root_dir):
#         subdir_path = os.path.join(root_dir, subdir)
        
#         # 确保是目录
#         if not os.path.isdir(subdir_path):
#             continue

#         for filename in os.listdir(subdir_path):
#             if filename.lower().endswith(".png"):
#                 base_name = os.path.splitext(filename)[0]
#                 old_path = os.path.join(subdir_path, filename)
#                 new_path = os.path.join(subdir_path, base_name + ".jpg")
                
#                 os.rename(old_path, new_path)
#                 print(f"Renamed: {old_path} -> {new_path}")

# # 用法
# convert_png_to_jpg("/data/hj_data/CVPR/Mouse/MouseSIS-main/data/samData/images")  # 替换为你的 images 目录路径


# import os
# import shutil
# import math

# def group_images_every_n(src_root, group_size=6):
#     for seq_name in os.listdir(src_root):
#         seq_path = os.path.join(src_root, seq_name)

#         if not os.path.isdir(seq_path):
#             continue

#         # 获取图片名并排序
#         image_files = sorted([
#             f for f in os.listdir(seq_path)
#             if f.lower().endswith((".jpg", ".png"))
#         ])

#         total_groups = math.ceil(len(image_files) / group_size)

#         for i in range(total_groups):
#             # 生成子文件夹名，如 seq02_001
#             group_folder = f"{seq_name}_{i+1:03d}"
#             group_path = os.path.join(src_root, group_folder)
#             os.makedirs(group_path, exist_ok=True)

#             # 拿出当前这组的图片
#             start = i * group_size
#             end = start + group_size
#             for img_name in image_files[start:end]:
#                 src_img = os.path.join(seq_path, img_name)
#                 dst_img = os.path.join(group_path, img_name)
#                 shutil.move(src_img, dst_img)
#                 print(f"Moved: {src_img} -> {dst_img}")
#         os.rmdir(os.path.join(src_root,seq_name))
# # 用法
# group_images_every_n("/data/hj_data/CVPR/Mouse/MouseSIS-main/data/samData/masks")  # 替换为你的 images 根目录

import os

folder_root = '/data/hj_data/CVPR/Mouse/MouseSIS-main/data/samData/masks'

for name in os.listdir(folder_root):
    full_path = os.path.join(folder_root, name)

    # 只处理目录并且以 _001 结尾的
    if os.path.isdir(full_path) and name.endswith('_001'):
        new_name = name[:-4]  # 去掉最后 4 个字符 "_001"
        new_path = os.path.join(folder_root, new_name)

        # 避免重名冲突
        if not os.path.exists(new_path):
            os.rename(full_path, new_path)
            print(f"重命名: {name} → {new_name}")
        else:
            print(f"目标目录已存在，跳过: {new_name}")
