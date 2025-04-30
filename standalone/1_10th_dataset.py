import os
import shutil
import random

def copy_random_subset(src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir, fraction=0.1):
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    all_files = os.listdir(src_img_dir)
    selected_files = random.sample(all_files, int(len(all_files) * fraction))

    for file in selected_files:
        shutil.copy(os.path.join(src_img_dir, file), dst_img_dir)
        label_file = file.replace('.jpg', '.txt')
        shutil.copy(os.path.join(src_lbl_dir, label_file), dst_lbl_dir)

    print(f"✅ {len(selected_files)} bestanden gekopieerd naar {dst_img_dir}")

# Paden instellen
base = "C:/Users/robbe/Desktop/VUB/master/thesis_github"
output = os.path.join(base, "standalone")

# 10% van de train data
copy_random_subset(
    src_img_dir=os.path.join(base, "OOD.v1i.yolov5pytorch/train/images"),
    src_lbl_dir=os.path.join(base, "OOD.v1i.yolov5pytorch/train/labels"),
    dst_img_dir=os.path.join(output, "train/images"),
    dst_lbl_dir=os.path.join(output, "train/labels")
)

# 10% van de test_combined data
copy_random_subset(
    src_img_dir=os.path.join(base, "OOD.v1i.yolov5pytorch/test_combined/images"),
    src_lbl_dir=os.path.join(base, "OOD.v1i.yolov5pytorch/test_combined/labels"),
    dst_img_dir=os.path.join(output, "test_combined/images"),
    dst_lbl_dir=os.path.join(output, "test_combined/labels")
)
