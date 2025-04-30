import os
import shutil

# Pad naar je bestaande datasetmappen
base_path = "C:/Users/robbe/Desktop/VUB/master/thesis_github/OOD.v1i.yolov5pytorch"
val_img = os.path.join(base_path, "valid", "images")
val_lbl = os.path.join(base_path, "valid", "labels")
test_img = os.path.join(base_path, "test", "images")
test_lbl = os.path.join(base_path, "test", "labels")

# Nieuwe map voor test + validatie
combined_img = os.path.join(base_path, "test_combined", "images")
combined_lbl = os.path.join(base_path, "test_combined", "labels")
os.makedirs(combined_img, exist_ok=True)
os.makedirs(combined_lbl, exist_ok=True)

# Kopieer alle bestanden
for src_folder in [val_img, test_img]:
    for file in os.listdir(src_folder):
        shutil.copy(os.path.join(src_folder, file), combined_img)

for src_folder in [val_lbl, test_lbl]:
    for file in os.listdir(src_folder):
        shutil.copy(os.path.join(src_folder, file), combined_lbl)

print("✅ Validatie + test data gekopieerd naar test_combined/")
