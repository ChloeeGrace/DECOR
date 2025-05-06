import os
import random

# 设置路径
train_root = '/data/Datasets/Imagenet/train'
output_txt = 'self_con.txt'

# 获取并排序类别文件夹（保证标签从0到999）
class_folders = sorted(os.listdir(train_root))
assert len(class_folders) == 1000, "类别数量不为1000，请检查路径是否正确"

all_entries = []

for label, class_name in enumerate(class_folders):
    class_path = os.path.join(train_root, class_name)
    # 只保留图像文件
    images = [img for img in os.listdir(class_path) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
    # 随机选10张图像
    if len(images) < 10:
        raise ValueError(f"类别 {class_name} 中的图像少于10张")
    sampled_images = random.sample(images, 10)
    for img_name in sampled_images:
        img_path = os.path.join(class_path, img_name)
        all_entries.append(f'{img_path}\t{label}')

# 打乱所有条目
random.shuffle(all_entries)

# 写入文件
with open(output_txt, 'w') as f_out:
    for entry in all_entries:
        f_out.write(entry + '\n')
