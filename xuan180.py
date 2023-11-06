import os
from PIL import Image

directory = "./data/gestrue/right"  # 目录路径

# 遍历目录下的所有文件
for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        filepath = os.path.join(directory, filename)
        
        # 打开图像文件
        image = Image.open(filepath)
        
        # 旋转图像
        rotated_image = image.rotate(90)
        
        # 保存旋转后的图像
        rotated_image.save(filepath)

        print(f"已旋转并保存文件: {filename}")
