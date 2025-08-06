import os
import glob
import json
import numpy as np
import cv2
from PIL import Image
from pycocotools import mask as coco_mask
import argparse
from collections import defaultdict

def create_coco_dataset(render_output_dir, output_file):
    """
    将实例分割掩码转换为COCO格式的标注文件
    
    Args:
        render_output_dir: 渲染输出的根目录，包含多个对象的子目录
        output_file: 输出的COCO格式JSON文件路径
    """
    print(f"正在生成COCO格式数据集，输出到: {output_file}")
    
    # 初始化COCO格式数据结构
    coco_data = {
        "images": [],
        "categories": [],
        "annotations": []
    }
    
    image_id = 0
    annotation_id = 0
    category_map = {}  # 部件名称到category_id的映射
    next_category_id = 1
    
    # 查找所有对象目录
    object_dirs = [d for d in os.listdir(render_output_dir) if os.path.isdir(os.path.join(render_output_dir, d))]
    print(f"找到 {len(object_dirs)} 个对象目录")
    
    # 处理每个对象
    for obj_dir in object_dirs:
        obj_path = os.path.join(render_output_dir, obj_dir)
        obj_id = obj_dir  # 使用目录名作为对象ID
        
        # 查找实例掩码目录
        instance_mask_dir = os.path.join(obj_path, "instance_mask")
        if not os.path.exists(instance_mask_dir):
            print(f"跳过 {obj_id}，未找到实例掩码目录")
            continue
        
        # 查找实例映射文件
        instance_mapping_path = os.path.join(render_output_dir, obj_dir, "instance_mapping", "instance_mapping.json")
        if not os.path.exists(instance_mapping_path):
            print(f"跳过 {obj_id}，未找到实例映射文件")
            continue
        
        # 加载实例映射
        try:
            with open(instance_mapping_path, 'r') as f:
                instance_mapping = json.load(f)
        except Exception as e:
            print(f"无法加载 {instance_mapping_path}: {e}")
            continue
        
        # 获取部件名称列表
        part_names = instance_mapping.get("part_names", [])
        instances_info = instance_mapping.get("instances", {})
        
        # 确保所有部件名称都有对应的类别ID
        for part_name in part_names:
            if part_name not in category_map:
                category_map[part_name] = next_category_id
                coco_data["categories"].append({
                    "id": next_category_id,
                    "name": part_name,
                    "supercategory": obj_id  # 使用对象ID作为超类别
                })
                next_category_id += 1
        
        # 创建颜色到部件名称的映射
        color_to_part = {}
        for obj_name, obj_info in instances_info.items():
            color_hex = obj_info.get("color", "")
            part_name = obj_info.get("part_name", "unknown")
            # 去掉 # 前缀并转换为RGB元组
            if color_hex.startswith("#"):
                color_hex = color_hex[1:]
            r = int(color_hex[0:2], 16)
            g = int(color_hex[2:4], 16)
            b = int(color_hex[4:6], 16)
            color_to_part[(r, g, b)] = part_name
        
        # 处理每个实例掩码图像
        mask_files = glob.glob(os.path.join(instance_mask_dir, "*.png"))
        print(f"对象 {obj_id} 有 {len(mask_files)} 个掩码文件")
        
        for mask_file in mask_files:
            mask_name = os.path.basename(mask_file)
            view_id = os.path.splitext(mask_name)[0]  # 不带扩展名的文件名
            
            # 读取掩码图像
            mask_img = cv2.imread(mask_file)
            if mask_img is None:
                print(f"无法读取掩码图像: {mask_file}")
                continue
            
            height, width, _ = mask_img.shape
            
            # 添加图像信息
            coco_data["images"].append({
                "id": image_id,
                "file_name": f"{obj_id}/instance_mask/{mask_name}",
                "width": width,
                "height": height,
                "obj_id": obj_id,
                "view_id": view_id
            })
            
            # 处理每个颜色(实例)
            for color, part_name in color_to_part.items():
                # 为这个颜色创建二进制掩码
                binary_mask = np.all(mask_img == color, axis=2).astype(np.uint8)
                
                # 检查掩码是否为空
                if not np.any(binary_mask):
                    continue
                
                # 计算RLE编码
                rle = coco_mask.encode(np.asfortranarray(binary_mask))
                # 将bytes转换为字符串
                counts = rle["counts"].decode('utf-8')
                
                # 获取类别ID
                if part_name in category_map:
                    category_id = category_map[part_name]
                else:
                    category_id = 0  # 未知类别
                
                # 计算边界框
                horizontal_indicies = np.where(np.any(binary_mask, axis=0))[0]
                vertical_indicies = np.where(np.any(binary_mask, axis=1))[0]
                if horizontal_indicies.shape[0] and vertical_indicies.shape[0]:
                    x1, x2 = horizontal_indicies[[0, -1]]
                    y1, y2 = vertical_indicies[[0, -1]]
                    bbox = [int(x1), int(y1), int(x2 - x1 + 1), int(y2 - y1 + 1)]
                else:
                    bbox = [0, 0, 0, 0]
                
                # 计算面积
                area = int(np.sum(binary_mask))
                
                # 添加标注信息
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": {
                        "counts": counts,
                        "size": [height, width]
                    },
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                    "part_name": part_name,
                    "obj_id": obj_id,
                    "view_id": view_id
                })
                
                annotation_id += 1
            
            image_id += 1
    
    # 保存COCO格式数据
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"COCO数据集生成完成，包含 {len(coco_data['images'])} 张图像, {len(coco_data['categories'])} 个类别, {len(coco_data['annotations'])} 个标注")
    return coco_data

def create_coco_dataset_polygon(render_output_dir, output_file):
    """
    将实例分割掩码转换为COCO格式的标注文件，使用多边形轮廓替代RLE编码
    
    Args:
        render_output_dir: 渲染输出的根目录，包含多个对象的子目录
        output_file: 输出的COCO格式JSON文件路径
    """
    print(f"正在生成COCO格式数据集(多边形格式)，输出到: {output_file}")
    
    # 初始化COCO格式数据结构
    coco_data = {
        "images": [],
        "categories": [],
        "annotations": []
    }
    
    image_id = 0
    annotation_id = 0
    category_map = {}  # 部件名称到category_id的映射
    next_category_id = 1
    
    # 查找所有对象目录
    object_dirs = [d for d in os.listdir(render_output_dir) if os.path.isdir(os.path.join(render_output_dir, d))]
    print(f"找到 {len(object_dirs)} 个对象目录")
    
    # 处理每个对象
    for obj_dir in object_dirs:
        obj_path = os.path.join(render_output_dir, obj_dir)
        obj_id = obj_dir  # 使用目录名作为对象ID
        
        # 查找实例掩码目录
        instance_mask_dir = os.path.join(obj_path, "instance_mask")
        if not os.path.exists(instance_mask_dir):
            print(f"跳过 {obj_id}，未找到实例掩码目录")
            continue
        
        # 查找实例映射文件
        instance_mapping_path = os.path.join(render_output_dir, obj_dir, "instance_mapping", "instance_mapping.json")
        if not os.path.exists(instance_mapping_path):
            print(f"跳过 {obj_id}，未找到实例映射文件")
            continue
        
        # 加载实例映射
        try:
            with open(instance_mapping_path, 'r') as f:
                instance_mapping = json.load(f)
        except Exception as e:
            print(f"无法加载 {instance_mapping_path}: {e}")
            continue
        
        # 获取部件名称列表
        part_names = instance_mapping.get("part_names", [])
        instances_info = instance_mapping.get("instances", {})
        
        # 确保所有部件名称都有对应的类别ID
        for part_name in part_names:
            if part_name not in category_map:
                category_map[part_name] = next_category_id
                coco_data["categories"].append({
                    "id": next_category_id,
                    "name": part_name,
                    "supercategory": obj_id  # 使用对象ID作为超类别
                })
                next_category_id += 1
        
        # 创建颜色到部件名称的映射
        color_to_part = {}
        for obj_name, obj_info in instances_info.items():
            color_hex = obj_info.get("color", "")
            part_name = obj_info.get("part_name", "unknown")
            # 去掉 # 前缀并转换为RGB元组
            if color_hex.startswith("#"):
                color_hex = color_hex[1:]
            r = int(color_hex[0:2], 16)
            g = int(color_hex[2:4], 16)
            b = int(color_hex[4:6], 16)
            color_to_part[(r, g, b)] = part_name
        
        # 处理每个实例掩码图像
        mask_files = glob.glob(os.path.join(instance_mask_dir, "*.png"))
        print(f"对象 {obj_id} 有 {len(mask_files)} 个掩码文件")
        
        for mask_file in mask_files:
            mask_name = os.path.basename(mask_file)
            view_id = os.path.splitext(mask_name)[0]  # 不带扩展名的文件名
            
            # 读取掩码图像
            mask_img = cv2.imread(mask_file)
            if mask_img is None:
                print(f"无法读取掩码图像: {mask_file}")
                continue
            
            height, width, _ = mask_img.shape
            
            # 添加图像信息
            coco_data["images"].append({
                "id": image_id,
                "file_name": f"{obj_id}/instance_mask/{mask_name}",
                "width": width,
                "height": height,
                "obj_id": obj_id,
                "view_id": view_id
            })
            
            # 处理每个颜色(实例)
            for color, part_name in color_to_part.items():
                # 为这个颜色创建二进制掩码
                binary_mask = np.all(mask_img == color, axis=2).astype(np.uint8)
                
                # 检查掩码是否为空
                if not np.any(binary_mask):
                    continue
                
                # 查找轮廓
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # 转换轮廓为COCO格式的多边形
                polygons = []
                for contour in contours:
                    # 过滤掉太小的轮廓
                    if contour.shape[0] < 3:
                        continue
                    
                    # 将轮廓压平为一维数组并四舍五入为整数
                    polygon = np.round(contour.flatten()).astype(int).tolist()
                    polygons.append(polygon)
                
                # 如果没有有效的多边形，跳过此实例
                if not polygons:
                    continue
                
                # 获取类别ID
                if part_name in category_map:
                    category_id = category_map[part_name]
                else:
                    category_id = 0  # 未知类别
                
                # 计算边界框
                horizontal_indicies = np.where(np.any(binary_mask, axis=0))[0]
                vertical_indicies = np.where(np.any(binary_mask, axis=1))[0]
                if horizontal_indicies.shape[0] and vertical_indicies.shape[0]:
                    x1, x2 = horizontal_indicies[[0, -1]]
                    y1, y2 = vertical_indicies[[0, -1]]
                    bbox = [int(x1), int(y1), int(x2 - x1 + 1), int(y2 - y1 + 1)]
                else:
                    bbox = [0, 0, 0, 0]
                
                # 计算面积
                area = int(np.sum(binary_mask))
                
                # 添加标注信息
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": polygons,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                    "part_name": part_name,
                    "obj_id": obj_id,
                    "view_id": view_id
                })
                
                annotation_id += 1
            
            image_id += 1
    
    # 保存COCO格式数据
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"COCO数据集(多边形格式)生成完成，包含 {len(coco_data['images'])} 张图像, {len(coco_data['categories'])} 个类别, {len(coco_data['annotations'])} 个标注")
    return coco_data


def main():
    parser = argparse.ArgumentParser(description='将实例分割掩码转换为COCO格式的标注文件')
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='渲染输出的根目录，包含多个对象的子目录')
    parser.add_argument('--output_file', type=str, required=True,
                        help='输出的COCO格式JSON文件路径')
    parser.add_argument('--format', type=str, choices=['rle', 'polygon'], default='rle',
                        help='分割格式: RLE编码(rle)或多边形轮廓(polygon)')
    
    args = parser.parse_args()
    
    if args.format == 'rle':
        create_coco_dataset(args.input_dir, args.output_file)
    else:
        create_coco_dataset_polygon(args.input_dir, args.output_file)

if __name__ == "__main__":
    main() 