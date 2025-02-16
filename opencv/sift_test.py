import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import time

def filter_keypoints_by_scale(keypoints, descriptors, min_scale=0, max_scale=float('inf')):
    """
    根据尺度过滤特征点
    """
    if descriptors is None:
        return [], None
        
    filtered_pairs = [(kp, desc) for kp, desc in zip(keypoints, descriptors) 
                     if min_scale <= kp.size <= max_scale]
    
    if not filtered_pairs:
        return [], None
        
    filtered_keypoints, filtered_descriptors = zip(*filtered_pairs)
    return list(filtered_keypoints), np.array(filtered_descriptors)

def process_single_image(
    image_path,
    output_dir,
    nfeatures=0,
    contrastThreshold=0.04,
    edgeThreshold=10,
    sigma=1.6,
    nOctaveLayers=3,
    min_scale=0,
    max_scale=10
):
    """
    处理单张图像
    """
    # 读取图像
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 创建SIFT对象，设置参数
    sift = cv2.SIFT_create(
        nfeatures=nfeatures,
        contrastThreshold=contrastThreshold,
        edgeThreshold=edgeThreshold,
        sigma=sigma,
        nOctaveLayers=nOctaveLayers
    )
    
    # 检测关键点
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    # 过滤特征点
    original_count = len(keypoints)
    keypoints, descriptors = filter_keypoints_by_scale(
        keypoints, descriptors, min_scale, max_scale
    )
    
    # 在图像上绘制关键点
    img_with_keypoints = cv2.drawKeypoints(
        img,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        color=(0, 255, 0)
    )
    
    # 添加特征点信息
    param_text = [
        f'Features: {len(keypoints)}/{original_count}',
        f'Scale: {min_scale:.1f}-{max_scale:.1f}'
    ]
    
    for i, text in enumerate(param_text):
        cv2.putText(
            img_with_keypoints,
            text,
            (10, 30 + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
    
    # 保存结果
    output_path = output_dir / f'sift_{image_path.stem}.jpg'
    cv2.imwrite(str(output_path), img_with_keypoints)
    
    return {
        'filename': image_path.name,
        'original_count': original_count,
        'filtered_count': len(keypoints),
        'scales': [kp.size for kp in keypoints] if keypoints else []
    }

def batch_process_images(
    input_dir,
    output_dir,
    image_extensions=('.bmp', '.jpg', '.jpeg', '.png'),
    **sift_params
):
    """
    批量处理文件夹中的图像
    
    Parameters:
    input_dir (str): 输入图像文件夹路径
    output_dir (str): 输出结果文件夹路径
    image_extensions (tuple): 要处理的图像文件扩展名
    sift_params: SIFT参数
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像文件
    image_files = [
        f for f in input_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"在 {input_dir} 中没有找到支持的图像文件")
        return
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 处理结果统计
    results = []
    start_time = time.time()
    
    # 处理每个图像
    for i, image_path in enumerate(image_files, 1):
        print(f"\n处理图像 {i}/{len(image_files)}: {image_path.name}")
        try:
            result = process_single_image(image_path, output_dir, **sift_params)
            results.append(result)
            print(f"特征点数量: {result['filtered_count']}/{result['original_count']}")
        except Exception as e:
            print(f"处理 {image_path.name} 时出错: {str(e)}")
    
    # 生成统计报告
    total_time = time.time() - start_time
    
    print("\n处理完成!")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"平均每张耗时: {total_time/len(image_files):.2f} 秒")
    
    # 将统计结果保存到文本文件
    report_path = output_dir / 'sift_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("SIFT特征点检测报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"处理时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总图像数: {len(image_files)}\n")
        f.write(f"总耗时: {total_time:.2f} 秒\n")
        f.write(f"平均耗时: {total_time/len(image_files):.2f} 秒/张\n\n")
        
        f.write("参数设置:\n")
        for key, value in sift_params.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\n各图像处理结果:\n")
        for result in results:
            f.write(f"\n文件名: {result['filename']}\n")
            f.write(f"原始特征点数: {result['original_count']}\n")
            f.write(f"过滤后特征点数: {result['filtered_count']}\n")
            if result['scales']:
                f.write(f"尺度范围: {min(result['scales']):.2f} - {max(result['scales']):.2f}\n")
                f.write(f"平均尺度: {sum(result['scales'])/len(result['scales']):.2f}\n")

def main():
    # 使用示例
    input_dir = 'input_sift'    # 输入图像文件夹
    output_dir = 'output_sift'  # 输出结果文件夹
    
    # SIFT参数设置
    params = {
        'nfeatures': 0,            # 不限制特征点数量
        'contrastThreshold': 0.04, # 对比度阈值
        'edgeThreshold': 5,       # 边缘阈值
        'sigma': 1.6,             # 高斯金字塔初始层sigma值
        'min_scale': 1.0,         # 最小尺度阈值
        'max_scale': 10.0          # 最大尺度阈值
    }
    
    try:
        batch_process_images(input_dir, output_dir, **params)
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")

if __name__ == "__main__":
    main()