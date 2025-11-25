import cv2
import numpy as np
import json
from pathlib import Path
import os
import torch

def load_classes(classes_file):
    classes = [
        'module1',
        'module2',
        'defect1',
        'defect2'
    ]
    return classes

def preprocess(image_path, target_size=(640, 640)):
    """对输入图像进行预处理"""
    image = cv2.imread(image_path)
    original_img = image.copy()
    
    # 获取原始图像尺寸
    orig_h, orig_w = image.shape[:2]
    
    # 调整图像大小到目标尺寸
    img = cv2.resize(image, target_size)
    
    # 转换颜色通道 BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 转换为tensor并标准化
    img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
    
    # 添加batch维度
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor, original_img, orig_h, orig_w

def postprocess(results, original_img, orig_h, orig_w, confidence_threshold=0.25):
    """对模型输出进行后处理"""
    # 获取检测结果
    detections = results[0]
    
    # 过滤低置信度的检测结果
    keep = detections[:, 4] > confidence_threshold
    detections = detections[keep]
    
    # 如果没有检测结果，返回原图像
    if len(detections) == 0:
        return original_img, []
    
    # 提取边界框坐标
    boxes = detections[:, :4]
    
    # 将坐标从网络输出尺寸(640x640)缩放回原始图像尺寸
    scale_x = orig_w / 640
    scale_y = orig_h / 640
    
    boxes[:, 0] *= scale_x  # x_center
    boxes[:, 1] *= scale_y  # y_center  
    boxes[:, 2] *= scale_x  # width
    boxes[:, 3] *= scale_y  # height
    
    # 提取类别和置信度
    classes = detections[:, 5].long()
    confidences = detections[:, 4]
    
    # 绘制检测结果
    result_image = original_img.copy()
    class_names = load_classes(None)
    
    for i in range(len(boxes)):
        x_center, y_center, width, height = boxes[i]
        confidence = confidences[i]
        class_id = classes[i]
        
        # 计算边界框坐标（左上角和右下角）
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        
        # 确保坐标在图像范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(orig_w - 1, x2)
        y2 = min(orig_h - 1, y2)
        
        # 绘制边界框
        color = (0, 255, 0) if class_id == 0 else (0, 0, 255)
        thickness = 2
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, thickness)
        
        # 添加标签
        label = f"{class_names[class_id]}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 1)[0]
        cv2.rectangle(result_image, (x1, y1 - label_size[1] - 5), (x1 + label_size[0], y1), color, -1)
        cv2.putText(result_image, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    return result_image, [{'class': class_names[class_id], 'confidence': float(confidence), 'bbox': [x1, y1, x2, y2]} for x_center, y_center, width, height, confidence, class_id in zip(boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], confidences, classes)]

def save_detection_result(result_image, detections, save_path):
    """保存检测结果到文件"""
    try:
        cv2.imwrite(save_path, result_image)
        print(f"检测结果已保存到: {save_path}")
        
        # 保存检测信息到JSON文件
        json_path = save_path.replace('.jpg', '_detections.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(detections, f, ensure_ascii=False, indent=2)
        print(f"检测信息已保存到: {json_path}")
        
        return True
    except Exception as e:
        print(f"无法保存图像到 {save_path}: {str(e)}")
        return False

def main():
    # 设置模型路径和图像路径
    model_path = r'D:\Python-Git\runs\detect\yolo11n-arcuchi-swanlab\weights\best.pt'
    image_path = r'D:\Python-Git\datasets\dataset\images\val\module2_039.jpg'
    
    # 设置结果保存路径
    results_dir = r'D:\Python-Git\runs\detect\results'
    os.makedirs(results_dir, exist_ok=True)
    
    # 获取原始图像名称并创建保存路径
    image_name = os.path.basename(image_path)
    name_without_ext = os.path.splitext(image_name)[0]
    result_image_path = os.path.join(results_dir, f'{name_without_ext}_detections.jpg')
    
    # 检查文件是否存在
    if not Path(model_path).exists():
        print(f"错误：模型文件不存在 - {model_path}")
        return
    
    if not Path(image_path).exists():
        print(f"错误：图像文件不存在 - {image_path}")
        return
    
    print(f"正在加载模型: {model_path}")
    
    try:
        # 加载YOLOv8模型
        from ultralytics import YOLO
        
        # 创建模型实例
        model = YOLO(model_path)
        
        print(f"正在推理图像: {image_path}")
        
        # 进行推理
        results = model(image_path, verbose=False)
        
        # 获取第一个结果
        result = results[0]
        
        # 获取原始图像
        original_img = cv2.imread(image_path)
        orig_h, orig_w = original_img.shape[:2]
        
        # 提取检测结果
        detections = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            
            class_names = load_classes(None)
            
            for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                x1, y1, x2, y2 = box
                
                # 确保坐标在图像范围内
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(orig_w - 1, x2)
                y2 = min(orig_h - 1, y2)
                
                detection = {
                    'class': class_names[int(cls_id)],
                    'confidence': float(conf),
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                }
                detections.append(detection)
        
        # 绘制检测结果
        result_image = original_img.copy()
        class_names = load_classes(None)
        
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']
            
            # 绘制边界框
            color = (0, 255, 0)  # 绿色
            thickness = 2
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, thickness)
            
            # 添加标签
            label = f"{class_name}: {confidence:.3f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
            
            # 绘制标签背景
            cv2.rectangle(result_image, (x1, y1 - label_size[1] - 20), (x1 + label_size[0], y1), color, -1)
            cv2.putText(result_image, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        
        print(f"检测到 {len(detections)} 个目标")
        
        # 保存结果
        if save_detection_result(result_image, detections, result_image_path):
            print("\n=== 推理完成 ===")
            print(f"原始图像: {image_path}")
            print(f"检测结果: {result_image_path}")
            print(f"检测数量: {len(detections)}")
            
            if detections:
                print("\n检测详情:")
                for i, detection in enumerate(detections, 1):
                    print(f"  {i}. {detection['class']} - 置信度: {detection['confidence']:.3f}")
        else:
            print("保存失败")
            
    except Exception as e:
        print(f"运行时出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()