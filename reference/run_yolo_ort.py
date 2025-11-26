import onnxruntime as ort
import numpy as np
import cv2
import os

# 类别映射
CLASS_NAMES = {
    0: "missing_hole",    # 缺失孔
    1: "mouse_bite",      # 鼠咬状缺陷 spur
    2: "open_circuit",    # 开路 mouse bite
    3: "short",           # 短路 open_circuit
    4: "spur",            # 毛刺 missinghole
    5: "spurious_copper"  # 假铜/多余铜
}

class YOLODetector:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # 使用固定输入尺寸
        self.input_height = 640
        self.input_width = 640
        
        print(f"Model input size: {self.input_width}x{self.input_height}")
    
    def preprocess(self, image):
        """预处理图像"""
        self.orig_height, self.orig_width = image.shape[:2]
        print(f"Original image size: {self.orig_width}x{self.orig_height}")
        
        # 转换为RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 使用letterbox保持宽高比
        img_resized, self.ratio, (self.dw, self.dh) = self.letterbox(
            img_rgb, new_shape=(self.input_width, self.input_height)
        )
        
        # 归一化并调整维度
        blob = img_resized.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))  # HWC to CHW
        blob = np.expand_dims(blob, 0)        # 添加batch维度
        
        return blob
    
    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
        """保持宽高比的图像缩放"""
        shape = img.shape[:2]  # 当前图像形状 [height, width]
        
        # 确保new_shape是整数元组
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # 计算缩放比例
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)

        # 计算新尺寸
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2

        # 调整图像大小
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        # 添加边框
        top, bottom = int(round(dh)), int(round(dh))
        left, right = int(round(dw)), int(round(dw))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        
        print(f"Letterbox result: {img.shape}, ratio: {r}, padding: ({dw}, {dh})")
        return img, r, (dw, dh)
    
    def postprocess(self, outputs, confidence_thres=0.25, iou_thres=0.45):
        """后处理模型输出"""
        # 输出形状为 (1, 10, 8400)
        raw_output = outputs[0]
        print(f"Raw output shape: {raw_output.shape}")
        print(f"Raw output min/max: {np.min(raw_output):.4f}/{np.max(raw_output):.4f}")
        
        # 参考TestOnnx.py的正确实现：转置并压缩输出
        outputs_transposed = np.transpose(np.squeeze(raw_output))
        print(f"Transposed output shape: {outputs_transposed.shape}")
        
        rows = outputs_transposed.shape[0]
        boxes = []
        scores = []
        class_ids = []
        
        # 分析输出结构
        print("\nAnalyzing output structure...")
        
        for i in range(rows):
            # 从第5个元素开始是类别分数
            classes_scores = outputs_transposed[i][4:]
            max_score = np.amax(classes_scores)
            
            if max_score >= confidence_thres:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs_transposed[i][0], outputs_transposed[i][1], outputs_transposed[i][2], outputs_transposed[i][3]
                
                # 将中心坐标转换为角点坐标
                x1 = x - w / 2
                y1 = y - h / 2
                x2 = x + w / 2
                y2 = y + h / 2
                
                # 调整到letterbox前的坐标
                x1 = (x1 - self.dw) / self.ratio
                y1 = (y1 - self.dh) / self.ratio
                x2 = (x2 - self.dw) / self.ratio
                y2 = (y2 - self.dh) / self.ratio
                
                # 转换为整数像素坐标
                x1 = int(max(0, min(x1, self.orig_width - 1)))
                y1 = int(max(0, min(y1, self.orig_height - 1)))
                x2 = int(max(0, min(x2, self.orig_width - 1)))
                y2 = int(max(0, min(y2, self.orig_height - 1)))
                
                width = x2 - x1
                height = y2 - y1
                
                if width > 0 and height > 0:
                    boxes.append([x1, y1, width, height])
                    scores.append(float(max_score))
                    class_ids.append(int(class_id))
        
        print(f"Found {len(boxes)} detections before NMS")
        
        # 应用NMS
        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_thres, iou_thres)
            
            final_boxes = []
            final_scores = []
            final_class_ids = []
            
            if indices is not None and len(indices) > 0:
                for i in indices:
                    idx = i[0] if isinstance(i, (list, np.ndarray)) else int(i)
                    final_boxes.append(boxes[idx])
                    final_scores.append(scores[idx])
                    final_class_ids.append(class_ids[idx])
                
                print(f"After NMS: {len(final_boxes)} detections")
                return final_boxes, final_scores, final_class_ids
        
        return [], [], []
    
    def draw_detections(self, image, boxes, scores, class_ids):
        """在图像上绘制检测结果"""
        # 生成颜色调色板
        color_palette = np.random.uniform(0, 255, size=(len(CLASS_NAMES), 3))
        
        for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
            x, y, w, h = box
            
            # 获取颜色
            color = [int(c) for c in color_palette[class_id]]
            
            # 绘制边界框
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            
            # 创建标签
            label = f"{CLASS_NAMES.get(class_id, 'unknown')}: {score:.2f}"
            
            # 计算标签尺寸
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # 计算标签位置
            label_y = y - 10 if y - 10 > label_height else y + 10
            
            # 绘制标签背景
            cv2.rectangle(
                image, 
                (x, label_y - label_height), 
                (x + label_width, label_y + baseline), 
                color, 
                cv2.FILLED
            )
            
            # 绘制标签文本
            cv2.putText(
                image, label, (x, label_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
            )
            
            print(f"Detection {i+1}: {CLASS_NAMES.get(class_id, 'unknown')} at [{x}, {y}, {w}, {h}] with confidence {score:.3f}")
        
        return image

def main():
    model_path = r'D:\VisualRobot-Git\VisualRobot\models\yolopcb.onnx'
    image_path = r"D:\Python-Git\datasets\PKU-Market-PCB\raw\PKU-Market-PCB_coco\images\val_labeled\04_spurious_copper_19.jpg"
    output_path = r'Img/yolo_test_ort_out.jpg'
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    # 创建检测器
    detector = YOLODetector(model_path)
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
        
    original_image = image.copy()
    
    # 预处理
    input_data = detector.preprocess(image)
    
    # 推理
    outputs = detector.session.run([detector.output_name], {detector.input_name: input_data})
    
    # 后处理 - 使用更低的置信度阈值
    boxes, scores, class_ids = detector.postprocess(outputs, confidence_thres=0.5, iou_thres=0.45)
    
    # 绘制检测结果
    if boxes:
        result_image = detector.draw_detections(image, boxes, scores, class_ids)
        print(f"Found {len(boxes)} detections")
    else:
        result_image = image
        print("No detections found")
    
    # 保存结果
    cv2.imwrite(output_path, result_image)
    print(f"Saved result to {output_path}")
    
    # 保存对比图像
    if boxes:
        comparison = np.hstack([original_image, result_image])
        cv2.imwrite('Img/yolo_comparison.jpg', comparison)
        print("Saved comparison image: Img/yolo_comparison.jpg")

if __name__ == "__main__":
    main()