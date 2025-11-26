import onnxruntime as ort
import numpy as np
import cv2
import os

# 类别映射
# 类别映射 - 调整类别ID映射以修复标签匹配问题
CLASS_NAMES = {
    0: "missing_hole",
    1: "mouse_bite", 
    2: "open_circuit", 
    3: "short", 
    4: "spur", 
    5: "spurious_copper"  
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
    
    def save_raw_output(self, raw_output, filename="raw.csv"):
        """将原始模型输出保存到CSV文件，保持矩阵行列形状"""
        try:
            # 创建保存目录（如果不存在）
            save_dir = "d:/VisualRobot-Git/VisualRobot"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # 构建完整文件路径
            file_path = os.path.join(save_dir, filename)
            
            # 打印原始输出形状信息
            print(f"原始输出形状: {raw_output.shape}")
            
            # 对于YOLOv8输出，形状通常是 [1, num_outputs, num_anchors]
            # 我们需要调整格式以保持矩阵行列结构
            
            # 如果是三维张量 [1, num_outputs, num_anchors]，将其调整为 [num_outputs, num_anchors]
            if len(raw_output.shape) == 3 and raw_output.shape[0] == 1:
                output_2d = raw_output.squeeze(0)  # 移除batch维度
                print(f"调整后形状: {output_2d.shape}")
                
                # 保存为CSV格式，保持行列结构
                np.savetxt(file_path, output_2d, fmt='%.6f', delimiter=',')
                print(f"原始模型输出矩阵已保存到CSV文件: {file_path}")
                print(f"保存的矩阵形状: {output_2d.shape[0]}行 x {output_2d.shape[1]}列")
            else:
                # 如果是其他形状，保存为CSV格式
                np.savetxt(file_path, raw_output, fmt='%.6f', delimiter=',')
                print(f"原始模型输出已保存到CSV文件: {file_path}")
                print(f"保存的形状: {raw_output.shape}")
                
        except Exception as e:
            print(f"保存原始输出时出错: {str(e)}")
    
    def postprocess(self, outputs, conf_threshold=0.25, iou_threshold=0.45):
        """后处理模型输出，提取检测结果"""
        # 打印原始输出信息
        raw_output = outputs[0]
        print(f"Raw output shape: {raw_output.shape}")
        print(f"Raw output min/max: {np.min(raw_output):.4f}/{np.max(raw_output):.4f}")
        
        # 保存原始输出到文件
        self.save_raw_output(raw_output)
        
        # 使用squeeze压缩维度后再转置，与TestOnnx.py保持一致
        output_data = np.transpose(np.squeeze(raw_output))
        print(f"Processed output shape: {output_data.shape}")
        
        # 分析输出结构 - 使用类别分数作为置信度
        print("\nAnalyzing output structure (using class scores as confidence)...")
        
        # 获取所有候选框的类别分数
        class_scores = output_data[:, 4:]
        
        # 新增：对类别分数应用sigmoid激活函数
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))
        
        # 对类别分数应用sigmoid激活函数
        sigmoid_scores = sigmoid(class_scores)
        max_scores = np.max(sigmoid_scores, axis=1)
        # 新增：获取原始类别分数的最大值
        max_raw_scores = np.max(class_scores, axis=1)
        # 新增：获取原始类别分数中最大值对应的类别ID
        max_raw_classes = np.argmax(class_scores, axis=1)
        
        # 打印top10置信度
        top10_indices = np.argsort(max_scores)[::-1][:10]
        top10_scores = max_scores[top10_indices]
        top10_classes = np.argmax(sigmoid_scores[top10_indices], axis=1)
        # 新增：获取top10的原始分数
        top10_raw_scores = max_raw_scores[top10_indices]
        print(f"max sigmoid scores top10: {[f'{score:.6f}' for score in top10_scores]}")
        print(f"corresponding class IDs: {[int(cid) for cid in top10_classes]}")
        # 新增：打印原始分数
        print(f"max raw scores top10: {[f'{score:.6f}' for score in top10_raw_scores]}")
        
        # 统计不同阈值下的候选框数量
        print(f"max sigmoid scores count > 0.5: {(max_scores > 0.5).sum()}")
        print(f"max sigmoid scores count > 0.1: {(max_scores > 0.1).sum()}")
        print(f"max sigmoid scores count > 0.05: {(max_scores > 0.05).sum()}")
        # 新增：统计原始分数的分布
        print(f"max raw scores count > 0.5: {(max_raw_scores > 0.5).sum()}")
        print(f"max raw scores count > 0.1: {(max_raw_scores > 0.1).sum()}")
        print(f"max raw scores count > 0.05: {(max_raw_scores > 0.05).sum()}")
        
        # 显示前10个候选框的详细信息
        print("\nTop candidates (index, max_sigmoid_score, raw_score, class_id):")
        for i, idx in enumerate(top10_indices[:10]):
            score = max_scores[idx]
            raw_score = max_raw_scores[idx]
            class_id = np.argmax(sigmoid_scores[idx])
            print(f"{idx}: max_sigmoid_score={score:.7f}, raw_score={raw_score:.7f}, class_id={class_id}, class_name={CLASS_NAMES.get(class_id, 'unknown')}")
        
        # 过滤置信度 - 使用原始分数阈值0.5
        keep_indices = max_raw_scores >= 0.5
        filtered_boxes = output_data[keep_indices]
        filtered_raw_scores = class_scores[keep_indices]
        print(f"Found {len(filtered_boxes)} detections before NMS with raw score > 0.5")
        
        if len(filtered_boxes) == 0:
            print("No detections found")
            return []
        
        # 提取边界框坐标
        boxes = filtered_boxes[:, :4]
        
        # 转换边界框格式 (cx, cy, w, h) -> (x1, y1, x2, y2)
        boxes[:, 0] -= boxes[:, 2] / 2  # x1
        boxes[:, 1] -= boxes[:, 3] / 2  # y1
        boxes[:, 2] += boxes[:, 0]      # x2
        boxes[:, 3] += boxes[:, 1]      # y2
        
        # 应用NMS - 使用原始分数
        detections = []
        for class_id in range(filtered_raw_scores.shape[1]):
            class_scores_per_class = filtered_raw_scores[:, class_id]
            keep_indices = class_scores_per_class >= 0.5
            
            if np.sum(keep_indices) > 0:
                class_boxes = boxes[keep_indices]
                class_scores_filtered = class_scores_per_class[keep_indices]
                
                # 应用NMS
                indices = self.nms(class_boxes, class_scores_filtered, iou_threshold)
                
                for idx in indices:
                    x1, y1, x2, y2 = class_boxes[idx]
                    conf = class_scores_filtered[idx]  # 使用原始分数
                    detections.append([x1, y1, x2, y2, conf, class_id])
        
        # 转换为numpy数组并按原始分数排序
        detections = np.array(detections)
        if len(detections) > 0:
            detections = detections[detections[:, 4].argsort()[::-1]]
        
        return detections
    
    def draw_detections(self, image, detections):
        """
        在图像上绘制检测结果
        detections格式: [x1, y1, x2, y2, confidence, class_id]
        """
        # 绘制结果
        for detection in detections:
            x1, y1, x2, y2, conf, class_id = detection
            
            # 确保类别ID有效
            if class_id < 0 or class_id >= len(CLASS_NAMES):
                class_name = f"Unknown ({class_id})"
            else:
                class_name = CLASS_NAMES[class_id]
            
            # 绘制边界框
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # 绘制标签 - 显示原始分数
            label = f"{class_name}: raw={conf:.2f}"
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # 确保标签在图像范围内
            y_label = max(y1, label_size[1] + 10)
            
            # 绘制标签背景
            cv2.rectangle(image, (int(x1), int(y_label - label_size[1] - 10)), 
                          (int(x1 + label_size[0]), int(y_label + base_line - 10)), 
                          (255, 255, 255), cv2.FILLED)
            
            # 绘制标签文本
            cv2.putText(image, label, (int(x1), int(y_label - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return image
        
    def nms(self, boxes, scores, iou_threshold=0.45):
        """非最大抑制算法"""
        # 按置信度降序排序
        order = scores.argsort()[::-1]
        keep = []
        
        while order.size > 0:
            # 保留当前置信度最高的边界框
            i = order[0]
            keep.append(i)
            
            # 计算与其他边界框的IOU
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
            
            # 计算交集面积
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            # 计算并集面积
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_j = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
            union = area_i + area_j - inter
            
            # 计算IOU
            iou = inter / union
            
            # 保留IOU小于阈值的边界框
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep

def test_multiple_images():
    """
    测试6类缺陷图像，收集识别结果
    """
    # 从每个缺陷类别选择代表性图像
    test_images = [
        # {"path": "D:/VisualRobot-Git\VisualRobot\Img/test_pcb/04_missing_hole_14.jpg", "expected_class": "missing_hole"},
        # {"path": "D:/VisualRobot-Git\VisualRobot\Img/test_pcb/05_open_circuit_08.jpg", "expected_class": "open_circuit"},
        # {"path": "D:/VisualRobot-Git\VisualRobot\Img/test_pcb/04_short_20.jpg", "expected_class": "short"},
        # {"path": "D:/VisualRobot-Git\VisualRobot\Img/test_pcb/04_spurious_copper_09.jpg", "expected_class": "spurious_copper"},
        # {"path": "D:/VisualRobot-Git\VisualRobot\Img/test_pcb/01_spur_18.jpg", "expected_class": "spur"},
        {"path": "D:/VisualRobot-Git\VisualRobot\Img/test_pcb/01_mouse_bite_09.jpg", "expected_class": "mouse_bite"}
    ]
    
    print("开始测试6类缺陷图像...")
    
    # 设置模型路径
    model_path = "d:/VisualRobot-Git/VisualRobot/models/yolopcb.onnx"
    
    # 创建检测器实例
    detector = YOLODetector(model_path)
    
    # 遍历测试图像
    for i, test_img in enumerate(test_images):
        img_path = test_img["path"]
        expected_class = test_img["expected_class"]
        
        print(f"\n=== 测试图像 {i+1}/{len(test_images)} ===")
        print(f"图像路径: {img_path}")
        print(f"期望类别: {expected_class}")
        
        try:
            # 读取图像
            img = cv2.imread(img_path)
            if img is None:
                print(f"无法加载图像: {img_path}")
                continue
            
            # 预处理图像
            img_resized = detector.preprocess(img)
            
            # 从检测器实例获取缩放比例和填充值
            ratio = detector.ratio
            dw, dh = detector.dw, detector.dh
            
            # 执行推理 - 使用正确的变量名img_resized
            outputs = detector.session.run([detector.output_name], {detector.input_name: img_resized})
            
            # 后处理 - 使用原始分数阈值0.5
            detections = detector.postprocess(outputs, conf_threshold=0.5)
            
            # 调整边界框到原始图像尺寸
            adjusted_detections = []
            for detection in detections:
                x1, y1, x2, y2, conf, class_id = detection
                # 调整坐标回原始图像
                x1 = (x1 - dw) / ratio
                y1 = (y1 - dh) / ratio
                x2 = (x2 - dw) / ratio
                y2 = (y2 - dh) / ratio
                adjusted_detections.append([x1, y1, x2, y2, conf, class_id])
            
            # 绘制结果
            result_img = img.copy()
            result_img = detector.draw_detections(result_img, adjusted_detections)
            
            # 保存结果图像
            output_path = f"d:/VisualRobot-Git/VisualRobot/ImgData/{expected_class}_result_raw.jpg"
            cv2.imwrite(output_path, result_img)
            print(f"保存结果到 {output_path}")
            
            # 分析检测结果
            detected_classes = [CLASS_NAMES[int(d[5])] for d in adjusted_detections if d[5] < len(CLASS_NAMES)]
            print(f"检测到的类别: {detected_classes}")
            
        except Exception as e:
            print(f"处理图像时出错: {str(e)}")

if __name__ == "__main__":
    # 调用测试函数
    test_multiple_images()
    # 注释掉原来的main函数调用
    # main()