#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Info_OnnxModel.py - ONNX模型信息分析和可视化工具

功能：
1. 分析ONNX模型的输入输出信息
2. 测试推理并生成可视化结果
3. 提供C++代码部署的参考信息

使用方法：
python Info_OnnxModel.py --model path/to/model.onnx --image path/to/test.jpg

作者：Claude
日期：2024
"""

import onnxruntime as ort
import numpy as np
import cv2
import argparse
import json
import sys
import os
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class ONNXModelAnalyzer:
    """ONNX模型分析器"""
    
    def __init__(self, model_path: str):
        """
        初始化模型分析器
        
        Args:
            model_path: ONNX模型文件路径
        """
        self.model_path = model_path
        self.session = None
        self.input_info = {}
        self.output_info = {}
        self.input_names = []
        self.output_names = []
        
        # 加载模型
        self._load_model()
        
    def _load_model(self):
        """加载ONNX模型"""
        try:
            self.session = ort.InferenceSession(self.model_path)
            self.input_names = [input.name for input in self.session.get_inputs()]
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            print(f"✅ 模型加载成功: {self.model_path}")
            print(f"📊 输入节点数量: {len(self.input_names)}")
            print(f"📊 输出节点数量: {len(self.output_names)}")
            
            # 分析输入输出信息
            self._analyze_io_info()
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
    
    def _analyze_io_info(self):
        """分析输入输出信息"""
        # 分析输入
        for input_info in self.session.get_inputs():
            self.input_info[input_info.name] = {
                'shape': input_info.shape,
                'dtype': str(input_info.type),
                'name': input_info.name
            }
        
        # 分析输出
        for output_info in self.session.get_outputs():
            self.output_info[output_info.name] = {
                'shape': output_info.shape,
                'dtype': str(output_info.type),
                'name': output_info.name
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型基本信息"""
        return {
            'model_path': self.model_path,
            'input_nodes': self.input_info,
            'output_nodes': self.output_info,
            'input_names': self.input_names,
            'output_names': self.output_names,
            'providers': self.session.get_providers()
        }
    
    def preprocess_image(self, image_path: str, target_size: Tuple[int, int] = None) -> np.ndarray:
        """
        图像预处理
        
        Args:
            image_path: 图像路径
            target_size: 目标尺寸 (width, height)
            
        Returns:
            预处理后的图像numpy数组
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        original_height, original_width = image.shape[:2]
        print(f"📷 原始图像尺寸: {original_width}x{original_height}")
        
        # 调整尺寸
        if target_size is None:
            # 使用模型的默认输入尺寸
            input_shape = self.input_info[self.input_names[0]]['shape']
            if len(input_shape) == 4:  # NCHW or NHWC
                target_size = (input_shape[3], input_shape[2])  # (width, height)
            else:
                target_size = (640, 640)  # 默认尺寸
        
        # 缩放图像
        resized = cv2.resize(image, target_size)
        
        # 标准化 (假设使用ImageNet标准)
        normalized = resized.astype(np.float32) / 255.0
        
        # 通道转换 (BGR to RGB)
        rgb_image = cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)
        
        # 调整维度顺序 (HWC to CHW)
        transposed = np.transpose(rgb_image, (2, 0, 1))
        
        # 添加batch维度
        batch_image = np.expand_dims(transposed, axis=0)
        
        print(f"🔧 预处理完成，目标尺寸: {target_size[0]}x{target_size[1]}")
        print(f"📐 最终张量形状: {batch_image.shape}")
        
        return batch_image, (original_width, original_height)
    
    def run_inference(self, input_data: np.ndarray) -> List[np.ndarray]:
        """
        运行推理
        
        Args:
            input_data: 输入数据
            
        Returns:
            输出结果列表
        """
        try:
            # 构建输入字典
            inputs = {self.input_names[0]: input_data}
            
            # 运行推理
            outputs = self.session.run(None, inputs)
            
            print(f"🚀 推理完成，输出数量: {len(outputs)}")
            
            return outputs
            
        except Exception as e:
            print(f"❌ 推理失败: {e}")
            raise
    
    def analyze_yolo_output(self, outputs: List[np.ndarray]) -> Dict[str, Any]:
        """
        分析YOLO输出
        
        Args:
            outputs: 模型输出
            
        Returns:
            分析结果
        """
        analysis = {
            'num_outputs': len(outputs),
            'output_shapes': [],
            'detected_format': 'unknown',
            'suggested_postprocessing': {}
        }
        
        for i, output in enumerate(outputs):
            shape = output.shape
            analysis['output_shapes'].append({
                'output_index': i,
                'shape': shape,
                'dtype': str(output.dtype),
                'min_value': float(np.min(output)),
                'max_value': float(np.max(output)),
                'mean_value': float(np.mean(output))
            })
            
            print(f"📊 输出层 {i}:")
            print(f"   形状: {shape}")
            print(f"   数据类型: {output.dtype}")
            print(f"   数值范围: {np.min(output):.6f} ~ {np.max(output):.6f}")
            print(f"   平均值: {np.mean(output):.6f}")
            
            # 检测YOLO格式
            if len(shape) == 3:
                if shape[1] < shape[2]:  # attributes < num_detections
                    analysis['detected_format'] = 'yolov8_format1'  # [batch, attributes, num_detections]
                    analysis['suggested_postprocessing'] = {
                        'format': 'YOLOv8 ONNX格式1',
                        'attributes': shape[1],
                        'detections': shape[2],
                        'description': '[batch, attributes, num_detections] - attributes: [x,y,w,h,obj_conf,class_0_conf,...]'
                    }
                elif shape[1] > shape[2]:  # attributes > num_detections
                    analysis['detected_format'] = 'yolov8_format2'  # [batch, num_detections, attributes]
                    analysis['suggested_postprocessing'] = {
                        'format': 'YOLOv8 ONNX格式2',
                        'detections': shape[1],
                        'attributes': shape[2],
                        'description': '[batch, num_detections, attributes] - attributes: [x,y,w,h,obj_conf,class_0_conf,...]'
                    }
            elif len(shape) == 2:
                analysis['detected_format'] = 'traditional_yolo'  # [rows, cols]
                analysis['suggested_postprocessing'] = {
                    'format': '传统YOLO格式',
                    'rows': shape[0],
                    'cols': shape[1],
                    'description': '[rows, cols] - 每行一个检测，包含[x,y,w,h,confidence,class_probs...]'
                }
        
        return analysis
    
    def visualize_results(self, image_path: str, outputs: List[np.ndarray], 
                         output_path: str = "result.jpg", conf_threshold: float = 0.5):
        """
        可视化检测结果
        
        Args:
            image_path: 原始图像路径
            outputs: 模型输出
            output_path: 输出图像路径
            conf_threshold: 置信度阈值
        """
        # 读取并预处理图像
        input_data, original_size = self.preprocess_image(image_path)
        
        # 分析输出
        analysis = self.analyze_yolo_output(outputs)
        
        # 读取原始图像
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 创建可视化图像
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 显示原始图像
        ax1.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        ax1.set_title('原始图像')
        ax1.axis('off')
        
        # 解析检测结果并绘制
        detections = self._parse_detections(outputs, analysis, original_size)
        
        # 在图像上绘制检测结果
        image_with_detections = original_image.copy()
        for detection in detections:
            if detection['confidence'] > conf_threshold:
                bbox = detection['bbox']
                label = f"{detection['class_name']} {detection['confidence']:.2f}"
                
                # 绘制边界框
                cv2.rectangle(image_with_detections, 
                            (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                            (0, 255, 0), 2)
                
                # 绘制标签
                cv2.putText(image_with_detections, label,
                          (int(bbox[0]), int(bbox[1]) - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        ax2.imshow(cv2.cvtColor(image_with_detections, cv2.COLOR_BGR2RGB))
        ax2.set_title(f'检测结果 (阈值: {conf_threshold})')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"📊 可视化结果已保存: {output_path}")
        
        return detections, analysis
    
    def _parse_detections(self, outputs: List[np.ndarray], analysis: Dict, 
                         original_size: Tuple[int, int]) -> List[Dict]:
        """
        解析检测结果
        
        Args:
            outputs: 模型输出
            analysis: 分析结果
            original_size: 原始图像尺寸
            
        Returns:
            解析后的检测结果
        """
        detections = []
        width, height = original_size
        
        if analysis['detected_format'] == 'yolov8_format1':
            # YOLOv8格式1: [batch, attributes, num_detections]
            output = outputs[0]  # 假设第一个输出包含检测结果
            batch_size, num_attributes, num_detections = output.shape
            
            for d in range(min(num_detections, 1000)):  # 限制处理数量
                obj_conf = output[0, 4, d]  # 对象置信度
                if obj_conf > 0.5:  # 第一层过滤
                    # 找到最高类别置信度
                    class_id = -1
                    max_class_conf = 0
                    for c in range(5, num_attributes):
                        class_conf = output[0, c, d]
                        if class_conf > max_class_conf:
                            max_class_conf = class_conf
                            class_id = c - 5
                    
                    if class_id >= 0:
                        final_conf = obj_conf * max_class_conf
                        if final_conf > 0.5:  # 第二层过滤
                            # 解析边界框
                            x_center = output[0, 0, d]
                            y_center = output[0, 1, d]
                            w_norm = output[0, 2, d]
                            h_norm = output[0, 3, d]
                            
                            # 转换为像素坐标
                            x1 = (x_center - w_norm/2) * width
                            y1 = (y_center - h_norm/2) * height
                            w = w_norm * width
                            h = h_norm * height
                            
                            detections.append({
                                'bbox': [x1, y1, w, h],
                                'confidence': final_conf,
                                'class_id': class_id,
                                'class_name': f'Class_{class_id}'
                            })
        
        elif analysis['detected_format'] == 'yolov8_format2':
            # YOLOv8格式2: [batch, num_detections, attributes]
            output = outputs[0]
            batch_size, num_detections, num_attributes = output.shape
            
            for d in range(min(num_detections, 1000)):
                obj_conf = output[0, d, 4]
                if obj_conf > 0.5:
                    class_id = -1
                    max_class_conf = 0
                    for c in range(5, num_attributes):
                        class_conf = output[0, d, c]
                        if class_conf > max_class_conf:
                            max_class_conf = class_conf
                            class_id = c - 5
                    
                    if class_id >= 0:
                        final_conf = obj_conf * max_class_conf
                        if final_conf > 0.5:
                            x_center = output[0, d, 0]
                            y_center = output[0, d, 1]
                            w_norm = output[0, d, 2]
                            h_norm = output[0, d, 3]
                            
                            x1 = (x_center - w_norm/2) * width
                            y1 = (y_center - h_norm/2) * height
                            w = w_norm * width
                            h = h_norm * height
                            
                            detections.append({
                                'bbox': [x1, y1, w, h],
                                'confidence': final_conf,
                                'class_id': class_id,
                                'class_name': f'Class_{class_id}'
                            })
        
        return detections
    
    def generate_cpp_deployment_info(self) -> Dict[str, Any]:
        """
        生成C++部署参考信息
        
        Returns:
            C++代码修改建议
        """
        cpp_info = {
            'headers': [],
            'preprocessing': {},
            'model_loading': {},
            'inference': {},
            'postprocessing': {},
            'code_examples': {}
        }
        
        # 生成头文件信息
        cpp_info['headers'] = [
            '#include <opencv2/dnn/dnn.hpp>',
            '#include <onnxruntime/core/session/onnxruntime_cxx_api.h>',
            '#include <vector>',
            '#include <string>'
        ]
        
        # 预处理建议
        input_shape = self.input_info[self.input_names[0]]['shape']
        cpp_info['preprocessing'] = {
            'description': '根据分析结果生成的图像预处理代码',
            'suggested_code': f'''
// 图像预处理 (基于分析结果)
cv::Mat preprocess_image(const cv::Mat& image) {{
    // 调整尺寸到模型输入尺寸
    cv::Mat resized;
    cv::resize(image, resized, cv::Size({input_shape[3]}, {input_shape[2]}));
    
    // 转换为浮点类型并归一化
    resized.convertTo(resized, CV_32F);
    resized = resized / 255.0f;
    
    // 通道转换 (BGR to RGB) 和维度调整
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    cv::dnn::blobFromImage(resized, resized, 1.0, cv::Size(), cv::Scalar(), false, false);
    
    return resized;
}}'''
        }
        
        # 模型加载建议
        cpp_info['model_loading'] = {
            'description': 'ONNX模型加载代码',
            'suggested_code': f'''
// ONNX模型加载
ort::SessionOptions session_options;
session_options.SetInterOpNumThreads(1);
session_options.SetIntraOpNumThreads(1);

std::unique_ptr<ort::Session> session;
try {{
    session.reset(new ort::Session(env, "{self.model_path}", session_options));
    std::cout << "模型加载成功" << std::endl;
}} catch (const std::exception& e) {{
    std::cerr << "模型加载失败: " << e.what() << std::endl;
}}'''
        }
        
        # 推理建议
        cpp_info['inference'] = {
            'description': '模型推理代码',
            'suggested_code': '''
// 模型推理
std::vector<Ort::Value> input_tensors;
input_tensors.push_back(Ort::Value::CreateTensor<float>(
    memory_info, input_data.data(), input_size,
    input_shape.data(), input_shape.size()));

auto output_tensors = session->Run(Ort::RunOptions{nullptr},
    input_names.data(), input_tensors.data(), 1,
    output_names.data(), output_names.size());'''
        }
        
        # 后处理建议
        if self.output_info:
            output_shape = list(self.output_info[self.output_names[0]]['shape'].values())
            cpp_info['postprocessing'] = {
                'description': '基于检测结果的后处理代码',
                'detected_format': analysis.get('detected_format', 'unknown'),
                'suggested_code': '''
// YOLO检测结果后处理 (示例)
std::vector<DetectionResult> postprocess_yolo(
    const std::vector<Ort::Value>& output_tensors,
    float conf_threshold = 0.5f,
    float nms_threshold = 0.4f) {
    
    std::vector<DetectionResult> results;
    // 根据实际输出格式实现具体的解析逻辑
    // 参见上方analyze_yolo_output()的分析结果
    
    return results;
}'''
            }
        
        return cpp_info
    
    def print_summary(self):
        """打印模型摘要信息"""
        print("\n" + "="*60)
        print("📋 ONNX模型分析摘要")
        print("="*60)
        
        # 基本信息
        print(f"📁 模型路径: {self.model_path}")
        print(f"🔧 执行提供者: {', '.join(self.session.get_providers())}")
        
        # 输入信息
        print(f"\n📥 输入节点 ({len(self.input_names)}个):")
        for name in self.input_names:
            info = self.input_info[name]
            print(f"  - {name}: {info['shape']} ({info['dtype']})")
        
        # 输出信息
        print(f"\n📤 输出节点 ({len(self.output_names)}个):")
        for name in self.output_names:
            info = self.output_info[name]
            print(f"  - {name}: {info['shape']} ({info['dtype']})")
        
        print("="*60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='ONNX模型信息分析工具')
    parser.add_argument('--model', type=str, required=True,
                        help='ONNX模型文件路径')
    parser.add_argument('--image', type=str, required=True,
                        help='测试图像文件路径')
    parser.add_argument('--output', type=str, default='analysis_result.jpg',
                        help='输出结果图像路径')
    parser.add_argument('--conf_threshold', type=float, default=0.5,
                        help='置信度阈值')
    parser.add_argument('--save_json', type=str, default='model_analysis.json',
                        help='保存分析结果到JSON文件')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.model):
        print(f"❌ 模型文件不存在: {args.model}")
        sys.exit(1)
    
    if not os.path.exists(args.image):
        print(f"❌ 图像文件不存在: {args.image}")
        sys.exit(1)
    
    try:
        # 创建分析器
        analyzer = ONNXModelAnalyzer(args.model)
        
        # 打印摘要
        analyzer.print_summary()
        
        # 预处理图像
        input_data, original_size = analyzer.preprocess_image(args.image)
        
        # 运行推理
        outputs = analyzer.run_inference(input_data)
        
        # 分析输出
        analysis = analyzer.analyze_yolo_output(outputs)
        
        # 可视化结果
        detections, analysis = analyzer.visualize_results(
            args.image, outputs, args.output, args.conf_threshold)
        
        # 生成C++部署信息
        cpp_info = analyzer.generate_cpp_deployment_info()
        
        # 保存完整分析结果
        complete_analysis = {
            'model_info': analyzer.get_model_info(),
            'yolo_analysis': analysis,
            'detections': detections,
            'cpp_deployment_info': cpp_info
        }
        
        with open(args.save_json, 'w', encoding='utf-8') as f:
            json.dump(complete_analysis, f, ensure_ascii=False, indent=2)
        print(f"💾 完整分析结果已保存到: {args.save_json}")
        
        # 打印C++部署建议
        print(f"\n🔧 C++代码部署建议:")
        print("="*40)
        print(cpp_info['preprocessing']['description'])
        print(cpp_info['preprocessing']['suggested_code'])
        
        print(f"\n📋 建议修改的文件:")
        print("  - DataProcessor.cpp: 添加模型预处理逻辑")
        print("  - DLProcessor.cpp: 根据输出格式修改PostProcessYolo方法")
        print("  - 确保正确设置输入张量和后处理参数")
        
        print(f"\n✅ 分析完成!")
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()