import torch.onnx
from ultralytics import YOLO
from copy import deepcopy
import torch
import os

def load_yolo_model(modelpath, device='cuda'):
    """
    加载 YOLO 模型并进行预处理
    
    Args:
        modelpath: 模型文件路径
        device: 计算设备 ('cuda' 或 'cpu')
    
    Returns:
        处理后的模型
    """
    yolo = YOLO(modelpath)
    model = yolo.model
    
    model = deepcopy(model).to(device)
    for p in model.parameters():
        p.requires_grad = False
    model.fuse().eval()  # 卷积和 BN 融合
    model.to(device)
    return model

def convert_yolo_to_onnx(model_path, output_path, input_size=(640, 640), device='cuda'):
    """
    将 YOLO 模型转换为 ONNX 格式
    
    Args:
        model_path: 输入的 .pt 模型路径
        output_path: 输出的 .onnx 模型路径
        input_size: 输入图像大小 (height, width)
        device: 计算设备
    
    Returns:
        输出文件路径
    """
    print(f"正在加载模型: {model_path}")
    
    # 1. 加载 PyTorch 模型
    model = load_yolo_model(model_path, device)
    model.eval()  # 切换到推理模式
    
    # 2. 创建输入示例
    # YOLO 模型通常接受 [batch_size, 3, height, width] 的输入
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(device)
    
    # 3. 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 4. 将模型转换为 ONNX 格式
    print(f"正在转换为 ONNX 格式，输出路径: {output_path}")
    torch.onnx.export(
        model,                          # 模型
        dummy_input,                    # 模型输入
        output_path,                    # 输出文件路径
        export_params=True,             # 是否导出训练的参数
        opset_version=12,               # ONNX 的 opset 版本 (推荐使用 11 或更高)
        do_constant_folding=True,       # 是否执行常量折叠
        input_names=['images'],         # 输入节点名称
        output_names=['outputs'],       # 输出节点名称
        dynamic_axes=None,
        verbose=False                   # 是否显示详细信息
    )
    
    print(f"✅ 模型已成功转换为 ONNX 格式!")
    print(f"📁 保存路径: {output_path}")
    print(f"📏 输入大小: {input_size}")
    print(f"💻 设备: {device}")
    
    return output_path

def main():
    """主函数 - 示例用法"""
    # 配置参数
    model_path = r'D:\Python-Git\runs\detect\yolo11n-arcuchi-swanlab\weights\best.pt'
    output_path = r'D:\Python-Git\runs\detect\yolo11n-arcuchi-swanlab\weights\best.onnx'
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"❌ 错误: 模型文件不存在 - {model_path}")
        return
    
    try:
        # 转换为 ONNX
        convert_yolo_to_onnx(
            model_path=model_path,
            output_path=output_path,
            input_size=(640, 640),  # 可以根据需要调整输入大小
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # 验证输出文件
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            print(f"✅ ONNX 模型文件验证成功!")
            print(f"📊 文件大小: {file_size:.2f} MB")
        else:
            print("❌ 转换失败: 输出文件未生成")
            
    except Exception as e:
        print(f"❌ 转换过程中发生错误: {str(e)}")
        print("请检查模型文件是否损坏或路径是否正确")

if __name__ == "__main__":
    main()