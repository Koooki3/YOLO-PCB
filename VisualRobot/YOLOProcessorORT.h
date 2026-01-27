/**
 * @file YOLOProcessorORT.h
 * @brief YOLO处理器模块头文件（基于ONNX Runtime）
 * 
 * 该文件定义了YOLOProcessorORT类，提供基于ONNX Runtime的YOLO模型推理功能，
 * 包括模型加载、目标检测、结果绘制、延时统计和调试输出等功能。
 * 
 * @author VisualRobot Team
 * @date 2025-12-30
 * @version 1.0
 */

#ifndef YOLOPROCESSORORT_H
#define YOLOPROCESSORORT_H

#include <QObject>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include "DLProcessor.h" // for DetectionResult
#include <chrono>

using namespace cv;
using namespace std;

/**
 * @brief YOLO模型处理延时统计结构体
 * 
 * 记录YOLO模型处理流程中各个阶段的耗时情况，用于性能分析和优化
 * 
 * @note 时间单位为毫秒（ms）
 * @note FPS基于总处理时间计算
 */
struct YOLOTimingStats {
    double preprocessTime;    ///< 图像预处理时间（毫秒）
    double inferenceTime;     ///< 模型推理时间（毫秒）
    double postprocessTime;   ///< 后处理时间（毫秒）
    double totalTime;         ///< 总处理时间（毫秒）
    int fps;                  ///< 帧率（FPS）
};

/**
 * @brief YOLO调试信息结构体
 * 
 * 用于存储和传递YOLO检测过程中的调试信息，包含处理过程中的关键数据
 * 
 * @note 仅在启用调试输出时使用，避免影响正常处理性能
 */
struct YOLODebugInfo {
    double letterbox_r;               ///< letterbox缩放比例
    double letterbox_dw;              ///< letterbox宽度方向填充
    double letterbox_dh;              ///< letterbox高度方向填充
    size_t outputTensorCount;         ///< 模型输出张量数量
    vector<vector<int64_t>> outputShapes; ///< 模型输出形状
    vector<Mat> mats;                 ///< 转换后的输出矩阵
    Mat dets;                         ///< 检测结果矩阵
    double preprocessTime;            ///< 预处理时间
    double inferenceTime;             ///< 推理时间
    double postprocessTime;           ///< 后处理时间
    double totalTime;                 ///< 总时间
    int fps;                          ///< 帧率
};

/**
 * @brief YOLO处理器类，基于ONNX Runtime实现
 * 
 * 该类封装了基于ONNX Runtime的YOLO模型推理功能，支持目标检测、结果绘制等功能
 * 支持不同版本的YOLO模型，通过后处理适配不同的输出格式
 * 
 * @note 支持CUDA加速和OpenCL加速
 * @note 支持 RK3588 NPU：当 hardware_config.accelerators.npu 为 true 且以 CONFIG+=rknpu
 *       构建、并使用带 RKNPU EP 的 ONNX Runtime 时，优先使用 NPU 推理
 * @note 支持自动从配置管理器获取优化参数
 * @note 支持延时统计和调试输出
 * @see DetectionResult, YOLOTimingStats, YOLODebugInfo
 */
class YOLOProcessorORT : public QObject
{
    Q_OBJECT
public:
    /**
     * @brief 构造函数
     * 
     * 初始化YOLO处理器的各项参数，包括ONNX Runtime环境、会话选项、默认输入尺寸等
     * 从配置管理器获取优化参数，包括线程数、内存管理、图优化级别等
     * 
     * @param parent 父对象指针，默认为nullptr
     * @note 初始化内容：
     *       - ONNX Runtime环境（日志级别：警告）
     *       - 默认输入尺寸：640x640
     *       - 默认置信度阈值：50.0%
     *       - 默认NMS阈值：0.45
     *       - 图像缩放因子：1/255
     *       - 通道交换：BGR→RGB
     * @see ConfigManager
     */
    explicit YOLOProcessorORT(QObject* parent = nullptr);
    
    /**
     * @brief 析构函数
     * 
     * 释放ONNX Runtime会话资源
     * 
     * @note 会话资源会自动释放
     */
    ~YOLOProcessorORT();

    /**
     * @brief 初始化YOLO模型
     * 
     * 创建ONNX Runtime会话，加载模型文件，支持CUDA和OpenCL加速
     * 
     * @param modelPath ONNX模型文件路径
     * @param useCUDA 是否使用CUDA加速（默认false）
     * @return 初始化是否成功
     * @note 支持的加速方式：
     *       - CUDA：通过useCUDA参数启用
     *       - OpenCL：从配置管理器自动检测
     *       - NPU（RK3588）：当 accelerators.npu 为 true 且以 USE_RKNPU_EP 构建时，通过 RKNPU EP 推理
     * @note 异常处理：捕获Ort::Exception并发射errorOccurred信号
     * @see ConfigManager::isAcceleratorEnabled()
     */
    bool InitModel(const string& modelPath, bool useCUDA = false);
    
    /**
     * @brief 检测图像中的目标
     * 
     * 执行完整的目标检测流程，包括预处理、推理、后处理和性能统计
     * 
     * @param frame 输入图像
     * @param results 输出检测结果列表
     * @return 检测是否成功
     * @note 检测流程：
     *       1. 图像预处理（letterbox缩放、BGR转RGB、归一化、HWC转CHW）
     *       2. ONNX模型推理
     *       3. 输出结果转换
     *       4. 后处理生成检测结果
     *       5. 性能统计和调试输出
     * @note 异常处理：捕获Ort::Exception和cv::Exception
     * @see PostProcess(), OrtOutputToMats(), DebugOutput()
     */
    bool DetectObjects(const Mat& frame, vector<DetectionResult>& results);

    /**
     * @brief 设置模型输入尺寸
     * 
     * 更新模型的输入尺寸，用于预处理阶段的letterbox缩放
     * 
     * @param size 输入尺寸（宽×高）
     * @note 默认尺寸：640×640
     * @see DetectObjects()
     */
    void SetInputSize(const Size& size);
    
    /**
     * @brief 设置检测阈值
     * 
     * 更新置信度阈值和NMS阈值，用于后处理阶段的过滤
     * 
     * @param conf 置信度阈值（0-100%）
     * @param nms NMS（非极大值抑制）阈值（0-1）
     * @note 置信度阈值单位为百分比，内部会转换为0-1范围
     * @see PostProcess()
     */
    void SetThresholds(float conf, float nms);
    
    /**
     * @brief 设置类别标签
     * 
     * 更新类别标签，用于结果绘制和输出
     * 
     * @param labels 类别标签列表
     * @note 标签数量决定了模型的类别数
     * @see DrawDetectionResults()
     */
    void SetClassLabels(const vector<string>& labels);

    /**
     * @brief 检查模型是否已加载
     * 
     * 验证ONNX Runtime会话是否已成功创建
     * 
     * @return 模型是否已加载
     * @see InitModel()
     */
    bool IsModelLoaded() const { return session_ != nullptr; }
    
    /**
     * @brief 检查类别标签是否已加载
     * 
     * 验证类别标签列表是否非空
     * 
     * @return 类别标签是否已加载
     * @see SetClassLabels()
     */
    bool AreLabelsLoaded() const { return !classLabels_.empty(); }

    /**
     * @brief 在图像上绘制检测结果
     * 
     * 为每个检测结果绘制边界框和标签文本
     * 
     * @param frame 输入输出图像，将在其上绘制检测结果
     * @param results 检测结果列表
     * @note 绘制内容：
     *       - 边界框（线宽3px）
     *       - 类别名称和置信度
     *       - 半透明背景
     * @note 颜色生成：基于类别ID的哈希函数，支持动态扩展
     * @see DetectionResult
     */
    void DrawDetectionResults(Mat& frame, const vector<DetectionResult>& results);
    
    /**
     * @brief 获取最新的延时统计
     * 
     * 返回最近一次检测的性能统计数据
     * 
     * @return 延时统计结构体
     * @note 统计数据在DetectObjects()中更新
     * @see YOLOTimingStats
     */
    YOLOTimingStats GetTimingStats() const;
    
    /**
     * @brief 启用/禁用详细调试输出
     * 
     * 控制调试信息的输出，避免影响正常处理性能
     * 
     * @param enable 是否启用调试输出
     * @note 调试输出内容：处理时间统计
     * @see DebugOutput()
     */
    void SetDebugOutput(bool enable);

signals:
    /**
     * @brief 处理完成信号
     * 
     * 当目标检测完成时发射此信号
     * 
     * @param resultImage 处理后的图像（包含检测框）
     * @note 用于UI更新或结果展示
     */
    void processingComplete(const cv::Mat& resultImage);
    
    /**
     * @brief 错误发生信号
     * 
     * 当检测过程中发生错误时发射此信号
     * 
     * @param error 错误信息
     * @note 用于错误处理和用户提示
     */
    void errorOccurred(const QString& error);

private:
    Ort::Env env_;                          ///< ONNX Runtime环境
    std::unique_ptr<Ort::Session> session_; ///< ONNX Runtime会话
    Ort::SessionOptions sessionOptions_;    ///< ONNX Runtime会话选项

    Size inputSize_;       ///< 模型输入尺寸
    float confThreshold_;  ///< 置信度阈值（百分比）
    float nmsThreshold_;   ///< NMS阈值（0-1）
    double scaleFactor_;   ///< 缩放因子（1/255）
    bool swapRB_;          ///< 是否交换RB通道
    vector<string> classLabels_; ///< 类别标签列表
    
    // letterbox参数，用于预处理和后处理之间的坐标转换
    double letterbox_r_;   ///< letterbox缩放比例
    double letterbox_dw_;  ///< letterbox宽度方向填充（半填充）
    double letterbox_dh_;  ///< letterbox高度方向填充（半填充）
    
    // 延时统计
    YOLOTimingStats timingStats_; ///< 最新的延时统计
    bool enableDebug_;  ///< 是否启用详细调试输出

    /**
     * @brief 将ONNX Runtime输出转换为OpenCV Mat格式
     * 
     * 处理不同形状的ONNX Runtime输出，将其转换为适合后续处理的OpenCV Mat格式
     * 
     * @param outputs ONNX Runtime输出列表
     * @return 转换后的Mat列表
     * @note 支持的形状：
     *       - 4D张量 [1, C, H, W] → (H*W) × C
     *       - 3D张量 [1, C, L] → L × C
     *       - 2D张量 [R, C] → R × C
     *       - 其他形状 → 1 × elemCount
     * @note 数据布局转换：NCHW/NHWC → Mat格式
     */
    vector<Mat> OrtOutputToMats(const std::vector<Ort::Value>& outputs);
    
    /**
     * @brief 处理ONNX输出，生成检测结果
     * 
     * 解析模型输出，提取边界框和类别信息，应用阈值过滤和NMS
     * 
     * @param outputs ONNX Runtime输出列表
     * @param frameSize 原始图像尺寸
     * @param imagePath 图像路径（可选，用于调试）
     * @param expectedClass 期望类别（可选，用于过滤）
     * @return 检测结果列表
     * @note 后处理流程：
     *       1. 解析模型输出张量
     *       2. 提取候选框和类别分数
     *       3. 应用置信度阈值过滤
     *       4. 将坐标映射回原始图像
     *       5. 应用NMS（非极大值抑制）
     *       6. 如果无结果，尝试降低阈值重新过滤
     * @note 坐标映射：考虑letterbox参数，确保坐标准确性
     * @see SetThresholds(), letterbox_r_, letterbox_dw_, letterbox_dh_
     */
    std::vector<DetectionResult> PostProcess(const std::vector<Ort::Value>& outputs, const cv::Size& frameSize, 
                                             const std::string& imagePath = "", const std::string& expectedClass = "");
    
    /**
     * @brief 输出调试信息
     * 
     * 集中处理所有调试输出，避免调试输出影响核心处理逻辑的性能
     * 
     * @param debugInfo 调试信息结构体，包含需要输出的各种调试数据
     * @note 调试内容：处理时间统计（预处理、推理、后处理、总时间、FPS）
     * @note 仅在enableDebug_为true时输出
     * @see YOLODebugInfo, SetDebugOutput()
     */
    void DebugOutput(const YOLODebugInfo& debugInfo);
};

#endif // YOLOPROCESSORORT_H
