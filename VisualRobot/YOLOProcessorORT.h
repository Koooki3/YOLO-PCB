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
 * 记录YOLO模型处理流程中各个阶段的耗时情况
 */
struct YOLOTimingStats {
    double preprocessTime;    // 图像预处理时间（毫秒）
    double inferenceTime;     // 模型推理时间（毫秒）
    double postprocessTime;   // 后处理时间（毫秒）
    double totalTime;         // 总处理时间（毫秒）
    int fps;                  // 帧率（FPS）
};

/**
 * @brief YOLO调试信息结构体
 * 
 * 用于存储和传递YOLO检测过程中的调试信息
 */
struct YOLODebugInfo {
    double letterbox_r;               // letterbox缩放比例
    double letterbox_dw;              // letterbox宽度方向填充
    double letterbox_dh;              // letterbox高度方向填充
    size_t outputTensorCount;         // 模型输出张量数量
    vector<vector<int64_t>> outputShapes; // 模型输出形状
    vector<Mat> mats;                 // 转换后的输出矩阵
    Mat dets;                         // 检测结果矩阵
    double preprocessTime;            // 预处理时间
    double inferenceTime;             // 推理时间
    double postprocessTime;           // 后处理时间
    double totalTime;                 // 总时间
    int fps;                          // 帧率
};

/**
 * @brief YOLO处理器类，基于ONNX Runtime实现
 * 
 * 该类封装了基于ONNX Runtime的YOLO模型推理功能，支持目标检测、结果绘制等功能
 * 支持不同版本的YOLO模型，通过后处理适配不同的输出格式
 */
class YOLOProcessorORT : public QObject
{
    Q_OBJECT
public:
    /**
     * @brief 构造函数
     * @param parent 父对象指针
     */
    explicit YOLOProcessorORT(QObject* parent = nullptr);
    
    /**
     * @brief 析构函数
     */
    ~YOLOProcessorORT();

    /**
     * @brief 初始化YOLO模型
     * @param modelPath ONNX模型文件路径
     * @param useCUDA 是否使用CUDA加速（默认不使用）
     * @return 初始化是否成功
     */
    bool InitModel(const string& modelPath, bool useCUDA = false);
    
    /**
     * @brief 检测图像中的目标
     * @param frame 输入图像
     * @param results 输出检测结果列表
     * @return 检测是否成功
     */
    bool DetectObjects(const Mat& frame, vector<DetectionResult>& results);

    /**
     * @brief 设置模型输入尺寸
     * @param size 输入尺寸
     */
    void SetInputSize(const Size& size);
    
    /**
     * @brief 设置检测阈值
     * @param conf 置信度阈值
     * @param nms NMS（非极大值抑制）阈值
     */
    void SetThresholds(float conf, float nms);
    
    /**
     * @brief 设置类别标签
     * @param labels 类别标签列表
     */
    void SetClassLabels(const vector<string>& labels);

    /**
     * @brief 检查模型是否已加载
     * @return 模型是否已加载
     */
    bool IsModelLoaded() const { return session_ != nullptr; }
    
    /**
     * @brief 检查类别标签是否已加载
     * @return 类别标签是否已加载
     */
    bool AreLabelsLoaded() const { return !classLabels_.empty(); }

    /**
     * @brief 在图像上绘制检测结果
     * @param frame 输入输出图像，将在其上绘制检测结果
     * @param results 检测结果列表
     */
    void DrawDetectionResults(Mat& frame, const vector<DetectionResult>& results);
    
    /**
     * @brief 获取最新的延时统计
     * @return 延时统计结构体
     */
    YOLOTimingStats GetTimingStats() const;
    
    /**
     * @brief 启用/禁用详细调试输出
     * @param enable 是否启用
     */
    void SetDebugOutput(bool enable);

signals:
    /**
     * @brief 处理完成信号
     * @param resultImage 处理后的图像
     */
    void processingComplete(const cv::Mat& resultImage);
    
    /**
     * @brief 错误发生信号
     * @param error 错误信息
     */
    void errorOccurred(const QString& error);

private:
    Ort::Env env_;                          // ONNX Runtime环境
    std::unique_ptr<Ort::Session> session_; // ONNX Runtime会话
    Ort::SessionOptions sessionOptions_;    // ONNX Runtime会话选项

    Size inputSize_;       // 模型输入尺寸
    float confThreshold_;  // 置信度阈值
    float nmsThreshold_;   // NMS阈值
    double scaleFactor_;   // 缩放因子
    bool swapRB_;          // 是否交换RB通道
    vector<string> classLabels_; // 类别标签列表
    
    // letterbox参数，用于预处理和后处理之间的坐标转换
    double letterbox_r_;   // 缩放比例
    double letterbox_dw_;  // 宽度方向的填充
    double letterbox_dh_;  // 高度方向的填充
    
    // 延时统计
    YOLOTimingStats timingStats_; // 最新的延时统计
    bool enableDebug_;  // 是否启用详细调试输出
    
    // 调试输出频率控制
    std::chrono::high_resolution_clock::time_point lastDebugOutputTime_; // 上次输出调试信息的时间
    const double DEBUG_OUTPUT_INTERVAL_SECONDS = 30.0; // 调试输出间隔（秒）

    /**
     * @brief 将ONNX Runtime输出转换为OpenCV Mat格式
     * @param outputs ONNX Runtime输出列表
     * @return 转换后的Mat列表
     */
    vector<Mat> OrtOutputToMats(const std::vector<Ort::Value>& outputs);
    
    /**
     * @brief 处理ONNX输出，生成检测结果
     * @param outputs ONNX Runtime输出列表
     * @param frameSize 原始图像尺寸
     * @param imagePath 图像路径（可选）
     * @param expectedClass 期望类别（可选）
     * @return 检测结果列表
     */
    std::vector<DetectionResult> PostProcess(const std::vector<Ort::Value>& outputs, const cv::Size& frameSize, 
                                             const std::string& imagePath = "", const std::string& expectedClass = "");
    
    /**
     * @brief 输出调试信息
     * @param debugInfo 调试信息结构体，包含需要输出的各种调试数据
     * 
     * 集中处理所有调试输出，避免调试输出影响核心处理逻辑的性能
     */
    void DebugOutput(const YOLODebugInfo& debugInfo);
};

#endif // YOLOPROCESSORORT_H
