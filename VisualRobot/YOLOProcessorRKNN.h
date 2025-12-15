#ifndef YOLOPROCESSORRKNN_H
#define YOLOPROCESSORRKNN_H

#include <QObject>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>
#include "rknn_api.h"
#include "DLProcessor.h" // for DetectionResult
#include <chrono>
#include <sys/time.h>  // 添加gettimeofday需要的头文件

using namespace cv;
using namespace std;

/**
 * @brief RKNN处理延时统计结构体
 */
struct RKNNTimingStats {
    double preprocessTime;    // 图像预处理时间（毫秒）
    double inferenceTime;     // 模型推理时间（毫秒）
    double postprocessTime;   // 后处理时间（毫秒）
    double totalTime;         // 总处理时间（毫秒）
    int fps;                  // 帧率（FPS）
};

/**
 * @brief RKNN调试信息结构体
 */
struct RKNNImageInfo {
    int width;                // 原始图像宽度
    int height;               // 原始图像高度
    int channels;             // 原始图像通道数
    int targetWidth;          // 目标宽度
    int targetHeight;         // 目标高度
    double letterboxR;        // 缩放比例
    double letterboxDw;       // 宽度填充
    double letterboxDh;       // 高度填充
};

/**
 * @brief RKNN输出张量信息
 */
struct RKNNTensorInfo {
    int index;                // 张量索引
    string name;             // 张量名称
    vector<int64_t> shape;   // 张量形状
    rknn_tensor_type type;   // 数据类型
    rknn_tensor_format fmt;  // 数据格式
    float scale;             // 量化缩放系数
    int zero_point;          // 量化零点
};

/**
 * @brief RKNN处理器类，基于Rockchip RKNN实现
 */
class YOLOProcessorRKNN : public QObject
{
    Q_OBJECT

public:
    /**
     * @brief 构造函数
     * @param parent 父对象指针
     */
    explicit YOLOProcessorRKNN(QObject* parent = nullptr);

    /**
     * @brief 析构函数
     */
    ~YOLOProcessorRKNN();

    /**
     * @brief 初始化RKNN模型
     * @param modelPath RKNN模型文件路径
     * @param labelsPath 标签文件路径（可选）
     * @return 初始化是否成功
     */
    bool InitModel(const string& modelPath, const string& labelsPath = "");

    /**
     * @brief 检测图像中的目标
     * @param frame 输入图像
     * @param results 输出检测结果列表
     * @return 检测是否成功
     */
    bool DetectObjects(const Mat& frame, vector<DetectionResult>& results);

    /**
     * @brief 处理图像并保存结果
     * @param imagePath 输入图像路径
     * @param outputPath 输出结果图像路径（可选）
     * @param saveResult 是否保存结果图像
     * @return 处理是否成功
     */
    bool ProcessImage(const string& imagePath, const string& outputPath = "", bool saveResult = true);

    /**
     * @brief 设置检测阈值
     * @param conf 置信度阈值
     * @param nms NMS阈值
     */
    void SetThresholds(float conf, float nms);

    /**
     * @brief 设置输入图像尺寸
     * @param width 宽度
     * @param height 高度
     */
    void SetInputSize(int width, int height);

    /**
     * @brief 获取模型输入尺寸
     * @return 输入尺寸
     */
    cv::Size GetInputSize() const;

    /**
     * @brief 加载类别标签
     * @param labelsPath 标签文件路径
     * @return 加载是否成功
     */
    bool LoadLabels(const string& labelsPath);

    /**
     * @brief 在图像上绘制检测结果
     * @param frame 输入输出图像
     * @param results 检测结果列表
     * @param outputPath 保存路径（可选）
     */
    void DrawDetectionResults(Mat& frame, const vector<DetectionResult>& results, const string& outputPath = "");

    /**
     * @brief 获取延时统计信息
     * @return 延时统计结构体
     */
    RKNNTimingStats GetTimingStats() const;

    /**
     * @brief 启用/禁用调试输出
     * @param enable 是否启用
     */
    void SetDebugOutput(bool enable);

    /**
     * @brief 检查模型是否已加载
     * @return 模型是否已加载
     */
    bool IsModelLoaded() const;

    /**
     * @brief 检查标签是否已加载
     * @return 标签是否已加载
     */
    bool AreLabelsLoaded() const;

    /**
     * @brief 获取模型信息
     */
    void PrintModelInfo() const;

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

    /**
     * @brief 检测结果信号
     * @param results 检测结果
     */
    void detectionResults(const std::vector<DetectionResult>& results);

private:
    rknn_context ctx_;                      // RKNN上下文
    rknn_input_output_num io_num_;          // 输入输出数量
    rknn_tensor_attr* input_attrs_;         // 输入张量属性
    rknn_tensor_attr* output_attrs_;        // 输出张量属性
    rknn_tensor_mem** input_mems_;          // 输入内存
    rknn_tensor_mem** output_mems_;         // 输出内存

    cv::Size input_size_;                   // 模型输入尺寸
    float conf_threshold_;                  // 置信度阈值
    float nms_threshold_;                   // NMS阈值
    vector<string> class_labels_;           // 类别标签

    // letterbox参数
    double letterbox_r_;                    // 缩放比例
    double letterbox_dw_;                   // 宽度填充
    double letterbox_dh_;                   // 高度填充

    RKNNTimingStats timing_stats_;          // 延时统计
    bool enable_debug_;                     // 启用调试输出

    // 模型信息
    string model_path_;                     // 模型路径
    string sdk_version_;                    // SDK版本
    string driver_version_;                 // 驱动版本

    /**
     * @brief 预处理图像
     * @param frame 输入图像
     * @param preprocessed 预处理后的图像数据
     * @param img_info 图像信息（输出参数）
     * @return 预处理是否成功
     */
    bool PreprocessImage(const Mat& frame, vector<uint8_t>& preprocessed, RKNNImageInfo& img_info);

    /**
     * @brief 后处理输出
     * @param outputs 模型输出数据
     * @param img_info 图像信息
     * @return 检测结果列表
     */
    vector<DetectionResult> PostProcess(const vector<float*>& outputs, const RKNNImageInfo& img_info);

    /**
     * @brief 解析YOLO输出
     * @param output_data 输出数据指针
     * @param output_shape 输出形状
     * @param img_info 图像信息
     * @param boxes 输出边界框
     * @param scores 输出分数
     * @param class_ids 输出类别ID
     */
    void ParseYOLOOutput(const float* output_data, const vector<int64_t>& output_shape,
                         const RKNNImageInfo& img_info,
                         vector<Rect>& boxes, vector<float>& scores, vector<int>& class_ids);

    /**
     * @brief 将归一化坐标转换为像素坐标
     * @param x 归一化x坐标
     * @param y 归一化y坐标
     * @param w 归一化宽度
     * @param h 归一化高度
     * @param img_info 图像信息
     * @param rect 输出像素坐标矩形
     */
    void NormalizedToPixel(float x, float y, float w, float h,
                          const RKNNImageInfo& img_info, Rect& rect);

    /**
     * @brief 应用NMS
     * @param boxes 边界框列表
     * @param scores 分数列表
     * @param indices 输出索引
     */
    void ApplyNMS(const vector<Rect>& boxes, const vector<float>& scores, vector<int>& indices);

    /**
     * @brief 打印调试信息
     * @param img_info 图像信息
     * @param results 检测结果
     */
    void DebugOutput(const RKNNImageInfo& img_info, const vector<DetectionResult>& results);

    /**
     * @brief 释放资源
     */
    void Cleanup();

    /**
     * @brief 获取当前时间（微秒）
     * @return 当前时间戳
     */
    int64_t GetCurrentTimeUs();
};

#endif // YOLOPROCESSORRKNN_H
