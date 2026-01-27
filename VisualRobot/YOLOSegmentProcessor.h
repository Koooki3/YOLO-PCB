/**
 * @file YOLOSegmentProcessor.h
 * @brief YOLO实例分割处理器头文件
 * 
 * 该文件定义了YOLOSegmentProcessor类，提供YOLO实例分割模型的推理功能，
 * 支持生成目标检测框和分割掩码。
 * 
 * @author VisualRobot Team
 * @date 2026-01-04
 * @version 1.0
 */

#ifndef YOLOSEGMENTPROCESSOR_H
#define YOLOSEGMENTPROCESSOR_H

#include <QObject>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>

/**
 * @brief 分割检测结果结构体
 * 
 * 包含目标检测框、类别信息、置信度和分割掩码
 */
struct SegmentationResult
{
    cv::Rect boundingBox;           ///< 目标边界框
    int classId;                    ///< 类别ID
    std::string className;          ///< 类别名称
    float confidence;               ///< 置信度
    cv::Mat mask;                   ///< 分割掩码（单通道，0-255）
};

/**
 * @brief YOLO实例分割处理器类
 * 
 * 该类实现了YOLO实例分割模型的推理功能，支持：
 * - 加载ONNX格式的YOLO分割模型
 * - 目标检测和实例分割
 * - 生成彩色掩码可视化
 * - 支持多种后处理选项
 * 
 * @note 使用OpenCV DNN模块进行推理
 * @note 支持CPU和GPU推理
 */
class YOLOSegmentProcessor : public QObject
{
    Q_OBJECT

public:
    /**
     * @brief 构造函数
     * 
     * @param parent 父对象指针
     */
    explicit YOLOSegmentProcessor(QObject *parent = nullptr);
    
    /**
     * @brief 析构函数
     */
    ~YOLOSegmentProcessor();

    /**
     * @brief 初始化模型
     * 
     * @param modelPath ONNX模型文件路径
     * @param useGPU 是否使用GPU推理，默认false
     * @return 成功返回true，失败返回false
     */
    bool InitModel(const std::string& modelPath, bool useGPU = false);

    /**
     * @brief 检查模型是否已加载
     * 
     * @return 已加载返回true，否则返回false
     */
    bool IsModelLoaded() const;

    /**
     * @brief 设置类别标签
     * 
     * @param labels 类别标签列表
     */
    void SetClassLabels(const std::vector<std::string>& labels);

    /**
     * @brief 设置检测阈值
     * 
     * @param confThreshold 置信度阈值（0-1）
     * @param nmsThreshold NMS阈值（0-1）
     * @param maskThreshold 掩码二值化阈值（0-1）
     */
    void SetThresholds(float confThreshold, float nmsThreshold, float maskThreshold = 0.5f);

    /**
     * @brief 执行分割检测
     * 
     * @param image 输入图像（BGR格式）
     * @param results 输出检测结果（包含掩码）
     * @return 成功返回true，失败返回false
     */
    bool DetectAndSegment(const cv::Mat& image, std::vector<SegmentationResult>& results);

    /**
     * @brief 绘制分割结果
     * 
     * @param image 输入输出图像
     * @param results 分割结果
     * @param alpha 掩码透明度（0-1）
     */
    void DrawSegmentationResults(cv::Mat& image, const std::vector<SegmentationResult>& results, float alpha = 0.5f);

    /**
     * @brief 生成纯掩码图像
     * 
     * @param imageSize 图像尺寸
     * @param results 分割结果
     * @return 掩码图像（彩色，每个实例不同颜色）
     */
    cv::Mat GenerateMaskImage(const cv::Size& imageSize, const std::vector<SegmentationResult>& results);

signals:
    /**
     * @brief 处理完成信号
     * 
     * @param resultImage 处理后的图像
     */
    void processingComplete(const cv::Mat& resultImage);

    /**
     * @brief 错误发生信号
     * 
     * @param error 错误信息
     */
    void errorOccurred(const QString& error);

private:
    /**
     * @brief 预处理图像
     * 
     * @param image 输入图像
     * @return 预处理后的blob
     */
    cv::Mat PreprocessImage(const cv::Mat& image);

    /**
     * @brief 后处理网络输出
     * 
     * @param outputs 网络输出
     * @param imageSize 原始图像尺寸
     * @param results 输出结果
     */
    void PostProcess(const std::vector<cv::Mat>& outputs, const cv::Size& imageSize, std::vector<SegmentationResult>& results);

    /**
     * @brief 生成随机颜色
     * 
     * @param classId 类别ID
     * @return BGR颜色
     */
    cv::Scalar GetClassColor(int classId);

    cv::dnn::Net net_;                      ///< OpenCV DNN网络
    bool modelLoaded_;                      ///< 模型是否已加载
    std::vector<std::string> classLabels_;  ///< 类别标签
    float confThreshold_;                   ///< 置信度阈值
    float nmsThreshold_;                    ///< NMS阈值
    float maskThreshold_;                   ///< 掩码阈值
    int inputWidth_;                        ///< 输入宽度
    int inputHeight_;                       ///< 输入高度
    std::vector<cv::Scalar> classColors_;   ///< 类别颜色缓存
};

#endif // YOLOSEGMENTPROCESSOR_H

