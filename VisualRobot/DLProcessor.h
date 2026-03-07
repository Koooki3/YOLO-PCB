/**
 * @file DLProcessor.h
 * @brief 深度学习处理器头文件
 * 
 * 该文件定义了DLProcessor类，提供基于OpenCV DNN模块的深度学习模型加载、
 * 推理和处理功能，支持二分类、批量处理、模型量化和GPU加速等功能。
 * 
 * @author VisualRobot Team
 * @date 2025-12-30
 * @version 1.0
 */

#ifndef DLPROCESSOR_H
#define DLPROCESSOR_H

#include <QObject>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>
#include <memory>
#include "DataProcessor.h"

using namespace std;
using namespace cv;

/**
 * @brief 二分类结果结构体
 * 
 * 存储图像二分类的结果信息，包括类别ID、置信度、类别名称和有效性标志
 */
struct ClassificationResult 
{
    int classId;      ///< 分类ID (0或1)
    float confidence; ///< 置信度 (0.0-1.0)
    string className; ///< 类别名称
    bool isValid;     ///< 结果是否有效（置信度是否超过阈值）
    
    /**
     * @brief 构造函数
     * 
     * 初始化为默认值：
     * - classId: -1
     * - confidence: 0.0
     * - className: "unknown"
     * - isValid: false
     */
    ClassificationResult() : classId(-1), confidence(0.0f), className("unknown"), isValid(false) {}
};

/**
 * @brief 检测结果结构体（支持YOLO）
 * 
 * 存储目标检测的结果信息，包括类别ID、置信度、类别名称、边界框和分割掩码
 */
struct DetectionResult
{
    int classId;      ///< 类别ID
    float confidence; ///< 置信度 (0.0-1.0)
    string className; ///< 类别名称
    Rect boundingBox; ///< 边界框 [x, y, width, height]
    Mat mask;         ///< 分割掩码（用于实例分割）
    
    /**
     * @brief 构造函数
     * 
     * 初始化为默认值：
     * - classId: -1
     * - confidence: 0.0
     * - className: "unknown"
     * - boundingBox: 空矩形
     */
    DetectionResult() : classId(-1), confidence(0.0f), className("unknown"), boundingBox() {}
};

/**
 * @brief 深度学习处理器类
 * 
 * 提供完整的深度学习模型处理功能，包括：
 * - 模型加载和初始化（支持ONNX、TensorFlow、Caffe等格式）
 * - 图像预处理和后处理
 * - 单张图像和批量图像分类
 * - 模型参数配置和优化
 * - GPU加速支持
 * - 模型量化（FP16、INT8、UINT8）
 * - 实时信号反馈
 * 
 * @note 该类继承自QObject，支持Qt的信号槽机制
 * @see ClassificationResult, DetectionResult
 */
class DLProcessor : public QObject
{
    Q_OBJECT

public:
    /**
     * @brief 构造函数
     * 
     * 初始化深度学习处理器，创建DataProcessor实例，设置默认参数
     * 
     * @param parent 父对象指针，默认为nullptr
     */
    explicit DLProcessor(QObject *parent = nullptr);
    
    /**
     * @brief 析构函数
     * 
     * 清理资源，释放内存
     */
    ~DLProcessor();

    /**
     * @brief 初始化深度学习模型
     * 
     * 从文件加载深度学习模型，支持多种格式，并自动配置网络参数
     * 
     * @param modelPath 模型文件路径（支持.onnx、.pb、.caffemodel等）
     * @param configPath 配置文件路径（可选，用于Caffe/YOLO等需要配置文件的模型）
     * @return bool 模型加载成功返回true，失败返回false
     * 
     * @note 支持的模型格式：
     *       - ONNX (.onnx)
     *       - TensorFlow (.pb)
     *       - Caffe (.caffemodel + .prototxt)
     *       - Darknet (.weights + .cfg)
     * @note 会自动设置计算后端为OpenCV DNN，目标设备为CPU
     * @see SetInputSize(), SetPreprocessParams()
     */
    bool InitModel(const string& modelPath, const string& configPath = "");
    
    /**
     * @brief 设置模型参数
     * 
     * 配置置信度阈值和NMS（非极大值抑制）阈值
     * 
     * @param confThreshold 置信度阈值，默认0.5，范围0.1-1.0
     * @param nmsThreshold NMS阈值，默认0.4，范围0.1-1.0
     * 
     * @note 参数会被自动限制在合理范围内以避免错误
     * @see GetConfidenceThreshold(), GetNMSThreshold()
     */
    void SetModelParams(float confThreshold = 0.5f, float nmsThreshold = 0.4f);
    
    /**
     * @brief 设置类别标签
     * 
     * 直接设置类别标签列表
     * 
     * @param labels 类别标签字符串向量
     * @see LoadClassLabels(), GetClassLabels()
     */
    void SetClassLabels(const vector<string>& labels);
    
    /**
     * @brief 从文件加载类别标签
     * 
     * 从文本文件读取类别标签，每行一个标签
     * 
     * @param labelPath 标签文件路径
     * @return bool 加载成功返回true，失败返回false
     * 
     * @note 标签文件格式：每行一个类别名称
     * @see SetClassLabels(), SaveClassLabels()
     */
    bool LoadClassLabels(const string& labelPath);
    
    /**
     * @brief 对单张图像进行分类
     * 
     * 对输入图像执行预处理、前向传播和后处理，得到分类结果
     * 
     * @param frame 输入图像（BGR格式）
     * @param result 输出参数，存储分类结果
     * @return bool 分类成功返回true，失败返回false
     * 
     * @note 会自动发射classificationComplete信号
     * @see PreProcess(), PostProcessClassification()
     */
    bool ClassifyImage(const Mat& frame, ClassificationResult& result);
    
    /**
     * @brief 批量分类多张图像
     * 
     * 对多张图像执行批量分类处理
     * 
     * @param frames 输入图像列表
     * @param results 输出参数，存储分类结果列表
     * @return bool 所有图像分类成功返回true，部分失败返回false
     * 
     * @note 会自动发射batchProcessingComplete信号
     * @see ClassifyImage()
     */
    bool ClassifyBatch(const vector<Mat>& frames, vector<ClassificationResult>& results);
    
    /**
     * @brief 处理单张图像（兼容原接口）
     * 
     * 对输入图像进行分类，并在图像上绘制结果
     * 
     * @param frame 输入图像
     * @param output 输出图像（带标注）
     * @return bool 处理成功返回true，失败返回false
     * 
     * @note 会自动发射processingComplete信号
     * @see ClassifyImage()
     */
    bool ProcessFrame(const Mat& frame, Mat& output);
    
    /**
     * @brief 获取输入尺寸
     * @return Size 模型输入尺寸
     * @see SetInputSize()
     */
    Size GetInputSize() const { return inputSize_; }
    
    /**
     * @brief 获取类别标签列表
     * @return vector<string> 类别标签向量
     * @see SetClassLabels()
     */
    vector<string> GetClassLabels() const { return classLabels_; }
    
    /**
     * @brief 检查模型是否已加载
     * @return bool 模型已加载返回true，否则返回false
     * @see InitModel()
     */
    bool IsModelLoaded() const { return isModelLoaded_; }
    
    /**
     * @brief 设置输入尺寸
     * 
     * 配置模型的输入图像尺寸
     * 
     * @param size 输入尺寸（宽度, 高度）
     * @see GetInputSize()
     */
    void SetInputSize(const Size& size);

    /**
     * @brief 设置预处理参数
     * 
     * 配置图像预处理的标准化参数
     * 
     * @param mean 均值向量（BGR顺序）
     * @param std 标准差向量
     * @param scaleFactor 缩放因子
     * @param swapRB 是否交换R和B通道
     * 
     * @note 默认参数适合大多数二分类模型
     */
    void SetPreprocessParams(const Scalar& mean, const Scalar& std, double scaleFactor, bool swapRB);
    
    /**
     * @brief 获取置信度阈值
     * @return float 当前置信度阈值
     * @see SetModelParams()
     */
    float GetConfidenceThreshold() const;
    
    /**
     * @brief 获取NMS阈值
     * @return float 当前NMS阈值
     * @see SetModelParams()
     */
    float GetNMSThreshold() const;
    
    /**
     * @brief 启用/禁用GPU加速
     * 
     * 切换计算后端（CPU/GPU）
     * 
     * @param enable true启用GPU，false使用CPU
     * 
     * @note 需要OpenCV编译时支持CUDA
     * @see InitModel()
     */
    void EnableGPU(bool enable);
    
    /**
     * @brief 模型量化优化
     * 
     * 对已加载的模型进行量化，减小模型大小，加速推理
     * 
     * @param quantizationType 量化类型（"FP16"、"INT8"、"UINT8"）
     * @param calibrationImages 校准图像列表（INT8/UINT8需要）
     * @return bool 量化成功返回true，失败返回false
     * 
     * @note INT8和UINT8量化需要校准图像来确定量化参数
     * @see IsModelQuantized(), GetQuantizationType()
     */
    bool QuantizeModel(const string& quantizationType = "INT8", const vector<Mat>& calibrationImages = {});
    
    /**
     * @brief 检查模型是否已量化
     * @return bool 模型已量化返回true，否则返回false
     * @see QuantizeModel()
     */
    bool IsModelQuantized() const { return isQuantized_; }
    
    /**
     * @brief 获取量化类型
     * @return string 量化类型字符串
     * @see QuantizeModel()
     */
    string GetQuantizationType() const { return quantizationType_; }
    
    /**
     * @brief 重置参数到默认值
     * 
     * 重置所有参数为默认设置，清除量化状态
     * @see InitDefaultParams()
     */
    void ResetToDefaults();
    
    /**
     * @brief 获取模型信息
     * 
     * 返回模型的详细信息字符串
     * 
     * @return string 模型信息（多行文本）
     * 
     * @note 包含输入尺寸、类别数、阈值、量化状态等
     */
    string GetModelInfo() const;
    
    /**
     * @brief 保存类别标签到文件
     * 
     * 将当前类别标签列表保存到文本文件
     * 
     * @param labelPath 保存路径
     * @return bool 保存成功返回true，失败返回false
     * @see LoadClassLabels()
     */
    bool SaveClassLabels(const string& labelPath) const;
    
    /**
     * @brief 清除模型
     * 
     * 释放模型资源，重置所有状态
     * @see InitModel()
     */
    void ClearModel();
    
    /**
     * @brief 模型预热
     * 
     * 使用虚拟输入执行一次推理，预热模型
     * 
     * @return bool 预热成功返回true，失败返回false
     * 
     * @note 用于减少首次推理的延迟
     */
    bool WarmUp();

signals:
    /**
     * @brief 分类完成信号
     * 
     * 单张图像分类完成时发射
     * 
     * @param result 分类结果
     * @see ClassifyImage()
     */
    void classificationComplete(const ClassificationResult& result);
    
    /**
     * @brief 图像处理完成信号
     * 
     * 图像处理完成时发射（用于检测和分割）
     * 
     * @param resultImage 处理后的图像
     * @see ProcessFrame()
     */
    void processingComplete(const cv::Mat& resultImage);
    
    /**
     * @brief 批量处理完成信号
     * 
     * 批量图像处理完成时发射
     * 
     * @param results 批量处理结果列表
     * @see ClassifyBatch()
     */
    void batchProcessingComplete(const vector<ClassificationResult>& results);
    
    /**
     * @brief 错误发生信号
     * 
     * 发生错误时发射
     * 
     * @param error 错误信息
     */
    void errorOccurred(const QString& error);

private:
    // OpenCV DNN模块相关
    dnn::Net net_;                  ///< OpenCV DNN网络实例
    float confThreshold_;           ///< 置信度阈值
    float nmsThreshold_;            ///< NMS阈值
    bool isModelLoaded_;            ///< 模型加载状态
    bool isQuantized_;              ///< 模型是否已量化
    string quantizationType_;       ///< 量化类型 (INT8, UINT8, FP16等)
    
    // 数据处理相关
    unique_ptr<DataProcessor> dataProcessor_;  ///< 数据处理器实例
    
    // 二分类相关参数
    Size inputSize_;                ///< 输入图像尺寸
    Scalar meanValues_;             ///< 均值
    Scalar stdValues_;              ///< 标准差
    float scaleFactor_;             ///< 缩放因子
    vector<string> classLabels_;    ///< 类别标签
    bool swapRB_;                   ///< 是否交换R和B通道
    
    /**
     * @brief 预处理方法
     * 
     * 对输入图像进行预处理，转换为模型输入格式
     * 
     * @param frame 输入图像
     * @return Mat 预处理后的blob数据
     * @see PostProcessClassification()
     */
    Mat PreProcess(const Mat& frame);
    
    /**
     * @brief 后处理方法 - 二分类
     * 
     * 处理模型输出，得到分类结果
     * 
     * @param outs 模型输出向量
     * @return ClassificationResult 分类结果
     * @see PreProcess()
     */
    ClassificationResult PostProcessClassification(const vector<Mat>& outs);
    
    /**
     * @brief 后处理方法（兼容原接口）
     * 
     * 处理模型输出并在图像上绘制结果
     * 
     * @param frame 输入图像
     * @param outs 模型输出向量
     * @see PostProcessClassification()
     */
    void PostProcess(Mat& frame, const vector<Mat>& outs);
    
    /**
     * @brief 初始化默认参数
     * 
     * 设置默认的预处理参数和类别标签
     * @see ResetToDefaults()
     */
    void InitDefaultParams();
    
    /**
     * @brief 验证输入图像
     * 
     * 检查输入图像是否有效
     * 
     * @param frame 输入图像
     * @return bool 有效返回true，无效返回false
     */
    bool ValidateInput(const Mat& frame);
};

#endif // DLPROCESSOR_H
