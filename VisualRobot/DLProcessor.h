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

// 二分类结果结构体
struct ClassificationResult 
{
    int classId;      // 分类ID (0或1)
    float confidence; // 置信度
    string className; // 类别名称
    bool isValid;     // 结果是否有效
    
    ClassificationResult() : classId(-1), confidence(0.0f), className("unknown"), isValid(false) {}
};

// 检测结果结构体（支持YOLO）
struct DetectionResult
{
    int classId;      // 类别ID
    float confidence; // 置信度
    string className; // 类别名称
    Rect boundingBox; // 边界框
    Mat mask;         // 分割掩码（用于实例分割）
    
    DetectionResult() : classId(-1), confidence(0.0f), className("unknown"), boundingBox() {}
};

class DLProcessor : public QObject
{
    Q_OBJECT

public:
    explicit DLProcessor(QObject *parent = nullptr);
    ~DLProcessor();

    // 初始化深度学习模型
    bool InitModel(const string& modelPath, const string& configPath = "");
    
    // 设置模型参数
    void SetModelParams(float confThreshold = 0.5f, float nmsThreshold = 0.4f);
    
    // 设置类别标签
    void SetClassLabels(const vector<string>& labels);
    bool LoadClassLabels(const string& labelPath);
    
    // YOLO模型配置
    // (YOLO 相关支持已迁移到独立的 YOLOProcessor 模块)
    
    // 处理单张图像 - 二分类
    bool ClassifyImage(const Mat& frame, ClassificationResult& result);
    
    // 批量处理图像
    bool ClassifyBatch(const vector<Mat>& frames, vector<ClassificationResult>& results);
    
    // 处理单张图像 (兼容原接口) 
    bool ProcessFrame(const Mat& frame, Mat& output);
    
    // YOLO目标检测与实例分割
    // (YOLO 相关支持已迁移到独立的 YOLOProcessor 模块)
    
    // 获取模型信息
    Size GetInputSize() const { return inputSize_; }
    vector<string> GetClassLabels() const { return classLabels_; }
    bool IsModelLoaded() const { return isModelLoaded_; }

    // 启用/禁用YOLO模式（使用YOLO专用预处理与后处理）
    void EnableYOLOMode(bool enable);

    // YOLO 目标检测接口
    bool DetectObjects(const Mat& frame, vector<DetectionResult>& results);
    bool ProcessYoloFrame(const Mat& frame, Mat& output);
    vector<DetectionResult> PostProcessYolo(const Mat& frame, const vector<Mat>& outs, float confThreshold, float nmsThreshold);
    void DrawDetectionResults(Mat& frame, const vector<DetectionResult>& results);
    
    // 设置输入尺寸
    void SetInputSize(const Size& size);

    // 设置预处理参数
    void SetPreprocessParams(const Scalar& mean, const Scalar& std, double scaleFactor, bool swapRB);
    
    // 获取模型参数
    float GetConfidenceThreshold() const;
    float GetNMSThreshold() const;
    
    // GPU支持
    void EnableGPU(bool enable);
    
    // 模型量化优化
    bool QuantizeModel(const string& quantizationType = "INT8", const vector<Mat>& calibrationImages = {});
    bool IsModelQuantized() const { return isQuantized_; }
    string GetQuantizationType() const { return quantizationType_; }
    
    // 重置参数
    void ResetToDefaults();
    
    // 获取模型信息
    string GetModelInfo() const;
    
    // 保存类别标签
    bool SaveClassLabels(const string& labelPath) const;
    
    // 清除模型
    void ClearModel();
    
    // 模型预热
    bool WarmUp();

signals:
      /**
       * @brief 分类完成信号
       * @param result 分类结果
       */
      void classificationComplete(const ClassificationResult& result);
      
      /**
       * @brief 图像处理完成信号（用于检测和分割）
       * @param resultImage 处理后的图像
       */
      void processingComplete(const cv::Mat& resultImage);
      
      /**
       * @brief 批量处理完成信号
       * @param results 批量处理结果列表
       */
      void batchProcessingComplete(const vector<ClassificationResult>& results);
      
      /**
       * @brief 错误发生信号
       * @param error 错误信息
       */
      void errorOccurred(const QString& error);

private:
    // OpenCV DNN模块相关
    dnn::Net net_;
    float confThreshold_;
    float nmsThreshold_;
    bool isModelLoaded_;
    bool isQuantized_;          // 模型是否已量化
    string quantizationType_;   // 量化类型 (INT8, UINT8, FP16等)
    
    // 数据处理相关
    unique_ptr<DataProcessor> dataProcessor_;  // 数据处理器实例
    
    // 二分类相关参数
    Size inputSize_;              // 输入图像尺寸
    Scalar meanValues_;           // 均值
    Scalar stdValues_;            // 标准差
    float scaleFactor_;           // 缩放因子
    vector<string> classLabels_;  // 类别标签
    bool swapRB_;                 // 是否交换R和B通道
    bool useYOLOPreprocessing_;   // 是否使用YOLO预处理
    
    // 预处理方法
    // 预处理方法
    Mat PreProcess(const Mat& frame);
    
    // 后处理方法 - 二分类
    ClassificationResult PostProcessClassification(const vector<Mat>& outs);
    
    // 后处理方法 (兼容原接口) 
    void PostProcess(Mat& frame, const vector<Mat>& outs);
    
    // YOLO后处理方法
    // (YOLO 相关后处理与绘制已迁移到 `YOLOProcessor`)
        // bool useYOLOPreprocessing_;   // YOLO 预处理相关设置已迁移
    
    // 辅助方法
    void InitDefaultParams();
    bool ValidateInput(const Mat& frame);
};

#endif // DLPROCESSOR_H
