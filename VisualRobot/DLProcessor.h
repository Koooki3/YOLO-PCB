#ifndef DLPROCESSOR_H
#define DLPROCESSOR_H

#include <QObject>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>
#include <memory>

// 二分类结果结构体
struct ClassificationResult {
    int classId;           // 分类ID (0或1)
    float confidence;      // 置信度
    std::string className; // 类别名称
    bool isValid;          // 结果是否有效
    
    ClassificationResult() : classId(-1), confidence(0.0f), className("unknown"), isValid(false) {}
};

class DLProcessor : public QObject
{
    Q_OBJECT

public:
    explicit DLProcessor(QObject *parent = nullptr);
    ~DLProcessor();

    // 初始化深度学习模型
    bool initModel(const std::string& modelPath, const std::string& configPath = "");
    
    // 设置模型参数
    void setModelParams(float confThreshold = 0.5f, float nmsThreshold = 0.4f);
    
    // 设置类别标签
    void setClassLabels(const std::vector<std::string>& labels);
    bool loadClassLabels(const std::string& labelPath);
    
    // 处理单张图像 - 二分类
    bool classifyImage(const cv::Mat& frame, ClassificationResult& result);
    
    // 批量处理图像
    bool classifyBatch(const std::vector<cv::Mat>& frames, std::vector<ClassificationResult>& results);
    
    // 处理单张图像（兼容原接口）
    bool processFrame(const cv::Mat& frame, cv::Mat& output);
    
    // 获取模型信息
    cv::Size getInputSize() const { return inputSize_; }
    std::vector<std::string> getClassLabels() const { return classLabels_; }
    bool isModelLoaded() const { return isModelLoaded_; }
    
    // 设置输入尺寸
    void setInputSize(const cv::Size& size);

    // 设置预处理参数
    void setPreprocessParams(const cv::Scalar& mean, const cv::Scalar& std, 
                           double scaleFactor, bool swapRB);
    
    // 获取模型参数
    float getConfidenceThreshold() const;
    float getNMSThreshold() const;
    
    // GPU支持
    void enableGPU(bool enable);
    
    // 重置参数
    void resetToDefaults();
    
    // 获取模型信息
    std::string getModelInfo() const;
    
    // 保存类别标签
    bool saveClassLabels(const std::string& labelPath) const;
    
    // 清除模型
    void clearModel();
    
    // 模型预热
    bool warmUp();

signals:
    // 处理结果信号
    void processingComplete(const cv::Mat& result);
    void classificationComplete(const ClassificationResult& result);
    void batchProcessingComplete(const std::vector<ClassificationResult>& results);
    void errorOccurred(const QString& error);

private:
    // OpenCV DNN模块相关
    cv::dnn::Net net_;
    float confThreshold_;
    float nmsThreshold_;
    bool isModelLoaded_;
    
    // 二分类相关参数
    cv::Size inputSize_;                    // 输入图像尺寸
    cv::Scalar meanValues_;                 // 均值
    cv::Scalar stdValues_;                  // 标准差
    float scaleFactor_;                     // 缩放因子
    std::vector<std::string> classLabels_;  // 类别标签
    bool swapRB_;                          // 是否交换R和B通道
    
    // 预处理方法
    cv::Mat preProcess(const cv::Mat& frame);
    
    // 后处理方法 - 二分类
    ClassificationResult postProcessClassification(const std::vector<cv::Mat>& outs);
    
    // 后处理方法（兼容原接口）
    void postProcess(cv::Mat& frame, const std::vector<cv::Mat>& outs);
    
    // 辅助方法
    void initDefaultParams();
    bool validateInput(const cv::Mat& frame);
};

#endif // DLPROCESSOR_H
