#ifndef DLPROCESSOR_H
#define DLPROCESSOR_H

#include <QObject>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>
#include <memory>

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
    
    // 处理单张图像 - 二分类
    bool ClassifyImage(const Mat& frame, ClassificationResult& result);
    
    // 批量处理图像
    bool ClassifyBatch(const vector<Mat>& frames, vector<ClassificationResult>& results);
    
    // 处理单张图像（兼容原接口）
    bool ProcessFrame(const Mat& frame, Mat& output);
    
    // 获取模型信息
    Size GetInputSize() const { return inputSize_; }
    vector<string> GetClassLabels() const { return classLabels_; }
    bool IsModelLoaded() const { return isModelLoaded_; }
    
    // 设置输入尺寸
    void SetInputSize(const Size& size);

    // 设置预处理参数
    void SetPreprocessParams(const Scalar& mean, const Scalar& std, double scaleFactor, bool swapRB);
    
    // 获取模型参数
    float GetConfidenceThreshold() const;
    float GetNMSThreshold() const;
    
    // GPU支持
    void EnableGPU(bool enable);
    
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
    // 处理结果信号
    void processingComplete(const Mat& result);
    void classificationComplete(const ClassificationResult& result);
    void batchProcessingComplete(const vector<ClassificationResult>& results);
    void errorOccurred(const QString& error);

private:
    // OpenCV DNN模块相关
    dnn::Net net_;
    float confThreshold_;
    float nmsThreshold_;
    bool isModelLoaded_;
    
    // 二分类相关参数
    Size inputSize_;              // 输入图像尺寸
    Scalar meanValues_;           // 均值
    Scalar stdValues_;            // 标准差
    float scaleFactor_;           // 缩放因子
    vector<string> classLabels_;  // 类别标签
    bool swapRB_;                 // 是否交换R和B通道
    
    // 预处理方法
    Mat PreProcess(const Mat& frame);
    
    // 后处理方法 - 二分类
    ClassificationResult PostProcessClassification(const vector<Mat>& outs);
    
    // 后处理方法（兼容原接口）
    void PostProcess(Mat& frame, const vector<Mat>& outs);
    
    // 辅助方法
    void InitDefaultParams();
    bool ValidateInput(const Mat& frame);
};

#endif // DLPROCESSOR_H
