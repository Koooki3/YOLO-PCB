#ifndef DLPROCESSOR_H
#define DLPROCESSOR_H

#include <QObject>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>
#include <memory>

class DLProcessor : public QObject
{
    Q_OBJECT

public:
    explicit DLProcessor(QObject *parent = nullptr);
    ~DLProcessor();

    // 初始化深度学习模型
    bool initModel(const std::string& modelPath, const std::string& configPath);
    
    // 设置模型参数
    void setModelParams(float confThreshold = 0.5f, float nmsThreshold = 0.4f);
    
    // 处理单张图像
    bool processFrame(const cv::Mat& frame, cv::Mat& output);

signals:
    // 处理结果信号
    void processingComplete(const cv::Mat& result);
    void errorOccurred(const QString& error);

private:
    // OpenCV DNN模型相关
    cv::dnn::Net net_;
    float confThreshold_;
    float nmsThreshold_;
    bool isModelLoaded_;
    
    // 预处理方法
    cv::Mat preProcess(const cv::Mat& frame);
    
    // 后处理方法
    void postProcess(cv::Mat& frame, const std::vector<cv::Mat>& outs);
};

#endif // DLPROCESSOR_H
