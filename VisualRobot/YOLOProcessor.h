#ifndef YOLOPROCESSOR_H
#define YOLOPROCESSOR_H

#include <QObject>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>
#include <memory>
#include "DLProcessor.h" // reuse DetectionResult struct

using namespace std;
using namespace cv;

class YOLOProcessor : public QObject
{
    Q_OBJECT
public:
    explicit YOLOProcessor(QObject* parent = nullptr);
    ~YOLOProcessor();
    
    bool InitModel(const std::string& modelPath, bool useGPU = false);
    std::vector<DetectionResult> DetectObjects(const cv::Mat& frame);
    void DrawDetectionResults(cv::Mat& frame, const std::vector<DetectionResult>& results);
    void SetClassLabels(const std::vector<std::string>& labels);
    void SetInputSize(const cv::Size& size);
    void SetThresholds(float conf, float nms);
    
    bool IsModelLoaded() const { return isModelLoaded_; }
    
protected:
    std::vector<DetectionResult> PostProcess(const std::vector<cv::Mat>& outputs, const cv::Size& frameSize);
    cv::Mat PreProcess(const cv::Mat& frame);
    
    cv::dnn::Net net_;
    float confThreshold_;  // 不再使用，保留兼容
    float nmsThreshold_;
    std::vector<std::string> classLabels_;
    bool isModelLoaded_;
    float scaleFactor_;
    cv::Size inputSize_;
    cv::Scalar meanValues_;
    bool swapRB_;
    
    // 用于缩放边界框
    float letterbox_r_;
    float letterbox_dw_;
    float letterbox_dh_;

    // helpers
    
signals:
    void errorOccurred(const QString& message);
};

#endif // YOLOPROCESSOR_H