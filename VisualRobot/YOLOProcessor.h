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

    // 初始化 ONNX/ONNXRuntime via OpenCV DNN
    bool InitModel(const string& modelPath, bool useCUDA = false);

    // 检测接口
    bool DetectObjects(const Mat& frame, vector<DetectionResult>& results);

    // 设置参数
    void SetInputSize(const Size& size);
    void SetThresholds(float conf, float nms);
    void SetClassLabels(const vector<string>& labels);

    bool IsModelLoaded() const { return isModelLoaded_; }

    // 绘制结果（对外公开，方便UI直接调用）
    void DrawDetectionResults(Mat& frame, const vector<DetectionResult>& results);

signals:
    void processingComplete(const cv::Mat& resultImage);
    void errorOccurred(const QString& error);

private:
    dnn::Net net_;
    bool isModelLoaded_;
    Size inputSize_;
    float confThreshold_;
    float nmsThreshold_;
    double scaleFactor_;
    Scalar meanValues_;
    bool swapRB_;
    vector<string> classLabels_;

    // helpers
    vector<DetectionResult> PostProcess(const Mat& frame, const Mat& outputs);
};

#endif // YOLOPROCESSOR_H
