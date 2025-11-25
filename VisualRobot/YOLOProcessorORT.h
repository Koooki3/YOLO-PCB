#ifndef YOLOPROCESSORORT_H
#define YOLOPROCESSORORT_H

#include <QObject>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include "DLProcessor.h" // for DetectionResult

using namespace cv;
using namespace std;

class YOLOProcessorORT : public QObject
{
    Q_OBJECT
public:
    explicit YOLOProcessorORT(QObject* parent = nullptr);
    ~YOLOProcessorORT();

    bool InitModel(const string& modelPath, bool useCUDA = false);
    bool DetectObjects(const Mat& frame, vector<DetectionResult>& results);

    void SetInputSize(const Size& size);
    void SetThresholds(float conf, float nms);
    void SetClassLabels(const vector<string>& labels);

    bool IsModelLoaded() const { return session_ != nullptr; }

    void DrawDetectionResults(Mat& frame, const vector<DetectionResult>& results);

signals:
    void processingComplete(const cv::Mat& resultImage);
    void errorOccurred(const QString& error);

private:
    Ort::Env env_{nullptr};
    std::unique_ptr<Ort::Session> session_;
    Ort::SessionOptions sessionOptions_;

    Size inputSize_;
    float confThreshold_;
    float nmsThreshold_;
    double scaleFactor_;
    bool swapRB_;
    vector<string> classLabels_;

    vector<Mat> OrtOutputToMats(const std::vector<Ort::Value>& outputs);
    vector<DetectionResult> PostProcess(const Mat& frame, const Mat& dets);
};

#endif // YOLOPROCESSORORT_H
