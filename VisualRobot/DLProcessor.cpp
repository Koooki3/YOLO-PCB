#include "DLProcessor.h"
#include <QDebug>

DLProcessor::DLProcessor(QObject *parent)
    : QObject(parent)
    , confThreshold_(0.5f)
    , nmsThreshold_(0.4f)
    , isModelLoaded_(false)
{
}

DLProcessor::~DLProcessor()
{
}

bool DLProcessor::initModel(const std::string& modelPath, const std::string& configPath)
{
    try {
        // 加载深度学习模型
        net_ = cv::dnn::readNet(modelPath, configPath);
        
        // 设置计算后端（根据实际硬件选择）
        // net_.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
        // net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        
        isModelLoaded_ = true;
        return true;
    }
    catch (const cv::Exception& e) {
        qDebug() << "Error loading model:" << QString::fromStdString(e.msg);
        emit errorOccurred(QString::fromStdString(e.msg));
        return false;
    }
}

void DLProcessor::setModelParams(float confThreshold, float nmsThreshold)
{
    confThreshold_ = confThreshold;
    nmsThreshold_ = nmsThreshold;
}

bool DLProcessor::processFrame(const cv::Mat& frame, cv::Mat& output)
{
    if (!isModelLoaded_) {
        emit errorOccurred("Model not loaded!");
        return false;
    }

    try {
        // 预处理
        cv::Mat blob = preProcess(frame);
        
        // 前向传播
        net_.setInput(blob);
        std::vector<cv::Mat> outs;
        net_.forward(outs, net_.getUnconnectedOutLayersNames());
        
        // 后处理
        output = frame.clone();
        postProcess(output, outs);
        
        emit processingComplete(output);
        return true;
    }
    catch (const cv::Exception& e) {
        qDebug() << "Error processing frame:" << QString::fromStdString(e.msg);
        emit errorOccurred(QString::fromStdString(e.msg));
        return false;
    }
}

cv::Mat DLProcessor::preProcess(const cv::Mat& frame)
{
    // 这里添加具体的预处理步骤
    // 示例：调整大小并进行归一化
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1.0/255.0, cv::Size(416, 416), cv::Scalar(), true, false);
    return blob;
}

void DLProcessor::postProcess(cv::Mat& frame, const std::vector<cv::Mat>& outs)
{
    // 这里添加具体的后处理步骤
    // 示例：解析网络输出，绘制检测框等
    // 具体实现取决于您使用的模型类型（分类、检测、分割等）
}
