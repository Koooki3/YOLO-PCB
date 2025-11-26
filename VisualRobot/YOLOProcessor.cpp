#include "YOLOProcessor.h"
#include <QDebug>

using namespace cv;
using namespace std;

YOLOProcessor::YOLOProcessor(QObject* parent)
    : QObject(parent)
    , isModelLoaded_(false)
    , inputSize_(640, 640)
    , confThreshold_(0.25f)
    , nmsThreshold_(0.45f)
    , scaleFactor_(1.0/255.0)
    , meanValues_(0.0,0.0,0.0)
    , swapRB_(true)
{
}

YOLOProcessor::~YOLOProcessor()
{
}

bool YOLOProcessor::InitModel(const string& modelPath, bool useCUDA)
{
    try {
        net_ = dnn::readNet(modelPath);
        if (useCUDA) {
            net_.setPreferableBackend(dnn::DNN_BACKEND_CUDA);
            net_.setPreferableTarget(dnn::DNN_TARGET_CUDA_FP16);
        } else {
            net_.setPreferableBackend(dnn::DNN_BACKEND_DEFAULT);
            net_.setPreferableTarget(dnn::DNN_TARGET_CPU);
        }
        isModelLoaded_ = true;
        return true;
    } catch (const cv::Exception& e) {
        isModelLoaded_ = false;
        QString msg = QString("YOLO InitModel error: %1").arg(e.what());
        qDebug() << msg;
        emit errorOccurred(msg);
        return false;
    }
}

void YOLOProcessor::SetInputSize(const Size& size)
{
    inputSize_ = size;
}

void YOLOProcessor::SetThresholds(float conf, float nms)
{
    confThreshold_ = conf;
    nmsThreshold_ = nms;
}

void YOLOProcessor::SetClassLabels(const vector<string>& labels)
{
    classLabels_ = labels;
}

std::vector<DetectionResult> YOLOProcessor::DetectObjects(const cv::Mat& frame)
{
    if (net_.empty()) {
        return {};
    }
    
    // 预处理
    cv::Mat blob = PreProcess(frame);
    
    // 设置输入
    net_.setInput(blob);
    
    // 前向传播
    std::vector<cv::Mat> outputs;
    net_.forward(outputs);
    
    // 后处理
    return PostProcess(outputs, frame.size());
}

cv::Mat YOLOProcessor::PreProcess(const cv::Mat& frame)
{
    cv::Mat blob;
    // 创建网络输入blob
    cv::dnn::blobFromImage(frame, blob, scaleFactor_, inputSize_, meanValues_, swapRB_, false);
    return blob;
}

// 后处理实现已在文件后面定义

std::vector<DetectionResult> YOLOProcessor::PostProcess(const std::vector<cv::Mat>& outputs, const cv::Size& frameSize)
{
    std::vector<DetectionResult> results;
    
    // 假设输出格式为 [batch_size, num_boxes, num_attributes]
    // num_attributes = 4 (bounding box) + num_classes (class scores)
    cv::Mat output = outputs[0];
    int numBoxes = output.size[1];
    int numAttributes = output.size[2];
    int numClasses = numAttributes - 4;
    
    // 提取边界框和类别分数
    for (int i = 0; i < numBoxes; ++i) {
        cv::Mat boxAttributes = output.row(0).col(i);
        
        // 提取边界框坐标
        float x = boxAttributes.at<float>(0, 0);
        float y = boxAttributes.at<float>(0, 1);
        float width = boxAttributes.at<float>(0, 2);
        float height = boxAttributes.at<float>(0, 3);
        
        // 提取类别分数 (raw scores before sigmoid)
        cv::Mat classScores = boxAttributes.colRange(4, 4 + numClasses);
        
        // 找到最高分和对应类别
        double maxVal;
        cv::Point maxLoc;
        cv::minMaxLoc(classScores, nullptr, &maxVal, nullptr, &maxLoc);
        
        // 使用原始分数(max_raw_scores)作为置信度，阈值为0.5
        float rawScore = static_cast<float>(maxVal);
        if (rawScore >= 0.5) {  // 使用0.5作为max_raw_scores的阈值
            int classId = maxLoc.x;
            
            // 计算实际边界框坐标
            float x1 = (x - width / 2.0f) * frameSize.width;
            float y1 = (y - height / 2.0f) * frameSize.height;
            float x2 = (x + width / 2.0f) * frameSize.width;
            float y2 = (y + height / 2.0f) * frameSize.height;
            
            // 确保边界框在图像范围内
            x1 = std::max(0.0f, x1);
            y1 = std::max(0.0f, y1);
            x2 = std::min(static_cast<float>(frameSize.width - 1), x2);
            y2 = std::min(static_cast<float>(frameSize.height - 1), y2);
            
            // 创建检测结果
            DetectionResult result;
            result.boundingBox = cv::Rect2i(static_cast<int>(x1), static_cast<int>(y1), 
                                          static_cast<int>(x2 - x1), static_cast<int>(y2 - y1));
            result.classId = classId;
            result.confidence = rawScore;  // 使用原始分数作为置信度
            
            // 分配类别名称（如果有）
            if (!classLabels_.empty() && classId < classLabels_.size()) {
                result.className = classLabels_[classId];
            } else {
                // 没有用户标签时，使用空字符串，后续DrawDetectionResults会处理显示数字标签
                result.className = "";
            }
            
            results.push_back(result);
        }
    }
    
    // 应用NMS
    if (!results.empty()) {
        std::vector<int> indices;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        
        for (const auto& result : results) {
            boxes.push_back(result.boundingBox);
            confidences.push_back(result.confidence);
        }
        
        cv::dnn::NMSBoxes(boxes, confidences, 0.5f, nmsThreshold_, indices);  // 使用0.5作为NMS置信度阈值
        
        std::vector<DetectionResult> filteredResults;
        for (int idx : indices) {
            filteredResults.push_back(results[idx]);
        }
        results = filteredResults;
    }
    
    return results;
}

void YOLOProcessor::DrawDetectionResults(cv::Mat& frame, const std::vector<DetectionResult>& results)
{
    // 为每个类别生成不同颜色
    std::vector<cv::Scalar> colors;
    for (int i = 0; i < 80; ++i) {  // 假设最多80个类别
        int b = rand() % 256;
        int g = rand() % 256;
        int r = rand() % 256;
        colors.push_back(cv::Scalar(b, g, r));
    }
    
    for (const auto& result : results) {
        // 绘制边界框
        cv::Scalar color = colors[result.classId % colors.size()];
        cv::rectangle(frame, result.boundingBox, color, 2);
        
        // 准备标签文本
        std::string label;
        if (!result.className.empty()) {
            // 有用户标签，使用用户提供的标签
            label = result.className + " " + cv::format("%.2f", result.confidence);
        } else {
            // 没有用户标签，显示数字标签
            label = "class_" + std::to_string(result.classId) + " " + cv::format("%.2f", result.confidence);
        }
        
        // 绘制标签
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        int top = std::max(result.boundingBox.y, labelSize.height);
        
        // 绘制标签背景
        cv::rectangle(frame, 
                     cv::Point(result.boundingBox.x, top - labelSize.height),
                     cv::Point(result.boundingBox.x + labelSize.width, top + baseLine),
                     color, cv::FILLED);
        
        // 绘制文本
        cv::putText(frame, label, 
                   cv::Point(result.boundingBox.x, top),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}