#include "DLProcessor.h"
#include <QDebug>
#include <QFile>
#include <QTextStream>
#include <algorithm>
#include <fstream>

DLProcessor::DLProcessor(QObject *parent)
    : QObject(parent)
    , confThreshold_(0.5f)
    , nmsThreshold_(0.4f)
    , isModelLoaded_(false)
{
    initDefaultParams();
}

DLProcessor::~DLProcessor()
{
}

void DLProcessor::initDefaultParams()
{
    // 初始化默认参数
    inputSize_ = cv::Size(224, 224);  // 常用的分类模型输入尺寸
    meanValues_ = cv::Scalar(104.0, 177.0, 123.0);  // ImageNet均值
    scaleFactor_ = 1.0;
    swapRB_ = true;  // OpenCV默认BGR，很多模型需要RGB

    // 默认二分类标签
    classLabels_ = {"Class_0", "Class_1"};
}

bool DLProcessor::initModel(const std::string& modelPath, const std::string& configPath)
{
    try {
        // 加载深度学习模型
        if (configPath.empty()) {
            net_ = cv::dnn::readNet(modelPath);
        } else {
            net_ = cv::dnn::readNet(modelPath, configPath);
        }

        if (net_.empty()) {
            qDebug() << "Failed to load model from:" << QString::fromStdString(modelPath);
            emit errorOccurred("Failed to load model: empty network");
            return false;
        }

        // 设置计算后端（根据实际硬件选择）
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        // 可选：如果有GPU支持
        // net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        // net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

        isModelLoaded_ = true;
        qDebug() << "Model loaded successfully from:" << QString::fromStdString(modelPath);
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

void DLProcessor::setClassLabels(const std::vector<std::string>& labels)
{
    classLabels_ = labels;
    qDebug() << "Class labels updated. Total classes:" << classLabels_.size();
}

bool DLProcessor::loadClassLabels(const std::string& labelPath)
{
    try {
        std::ifstream file(labelPath);
        if (!file.is_open()) {
            qDebug() << "Cannot open label file:" << QString::fromStdString(labelPath);
            return false;
        }

        classLabels_.clear();
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty()) {
                classLabels_.push_back(line);
            }
        }
        file.close();

        qDebug() << "Loaded" << classLabels_.size() << "class labels from:" << QString::fromStdString(labelPath);

        // 调试输出：显示加载的标签内容
        qDebug() << "Loaded class labels:";
        for (size_t i = 0; i < classLabels_.size(); i++) {
            qDebug() << "  Class" << i << ":" << QString::fromStdString(classLabels_[i]);
        }

        return true;
    }
    catch (const std::exception& e) {
        qDebug() << "Error loading class labels:" << QString::fromStdString(e.what());
        return false;
    }
}

bool DLProcessor::validateInput(const cv::Mat& frame)
{
    if (frame.empty()) {
        emit errorOccurred("Input frame is empty");
        return false;
    }

    if (frame.channels() != 1 && frame.channels() != 3) {
        emit errorOccurred("Input frame must have 1 or 3 channels");
        return false;
    }

    return true;
}

bool DLProcessor::classifyImage(const cv::Mat& frame, ClassificationResult& result)
{
    if (!isModelLoaded_) {
        emit errorOccurred("Model not loaded!");
        return false;
    }

    if (!validateInput(frame)) {
        return false;
    }

    try {
        // 预处理
        cv::Mat blob = preProcess(frame);

        // 前向传播
        net_.setInput(blob);
        std::vector<cv::Mat> outs;
        net_.forward(outs, net_.getUnconnectedOutLayersNames());

        // 后处理 - 二分类
        result = postProcessClassification(outs);

        if (result.isValid) {
            emit classificationComplete(result);
            qDebug() << "Classification result: Class" << result.classId
                     << "(" << QString::fromStdString(result.className) << ")"
                     << "Confidence:" << result.confidence;
        }

        return result.isValid;
    }
    catch (const cv::Exception& e) {
        qDebug() << "Error classifying image:" << QString::fromStdString(e.msg);
        emit errorOccurred(QString::fromStdString(e.msg));
        return false;
    }
}

bool DLProcessor::classifyBatch(const std::vector<cv::Mat>& frames, std::vector<ClassificationResult>& results)
{
    if (!isModelLoaded_) {
        emit errorOccurred("Model not loaded!");
        return false;
    }

    results.clear();
    results.reserve(frames.size());

    bool allSuccess = true;
    for (const auto& frame : frames) {
        ClassificationResult result;
        if (classifyImage(frame, result)) {
            results.push_back(result);
        } else {
            allSuccess = false;
            // 添加无效结果以保持索引对应
            results.push_back(ClassificationResult());
        }
    }

    if (!results.empty()) {
        emit batchProcessingComplete(results);
    }

    return allSuccess;
}

bool DLProcessor::processFrame(const cv::Mat& frame, cv::Mat& output)
{
    ClassificationResult result;
    if (!classifyImage(frame, result)) {
        return false;
    }

    // 在图像上绘制分类结果
    output = frame.clone();

    // 添加文本标注
    std::string text = result.className + ": " + std::to_string(result.confidence);
    cv::putText(output, text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                1.0, cv::Scalar(0, 255, 0), 2);

    // 根据分类结果添加边框颜色
    cv::Scalar borderColor = (result.classId == 1) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
    cv::rectangle(output, cv::Point(0, 0), cv::Point(output.cols-1, output.rows-1),
                  borderColor, 3);

    emit processingComplete(output);
    return true;
}

cv::Mat DLProcessor::preProcess(const cv::Mat& frame)
{
    cv::Mat blob;

    try {
        // 转换颜色空间（如果需要）
        cv::Mat processedFrame = frame;
        if (frame.channels() == 1 && inputSize_.width > 0) {
            // 灰度图转RGB
            cv::cvtColor(frame, processedFrame, cv::COLOR_GRAY2RGB);
        }

        // 创建blob
        cv::dnn::blobFromImage(processedFrame, blob, scaleFactor_, inputSize_,
                               meanValues_, swapRB_, false, CV_32F);

        qDebug() << "Preprocessed blob shape: [" << blob.size[0] << ","
                 << blob.size[1] << "," << blob.size[2] << "," << blob.size[3] << "]";

    } catch (const cv::Exception& e) {
        qDebug() << "Error in preprocessing:" << QString::fromStdString(e.msg);
        throw;
    }

    return blob;
}

ClassificationResult DLProcessor::postProcessClassification(const std::vector<cv::Mat>& outs)
{
    ClassificationResult result;

    if (outs.empty()) {
        qDebug() << "No output from network";
        return result;
    }

    try {
        // 获取第一个输出（通常分类模型只有一个输出）
        cv::Mat output = outs[0];

        // 确保输出是1D向量
        if (output.dims > 2) {
            // 如果是多维，reshape为2D
            int totalSize = 1;
            for (int i = 0; i < output.dims; i++) {
                totalSize *= output.size[i];
            }
            output = output.reshape(1, totalSize);
        }

        // 调试输出：显示原始模型输出
        qDebug() << "Raw model output:";
        if (output.total() == 1) {
            float rawValue = output.at<float>(0);
            qDebug() << "Single output value:" << rawValue;

            // 如果输出值很大（可能是logits），使用用户建议的100阈值
            if (std::abs(rawValue) > 10.0f) {
                result.classId = (rawValue > 100.0f) ? 1 : 0;
                result.confidence = std::abs(rawValue) / 200.0f;  // 粗略归一化到0-1
                qDebug() << "Using threshold-based classification (raw value > 100 = Class 1)";
            } else {
                // 小值情况，使用sigmoid
                result.confidence = 1.0f / (1.0f + std::exp(-rawValue));  // sigmoid
                result.classId = (result.confidence > confThreshold_) ? 1 : 0;
                qDebug() << "Using sigmoid-based classification";
            }
        } else if (output.total() >= 2) {
            // 多输出情况
            qDebug() << "Multiple output values:";
            for (size_t i = 0; i < output.total(); i++) {
                qDebug() << "  Output[" << i << "]:" << output.at<float>(i);
            }

            // 检查是否是logits（值较大）
            bool isLogits = false;
            for (size_t i = 0; i < output.total(); i++) {
                if (std::abs(output.at<float>(i)) > 10.0f) {
                    isLogits = true;
                    break;
                }
            }

            if (isLogits) {
                // 如果是logits，使用softmax转换为概率
                double maxVal;
                cv::minMaxLoc(output, nullptr, &maxVal);

                cv::Mat softmaxOutput;
                cv::exp(output - maxVal, softmaxOutput);  // 数值稳定性
                softmaxOutput /= cv::sum(softmaxOutput)[0];

                cv::Point maxLoc;
                double probVal;
                cv::minMaxLoc(softmaxOutput, nullptr, &probVal, nullptr, &maxLoc);

                result.classId = maxLoc.x;
                result.confidence = static_cast<float>(probVal);
                qDebug() << "Using softmax classification (logits to probabilities)";

                // 调试输出softmax后的概率
                qDebug() << "Softmax probabilities:";
                for (size_t i = 0; i < softmaxOutput.total(); i++) {
                    qDebug() << "  Class" << i << "probability:" << softmaxOutput.at<float>(i);
                }
            } else {
                // 如果是概率，直接使用
                cv::Point maxLoc;
                double probVal;
                cv::minMaxLoc(output, nullptr, &probVal, nullptr, &maxLoc);

                result.classId = maxLoc.x;
                result.confidence = static_cast<float>(probVal);
                qDebug() << "Using direct probability classification";
            }
        } else {
            qDebug() << "Unexpected output size:" << output.total();
            return result;
        }

        // 设置类别名称
        if (result.classId >= 0 && result.classId < static_cast<int>(classLabels_.size())) {
            result.className = classLabels_[result.classId];
        } else {
            result.className = "Class_" + std::to_string(result.classId);
        }

        // 检查置信度阈值
        result.isValid = (result.confidence >= confThreshold_);

        qDebug() << "Classification output - Class:" << result.classId
                 << "Confidence:" << result.confidence
                 << "Valid:" << result.isValid;

    } catch (const cv::Exception& e) {
        qDebug() << "Error in post-processing:" << QString::fromStdString(e.msg);
    }

    return result;
}

void DLProcessor::postProcess(cv::Mat& frame, const std::vector<cv::Mat>& outs)
{
    // 兼容原接口的后处理方法
    ClassificationResult result = postProcessClassification(outs);

    if (result.isValid) {
        // 在图像上绘制分类结果
        std::string text = result.className + ": " + std::to_string(result.confidence);
        cv::putText(frame, text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                    1.0, cv::Scalar(0, 255, 0), 2);

        // 根据分类结果添加边框颜色
        cv::Scalar borderColor = (result.classId == 1) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        cv::rectangle(frame, cv::Point(0, 0), cv::Point(frame.cols-1, frame.rows-1),
                      borderColor, 3);
    }
}

void DLProcessor::setInputSize(const cv::Size& size)
{
    inputSize_ = size;
    qDebug() << "Input size set to:" << size.width << "x" << size.height;
}

void DLProcessor::setPreprocessParams(const cv::Scalar& mean, const cv::Scalar& std,
                                     double scaleFactor, bool swapRB)
{
    meanValues_ = mean;
    stdValues_ = std;
    scaleFactor_ = scaleFactor;
    swapRB_ = swapRB;

    qDebug() << "Preprocessing parameters updated:";
    qDebug() << "  Mean:" << mean[0] << "," << mean[1] << "," << mean[2];
    qDebug() << "  Std:" << std[0] << "," << std[1] << "," << std[2];
    qDebug() << "  Scale factor:" << scaleFactor;
    qDebug() << "  Swap RB:" << swapRB;
}

float DLProcessor::getConfidenceThreshold() const
{
    return confThreshold_;
}

float DLProcessor::getNMSThreshold() const
{
    return nmsThreshold_;
}

void DLProcessor::enableGPU(bool enable)
{
    if (!isModelLoaded_) {
        qDebug() << "Cannot change backend: model not loaded";
        return;
    }

    try {
        if (enable) {
            // 尝试启用GPU后端
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            qDebug() << "GPU backend enabled";
        } else {
            // 使用CPU后端
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            qDebug() << "CPU backend enabled";
        }
    } catch (const cv::Exception& e) {
        qDebug() << "Error setting backend:" << QString::fromStdString(e.msg);
        emit errorOccurred(QString::fromStdString(e.msg));
    }
}

void DLProcessor::resetToDefaults()
{
    confThreshold_ = 0.5f;
    nmsThreshold_ = 0.4f;
    initDefaultParams();
    qDebug() << "Parameters reset to defaults";
}

std::string DLProcessor::getModelInfo() const
{
    if (!isModelLoaded_) {
        return "No model loaded";
    }

    std::string info = "Model Information:\n";
    info += "- Input size: " + std::to_string(inputSize_.width) + "x" + std::to_string(inputSize_.height) + "\n";
    info += "- Classes: " + std::to_string(classLabels_.size()) + "\n";
    info += "- Confidence threshold: " + std::to_string(confThreshold_) + "\n";
    info += "- Scale factor: " + std::to_string(scaleFactor_) + "\n";

    return info;
}

bool DLProcessor::saveClassLabels(const std::string& labelPath) const
{
    try {
        std::ofstream file(labelPath);
        if (!file.is_open()) {
            qDebug() << "Cannot create label file:" << QString::fromStdString(labelPath);
            return false;
        }

        for (const auto& label : classLabels_) {
            file << label << std::endl;
        }
        file.close();

        qDebug() << "Saved" << classLabels_.size() << "class labels to:" << QString::fromStdString(labelPath);
        return true;
    } catch (const std::exception& e) {
        qDebug() << "Error saving class labels:" << QString::fromStdString(e.what());
        return false;
    }
}

void DLProcessor::clearModel()
{
    net_ = cv::dnn::Net();
    isModelLoaded_ = false;
    qDebug() << "Model cleared";
}

bool DLProcessor::warmUp()
{
    if (!isModelLoaded_) {
        qDebug() << "Cannot warm up: model not loaded";
        return false;
    }

    try {
        // 创建一个虚拟输入进行预热
        cv::Mat dummyInput = cv::Mat::zeros(inputSize_, CV_8UC3);
        ClassificationResult result;

        qDebug() << "Warming up model...";
        bool success = classifyImage(dummyInput, result);

        if (success) {
            qDebug() << "Model warm-up completed successfully";
        } else {
            qDebug() << "Model warm-up failed";
        }

        return success;
    } catch (const cv::Exception& e) {
        qDebug() << "Error during warm-up:" << QString::fromStdString(e.msg);
        return false;
    }
}
