#include "DLProcessor.h"
#include <QDebug>
#include <QFile>
#include <QTextStream>
#include <algorithm>
#include <fstream>

using namespace cv;
using namespace std;

DLProcessor::DLProcessor(QObject *parent)
    : QObject(parent)
    , confThreshold_(0.5f)
    , nmsThreshold_(0.4f)
    , isModelLoaded_(false)
{
    InitDefaultParams();
}

DLProcessor::~DLProcessor()
{

}

void DLProcessor::InitDefaultParams()
{
    // 初始化默认参数
    inputSize_ = Size(224, 224);                // 常用的分类模型输入尺寸
    meanValues_ = Scalar(104.0, 177.0, 123.0);  // ImageNet均值
    scaleFactor_ = 1.0;
    swapRB_ = true;                             // OpenCV默认BGR，很多模型需要RGB

    // 默认二分类标签
    classLabels_ = {"Class_0", "Class_1"};
}

bool DLProcessor::InitModel(const string& modelPath, const string& configPath)
{
    // 变量定义
    bool modelLoaded = false;  // 模型加载状态

    try 
    {
        // 加载深度学习模型
        if (configPath.empty()) 
        {
            net_ = dnn::readNet(modelPath);
        } 
        else 
        {
            net_ = dnn::readNet(modelPath, configPath);
        }

        if (net_.empty()) 
        {
            qDebug() << "Failed to load model from:" << QString::fromStdString(modelPath);
            emit errorOccurred("Failed to load model: empty network");
            return false;
        }

        // 设置计算后端（根据实际硬件选择）
        net_.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(dnn::DNN_TARGET_CPU);

        // 可选：如果有GPU支持
        // net_.setPreferableBackend(dnn::DNN_BACKEND_CUDA);
        // net_.setPreferableTarget(dnn::DNN_TARGET_CUDA);

        isModelLoaded_ = true;
        modelLoaded = true;
        qDebug() << "Model loaded successfully from:" << QString::fromStdString(modelPath);
        return true;
    }
    catch (const Exception& e) 
    {
        qDebug() << "Error loading model:" << QString::fromStdString(e.msg);
        emit errorOccurred(QString::fromStdString(e.msg));
        return false;
    }
}

void DLProcessor::SetModelParams(float confThreshold, float nmsThreshold)
{
    confThreshold_ = confThreshold;
    nmsThreshold_ = nmsThreshold;
}

void DLProcessor::SetClassLabels(const vector<string>& labels)
{
    classLabels_ = labels;
    qDebug() << "Class labels updated. Total classes:" << classLabels_.size();
}

bool DLProcessor::LoadClassLabels(const string& labelPath)
{
    // 变量定义
    ifstream file(labelPath);  // 文件流对象
    string line;               // 存储每行读取的内容
    bool loadSuccess = false;  // 加载成功标志

    try 
    {
        if (!file.is_open()) 
        {
            qDebug() << "Cannot open label file:" << QString::fromStdString(labelPath);
            return false;
        }

        classLabels_.clear();
        while (getline(file, line)) 
        {
            if (!line.empty()) 
            {
                classLabels_.push_back(line);
            }
        }
        file.close();

        qDebug() << "Loaded" << classLabels_.size() << "class labels from:" << QString::fromStdString(labelPath);

        // 调试输出：显示加载的标签内容
        qDebug() << "Loaded class labels:";
        for (size_t i = 0; i < classLabels_.size(); i++) 
        {
            qDebug() << "  Class" << i << ":" << QString::fromStdString(classLabels_[i]);
        }

        loadSuccess = true;
        return true;
    }
    catch (const exception& e) 
    {
        qDebug() << "Error loading class labels:" << QString::fromStdString(e.what());
        return false;
    }
}

bool DLProcessor::ValidateInput(const Mat& frame)
{
    if (frame.empty()) 
    {
        emit errorOccurred("Input frame is empty");
        return false;
    }

    if (frame.channels() != 1 && frame.channels() != 3) 
    {
        emit errorOccurred("Input frame must have 1 or 3 channels");
        return false;
    }

    return true;
}

bool DLProcessor::ClassifyImage(const Mat& frame, ClassificationResult& result)
{
    // 变量定义
    Mat blob;                            // 预处理后的blob数据
    vector<Mat> outs;                    // 网络输出结果
    bool classificationSuccess = false;  // 分类成功标志

    if (!isModelLoaded_) 
    {
        emit errorOccurred("Model not loaded!");
        return false;
    }

    if (!ValidateInput(frame)) 
    {
        return false;
    }

    try 
    {
        // 预处理
        blob = PreProcess(frame);

        // 前向传播
        net_.setInput(blob);
        net_.forward(outs, net_.getUnconnectedOutLayersNames());

        // 后处理 - 二分类
        result = PostProcessClassification(outs);

        if (result.isValid) 
        {
            emit classificationComplete(result);
            qDebug() << "Classification result: Class" << result.classId
                     << "(" << QString::fromStdString(result.className) << ")"
                     << "Confidence:" << result.confidence;
            classificationSuccess = true;
        }

        return result.isValid;
    }
    catch (const Exception& e) 
    {
        qDebug() << "Error classifying image:" << QString::fromStdString(e.msg);
        emit errorOccurred(QString::fromStdString(e.msg));
        return false;
    }
}

bool DLProcessor::ClassifyBatch(const vector<Mat>& frames, vector<ClassificationResult>& results)
{
    // 变量定义
    bool allSuccess = true;             // 所有图像分类是否成功
    ClassificationResult singleResult;  // 单张图像分类结果

    if (!isModelLoaded_) 
    {
        emit errorOccurred("Model not loaded!");
        return false;
    }

    results.clear();
    results.reserve(frames.size());

    // 批量处理所有图像
    for (const auto& frame : frames) 
    {
        if (ClassifyImage(frame, singleResult)) 
        {
            results.push_back(singleResult);
        } 
        else 
        {
            allSuccess = false;
            // 添加无效结果以保持索引对应
            results.push_back(ClassificationResult());
        }
    }

    if (!results.empty()) 
    {
        emit batchProcessingComplete(results);
    }

    return allSuccess;
}

bool DLProcessor::ProcessFrame(const Mat& frame, Mat& output)
{
    // 变量定义
    ClassificationResult result;  // 分类结果
    string text;                  // 显示的文本内容
    Scalar borderColor;           // 边框颜色

    if (!ClassifyImage(frame, result)) 
    {
        return false;
    }

    // 在图像上绘制分类结果
    output = frame.clone();

    // 添加文本标注
    text = result.className + ": " + to_string(result.confidence);
    putText(output, text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);

    // 根据分类结果添加边框颜色
    borderColor = (result.classId == 1) ? Scalar(0, 255, 0) : Scalar(0, 0, 255);
    rectangle(output, Point(0, 0), Point(output.cols-1, output.rows-1), borderColor, 3);

    emit processingComplete(output);
    return true;
}

Mat DLProcessor::PreProcess(const Mat& frame)
{
    Mat blob;  // 存储预处理后的blob数据

    try 
    {
        // 转换颜色空间（如果需要）
        Mat processedFrame = frame;
        if (frame.channels() == 1 && inputSize_.width > 0) 
        {
            // 灰度图转RGB
            cvtColor(frame, processedFrame, COLOR_GRAY2RGB);
        }

        // 创建blob
        dnn::blobFromImage(processedFrame, blob, scaleFactor_, inputSize_, meanValues_, swapRB_, false, CV_32F);

        qDebug() << "Preprocessed blob shape: [" << blob.size[0] << ","
                 << blob.size[1] << "," << blob.size[2] << "," << blob.size[3] << "]";

    } 
    catch (const Exception& e) 
    {
        qDebug() << "Error in preprocessing:" << QString::fromStdString(e.msg);
        throw;
    }

    return blob;
}

ClassificationResult DLProcessor::PostProcessClassification(const vector<Mat>& outs)
{
    // 变量定义
    ClassificationResult result;  // 存储分类结果
    Mat output;                   // 网络输出数据
    bool isLogits = false;        // 是否为logits输出

    if (outs.empty()) 
    {
        qDebug() << "No output from network";
        return result;
    }

    try 
    {
        // 获取第一个输出（通常分类模型只有一个输出）
        output = outs[0];

        // 确保输出是1D向量
        if (output.dims > 2) 
        {
            // 如果是多维，reshape为2D
            int totalSize = 1;
            for (int i = 0; i < output.dims; i++) 
            {
                totalSize *= output.size[i];
            }
            output = output.reshape(1, totalSize);
        }

        // 调试输出：显示原始模型输出
        qDebug() << "Raw model output:";
        if (output.total() == 1) 
        {
            float rawValue = output.at<float>(0);
            qDebug() << "Single output value:" << rawValue;

            // 如果输出值很大（可能是logits），使用用户建议的100阈值
            if (abs(rawValue) > 10.0f) 
            {
                result.classId = (rawValue > 100.0f) ? 1 : 0;
                result.confidence = abs(rawValue) / 200.0f;  // 粗略归一化到0-1
                qDebug() << "Using threshold-based classification (raw value > 100 = Class 1)";
            } 
            else 
            {
                // 小值情况，使用sigmoid
                result.confidence = 1.0f / (1.0f + exp(-rawValue));  // sigmoid
                result.classId = (result.confidence > confThreshold_) ? 1 : 0;
                qDebug() << "Using sigmoid-based classification";
            }
        } 
        else if (output.total() >= 2) 
        {
            // 多输出情况
            qDebug() << "Multiple output values:";
            for (size_t i = 0; i < output.total(); i++) 
            {
                qDebug() << "  Output[" << i << "]:" << output.at<float>(i);
            }

            // 检查是否是logits（值较大）
            for (size_t i = 0; i < output.total(); i++) 
            {
                if (abs(output.at<float>(i)) > 10.0f) 
                {
                    isLogits = true;
                    break;
                }
            }

            if (isLogits) 
            {
                // 如果是logits，使用softmax转换为概率
                double maxVal;
                minMaxLoc(output, nullptr, &maxVal);

                Mat softmaxOutput;
                exp(output - maxVal, softmaxOutput);  // 数值稳定性
                softmaxOutput /= sum(softmaxOutput)[0];

                Point maxLoc;
                double probVal;
                minMaxLoc(softmaxOutput, nullptr, &probVal, nullptr, &maxLoc);

                result.classId = maxLoc.x;
                result.confidence = static_cast<float>(probVal);
                qDebug() << "Using softmax classification (logits to probabilities)";

                // 调试输出softmax后的概率
                qDebug() << "Softmax probabilities:";
                for (size_t i = 0; i < softmaxOutput.total(); i++) 
                {
                    qDebug() << "  Class" << i << "probability:" << softmaxOutput.at<float>(i);
                }
            } 
            else 
            {
                // 如果是概率，直接使用
                Point maxLoc;
                double probVal;
                minMaxLoc(output, nullptr, &probVal, nullptr, &maxLoc);

                result.classId = maxLoc.x;
                result.confidence = static_cast<float>(probVal);
                qDebug() << "Using direct probability classification";
            }
        } 
        else 
        {
            qDebug() << "Unexpected output size:" << output.total();
            return result;
        }

        // 设置类别名称
        if (result.classId >= 0 && result.classId < static_cast<int>(classLabels_.size()))
        {
            result.className = classLabels_[result.classId];
        } 
        else 
        {
            result.className = "Class_" + to_string(result.classId);
        }

        // 检查置信度阈值
        result.isValid = (result.confidence >= confThreshold_);

        qDebug() << "Classification output - Class:" << result.classId
                 << "Confidence:" << result.confidence
                 << "Valid:" << result.isValid;

    } 
    catch (const Exception& e) 
    {
        qDebug() << "Error in post-processing:" << QString::fromStdString(e.msg);
    }

    return result;
}

void DLProcessor::PostProcess(Mat& frame, const vector<Mat>& outs)
{
    // 变量定义
    ClassificationResult result;  // 分类结果
    string text;                  // 显示的文本内容
    Scalar borderColor;           // 边框颜色

    // 兼容原接口的后处理方法
    result = PostProcessClassification(outs);

    if (result.isValid) 
    {
        // 在图像上绘制分类结果
        text = result.className + ": " + to_string(result.confidence);
        putText(frame, text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);

        // 根据分类结果添加边框颜色
        borderColor = (result.classId == 1) ? Scalar(0, 255, 0) : Scalar(0, 0, 255);
        rectangle(frame, Point(0, 0), Point(frame.cols-1, frame.rows-1), borderColor, 3);
    }
}

void DLProcessor::SetInputSize(const Size& size)
{
    inputSize_ = size;
    qDebug() << "Input size set to:" << size.width << "x" << size.height;
}

void DLProcessor::SetPreprocessParams(const Scalar& mean, const Scalar& std, double scaleFactor, bool swapRB)
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

float DLProcessor::GetConfidenceThreshold() const
{
    return confThreshold_;
}

float DLProcessor::GetNMSThreshold() const
{
    return nmsThreshold_;
}

void DLProcessor::EnableGPU(bool enable)
{
    if (!isModelLoaded_) 
    {
        qDebug() << "Cannot change backend: model not loaded";
        return;
    }

    try 
    {
        if (enable) 
        {
            // 尝试启用GPU后端
            net_.setPreferableBackend(dnn::DNN_BACKEND_CUDA);
            net_.setPreferableTarget(dnn::DNN_TARGET_CUDA);
            qDebug() << "GPU backend enabled";
        } 
        else 
        {
            // 使用CPU后端
            net_.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
            net_.setPreferableTarget(dnn::DNN_TARGET_CPU);
            qDebug() << "CPU backend enabled";
        }
    } 
    catch (const Exception& e) 
    {
        qDebug() << "Error setting backend:" << QString::fromStdString(e.msg);
        emit errorOccurred(QString::fromStdString(e.msg));
    }
}

void DLProcessor::ResetToDefaults()
{
    confThreshold_ = 0.5f;
    nmsThreshold_ = 0.4f;
    InitDefaultParams();
    qDebug() << "Parameters reset to defaults";
}

string DLProcessor::GetModelInfo() const
{
    if (!isModelLoaded_) 
    {
        return "No model loaded";
    }

    string info = "Model Information:\n";
    info += "- Input size: " + to_string(inputSize_.width) + "x" + to_string(inputSize_.height) + "\n";
    info += "- Classes: " + to_string(classLabels_.size()) + "\n";
    info += "- Confidence threshold: " + to_string(confThreshold_) + "\n";
    info += "- Scale factor: " + to_string(scaleFactor_) + "\n";

    return info;
}

bool DLProcessor::SaveClassLabels(const string& labelPath) const
{
    // 变量定义
    ofstream file(labelPath);  // 输出文件流
    bool saveSuccess = false;  // 保存成功标志

    try 
    {
        if (!file.is_open()) 
        {
            qDebug() << "Cannot create label file:" << QString::fromStdString(labelPath);
            return false;
        }

        for (const auto& label : classLabels_) 
        {
            file << label << endl;
        }
        file.close();

        qDebug() << "Saved" << classLabels_.size() << "class labels to:" << QString::fromStdString(labelPath);
        saveSuccess = true;
        return true;
    } 
    catch (const exception& e) 
    {
        qDebug() << "Error saving class labels:" << QString::fromStdString(e.what());
        return false;
    }
}

void DLProcessor::ClearModel()
{
    net_ = dnn::Net();
    isModelLoaded_ = false;
    qDebug() << "Model cleared";
}

bool DLProcessor::WarmUp()
{
    // 变量定义
    Mat dummyInput;               // 虚拟输入图像
    ClassificationResult result;  // 分类结果
    bool warmUpSuccess = false;   // 预热成功标志

    if (!isModelLoaded_) 
    {
        qDebug() << "Cannot warm up: model not loaded";
        return false;
    }

    try 
    {
        // 创建一个虚拟输入进行预热
        dummyInput = Mat::zeros(inputSize_, CV_8UC3);

        qDebug() << "Warming up model...";
        warmUpSuccess = ClassifyImage(dummyInput, result);

        if (warmUpSuccess) 
        {
            qDebug() << "Model warm-up completed successfully";
        } 
        else 
        {
            qDebug() << "Model warm-up failed";
        }

        return warmUpSuccess;
    } 
    catch (const Exception& e) 
    {
        qDebug() << "Error during warm-up:" << QString::fromStdString(e.msg);
        return false;
    }
}
