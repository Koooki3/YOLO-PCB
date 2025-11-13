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
    , isQuantized_(false)
    , quantizationType_("")
{
    InitDefaultParams();
}

DLProcessor::~DLProcessor()
{

}

void DLProcessor::InitDefaultParams()
{
    // 初始化默认参数 - 调整为适合YOLO模型的参数
    inputSize_ = Size(640, 640);                // YOLOv8默认输入尺寸
    meanValues_ = Scalar(0.0, 0.0, 0.0);        // YOLO通常使用0均值
    scaleFactor_ = 1.0 / 255.0;                 // YOLO通常使用1/255归一化
    swapRB_ = true;                             // OpenCV默认BGR，YOLO需要RGB

    // 默认二分类标签
    classLabels_ = {"Class_0", "Class_1"};
}

bool DLProcessor::InitModel(const string& modelPath, const string& configPath)
{
    // 变量定义
    bool modelLoaded = false;  // 模型加载状态

    try 
    {
        // 检测是否为ONNX模型
        bool isOnnxModel = false;
        string lowerModelPath = modelPath;
        transform(lowerModelPath.begin(), lowerModelPath.end(), lowerModelPath.begin(), ::tolower);
        if (lowerModelPath.find(".onnx") != string::npos)
        {
            isOnnxModel = true;
            qDebug() << "Detected ONNX model format";
        }

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

        // 设置计算后端 (根据实际硬件选择) 
        net_.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(dnn::DNN_TARGET_CPU);

        // 可选: 如果有GPU支持
        // net_.setPreferableBackend(dnn::DNN_BACKEND_CUDA);
        // net_.setPreferableTarget(dnn::DNN_TARGET_CUDA);

        // 如果是ONNX模型，可能需要额外的设置
        if (isOnnxModel)
        {
            qDebug() << "Configuring for ONNX model compatibility";
            // 对于YOLO ONNX模型，确保输入尺寸正确设置
            // 这里可以根据需要添加更多针对ONNX模型的特殊处理
        }

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

        // 调试输出: 显示加载的标签内容
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
        // 转换颜色空间 (如果需要) 
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

bool DLProcessor::DetectObjects(const Mat& frame, vector<DetectionResult>& results)
{
    // 变量定义
    Mat blob;                            // 预处理后的blob数据
    vector<Mat> outs;                    // 网络输出结果

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
        // 预处理 - 注意YOLO通常使用不同的预处理参数
        blob = PreProcess(frame);

        // 前向传播
        net_.setInput(blob);
        net_.forward(outs, net_.getUnconnectedOutLayersNames());

        // YOLO后处理 - 包括边界框检测和实例分割
        results = PostProcessYolo(frame, outs, confThreshold_, nmsThreshold_);

        qDebug() << "Detected" << results.size() << "objects";
        return !results.empty();
    }
    catch (const Exception& e)
    {
        qDebug() << "Error detecting objects:" << QString::fromStdString(e.msg);
        emit errorOccurred(QString::fromStdString(e.msg));
        return false;
    }
}

bool DLProcessor::ProcessYoloFrame(const Mat& frame, Mat& output)
{
    // 变量定义
    vector<DetectionResult> results;

    // 执行检测
    if (!DetectObjects(frame, results))
    {
        return false;
    }

    // 复制原始图像并绘制结果
    output = frame.clone();
    DrawDetectionResults(output, results);

    // 发送处理完成信号
    emit processingComplete(output);
    return true;
}

vector<DetectionResult> DLProcessor::PostProcessYolo(const Mat& frame, const vector<Mat>& outs, float confThreshold, float nmsThreshold)
{
    vector<DetectionResult> results;
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    vector<Mat> masks;

    // YOLO模型输出解析
    for (size_t i = 0; i < outs.size(); ++i)
    {
        qDebug() << "Processing output layer" << i;
        qDebug() << "Output shape: dimensions=" << outs[i].dims;
        for (int d = 0; d < outs[i].dims; ++d)
        {
            qDebug() << "  dim[" << d << "]=" << outs[i].size[d];
        }

        // 处理YOLOv8 ONNX输出格式
        // YOLOv8导出为ONNX时通常会有一个输出，形状为[N, num_boxes, num_attributes]
        if (outs[i].dims == 3)
        {
            // 这是YOLOv8的典型输出格式
            int batchSize = outs[i].size[0];
            int numDetections = outs[i].size[1];
            int detectionSize = outs[i].size[2];
            
            qDebug() << "YOLOv8 ONNX format detected: batch=" << batchSize 
                     << " detections=" << numDetections 
                     << " detection_size=" << detectionSize;
            
            // 假设输出格式: [batch, num_detections, (x, y, w, h, conf, class0, class1, ...)]
            for (int b = 0; b < batchSize; ++b)
            {
                for (int d = 0; d < numDetections; ++d)
                {
                    // 获取当前检测的起始指针
                    float* detectionPtr = (float*)outs[i].ptr<float>(b, d);
                    
                    // 提取边界框信息
                    float x = detectionPtr[0];  // 中心点x坐标
                    float y = detectionPtr[1];  // 中心点y坐标
                    float w = detectionPtr[2];  // 宽度
                    float h = detectionPtr[3];  // 高度
                    float confidence = detectionPtr[4];  // 置信度
                    
                    // 应用置信度阈值
                    if (confidence > confThreshold)
                    {
                        // 找到最高置信度的类别
                        int classId = -1;
                        float maxClassConf = 0.0f;
                        
                        // 从第5个元素开始是类别置信度
                        for (int c = 0; c < detectionSize - 5; ++c)
                        {
                            if (detectionPtr[5 + c] > maxClassConf)
                            {
                                maxClassConf = detectionPtr[5 + c];
                                classId = c;
                            }
                        }
                        
                        // 计算边界框坐标（YOLOv8输出是归一化的坐标）
                        int left = static_cast<int>((x - w / 2) * frame.cols);
                        int top = static_cast<int>((y - h / 2) * frame.rows);
                        int width = static_cast<int>(w * frame.cols);
                        int height = static_cast<int>(h * frame.rows);
                        
                        // 确保边界框在图像范围内
                        left = max(0, left);
                        top = max(0, top);
                        width = min(frame.cols - left, width);
                        height = min(frame.rows - top, height);
                        
                        classIds.push_back(classId);
                        confidences.push_back(confidence * maxClassConf);  // 乘以类别置信度
                        boxes.push_back(Rect(left, top, width, height));
                    }
                }
            }
        }
        else if (outs[i].dims == 2)
        {
            // 传统的YOLO输出格式
            float* data = (float*)outs[i].data;
            
            // 对于每个检测结果
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
            {
                // 提取边界框信息
                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];
                float confidence = data[4];
                
                // 应用置信度阈值
                if (confidence > confThreshold)
                {
                    // 获取置信度最高的类别
                    int classId = -1;
                    float maxClassConf = 0.0f;
                    
                    for (int c = 0; c < outs[i].cols - 5; ++c)
                    {
                        if (data[5 + c] > maxClassConf)
                        {
                            maxClassConf = data[5 + c];
                            classId = c;
                        }
                    }
                    
                    // 计算边界框坐标
                    int left = static_cast<int>((x - w / 2) * frame.cols);
                    int top = static_cast<int>((y - h / 2) * frame.rows);
                    int width = static_cast<int>(w * frame.cols);
                    int height = static_cast<int>(h * frame.rows);
                    
                    // 确保边界框在图像范围内
                    left = max(0, left);
                    top = max(0, top);
                    width = min(frame.cols - left, width);
                    height = min(frame.rows - top, height);
                    
                    classIds.push_back(classId);
                    confidences.push_back(confidence * maxClassConf);
                    boxes.push_back(Rect(left, top, width, height));
                }
            }
        }
    }
    
    // 应用非极大值抑制
    vector<int> indices;
    if (!boxes.empty())
    {
        dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
        
        // 构建最终结果
        for (size_t i = 0; i < indices.size(); ++i)
        {
            int idx = indices[i];
            DetectionResult result;
            result.classId = classIds[idx];
            result.confidence = confidences[idx];
            result.boundingBox = boxes[idx];
            
            // 设置类别名称
            if (result.classId >= 0 && result.classId < static_cast<int>(classLabels_.size()))
            {
                result.className = classLabels_[result.classId];
            }
            else
            {
                result.className = "Class_" + to_string(result.classId);
            }
            
            results.push_back(result);
        }
    }
    
    qDebug() << "Final detection results count:" << results.size();
    return results;
}

void DLProcessor::DrawDetectionResults(Mat& frame, const vector<DetectionResult>& results)
{
    // 颜色列表，用于不同类别
    vector<Scalar> colors = {
        Scalar(0, 0, 255),   // 红色
        Scalar(0, 255, 0),   // 绿色
        Scalar(255, 0, 0),   // 蓝色
        Scalar(0, 255, 255), // 黄色
        Scalar(255, 0, 255), // 紫色
        Scalar(255, 255, 0)  // 青色
    };
    
    // 绘制每个检测结果
    for (size_t i = 0; i < results.size(); ++i)
    {
        const DetectionResult& result = results[i];
        
        // 选择颜色
        Scalar color = colors[i % colors.size()];
        
        // 绘制边界框
        rectangle(frame, result.boundingBox, color, 2);
        
        // 绘制类别和置信度
        string label = result.className + ": " + to_string(result.confidence).substr(0, 4);
        int baseline = 0;
        Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        
        // 绘制标签背景
        Rect labelRect(result.boundingBox.x, result.boundingBox.y - labelSize.height, 
                      labelSize.width, labelSize.height + baseline);
        rectangle(frame, labelRect, color, FILLED);
        
        // 绘制标签文本
        putText(frame, label, Point(result.boundingBox.x, result.boundingBox.y - 5),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
        
        // 如果有分割掩码，绘制掩码
        if (!result.mask.empty())
        {
            // 创建彩色掩码
            Mat coloredMask(result.mask.size(), CV_8UC3, color);
            
            // 将掩码应用到原始图像的相应区域
            Mat roi = frame(result.boundingBox);
            Mat maskROI;
            coloredMask.copyTo(maskROI, result.mask);
            
            // 添加半透明效果
            addWeighted(maskROI, 0.3, roi, 0.7, 0, roi);
        }
    }
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
        // 获取第一个输出 (通常分类模型只有一个输出) 
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

        // 调试输出: 显示原始模型输出
        qDebug() << "Raw model output:";
        if (output.total() == 1) 
        {
            float rawValue = output.at<float>(0);
            qDebug() << "Single output value:" << rawValue;

            // 如果输出值很大 (可能是logits) ，使用用户建议的100阈值
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

            // 检查是否是logits (值较大) 
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
    try
    {
        if (!isModelLoaded_)
        {
            qDebug() << "Cannot set GPU mode: model not loaded";
            emit errorOccurred("Cannot set GPU mode: model not loaded");
            return;
        }

        if (enable)
        {
            #ifdef OPENCV_DNN_CUDA
            net_.setPreferableBackend(dnn::DNN_BACKEND_CUDA);
            net_.setPreferableTarget(dnn::DNN_TARGET_CUDA);
            qDebug() << "GPU backend enabled";
            #else
            qDebug() << "CUDA backend not available in this OpenCV build";
            emit errorOccurred("CUDA backend not available in this OpenCV build");
            #endif
        }
        else
        {
            net_.setPreferableBackend(dnn::DNN_BACKEND_DEFAULT);
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
    isQuantized_ = false;
    quantizationType_ = "";
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
    info += "- Quantization: " + (isQuantized_ ? quantizationType_ : "None") + "\n";

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
    isQuantized_ = false;
    quantizationType_ = "";
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
        qDebug() << "OpenCV exception during warm-up: " << e.what();
        emit errorOccurred(QString("Warm-up failed: %1").arg(e.what()));
        return false;
    }
    catch (const std::exception& e)
    {
        qDebug() << "Standard exception during warm-up: " << e.what();
        emit errorOccurred(QString("Warm-up failed: %1").arg(e.what()));
        return false;
    }
    catch(...)
    {
        qDebug() << "Unknown exception during warm-up";
        emit errorOccurred("Warm-up failed: Unknown error");
        return false;
    }
}

bool DLProcessor::QuantizeModel(const string& quantizationType, const vector<Mat>& calibrationImages)
{
//    // 检查模型是否已加载
//    if (!isModelLoaded_)
//    {
//        qDebug() << "Cannot quantize: model not loaded";
//        emit errorOccurred("Cannot quantize: model not loaded");
//        return false;
//    }

//    try
//    {
//        qDebug() << "Starting model quantization with type:" << QString::fromStdString(quantizationType);

//        // 如果模型已经量化，则先清除
//        if (isQuantized_)
//        {
//            qDebug() << "Model is already quantized, will re-quantize";
//        }

//        // 量化配置
//        cv::dnn::Net quantizedNet;

//        // 根据量化类型选择不同的量化策略
//        if (quantizationType == "INT8")
//        {
//            // INT8量化需要校准图像
//            if (calibrationImages.empty())
//            {
//                qDebug() << "INT8 quantization requires calibration images";
//                emit errorOccurred("INT8 quantization requires calibration images");
//                return false;
//            }

//            // 准备校准数据
//            std::vector<cv::Mat> calibrationBlobs;
//            for (const auto& img : calibrationImages)
//            {
//                if (!img.empty())
//                {
//                    cv::Mat blob = PreProcess(img);
//                    calibrationBlobs.push_back(blob);
//                }
//            }

//            // 检查校准数据是否有效
//            if (calibrationBlobs.empty())
//            {
//                qDebug() << "No valid calibration images provided";
//                emit errorOccurred("No valid calibration images provided");
//                return false;
//            }

//            qDebug() << "Using" << calibrationBlobs.size() << "calibration images for INT8 quantization";

//            // OpenCV DNN的INT8量化方法 (使用dnn::writeTextGraph和dnn::readNetFromONNX/Protobuf等)
//            // 注意：这里使用简化的量化实现，实际项目中可能需要更复杂的校准过程

//            // 保存原始模型的网络结构到临时文件
//            string tempPrototxt = "temp_quantization_model.prototxt";
//            if (cv::dnn::writeTextGraph(net_, tempPrototxt))
//            {
//                qDebug() << "Successfully wrote network graph for quantization";

//                // 这里是简化的量化处理，实际的INT8量化通常需要：
//                // 1. 运行校准数据获取激活值的范围
//                // 2. 计算量化参数
//                // 3. 应用量化到模型权重和激活值

//                // 为了演示，我们直接使用原始网络但标记为已量化
//                // 实际应用中应该调用相应的量化API
//                quantizedNet = net_.clone();

//                // 删除临时文件
//                std::remove(tempPrototxt.c_str());
//            }
//        }
//        else if (quantizationType == "FP16")
//        {
//            // FP16量化相对简单，大多数后端都支持
//            qDebug() << "Applying FP16 precision optimization";
//            quantizedNet = net_.clone();

//            // 设置FP16推理模式（如果后端支持）
//            #ifdef OPENCV_DNN_CUDA
//            quantizedNet.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
//            quantizedNet.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
//            #endif
//        }
//        else if (quantizationType == "UINT8")
//        {
//            // UINT8量化类似于INT8，但使用无符号整数
//            qDebug() << "Applying UINT8 quantization";
//            if (calibrationImages.empty())
//            {
//                qDebug() << "UINT8 quantization requires calibration images";
//                emit errorOccurred("UINT8 quantization requires calibration images");
//                return false;
//            }

//            // 类似INT8的实现，但使用无符号整数范围
//            quantizedNet = net_.clone();
//        }
//        else
//        {
//            qDebug() << "Unsupported quantization type:" << QString::fromStdString(quantizationType);
//            emit errorOccurred("Unsupported quantization type");
//            return false;
//        }

//        // 验证量化后的模型是否有效
//        if (quantizedNet.empty())
//        {
//            qDebug() << "Failed to create quantized network";
//            emit errorOccurred("Failed to create quantized network");
//            return false;
//        }

//        // 替换原始网络
//        net_ = quantizedNet;
//        isQuantized_ = true;
//        quantizationType_ = quantizationType;

//        qDebug() << "Model quantization successful. Type:" << QString::fromStdString(quantizationType);

//        // 预热量化后的模型
//        WarmUp();

//        return true;
//    }
//    catch (const Exception& e)
//    {
//        qDebug() << "Error during model quantization:" << QString::fromStdString(e.msg);
//        emit errorOccurred(QString::fromStdString(e.msg));
//        return false;
//    }
//    catch (const Exception& e)
//    {
//        qDebug() << "Error during warm-up:" << QString::fromStdString(e.msg);
//        return false;
//    }
}
