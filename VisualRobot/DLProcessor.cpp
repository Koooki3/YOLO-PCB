#include "DLProcessor.h"
#include <QDebug>
#include <QFile>
#include <QTextStream>
#include <algorithm>
#include <fstream>
#ifdef _WIN32
    #include <direct.h>
    #include <windows.h>
#else
    #include <sys/stat.h>
    #include <sys/types.h>
#endif

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
    // 验证并限制置信度阈值在0.1-1.0范围内
    confThreshold_ = max(0.1f, min(1.0f, confThreshold));
    
    // 验证并限制NMS阈值在0.1-1.0范围内
    nmsThreshold_ = max(0.1f, min(1.0f, nmsThreshold));
    
    qDebug() << "Model parameters updated:";
    qDebug() << "  Confidence threshold:" << confThreshold_;
    qDebug() << "  NMS threshold:" << nmsThreshold_;
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
// 添加调试输出，确认检测结果数量
    qDebug() << "Number of detection results before drawing:" << results.size();
    
    // 绘制检测结果
    DrawDetectionResults(output, results);
    
    // 添加调试输出，确认绘制后图像尺寸
    qDebug() << "Output image size after drawing: width=" << output.cols << " height=" << output.rows;
    
    // 保存检测结果图片
    string savePath = "/home/orangepi/Desktop/VisualRobot_Local/Img/Yolo_Detected.jpg";
    try {
        // 确保保存目录存在
        string directory = "/home/orangepi/Desktop/VisualRobot_Local/Img";
        
        // 创建目录（如果不存在）
        #ifdef _WIN32
            // Windows系统
            CreateDirectoryA(directory.c_str(), NULL);
        #else
            // Linux/macOS系统
            system((string("mkdir -p ") + directory).c_str());
        #endif
        
        // 使用OpenCV的imwrite保存图片
        if (imwrite(savePath, output)) {
            qDebug() << "检测结果已保存到:" << QString::fromStdString(savePath);
        } else {
            qDebug() << "保存检测结果失败:" << QString::fromStdString(savePath);
        }
    } catch (const Exception& e) {
        qDebug() << "保存图片时出错:" << QString::fromStdString(e.msg);
    }

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

        // 处理YOLOv8 ONNX输出格式 - 格式1: [1, 84, 8400]
        if (outs[i].dims == 3)
        {
            int batchSize = outs[i].size[0];
            int numAttributes = outs[i].size[1];  // 通常是84 (4+1+80)或80+5
            int numDetections = outs[i].size[2];  // 通常是8400或25200
            
            qDebug() << "YOLOv8 ONNX format detected: batch=" << batchSize 
                     << " attributes=" << numAttributes 
                     << " detections=" << numDetections;
            
            // 处理格式1: [batch, attributes, num_detections]
            // attributes: [x_center, y_center, width, height, object_conf, class_0_conf, class_1_conf, ...]
            if (numAttributes < numDetections)  // 通常84 < 8400或25200
            {
                for (int b = 0; b < batchSize; ++b)
                {
                    for (int d = 0; d < numDetections; ++d)
                    {
                        // 获取当前检测的所有属性
                        float x_center = outs[i].at<float>(b, 0, d);  // 中心点x坐标（归一化）
                        float y_center = outs[i].at<float>(b, 1, d);  // 中心点y坐标（归一化）
                        float width_norm = outs[i].at<float>(b, 2, d);   // 宽度（归一化）
                        float height_norm = outs[i].at<float>(b, 3, d);  // 高度（归一化）
                        float object_conf = outs[i].at<float>(b, 4, d);  // 对象置信度
                        
                        // 应用置信度阈值 - 第一层过滤
                        if (object_conf > confThreshold)
                        {
                            // 找到最高置信度的类别
                            int classId = -1;
                            float maxClassConf = 0.0f;
                            
                            // 从第5个元素开始是类别置信度
                            int numClasses = max(1, numAttributes - 5); // 至少有1个类别
                            for (int c = 0; c < numClasses; ++c)
                            {
                                float classConf = outs[i].at<float>(b, 5 + c, d);
                                if (classConf > maxClassConf)
                                {
                                    maxClassConf = classConf;
                                    classId = c;
                                }
                            }
                            
                            // 计算最终置信度 = 对象置信度 * 类别置信度
                            float finalConfidence = object_conf * maxClassConf;
                            
                            // 应用最终置信度阈值 - 第二层过滤
                            if (finalConfidence > confThreshold)
                            {
                                // 坐标反归一化和转换 (与Python实现一致)
                                // 从中心点+宽高格式转换为左上角+宽高格式
                                float x_min = x_center - width_norm / 2;
                                float y_min = y_center - height_norm / 2;
                                
                                // 转换为像素坐标并缩放
                                int left = static_cast<int>(x_min * frame.cols);
                                int top = static_cast<int>(y_min * frame.rows);
                                int width = static_cast<int>(width_norm * frame.cols);
                                int height = static_cast<int>(height_norm * frame.rows);
                                
                                qDebug() << "  Detection " << d << ":";
                                qDebug() << "    Raw normalized values: x_center=" << x_center << " y_center=" << y_center 
                                         << " width=" << width_norm << " height=" << height_norm;
                                qDebug() << "    Calculated box: left=" << left << " top=" << top 
                                         << " width=" << width << " height=" << height;
                                qDebug() << "    Class ID=" << classId << " Confidence=" << finalConfidence;
                                
                                classIds.push_back(classId);
                                confidences.push_back(finalConfidence);
                                boxes.push_back(Rect(left, top, width, height));
                            }
                        }
                    }
                }
            }
            // 处理格式2: [1, 25200, 84] (numAttributes > numDetections)
            else
            {
                qDebug() << "Processing YOLOv8 format 2: [batch, num_detections, attributes]";
                for (int b = 0; b < batchSize; ++b)
                {
                    for (int d = 0; d < numDetections; ++d)
                    {
                        // 获取当前检测的所有属性
                        float x_center = outs[i].at<float>(b, d, 0);  // 中心点x坐标（归一化）
                        float y_center = outs[i].at<float>(b, d, 1);  // 中心点y坐标（归一化）
                        float width_norm = outs[i].at<float>(b, d, 2);   // 宽度（归一化）
                        float height_norm = outs[i].at<float>(b, d, 3);  // 高度（归一化）
                        float object_conf = outs[i].at<float>(b, d, 4);  // 对象置信度
                        
                        // 应用置信度阈值 - 第一层过滤
                        if (object_conf > confThreshold)
                        {
                            // 找到最高置信度的类别
                            int classId = -1;
                            float maxClassConf = 0.0f;
                            
                            // 从第5个元素开始是类别置信度
                            int numClasses = max(1, numAttributes - 5); // 至少有1个类别
                            for (int c = 0; c < numClasses; ++c)
                            {
                                float classConf = outs[i].at<float>(b, d, 5 + c);
                                if (classConf > maxClassConf)
                                {
                                    maxClassConf = classConf;
                                    classId = c;
                                }
                            }
                            
                            // 计算最终置信度 = 对象置信度 * 类别置信度
                            float finalConfidence = object_conf * maxClassConf;
                            
                            // 应用最终置信度阈值 - 第二层过滤
                            if (finalConfidence > confThreshold)
                            {
                                // 坐标反归一化和转换 (与Python实现一致)
                                // 从中心点+宽高格式转换为左上角+宽高格式
                                float x_min = x_center - width_norm / 2;
                                float y_min = y_center - height_norm / 2;
                                
                                // 转换为像素坐标并缩放
                                int left = static_cast<int>(x_min * frame.cols);
                                int top = static_cast<int>(y_min * frame.rows);
                                int width = static_cast<int>(width_norm * frame.cols);
                                int height = static_cast<int>(height_norm * frame.rows);
                                
                                qDebug() << "  Detection " << d << ":";
                                qDebug() << "    Raw normalized values: x_center=" << x_center << " y_center=" << y_center 
                                         << " width=" << width_norm << " height=" << height_norm;
                                qDebug() << "    Calculated box: left=" << left << " top=" << top 
                                         << " width=" << width << " height=" << height;
                                qDebug() << "    Class ID=" << classId << " Confidence=" << finalConfidence;
                                
                                classIds.push_back(classId);
                                confidences.push_back(finalConfidence);
                                boxes.push_back(Rect(left, top, width, height));
                            }
                        }
                    }
                }
            }
            
            qDebug() << "Total detections after filtering:" << boxes.size();
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
                
                // 应用置信度阈值 - 第一层过滤
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
                    
                    // 计算最终置信度 = 对象置信度 * 类别置信度
                    float finalConfidence = confidence * maxClassConf;
                    
                    // 应用最终置信度阈值 - 第二层过滤
                    if (finalConfidence > confThreshold)
                    {
                        // 坐标反归一化和转换 (与Python实现一致)
                        float x_min = x - w / 2;
                        float y_min = y - h / 2;
                        
                        // 转换为像素坐标并缩放
                        int left = static_cast<int>(x_min * frame.cols);
                        int top = static_cast<int>(y_min * frame.rows);
                        int width = static_cast<int>(w * frame.cols);
                        int height = static_cast<int>(h * frame.rows);
                        
                        // 确保边界框在图像范围内
                        left = max(0, left);
                        top = max(0, top);
                        width = min(frame.cols - left, width);
                        height = min(frame.rows - top, height);
                        
                        classIds.push_back(classId);
                        confidences.push_back(finalConfidence);
                        boxes.push_back(Rect(left, top, width, height));
                    }
                }
            }
        }
    }
    
    // 应用非极大值抑制 (NMS) - 与Python实现保持一致
    vector<int> indices;
    if (!boxes.empty())
    {
        qDebug() << "Applying NMS with threshold" << nmsThreshold;
        // NMSBoxes参数说明：boxes, confidences, score_threshold, nms_threshold, indices
        dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
        
        qDebug() << "NMS selected" << indices.size() << "detections";  
        
        // 构建最终结果
        for (size_t i = 0; i < indices.size(); ++i)
        {
            int idx = indices[i];
            DetectionResult result;
            
            // 验证类别ID，确保在合理范围内
            if (classIds[idx] >= 0 && classIds[idx] < 1000) // 限制最大类别ID为1000
            {
                result.classId = classIds[idx];
            } else {
                qDebug() << "Warning: Invalid class ID" << classIds[idx] << "detected, setting to 0";
                result.classId = 0;
            }
            
            // 验证置信度，确保在0-1范围内
            if (confidences[idx] >= 0.0f && confidences[idx] <= 1.0f)
            {
                result.confidence = confidences[idx];
            } else {
                // 如果置信度异常，将其归一化到合理范围
                result.confidence = max(0.0f, min(1.0f, confidences[idx]));
                qDebug() << "Warning: Abnormal confidence value" << confidences[idx] << "detected, normalized to" << result.confidence;
            }
            
            // 获取边界框
            Rect bbox = boxes[idx];
            
            // 增强的边界框有效性检查和修复 - 与Python实现完全一致
            int frameWidth = frame.cols;
            int frameHeight = frame.rows;
            int min_size = 2; // 增加最小尺寸要求，太小的边界框没有实际意义
            
            // 计算边界框的四个角坐标
            int x1 = bbox.x;
            int y1 = bbox.y;
            int x2 = bbox.x + bbox.width;
            int y2 = bbox.y + bbox.height;
            
            // 1. 首先检查边界框是否有基本有效性
            if (bbox.width <= 0 || bbox.height <= 0 ||  // 无效尺寸
                x1 >= frameWidth || y1 >= frameHeight || // 完全在图像外
                x2 <= 0 || y2 <= 0) { // 完全在图像左侧/上方
                qDebug() << "Invalid detection skipped: Box dimensions or position invalid";
                qDebug() << "  Box:" << bbox.x << "," << bbox.y 
                         << "," << bbox.width << "," << bbox.height;
                continue;
            }
            
            // 2. 修复边界框使其完全在图像范围内 - 采用Python风格的边界裁剪
            x1 = max(0, x1);
            y1 = max(0, y1);
            x2 = min(frameWidth, x2);
            y2 = min(frameHeight, y2);
            
            // 3. 重新计算边界框参数
            bbox.x = x1;
            bbox.y = y1;
            bbox.width = x2 - x1;
            bbox.height = y2 - y1;
            
            // 4. 最终验证边界框是否有效且具有最小尺寸
            if (bbox.width < min_size || bbox.height < min_size || 
                bbox.x >= frameWidth || bbox.y >= frameHeight ||
                bbox.x + bbox.width <= 0 || bbox.y + bbox.height <= 0) {
                qDebug() << "Invalid detection skipped after adjustment: Box too small or still out of bounds";
                qDebug() << "  Box:" << bbox.x << "," << bbox.y 
                         << "," << bbox.width << "," << bbox.height;
                qDebug() << "  Frame size:" << frameWidth << "x" << frameHeight;
                continue;
            }
            
            result.boundingBox = bbox;
            
            // 设置类别名称
            if (result.classId >= 0 && result.classId < static_cast<int>(classLabels_.size()))
            {
                result.className = classLabels_[result.classId];
            }
            else
            {
                result.className = "Class_" + to_string(result.classId);
            }
            
            qDebug() << "Added detection result" << i << ":";
            qDebug() << "  Class:" << QString::fromStdString(result.className) << "(ID:" << result.classId << ")";
            qDebug() << "  Confidence:" << result.confidence;
            qDebug() << "  Bounding box:" << result.boundingBox.x << "," << result.boundingBox.y 
                     << "," << result.boundingBox.width << "," << result.boundingBox.height;
            
            results.push_back(result);
        }
    }
    
    qDebug() << "Final detection results count:" << results.size();
    return results;
}

void DLProcessor::DrawDetectionResults(Mat& frame, const vector<DetectionResult>& results)
{
    // 添加调试输出
    qDebug() << "DrawDetectionResults called with" << results.size() << "results, target frame size:" << frame.cols << "x" << frame.rows;
    
    // 绘制每个检测结果 - 与Python实现保持一致
    for (size_t i = 0; i < results.size(); ++i)
    {
        const DetectionResult& result = results[i];
        
        // 添加调试输出，显示当前检测结果
        qDebug() << "  Drawing result" << i << ": class=" << QString::fromStdString(result.className)
                 << " confidence=" << result.confidence
                 << " bbox=" << result.boundingBox.x << "," << result.boundingBox.y
                 << "," << result.boundingBox.width << "," << result.boundingBox.height;
        
        // 使用与Python相同的颜色生成逻辑
        // 基于类别ID生成固定的颜色，而不是基于索引
        Scalar color = Scalar(255, 0, 0); // 默认红色
        if (result.classId >= 0)
        {
            // 使用哈希方式生成颜色，与Python实现一致
            int colorIndex = result.classId % 80; // 假设最多80个类别（COCO数据集标准）
            int r = (37 * colorIndex) % 255;
            int g = (179 * colorIndex) % 255;
            int b = (119 * colorIndex) % 255;
            color = Scalar(b, g, r); // OpenCV使用BGR格式
        }
        
        // 严格验证边界框有效性 - 与Python实现一致
        Rect bbox = result.boundingBox;
        if (bbox.width <= 0 || bbox.height <= 0 || 
            bbox.x < 0 || bbox.y < 0 || 
            bbox.x + bbox.width > frame.cols || 
            bbox.y + bbox.height > frame.rows)
        {
            qDebug() << "  Warning: Skipping invalid bounding box for result" << i;
            qDebug() << "    Box:" << bbox.x << "," << bbox.y 
                     << "," << bbox.width << "," << bbox.height;
            qDebug() << "    Frame size:" << frame.cols << "x" << frame.rows;
            continue;
        }
        
        // 绘制边界框 - 使用抗锯齿线条，与Python一致
        rectangle(frame, bbox, color, 2, LINE_AA);
        qDebug() << "  Bounding box drawn";
        
        // 创建标签文本 - 与Python格式一致："类别 置信度"
        string labelText;
        if (!result.className.empty())
        {
            labelText = result.className;
        }
        else if (result.classId >= 0)
        {
            labelText = "Class " + to_string(result.classId);
        }
        else
        {
            labelText = "Unknown";
        }
        
        // 格式化置信度为两位小数 - 与Python实现一致
        char confStr[10];
        sprintf(confStr, "%.2f", max(0.0f, min(1.0f, result.confidence)));
        labelText += " " + string(confStr);
        
        // 计算标签大小和位置 - 与Python实现一致
        int fontFace = FONT_HERSHEY_SIMPLEX;
        double fontScale = 0.5;
        int thickness = 1;
        int baseline = 0;
        Size textSize = getTextSize(labelText, fontFace, fontScale, thickness, &baseline);
        
        // 标签位置和背景 - 确保在图像范围内
        Point labelOrg(bbox.x, max(0, bbox.y - 5)); // 标签在边界框上方
        Rect labelBg(bbox.x, max(0, bbox.y - textSize.height - 10), 
                    textSize.width + 10, textSize.height + 10);
        
        // 额外检查，确保标签完全在图像范围内
        if (labelBg.x < 0) labelBg.x = 0;
        if (labelBg.y < 0) labelBg.y = 0;
        if (labelBg.x + labelBg.width > frame.cols)
            labelBg.width = frame.cols - labelBg.x;
        if (labelBg.y + labelBg.height > frame.rows)
            labelBg.height = frame.rows - labelBg.y;
        
        // 绘制标签背景
        rectangle(frame, labelBg, color, FILLED);
        qDebug() << "  Label background drawn";
        
        // 绘制标签文本 - 使用抗锯齿
        putText(frame, labelText, 
                Point(labelBg.x + 5, labelBg.y + textSize.height + 5 - baseline), 
                fontFace, fontScale, Scalar(0, 0, 0), thickness, LINE_AA);
        qDebug() << "  Label text drawn";
        
        // 如果有分割掩码，绘制掩码
        if (!result.mask.empty())
        {
            qDebug() << "  Drawing mask";
            // 创建彩色掩码
            Mat coloredMask(result.mask.size(), CV_8UC3, color);
            
            // 确保ROI在图像范围内
            Rect safeRoi = bbox & Rect(0, 0, frame.cols, frame.rows);
            if (safeRoi.width > 0 && safeRoi.height > 0)
            {
                // 调整掩码大小以匹配边界框
                Mat resizedMask;
                resize(coloredMask, resizedMask, bbox.size(), 0, 0, INTER_LINEAR);
                
                // 将掩码应用到原始图像的相应区域
                Mat roi = frame(safeRoi);
                Mat maskROI;
                resizedMask.copyTo(maskROI, result.mask);
                
                // 添加半透明效果 - 与Python一致的透明度0.4
                addWeighted(maskROI, 0.4, roi, 0.6, 0, roi);
                qDebug() << "  Mask applied successfully";
            }
        }
        qDebug() << "  Result" << i << "drawing completed";
    }
    
    qDebug() << "Total results drawn:" << results.size();
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
