/**
 * @file YOLOSegmentProcessor.cpp
 * @brief YOLO实例分割处理器实现文件
 *
 * 该文件实现了YOLOSegmentProcessor类的所有方法，提供YOLO实例分割模型的推理功能。
 *
 * @author VisualRobot Team
 * @date 2026-01-04
 * @version 1.0
 */

#include "YOLOSegmentProcessor.h"
#include <QDebug>
#include <QFile>
#include <algorithm>
#include <random>

using namespace cv;
using namespace cv::dnn;
using namespace std;

/**
 * @brief 构造函数
 */
YOLOSegmentProcessor::YOLOSegmentProcessor(QObject *parent)
    : QObject(parent)
    , modelLoaded_(false)
    , confThreshold_(0.5f)
    , nmsThreshold_(0.45f)
    , maskThreshold_(0.5f)
    , inputWidth_(640)
    , inputHeight_(640)
{
    // 初始化随机颜色种子
    srand(42);
}

/**
 * @brief 析构函数
 */
YOLOSegmentProcessor::~YOLOSegmentProcessor()
{
}

/**
 * @brief 初始化模型
 */
bool YOLOSegmentProcessor::InitModel(const std::string& modelPath, bool useGPU)
{
    try
    {
        qDebug() << "正在加载模型:" << QString::fromStdString(modelPath);

        // 检查文件是否存在
        QFile modelFile(QString::fromStdString(modelPath));
        if (!modelFile.exists())
        {
            QString errorMsg = QString("模型文件不存在: %1").arg(QString::fromStdString(modelPath));
            emit errorOccurred(errorMsg);
            qDebug() << errorMsg;
            return false;
        }

        // 加载ONNX模型
        net_ = readNetFromONNX(modelPath);

        if (net_.empty())
        {
            emit errorOccurred("无法加载模型文件，网络为空");
            return false;
        }

        // 设置推理后端
        if (useGPU)
        {
            try
            {
                net_.setPreferableBackend(DNN_BACKEND_CUDA);
                net_.setPreferableTarget(DNN_TARGET_CUDA);
                qDebug() << "使用GPU推理";
            }
            catch (const cv::Exception& e)
            {
                qDebug() << "GPU不可用，切换到CPU:" << e.what();
                net_.setPreferableBackend(DNN_BACKEND_OPENCV);
                net_.setPreferableTarget(DNN_TARGET_CPU);
            }
        }
        else
        {
            net_.setPreferableBackend(DNN_BACKEND_OPENCV);
            net_.setPreferableTarget(DNN_TARGET_CPU);
            qDebug() << "使用CPU推理";
        }

        // 获取输出层信息
        vector<String> outLayerNames = net_.getUnconnectedOutLayersNames();
        qDebug() << "模型输出层数量:" << outLayerNames.size();
        for (size_t i = 0; i < outLayerNames.size(); ++i)
        {
            qDebug() << "  输出层[" << i << "]:" << QString::fromStdString(outLayerNames[i]);
        }

        modelLoaded_ = true;
        qDebug() << "YOLO模型加载成功（支持普通检测和分割模型）";
        return true;
    }
    catch (const cv::Exception& e)
    {
        QString errorMsg = QString("模型加载失败: %1").arg(e.what());
        emit errorOccurred(errorMsg);
        qDebug() << errorMsg;
        return false;
    }
    catch (const std::exception& e)
    {
        QString errorMsg = QString("模型加载异常: %1").arg(e.what());
        emit errorOccurred(errorMsg);
        qDebug() << errorMsg;
        return false;
    }
}

/**
 * @brief 检查模型是否已加载
 */
bool YOLOSegmentProcessor::IsModelLoaded() const
{
    return modelLoaded_;
}

/**
 * @brief 设置类别标签
 */
void YOLOSegmentProcessor::SetClassLabels(const std::vector<std::string>& labels)
{
    classLabels_ = labels;

    // 预生成类别颜色
    classColors_.clear();
    for (size_t i = 0; i < labels.size(); ++i)
    {
        classColors_.push_back(GetClassColor(i));
    }
}

/**
 * @brief 设置检测阈值
 */
void YOLOSegmentProcessor::SetThresholds(float confThreshold, float nmsThreshold, float maskThreshold)
{
    confThreshold_ = confThreshold;
    nmsThreshold_ = nmsThreshold;
    maskThreshold_ = maskThreshold;
}

/**
 * @brief 预处理图像
 */
Mat YOLOSegmentProcessor::PreprocessImage(const Mat& image)
{
    Mat blob;
    // 转换为blob格式：归一化到0-1，BGR转RGB，NCHW格式
    blobFromImage(image, blob, 1.0 / 255.0, Size(inputWidth_, inputHeight_), Scalar(), true, false);
    return blob;
}

/**
 * @brief 执行分割检测
 */
bool YOLOSegmentProcessor::DetectAndSegment(const Mat& image, std::vector<SegmentationResult>& results)
{
    if (!modelLoaded_)
    {
        emit errorOccurred("模型未加载");
        return false;
    }

    if (image.empty())
    {
        emit errorOccurred("输入图像为空");
        return false;
    }

    try
    {
        // 预处理
        Mat blob = PreprocessImage(image);
        net_.setInput(blob);

        // 前向推理
        vector<Mat> outputs;
        net_.forward(outputs, net_.getUnconnectedOutLayersNames());

        // 后处理
        PostProcess(outputs, image.size(), results);

        return true;
    }
    catch (const cv::Exception& e)
    {
        QString errorMsg = QString("检测失败: %1").arg(e.what());
        emit errorOccurred(errorMsg);
        qDebug() << errorMsg;
        return false;
    }
}

/**
 * @brief 后处理网络输出
 */
void YOLOSegmentProcessor::PostProcess(const std::vector<Mat>& outputs, const Size& imageSize, std::vector<SegmentationResult>& results)
{
    results.clear();

    if (outputs.empty())
    {
        qDebug() << "警告: 模型输出为空";
        return;
    }

    // YOLO模型输出：
    // 普通检测: [1, num_proposals, 4+num_classes] 或 [1, 4+num_classes, num_proposals]
    // 分割模型: [1, num_proposals, 4+num_classes+num_masks] + [1, num_masks, mask_h, mask_w]

    Mat detectOutput = outputs[0]; // 检测输出
    Mat protoOutput;               // 掩码原型（如果有）

    if (outputs.size() > 1)
    {
        protoOutput = outputs[1];
        qDebug() << "检测到分割模型输出";
    }
    else
    {
        qDebug() << "检测到普通检测模型输出，将生成矩形掩码";
    }

    // 解析检测输出的维度
    vector<int> detectShape;
    for (int i = 0; i < detectOutput.dims; ++i)
    {
        detectShape.push_back(detectOutput.size[i]);
    }

    qDebug() << "检测输出维度:";
    for (size_t i = 0; i < detectShape.size(); ++i)
    {
        qDebug() << "  dim[" << i << "] =" << detectShape[i];
    }

    // 通常格式为 [1, num_proposals, 4+num_classes+num_masks]
    // 或者转置后的 [1, 4+num_classes+num_masks, num_proposals]
    int numProposals = 0;
    int numChannels = 0;
    Mat detectData;

    if (detectShape.size() == 3)
    {
        if (detectShape[2] > detectShape[1])
        {
            // 格式: [1, num_proposals, channels]
            numProposals = detectShape[1];
            numChannels = detectShape[2];
            detectData = detectOutput.reshape(1, numProposals);
            qDebug() << "格式: [batch, proposals, channels]";
        }
        else
        {
            // 格式: [1, channels, num_proposals] - 需要转置
            numChannels = detectShape[1];
            numProposals = detectShape[2];
            detectData = detectOutput.reshape(1, numChannels);
            transpose(detectData, detectData);
            qDebug() << "格式: [batch, channels, proposals] - 已转置";
        }
    }
    else if (detectShape.size() == 2)
    {
        numProposals = detectShape[0];
        numChannels = detectShape[1];
        detectData = detectOutput.reshape(1, numProposals);
        qDebug() << "格式: [proposals, channels]";
    }
    else
    {
        qDebug() << "错误: 不支持的输出维度";
        return;
    }

    qDebug() << "提案数量:" << numProposals << ", 通道数:" << numChannels;

    // 计算类别数量和掩码系数数量
    int numClasses = classLabels_.empty() ? 80 : classLabels_.size();
    int numMaskCoeffs = numChannels - 4 - numClasses;

    if (numMaskCoeffs < 0)
    {
        numMaskCoeffs = 0; // 如果没有掩码输出，当作普通检测
        qDebug() << "普通检测模型，类别数:" << numClasses;
    }
    else
    {
        qDebug() << "分割模型，类别数:" << numClasses << ", 掩码系数数:" << numMaskCoeffs;
    }

    // 存储候选框
    vector<Rect> boxes;
    vector<float> confidences;
    vector<int> classIds;
    vector<Mat> maskCoeffs;

    // 计算缩放比例
    float xScale = static_cast<float>(imageSize.width) / inputWidth_;
    float yScale = static_cast<float>(imageSize.height) / inputHeight_;

    // 遍历所有提案
    for (int i = 0; i < numProposals; ++i)
    {
        float* data = detectData.ptr<float>(i);

        // 解析边界框 (cx, cy, w, h)
        float cx = data[0];
        float cy = data[1];
        float w = data[2];
        float h = data[3];

        // 查找最大置信度和对应类别
        float maxConf = 0.0f;
        int maxClassId = 0;

        for (int c = 0; c < numClasses; ++c)
        {
            float conf = data[4 + c];
            if (conf > maxConf)
            {
                maxConf = conf;
                maxClassId = c;
            }
        }

        // 过滤低置信度
        if (maxConf < confThreshold_)
        {
            continue;
        }

        // 转换为左上角坐标
        float x1 = (cx - w / 2.0f) * xScale;
        float y1 = (cy - h / 2.0f) * yScale;
        float x2 = (cx + w / 2.0f) * xScale;
        float y2 = (cy + h / 2.0f) * yScale;

        // 边界检查
        x1 = max(0.0f, min(x1, static_cast<float>(imageSize.width - 1)));
        y1 = max(0.0f, min(y1, static_cast<float>(imageSize.height - 1)));
        x2 = max(0.0f, min(x2, static_cast<float>(imageSize.width - 1)));
        y2 = max(0.0f, min(y2, static_cast<float>(imageSize.height - 1)));

        Rect box(static_cast<int>(x1), static_cast<int>(y1),
                 static_cast<int>(x2 - x1), static_cast<int>(y2 - y1));

        boxes.push_back(box);
        confidences.push_back(maxConf);
        classIds.push_back(maxClassId);

        // 提取掩码系数
        if (numMaskCoeffs > 0)
        {
            Mat coeffs(1, numMaskCoeffs, CV_32F);
            for (int m = 0; m < numMaskCoeffs; ++m)
            {
                coeffs.at<float>(0, m) = data[4 + numClasses + m];
            }
            maskCoeffs.push_back(coeffs);
        }
    }

    // NMS非极大值抑制
    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold_, nmsThreshold_, indices);

    // 生成最终结果
    for (int idx : indices)
    {
        SegmentationResult result;
        result.boundingBox = boxes[idx];
        result.classId = classIds[idx];
        result.confidence = confidences[idx];

        if (result.classId < static_cast<int>(classLabels_.size()))
        {
            result.className = classLabels_[result.classId];
        }
        else
        {
            result.className = "class_" + to_string(result.classId);
        }

        // 生成掩码
        if (!protoOutput.empty() && idx < static_cast<int>(maskCoeffs.size()))
        {
            // 掩码原型维度: [1, num_masks, mask_h, mask_w] 或 [num_masks, mask_h, mask_w]
            vector<int> protoShape;
            for (int i = 0; i < protoOutput.dims; ++i)
            {
                protoShape.push_back(protoOutput.size[i]);
            }

            int numMasks = 0;
            int maskH = 0;
            int maskW = 0;
            Mat protoData;

            if (protoShape.size() == 4)
            {
                numMasks = protoShape[1];
                maskH = protoShape[2];
                maskW = protoShape[3];
                protoData = protoOutput.reshape(1, {numMasks, maskH * maskW});
            }
            else if (protoShape.size() == 3)
            {
                numMasks = protoShape[0];
                maskH = protoShape[1];
                maskW = protoShape[2];
                protoData = protoOutput.reshape(1, {numMasks, maskH * maskW});
            }

            if (!protoData.empty() && !maskCoeffs[idx].empty())
            {
                // 计算掩码: mask = sigmoid(coeffs @ protos)
                Mat maskProto = maskCoeffs[idx] * protoData;
                maskProto = maskProto.reshape(1, {maskH, maskW});

                // Sigmoid激活
                Mat maskSigmoid;
                exp(-maskProto, maskSigmoid);
                maskSigmoid = 1.0 / (1.0 + maskSigmoid);

                // 调整到原图尺寸
                Mat maskResized;
                resize(maskSigmoid, maskResized, imageSize, 0, 0, INTER_LINEAR);

                // 裁剪到边界框区域
                Mat maskCropped = Mat::zeros(imageSize, CV_32F);
                Rect safeBox = result.boundingBox & Rect(0, 0, imageSize.width, imageSize.height);

                if (safeBox.width > 0 && safeBox.height > 0)
                {
                    Mat roi = maskResized(safeBox);
                    roi.copyTo(maskCropped(safeBox));
                }

                // 二值化
                Mat maskBinary;
                threshold(maskCropped, maskBinary, maskThreshold_, 255, THRESH_BINARY);
                maskBinary.convertTo(result.mask, CV_8U);
            }
        }

        // 如果没有掩码，创建一个矩形掩码
        if (result.mask.empty())
        {
            result.mask = Mat::zeros(imageSize, CV_8U);
            rectangle(result.mask, result.boundingBox, Scalar(255), FILLED);
        }

        results.push_back(result);
    }
}

/**
 * @brief 生成随机颜色
 */
Scalar YOLOSegmentProcessor::GetClassColor(int classId)
{
    // 使用固定种子生成一致的颜色
    mt19937 rng(classId * 12345);
    uniform_int_distribution<int> dist(0, 255);

    return Scalar(dist(rng), dist(rng), dist(rng));
}

/**
 * @brief 绘制分割结果
 */
void YOLOSegmentProcessor::DrawSegmentationResults(Mat& image, const std::vector<SegmentationResult>& results, float alpha)
{
    // 创建掩码叠加层
    Mat overlay = image.clone();

    for (const auto& result : results)
    {
        // 获取颜色
        Scalar color = GetClassColor(result.classId);

        // 绘制掩码
        if (!result.mask.empty())
        {
            Mat colorMask = Mat::zeros(image.size(), image.type());
            colorMask.setTo(color, result.mask);
            addWeighted(overlay, 1.0, colorMask, alpha, 0, overlay);
        }

        // 绘制边界框
        rectangle(overlay, result.boundingBox, color, 2);

        // 绘制标签
        string label = result.className + " " + cv::format("%.2f", result.confidence);
        int baseLine;
        Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseLine);

        int top = max(result.boundingBox.y - 5, labelSize.height);
        rectangle(overlay, Point(result.boundingBox.x, top - labelSize.height - 5),
                  Point(result.boundingBox.x + labelSize.width, top), color, FILLED);
        putText(overlay, label, Point(result.boundingBox.x, top - 3),
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);
    }

    // 混合原图和叠加层
    addWeighted(image, 1.0 - alpha, overlay, alpha, 0, image);
}

/**
 * @brief 生成纯掩码图像
 */
Mat YOLOSegmentProcessor::GenerateMaskImage(const Size& imageSize, const std::vector<SegmentationResult>& results)
{
    Mat maskImage = Mat::zeros(imageSize, CV_8UC3);

    for (const auto& result : results)
    {
        if (!result.mask.empty())
        {
            Scalar color = GetClassColor(result.classId);
            maskImage.setTo(color, result.mask);
        }
    }

    return maskImage;
}

