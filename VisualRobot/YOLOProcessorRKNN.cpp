#include "YOLOProcessorRKNN.h"
#include <QDebug>
#include <QFile>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>

using namespace std;
using namespace cv;

/**
 * @brief 构造函数
 */
YOLOProcessorRKNN::YOLOProcessorRKNN(QObject* parent)
    : QObject(parent)
    , ctx_(0)
    , input_attrs_(nullptr)
    , output_attrs_(nullptr)
    , input_mems_(nullptr)
    , output_mems_(nullptr)
    , input_size_(640, 640)
    , conf_threshold_(0.25f)
    , nms_threshold_(0.45f)
    , letterbox_r_(1.0)
    , letterbox_dw_(0.0)
    , letterbox_dh_(0.0)
    , enable_debug_(false)
{
    // 初始化延时统计
    timing_stats_.preprocessTime = 0.0;
    timing_stats_.inferenceTime = 0.0;
    timing_stats_.postprocessTime = 0.0;
    timing_stats_.totalTime = 0.0;
    timing_stats_.fps = 0;
}

/**
 * @brief 析构函数
 */
YOLOProcessorRKNN::~YOLOProcessorRKNN()
{
    Cleanup();
}

/**
 * @brief 初始化RKNN模型
 */
bool YOLOProcessorRKNN::InitModel(const string& modelPath, const string& labelsPath)
{
    model_path_ = modelPath;

    // 1. 加载RKNN模型
    // 修复：需要将const string转换为非const的char*
    char* modelPathChar = const_cast<char*>(modelPath.c_str());
    int ret = rknn_init(&ctx_, modelPathChar, 0, 0, nullptr);

    if (ret < 0) {
        QString error = QString("rknn_init失败! ret=%1").arg(ret);
        qDebug() << error;
        emit errorOccurred(error);
        return false;
    }

    // 2. 获取SDK和驱动版本
    rknn_sdk_version sdk_ver;
    ret = rknn_query(ctx_, RKNN_QUERY_SDK_VERSION, &sdk_ver, sizeof(sdk_ver));
    if (ret != RKNN_SUCC) {
        QString error = QString("rknn_query SDK版本失败! ret=%1").arg(ret);
        qDebug() << error;
        Cleanup();
        emit errorOccurred(error);
        return false;
    }

    sdk_version_ = string(sdk_ver.api_version);
    driver_version_ = string(sdk_ver.drv_version);

    qDebug() << "RKNN SDK版本:" << sdk_version_.c_str();
    qDebug() << "驱动版本:" << driver_version_.c_str();

    // 3. 获取输入输出数量
    ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));
    if (ret != RKNN_SUCC) {
        QString error = QString("rknn_query输入输出数量失败! ret=%1").arg(ret);
        qDebug() << error;
        Cleanup();
        emit errorOccurred(error);
        return false;
    }

    qDebug() << "模型输入数量:" << io_num_.n_input;
    qDebug() << "模型输出数量:" << io_num_.n_output;

    // 4. 获取输入张量属性
    input_attrs_ = new rknn_tensor_attr[io_num_.n_input];
    for (uint32_t i = 0; i < io_num_.n_input; i++) {
        input_attrs_[i].index = i;
        ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &input_attrs_[i], sizeof(rknn_tensor_attr));
        if (ret < 0) {
            QString error = QString("rknn_query输入属性失败! ret=%1").arg(ret);
            qDebug() << error;
            Cleanup();
            emit errorOccurred(error);
            return false;
        }

        // 根据输入格式确定输入尺寸
        switch (input_attrs_[i].fmt) {
            case RKNN_TENSOR_NHWC:
                input_size_.width = input_attrs_[i].dims[2];
                input_size_.height = input_attrs_[i].dims[1];
                break;
            case RKNN_TENSOR_NCHW:
                input_size_.width = input_attrs_[i].dims[3];
                input_size_.height = input_attrs_[i].dims[2];
                break;
            default:
                break;
        }

        qDebug() << "输入张量" << i << ":";
        qDebug() << "  名称:" << input_attrs_[i].name;
        qDebug() << "  格式:" << input_attrs_[i].fmt;
        qDebug() << "  类型:" << input_attrs_[i].type;
        qDebug() << "  尺寸:" << input_attrs_[i].dims[0] << input_attrs_[i].dims[1]
                 << input_attrs_[i].dims[2] << input_attrs_[i].dims[3];
    }

    // 5. 获取输出张量属性
    output_attrs_ = new rknn_tensor_attr[io_num_.n_output];
    for (uint32_t i = 0; i < io_num_.n_output; i++) {
        output_attrs_[i].index = i;
        ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &output_attrs_[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            QString error = QString("rknn_query输出属性失败! ret=%1").arg(ret);
            qDebug() << error;
            Cleanup();
            emit errorOccurred(error);
            return false;
        }

        qDebug() << "输出张量" << i << ":";
        qDebug() << "  名称:" << output_attrs_[i].name;
        qDebug() << "  格式:" << output_attrs_[i].fmt;
        qDebug() << "  类型:" << output_attrs_[i].type;
        qDebug() << "  尺寸:" << output_attrs_[i].dims[0] << output_attrs_[i].dims[1]
                 << output_attrs_[i].dims[2] << output_attrs_[i].dims[3];
    }

    // 6. 创建输入输出内存
    input_mems_ = new rknn_tensor_mem*[io_num_.n_input];
    output_mems_ = new rknn_tensor_mem*[io_num_.n_output];

    // 创建输入内存
    for (uint32_t i = 0; i < io_num_.n_input; i++) {
        input_mems_[i] = rknn_create_mem(ctx_, input_attrs_[i].size_with_stride);
        if (!input_mems_[i]) {
            QString error = "创建输入内存失败!";
            qDebug() << error;
            Cleanup();
            emit errorOccurred(error);
            return false;
        }
        rknn_set_io_mem(ctx_, input_mems_[i], &input_attrs_[i]);
    }

    // 创建输出内存（浮点型，用于后处理）
    for (uint32_t i = 0; i < io_num_.n_output; i++) {
        output_attrs_[i].type = RKNN_TENSOR_FLOAT32;
        int output_size = output_attrs_[i].n_elems * sizeof(float);
        output_mems_[i] = rknn_create_mem(ctx_, output_size);
        if (!output_mems_[i]) {
            QString error = "创建输出内存失败!";
            qDebug() << error;
            Cleanup();
            emit errorOccurred(error);
            return false;
        }
        rknn_set_io_mem(ctx_, output_mems_[i], &output_attrs_[i]);
    }

    // 7. 加载标签文件
    if (!labelsPath.empty()) {
        if (!LoadLabels(labelsPath)) {
            qWarning() << "标签文件加载失败，将继续使用默认标签";
        }
    }

    qDebug() << "RKNN模型初始化成功!";
    qDebug() << "输入尺寸:" << input_size_.width << "x" << input_size_.height;

    return true;
}

/**
 * @brief 加载类别标签
 */
bool YOLOProcessorRKNN::LoadLabels(const string& labelsPath)
{
    QFile file(QString::fromStdString(labelsPath));
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qWarning() << "无法打开标签文件:" << labelsPath.c_str();
        return false;
    }

    class_labels_.clear();
    QTextStream in(&file);
    while (!in.atEnd()) {
        QString line = in.readLine().trimmed();
        if (!line.isEmpty()) {
            class_labels_.push_back(line.toStdString());
        }
    }
    file.close();

    qDebug() << "加载了" << class_labels_.size() << "个类别标签";
    return !class_labels_.empty();
}

/**
 * @brief 预处理图像
 */
bool YOLOProcessorRKNN::PreprocessImage(const Mat& frame, vector<uint8_t>& preprocessed, RKNNImageInfo& img_info)
{
    auto start_time = chrono::high_resolution_clock::now();

    // 保存原始图像信息
    img_info.width = frame.cols;
    img_info.height = frame.rows;
    img_info.channels = frame.channels();
    img_info.targetWidth = input_size_.width;
    img_info.targetHeight = input_size_.height;

    // 1. Letterbox缩放
    int orig_w = frame.cols;
    int orig_h = frame.rows;
    int target_w = input_size_.width;
    int target_h = input_size_.height;

    // 计算缩放比例
    double r = min(static_cast<double>(target_w) / orig_w,
                   static_cast<double>(target_h) / orig_h);
    img_info.letterboxR = r;

    // 计算新尺寸
    int new_w = static_cast<int>(round(orig_w * r));
    int new_h = static_cast<int>(round(orig_h * r));

    // 计算填充
    int dw = target_w - new_w;
    int dh = target_h - new_h;
    img_info.letterboxDw = dw / 2.0;
    img_info.letterboxDh = dh / 2.0;

    // 执行letterbox
    Mat resized;
    if (orig_w != new_w || orig_h != new_h) {
        resize(frame, resized, Size(new_w, new_h), 0, 0, INTER_LINEAR);
    } else {
        resized = frame.clone();
    }

    // 填充
    int top = dh / 2;
    int bottom = dh - top;
    int left = dw / 2;
    int right = dw - left;

    Mat padded;
    copyMakeBorder(resized, padded, top, bottom, left, right,
                   BORDER_CONSTANT, Scalar(114, 114, 114));

    // 2. BGR转RGB
    Mat rgb;
    cvtColor(padded, rgb, COLOR_BGR2RGB);

    // 3. 根据输入格式准备数据
    int stride = input_attrs_[0].w_stride;
    int width = input_attrs_[0].dims[2];
    int height = input_attrs_[0].dims[1];
    int channels = input_attrs_[0].dims[3];

    preprocessed.resize(input_attrs_[0].size_with_stride);

    if (width == stride) {
        // 连续内存，直接拷贝
        memcpy(preprocessed.data(), rgb.data, width * height * channels);
    } else {
        // 带stride的内存，逐行拷贝
        uint8_t* src_ptr = rgb.data;
        uint8_t* dst_ptr = preprocessed.data();
        int src_line_size = width * channels;
        int dst_line_size = stride * channels;

        for (int h = 0; h < height; h++) {
            memcpy(dst_ptr, src_ptr, src_line_size);
            src_ptr += src_line_size;
            dst_ptr += dst_line_size;
        }
    }

    // 复制到输入内存
    memcpy(input_mems_[0]->virt_addr, preprocessed.data(), preprocessed.size());

    auto end_time = chrono::high_resolution_clock::now();
    timing_stats_.preprocessTime = chrono::duration<double, milli>(end_time - start_time).count();

    return true;
}

/**
 * @brief 检测图像中的目标
 */
bool YOLOProcessorRKNN::DetectObjects(const Mat& frame, vector<DetectionResult>& results)
{
    results.clear();

    if (!ctx_) {
        emit errorOccurred("RKNN模型未初始化!");
        return false;
    }

    if (frame.empty()) {
        emit errorOccurred("输入图像为空!");
        return false;
    }

    auto total_start = chrono::high_resolution_clock::now();

    try {
        // 1. 预处理
        RKNNImageInfo img_info;
        vector<uint8_t> preprocessed_data;

        if (!PreprocessImage(frame, preprocessed_data, img_info)) {
            emit errorOccurred("图像预处理失败!");
            return false;
        }

        // 2. 推理
        auto inference_start = chrono::high_resolution_clock::now();

        int ret = rknn_run(ctx_, nullptr);
        if (ret < 0) {
            QString error = QString("rknn_run失败! ret=%1").arg(ret);
            emit errorOccurred(error);
            return false;
        }

        auto inference_end = chrono::high_resolution_clock::now();
        timing_stats_.inferenceTime = chrono::duration<double, milli>(inference_end - inference_start).count();

        // 3. 后处理
        auto postprocess_start = chrono::high_resolution_clock::now();

        // 收集输出数据
        vector<float*> outputs;
        for (uint32_t i = 0; i < io_num_.n_output; i++) {
            outputs.push_back(static_cast<float*>(output_mems_[i]->virt_addr));
        }

        // 执行后处理
        results = PostProcess(outputs, img_info);

        auto postprocess_end = chrono::high_resolution_clock::now();
        timing_stats_.postprocessTime = chrono::duration<double, milli>(postprocess_end - postprocess_start).count();

        // 4. 计算总时间
        auto total_end = chrono::high_resolution_clock::now();
        timing_stats_.totalTime = chrono::duration<double, milli>(total_end - total_start).count();
        timing_stats_.fps = (timing_stats_.totalTime > 0) ?
                            static_cast<int>(1000.0 / timing_stats_.totalTime) : 0;

        // 5. 调试输出
        if (enable_debug_) {
            DebugOutput(img_info, results);
        }

        return true;

    } catch (const exception& e) {
        QString error = QString("检测过程中发生异常: %1").arg(e.what());
        emit errorOccurred(error);
        return false;
    }
}

/**
 * @brief 后处理输出
 */
vector<DetectionResult> YOLOProcessorRKNN::PostProcess(const vector<float*>& outputs, const RKNNImageInfo& img_info)
{
    vector<DetectionResult> results;

    if (outputs.empty()) {
        return results;
    }

    // 假设只有一个输出（YOLO11n）
    const float* output_data = outputs[0];

    // 获取输出形状
    vector<int64_t> output_shape;
    for (int i = 0; i < 4; i++) {
        output_shape.push_back(output_attrs_[0].dims[i]);
    }

    // 解析YOLO输出
    vector<Rect> boxes;
    vector<float> scores;
    vector<int> class_ids;

    ParseYOLOOutput(output_data, output_shape, img_info, boxes, scores, class_ids);

    // 应用NMS
    vector<int> indices;
    ApplyNMS(boxes, scores, indices);

    // 生成检测结果
    for (int idx : indices) {
        DetectionResult result;
        result.boundingBox = boxes[idx];
        result.confidence = scores[idx];
        result.classId = class_ids[idx];

        if (result.classId >= 0 && result.classId < static_cast<int>(class_labels_.size())) {
            result.className = class_labels_[result.classId];
        } else {
            result.className = "class_" + to_string(result.classId);
        }

        results.push_back(result);
    }

    return results;
}

/**
 * @brief 解析YOLO输出
 */
void YOLOProcessorRKNN::ParseYOLOOutput(const float* output_data, const vector<int64_t>& output_shape,
                                       const RKNNImageInfo& img_info,
                                       vector<Rect>& boxes, vector<float>& scores, vector<int>& class_ids)
{
    // YOLO11n输出形状通常是[1, 84, 8400]
    // 其中84 = 4个坐标 + 80个类别（COCO数据集）

    int batch_size = output_shape[0];    // 通常为1
    int num_features = output_shape[1];  // 特征数
    int num_boxes = output_shape[2];     // 预测框数量

    // 计算类别数量
    int num_classes = num_features - 4;

    for (int i = 0; i < num_boxes; i++) {
        // 获取当前预测框的数据指针
        const float* box_data = output_data + i * num_features;

        // 提取坐标
        float x_center = box_data[0];
        float y_center = box_data[1];
        float width = box_data[2];
        float height = box_data[3];

        // 找到最大类别分数
        float max_score = 0.0f;
        int max_class_id = -1;

        for (int c = 0; c < num_classes; c++) {
            float score = box_data[4 + c];
            if (score > max_score) {
                max_score = score;
                max_class_id = c;
            }
        }

        // 应用置信度阈值
        if (max_score < conf_threshold_) {
            continue;
        }

        // 将归一化坐标转换为像素坐标
        Rect pixel_rect;
        NormalizedToPixel(x_center, y_center, width, height, img_info, pixel_rect);

        // 确保矩形有效
        if (pixel_rect.width <= 0 || pixel_rect.height <= 0) {
            continue;
        }

        // 确保矩形在图像范围内
        pixel_rect.x = max(0, pixel_rect.x);
        pixel_rect.y = max(0, pixel_rect.y);
        pixel_rect.width = min(pixel_rect.width, img_info.width - pixel_rect.x);
        pixel_rect.height = min(pixel_rect.height, img_info.height - pixel_rect.y);

        boxes.push_back(pixel_rect);
        scores.push_back(max_score);
        class_ids.push_back(max_class_id);
    }
}

/**
 * @brief 将归一化坐标转换为像素坐标
 */
void YOLOProcessorRKNN::NormalizedToPixel(float x, float y, float w, float h,
                                         const RKNNImageInfo& img_info, Rect& rect)
{
    // 模型输出是相对于640x640的坐标
    // 需要先去除letterbox填充，然后映射回原始图像

    // 1. 转换为letterbox图像上的像素坐标
    float x_center_pix = x * img_info.targetWidth;
    float y_center_pix = y * img_info.targetHeight;
    float width_pix = w * img_info.targetWidth;
    float height_pix = h * img_info.targetHeight;

    // 2. 去除letterbox填充
    float real_x_center = (x_center_pix - img_info.letterboxDw) / img_info.letterboxR;
    float real_y_center = (y_center_pix - img_info.letterboxDh) / img_info.letterboxR;
    float real_width = width_pix / img_info.letterboxR;
    float real_height = height_pix / img_info.letterboxR;

    // 3. 转换为左上角坐标
    int x1 = static_cast<int>(real_x_center - real_width / 2.0f);
    int y1 = static_cast<int>(real_y_center - real_height / 2.0f);
    int x2 = static_cast<int>(real_x_center + real_width / 2.0f);
    int y2 = static_cast<int>(real_y_center + real_height / 2.0f);

    rect.x = x1;
    rect.y = y1;
    rect.width = x2 - x1;
    rect.height = y2 - y1;
}

/**
 * @brief 应用NMS
 */
void YOLOProcessorRKNN::ApplyNMS(const vector<Rect>& boxes, const vector<float>& scores, vector<int>& indices)
{
    if (boxes.empty()) {
        return;
    }

    // 使用OpenCV的NMSBoxes
    dnn::NMSBoxes(boxes, scores, conf_threshold_, nms_threshold_, indices);
}

/**
 * @brief 处理图像并保存结果
 */
bool YOLOProcessorRKNN::ProcessImage(const string& imagePath, const string& outputPath, bool saveResult)
{
    // 读取图像
    Mat image = imread(imagePath);
    if (image.empty()) {
        QString error = QString("无法读取图像: %1").arg(imagePath.c_str());
        emit errorOccurred(error);
        return false;
    }

    // 检测目标
    vector<DetectionResult> results;
    if (!DetectObjects(image, results)) {
        return false;
    }

    // 绘制结果
    DrawDetectionResults(image, results, outputPath);

    // 发送信号
    emit detectionResults(results);
    emit processingComplete(image);

    return true;
}

/**
 * @brief 在图像上绘制检测结果
 */
void YOLOProcessorRKNN::DrawDetectionResults(Mat& frame, const vector<DetectionResult>& results, const string& outputPath)
{
    // 定义颜色
    vector<Scalar> colors = {
        Scalar(255, 0, 0),    // 蓝色
        Scalar(0, 255, 0),    // 绿色
        Scalar(0, 0, 255),    // 红色
        Scalar(255, 255, 0),  // 青色
        Scalar(255, 0, 255),  // 品红
        Scalar(0, 255, 255),  // 黄色
        Scalar(128, 0, 0),    // 深蓝
        Scalar(0, 128, 0),    // 深绿
        Scalar(0, 0, 128),    // 深红
        Scalar(128, 128, 0)   // 橄榄色
    };

    // 绘制每个检测结果
    for (const auto& result : results) {
        int color_idx = result.classId % colors.size();
        Scalar color = colors[color_idx];

        // 绘制边界框
        rectangle(frame, result.boundingBox, color, 2);

        // 准备标签文本
        string label;
        if (!result.className.empty()) {
            label = result.className + ": " + to_string(static_cast<int>(result.confidence * 100)) + "%";
        } else {
            label = "Class " + to_string(result.classId) + ": " + to_string(static_cast<int>(result.confidence * 100)) + "%";
        }

        // 计算文本大小
        int baseline = 0;
        Size text_size = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

        // 绘制文本背景
        Point text_org(result.boundingBox.x, result.boundingBox.y - 5);
        if (text_org.y < text_size.height + 5) {
            text_org.y = result.boundingBox.y + text_size.height + 5;
        }

        rectangle(frame,
                 Point(text_org.x, text_org.y - text_size.height - 5),
                 Point(text_org.x + text_size.width, text_org.y + 5),
                 color, FILLED);

        // 绘制文本
        putText(frame, label, text_org, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
    }

    // 添加统计信息
    string stats_text = "FPS: " + to_string(timing_stats_.fps) +
                       " | Objects: " + to_string(results.size());
    putText(frame, stats_text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);

    // 保存结果 - 修复：直接根据outputPath是否为空来判断
    if (!outputPath.empty()) {
        imwrite(outputPath, frame);
        qDebug() << "结果已保存到:" << outputPath.c_str();
    }
}

/**
 * @brief 获取延时统计信息
 */
RKNNTimingStats YOLOProcessorRKNN::GetTimingStats() const
{
    return timing_stats_;
}

/**
 * @brief 设置检测阈值
 */
void YOLOProcessorRKNN::SetThresholds(float conf, float nms)
{
    conf_threshold_ = conf;
    nms_threshold_ = nms;
}

/**
 * @brief 设置输入图像尺寸
 */
void YOLOProcessorRKNN::SetInputSize(int width, int height)
{
    input_size_ = Size(width, height);
}

/**
 * @brief 获取模型输入尺寸
 */
cv::Size YOLOProcessorRKNN::GetInputSize() const
{
    return input_size_;
}

/**
 * @brief 启用/禁用调试输出
 */
void YOLOProcessorRKNN::SetDebugOutput(bool enable)
{
    enable_debug_ = enable;
}

/**
 * @brief 检查模型是否已加载
 */
bool YOLOProcessorRKNN::IsModelLoaded() const
{
    return ctx_ != 0;
}

/**
 * @brief 检查标签是否已加载
 */
bool YOLOProcessorRKNN::AreLabelsLoaded() const
{
    return !class_labels_.empty();
}

/**
 * @brief 获取当前时间（微秒）
 */
int64_t YOLOProcessorRKNN::GetCurrentTimeUs()
{
    struct timeval tv;
    gettimeofday(&tv, nullptr);  // 使用nullptr代替NULL
    return tv.tv_sec * 1000000 + tv.tv_usec;
}

/**
 * @brief 打印调试信息
 */
void YOLOProcessorRKNN::DebugOutput(const RKNNImageInfo& img_info, const vector<DetectionResult>& results)
{
    qDebug() << "=== RKNN调试信息 ===";
    qDebug() << "图像信息:";
    qDebug() << "  原始尺寸:" << img_info.width << "x" << img_info.height;
    qDebug() << "  目标尺寸:" << img_info.targetWidth << "x" << img_info.targetHeight;
    qDebug() << "  Letterbox R:" << img_info.letterboxR;
    qDebug() << "  Letterbox Dw:" << img_info.letterboxDw;
    qDebug() << "  Letterbox Dh:" << img_info.letterboxDh;

    qDebug() << "模型信息:";
    qDebug() << "  输入数量:" << io_num_.n_input;
    qDebug() << "  输出数量:" << io_num_.n_output;

    qDebug() << "检测结果:";
    qDebug() << "  检测到" << results.size() << "个目标";

    for (size_t i = 0; i < results.size(); i++) {
        const auto& result = results[i];
        qDebug() << "  目标" << i + 1 << ":";
        qDebug() << "    类别:" << result.className.c_str() << "(ID:" << result.classId << ")";
        qDebug() << "    置信度:" << result.confidence;
        qDebug() << "    位置:" << result.boundingBox.x << "," << result.boundingBox.y
                 << " - " << result.boundingBox.width << "x" << result.boundingBox.height;
    }

    qDebug() << "性能统计:";
    qDebug() << "  预处理时间:" << timing_stats_.preprocessTime << "ms";
    qDebug() << "  推理时间:" << timing_stats_.inferenceTime << "ms";
    qDebug() << "  后处理时间:" << timing_stats_.postprocessTime << "ms";
    qDebug() << "  总时间:" << timing_stats_.totalTime << "ms";
    qDebug() << "  FPS:" << timing_stats_.fps;
    qDebug() << "=========================";
}

/**
 * @brief 释放资源
 */
void YOLOProcessorRKNN::Cleanup()
{
    // 释放输出内存
    if (output_mems_) {
        for (uint32_t i = 0; i < io_num_.n_output; i++) {
            if (output_mems_[i]) {
                rknn_destroy_mem(ctx_, output_mems_[i]);
                output_mems_[i] = nullptr;
            }
        }
        delete[] output_mems_;
        output_mems_ = nullptr;
    }

    // 释放输入内存
    if (input_mems_) {
        for (uint32_t i = 0; i < io_num_.n_input; i++) {
            if (input_mems_[i]) {
                rknn_destroy_mem(ctx_, input_mems_[i]);
                input_mems_[i] = nullptr;
            }
        }
        delete[] input_mems_;
        input_mems_ = nullptr;
    }

    // 释放张量属性
    if (input_attrs_) {
        delete[] input_attrs_;
        input_attrs_ = nullptr;
    }

    if (output_attrs_) {
        delete[] output_attrs_;
        output_attrs_ = nullptr;
    }

    // 销毁RKNN上下文
    if (ctx_) {
        rknn_destroy(ctx_);
        ctx_ = 0;
    }
}

/**
 * @brief 打印模型信息
 */
void YOLOProcessorRKNN::PrintModelInfo() const
{
    if (!ctx_) {
        qDebug() << "模型未加载!";
        return;
    }

    qDebug() << "=== RKNN模型信息 ===";
    qDebug() << "模型路径:" << model_path_.c_str();
    qDebug() << "SDK版本:" << sdk_version_.c_str();
    qDebug() << "驱动版本:" << driver_version_.c_str();
    qDebug() << "输入尺寸:" << input_size_.width << "x" << input_size_.height;
    qDebug() << "置信度阈值:" << conf_threshold_;
    qDebug() << "NMS阈值:" << nms_threshold_;
    qDebug() << "类别数量:" << class_labels_.size();
    qDebug() << "======================";
}
