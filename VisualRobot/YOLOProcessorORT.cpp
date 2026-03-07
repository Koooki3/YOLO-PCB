/**
 * @file YOLOProcessorORT.cpp
 * @brief YOLO处理器模块实现文件（基于ONNX Runtime）
 * 
 * 该文件实现了YOLOProcessorORT类的所有方法，提供基于ONNX Runtime的YOLO模型推理功能，
 * 包括模型加载、目标检测、结果绘制、延时统计和调试输出等功能。
 * 
 * @author VisualRobot Team
 * @date 2025-12-30
 * @version 1.0
 */

#include "YOLOProcessorORT.h"
#include <QDebug>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <tuple>
#include <omp.h>

using namespace std;

#include "configmanager.h"

#ifdef USE_RKNPU_EP
#include <onnxruntime_rknpu_provider_factory.h>
#endif

/**
 * @brief YOLOProcessorORT构造函数
 * 
 * 初始化YOLO处理器的各项参数，包括ONNX Runtime环境、会话选项、默认输入尺寸等
 * 从配置管理器获取优化参数，包括线程数、内存管理、图优化级别等
 * 
 * @param parent 父对象指针，默认为nullptr
 * @note 初始化内容：
 *       - ONNX Runtime环境（日志级别：警告）
 *       - 默认输入尺寸：640x640
 *       - 默认置信度阈值：50.0%
 *       - 默认NMS阈值：0.45
 *       - 图像缩放因子：1/255
 *       - 通道交换：BGR→RGB
 * @see ConfigManager
 */
YOLOProcessorORT::YOLOProcessorORT(QObject* parent)
    : QObject(parent)
    , env_(ORT_LOGGING_LEVEL_WARNING, "YOLOORT")  // 初始化ONNX Runtime环境，设置日志级别为警告
    , session_(nullptr)  // 初始化会话指针为空
    , inputSize_(640, 640)  // 默认输入尺寸为640x640
    , confThreshold_(50.0f)  // 默认置信度阈值为50.0
    , nmsThreshold_(0.45f)  // 默认NMS阈值为0.45
    , scaleFactor_(1.0/255.0)  // 图像缩放因子，将像素值归一化到[0,1]
    , swapRB_(true)  // 是否交换RB通道（OpenCV默认BGR，模型通常使用RGB）
    , letterbox_r_(1.0)  // letterbox缩放比例，初始化为1.0
    , letterbox_dw_(0.0)  // 宽度方向填充，初始化为0.0
    , letterbox_dh_(0.0)  // 高度方向填充，初始化为0.0
{
    // 从配置管理器获取优化参数
    ConfigManager* config = ConfigManager::instance();
    
    // 设置线程数
    sessionOptions_.SetIntraOpNumThreads(config->getIntraOpNumThreads());
    sessionOptions_.SetInterOpNumThreads(config->getInterOpNumThreads());

    // 设置内存管理
    if (config->getEnableCpuMemArena())
    {
        sessionOptions_.EnableCpuMemArena();
    }
    else
    {
        sessionOptions_.DisableCpuMemArena();
    }

    // 设置图优化级别
    QString graphOptLevel = config->getGraphOptimizationLevel();
    if (graphOptLevel == "ORT_ENABLE_ALL")
    {
        sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    }
    else if (graphOptLevel == "ORT_ENABLE_EXTENDED")
    {
        sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    }
    else if (graphOptLevel == "ORT_ENABLE_BASIC")
    {
        sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
    }
    else
    {
        sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    }
    
    // 设置执行模式
    QString execMode = config->getExecutionMode();
    if (execMode == "ORT_PARALLEL")
    {
        sessionOptions_.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
    }
    else
    {
        sessionOptions_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    }

    // 设置内存模式优化
    if (config->getEnableMemPattern())
    {
        sessionOptions_.EnableMemPattern();
    }
    else
    {
        sessionOptions_.DisableMemPattern();
    }

    // 可选：启用profiling（仅用于调试）
    // sessionOptions_.EnableProfiling("profile.json");
    
    // 初始化延时统计结构体
    timingStats_.preprocessTime = 0.0;
    timingStats_.inferenceTime = 0.0;
    timingStats_.postprocessTime = 0.0;
    timingStats_.totalTime = 0.0;
    timingStats_.fps = 0;
}

/**
 * @brief YOLOProcessorORT析构函数
 * 
 * 释放ONNX Runtime会话资源
 * 
 * @note 会话资源会自动释放
 */
YOLOProcessorORT::~YOLOProcessorORT()
{
    session_.reset();  // 释放会话资源
}

/**
 * @brief 初始化YOLO模型
 * @param modelPath ONNX模型文件路径
 * @param useCUDA 是否使用CUDA加速
 * @return 初始化是否成功
 * 
 * 创建ONNX Runtime会话，加载模型文件
 */
bool YOLOProcessorORT::InitModel(const string& modelPath, bool useCUDA)
{
    try 
    {   
        // 从配置管理器获取加速选项
        ConfigManager* config = ConfigManager::instance();
        if (config->isAcceleratorEnabled("opencl")) {
            sessionOptions_.AddConfigEntry("session.enable_opencl","1");
            sessionOptions_.AddConfigEntry("opencl_device_id", "0");  // 使用第一个OpenCL设备
            sessionOptions_.AddConfigEntry("opencl_mem_limit", "4096");  // 4GB内存限制
        } else {
            sessionOptions_.AddConfigEntry("session.enable_opencl","0");
        }

#ifdef USE_RKNPU_EP
        // RK3588 NPU：当 accelerators.npu 为 true 且 ONNX Runtime 带 RKNPU EP 时，优先使用 NPU 推理
        if (config->isAcceleratorEnabled("npu")) {
            Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_RKNPU(sessionOptions_));
        }
#endif

        session_ = std::make_unique<Ort::Session>(env_, modelPath.c_str(), sessionOptions_);
        return true;
    } 
    catch (const Ort::Exception& e) 
    {
        QString msg = QString("ORT InitModel error: %1").arg(e.what());
        qDebug() << msg;
        emit errorOccurred(msg);
        return false;
    }
}

/**
 * @brief 设置模型输入尺寸
 * @param size 输入尺寸
 * 
 * 更新模型的输入尺寸，用于预处理
 */
void YOLOProcessorORT::SetInputSize(const Size& size)
{
    inputSize_ = size;
}

/**
 * @brief 设置检测阈值
 * @param conf 置信度阈值
 * @param nms NMS（非极大值抑制）阈值
 * 
 * 更新置信度阈值和NMS阈值，用于后处理
 */
void YOLOProcessorORT::SetThresholds(float conf, float nms)
{
    confThreshold_ = conf;
    nmsThreshold_ = nms;
}

/**
 * @brief 设置类别标签
 * @param labels 类别标签列表
 * 
 * 更新类别标签，用于结果绘制和输出
 */
void YOLOProcessorORT::SetClassLabels(const vector<string>& labels)
{
    classLabels_ = labels;
}

/**
 * @brief 将ONNX Runtime输出转换为OpenCV Mat格式
 * @param outputs ONNX Runtime输出列表
 * @return 转换后的Mat列表
 * 
 * 处理不同形状的ONNX Runtime输出，将其转换为适合后续处理的OpenCV Mat格式
 * 支持4D、3D、2D和其他形状的张量转换
 */
vector<Mat> YOLOProcessorORT::OrtOutputToMats(const std::vector<Ort::Value>& outputs)
{
    vector<Mat> mats;
    Ort::AllocatorWithDefaultOptions allocator;
    
    // 遍历所有输出
    for (const auto& out : outputs) 
    {
        // 获取输出的类型和形状信息
        auto type_info = out.GetTensorTypeAndShapeInfo();
        vector<int64_t> shape = type_info.GetShape();
        
        // 计算元素总数
        size_t elemCount = 1;
        for (auto d : shape) 
        {
            elemCount *= (d > 0 ? d : 1);
        }

        // 获取输出数据指针
        const float* data = out.GetTensorData<float>();

        // 处理4D张量：常见形状为[1, C, H, W]，转换为(H*W) x C的Mat
        if (shape.size() == 4 && shape[0] == 1) 
        {
            int C = (int)shape[1];  // 通道数
            int H = (int)shape[2];  // 高度
            int W = (int)shape[3];  // 宽度
            int rows = H * W;       // 行数为高度乘宽度
            
            Mat m(rows, C, CV_32F);  // 创建Mat对象
            
            // 数据布局为N(=1),C,H,W -> 索引 = c*H*W + y*W + x
            for (int c = 0; c < C; ++c) 
            {
                const float* plane = data + (size_t)c * H * W;
                for (int y = 0; y < H; ++y) 
                {
                    for (int x = 0; x < W; ++x) 
                    {
                        int rowIdx = y * W + x;
                        m.at<float>(rowIdx, c) = plane[y * W + x];
                    }
                }
            }
            mats.push_back(m);
        }
        // 处理3D张量：
        // - [1, C, L] -> 解释为L行，C列（例如：(1,8,8400) -> 8400 x 8）
        // - [A,B,C]（其他） -> 转换为A x (B*C)
        else if (shape.size() == 3) 
        {
            int A = (int)shape[0];
            int B = (int)shape[1];
            int C = (int)shape[2];
            
            if (A == 1) 
            {
                int rows = C;
                int cols = B;
                Mat m(rows, cols, CV_32F);
                
                // 数据布局N(=1), B, C -> 索引 = b*C + c
                for (int b = 0; b < B; ++b) 
                {
                    const float* plane = data + (size_t)b * C;
                    for (int cidx = 0; cidx < C; ++cidx) 
                    {
                        m.at<float>(cidx, b) = plane[cidx];
                    }
                }
                mats.push_back(m);
            } 
            else 
            {
                Mat m(A, B*C, CV_32F);
                memcpy(m.ptr<float>(), data, elemCount * sizeof(float));
                mats.push_back(m);
            }
        } 
        // 处理2D张量：直接转换为R x C的Mat
        else if (shape.size() == 2) 
        {
            int R = (int)shape[0];  // 行数
            int C = (int)shape[1];  // 列数
            Mat m(R, C, CV_32F);
            memcpy(m.ptr<float>(), data, elemCount * sizeof(float));
            mats.push_back(m);
        } 
        // 处理其他形状：转换为1 x elemCount的Mat
        else 
        {
            Mat m(1, (int)elemCount, CV_32F);
            memcpy(m.ptr<float>(), data, elemCount * sizeof(float));
            mats.push_back(m);
        }
    }
    
    return mats;
}

/**
 * @brief 检测图像中的目标
 * @param frame 输入图像
 * @param results 输出检测结果列表
 * @return 检测是否成功
 * 
 * 完整的目标检测流程：
 * 1. 图像预处理（letterbox缩放、BGR转RGB、归一化、HWC转CHW）
 * 2. ONNX模型推理
 * 3. 输出结果转换
 * 4. 后处理生成检测结果
 */
bool YOLOProcessorORT::DetectObjects(const Mat& frame, vector<DetectionResult>& results)
{
    results.clear();
    if (!session_) 
    {
        emit errorOccurred("ORT会话未初始化");
        return false;
    }
    if (classLabels_.empty()) 
    {
        emit errorOccurred("类别标签未加载");
        return false;
    }
    if (frame.empty()) 
    {
        emit errorOccurred("空帧输入");
        return false;
    }

    try 
    {
        // 总处理时间开始计时
        auto totalStartTime = chrono::high_resolution_clock::now();
        
        // 1. 图像预处理阶段计时开始
        auto preprocessStartTime = chrono::high_resolution_clock::now();
        
        // Preprocess: letterbox (keep aspect ratio), BGR->RGB, float32, /255, NCHW
        int orig_w = frame.cols;
        int orig_h = frame.rows;
        int target_w = inputSize_.width;
        int target_h = inputSize_.height;
        double r = std::min((double)target_w / orig_w, (double)target_h / orig_h);
        // compute padding
        int new_unpad_w = (int)round(orig_w * r);
        int new_unpad_h = (int)round(orig_h * r);
        int dw = target_w - new_unpad_w;
        int dh = target_h - new_unpad_h;
        // divide padding into two sides
        int top = int(round(dh / 2.0));
        int bottom = dh - top;
        int left = int(round(dw / 2.0));
        int right = dw - left;

        // 使用UMat实现OpenCL加速
        cv::UMat frame_umat = frame.getUMat(cv::ACCESS_READ);
        cv::UMat resized_umat;
        
        if (orig_w != new_unpad_w || orig_h != new_unpad_h)
        {
            cv::resize(frame_umat, resized_umat, Size(new_unpad_w, new_unpad_h));
        }
        else
        {
            resized_umat = frame_umat.clone();
        }

        // pad with 114 (common YOLO) to reach target size
        cv::UMat padded_umat;
        copyMakeBorder(resized_umat, padded_umat, top, bottom, left, right, BORDER_CONSTANT, Scalar(114,114,114));

        // store letterbox params for postprocess
        // store dw/dh as float half-padding (before integer rounding) to match Python letterbox
        letterbox_r_ = r;
        letterbox_dw_ = dw / 2.0; // half padding as float
        letterbox_dh_ = dh / 2.0;
        
        

        // 使用UMat进行颜色转换和归一化
        cv::UMat rgb_umat;
        cv::cvtColor(padded_umat, rgb_umat, cv::COLOR_BGR2RGB);
        rgb_umat.convertTo(rgb_umat, CV_32F, (float)scaleFactor_);
        
        // 将UMat转换为Mat以便访问数据（数据格式转换需要CPU访问）
        cv::Mat rgb;
        rgb_umat.copyTo(rgb);

        // HWC -> CHW - 使用OpenMP并行化加速
        vector<int64_t> inputShape = {1, 3, target_h, target_w};
        size_t inputTensorSize = 1 * 3 * target_h * target_w;
        vector<float> inputTensorValues(inputTensorSize);
        
        // 并行化HWC转CHW的过程
        #pragma omp parallel for collapse(3)
        for (int c = 0; c < 3; ++c) 
        {
            for (int y = 0; y < target_h; ++y) 
            {
                for (int x = 0; x < target_w; ++x) 
                {
                    size_t idx = c * target_h * target_w + y * target_w + x;
                    Vec3f v = rgb.at<Vec3f>(y, x);
                    inputTensorValues[idx] = v[c];
                }
            }
        }
        
        // 图像预处理阶段计时结束
        auto preprocessEndTime = chrono::high_resolution_clock::now();
        double preprocessTime = chrono::duration<double, milli>(preprocessEndTime - preprocessStartTime).count();
        
        // 2. 模型推理阶段计时开始
        auto inferenceStartTime = chrono::high_resolution_clock::now();

        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(mem_info, inputTensorValues.data(), inputTensorSize, inputShape.data(), inputShape.size());

        // prepare input and output name arrays (use GetInputNames/GetOutputNames to get std::vector<string>)
        auto inputNamesVec = session_->GetInputNames();
        vector<const char*> inputNames;
        inputNames.reserve(inputNamesVec.size());
        for (auto &s : inputNamesVec) 
        {
            inputNames.push_back(s.c_str());
        }

        auto outputNamesVec = session_->GetOutputNames();
        vector<const char*> outputNames;
        outputNames.reserve(outputNamesVec.size());
        for (auto &s : outputNamesVec) 
        {
            outputNames.push_back(s.c_str());
        }

        // run (assume single input tensor)
        size_t numOutputNodes = outputNames.size();
        auto outputTensors = session_->Run(Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor, 1, outputNames.data(), numOutputNodes);
        
        // 模型推理阶段计时结束
        auto inferenceEndTime = chrono::high_resolution_clock::now();
        double inferenceTime = chrono::duration<double, milli>(inferenceEndTime - inferenceStartTime).count();
        
        // 3. 后处理阶段计时开始
        auto postprocessStartTime = chrono::high_resolution_clock::now();

        // 准备调试信息（仅当启用调试时才会使用）
        YOLODebugInfo debugInfo;

        // convert FIRST output to mat (mimic TestOnnx.py behavior)
        std::vector<Ort::Value> singleOuts;
        // Ort::Value is move-only; avoid copy by moving the element into the new vector
        singleOuts.emplace_back(std::move(outputTensors[0]));
        auto mats = OrtOutputToMats(singleOuts);

        // Use only the first output's mat (this matches TestOnnx.py / run_yolo_ort.py)
        Mat dets;
        if (!mats.empty()) 
        {
            dets = mats[0];
        }

        // now call PostProcess with the correct parameters
        // 临时添加默认的图像路径和期望类别，实际使用时应该从外部传入
        string imagePath = "test_image.jpg";
        string expectedClass = "unknown";
        vector<DetectionResult> dr = PostProcess(singleOuts, frame.size(), imagePath, expectedClass);
        results = dr;
        
        // 后处理阶段计时结束
        auto postprocessEndTime = chrono::high_resolution_clock::now();
        double postprocessTime = chrono::duration<double, milli>(postprocessEndTime - postprocessStartTime).count();
        
        // 总处理时间结束计时
        auto totalEndTime = chrono::high_resolution_clock::now();
        double totalTime = chrono::duration<double, milli>(totalEndTime - totalStartTime).count();
        
        // 计算FPS
        int fps = totalTime > 0 ? static_cast<int>(1000.0 / totalTime) : 0;
        
        // 更新延时统计
        timingStats_.preprocessTime = preprocessTime;
        timingStats_.inferenceTime = inferenceTime;
        timingStats_.postprocessTime = postprocessTime;
        timingStats_.totalTime = totalTime;
        timingStats_.fps = fps;
        
        // 完成调试信息收集并调用调试输出函数
        if (enableDebug_)
        {
            debugInfo.mats = mats;
            debugInfo.dets = dets;
            debugInfo.preprocessTime = preprocessTime;
            debugInfo.inferenceTime = inferenceTime;
            debugInfo.postprocessTime = postprocessTime;
            debugInfo.totalTime = totalTime;
            debugInfo.fps = fps;
            
            // 调用独立的调试输出函数，避免影响计时
            DebugOutput(debugInfo);
        }

        return true;
    } 
    catch (const Ort::Exception& e) 
    {
        QString msg = QString("ORT DetectObjects error: %1").arg(e.what());
        emit errorOccurred(msg);
        return false;
    } 
    catch (const cv::Exception& e) 
    {
        QString msg = QString("CV error: %1").arg(e.what());
        emit errorOccurred(msg);
        return false;
    }
}

/**
 * @brief YOLO模型后处理方法，根据标签文件动态适配类别数量
 * @param outputs ONNX Runtime输出列表
 * @param frameSize 原始图像尺寸
 * @param imagePath 图像路径（可选）
 * @param expectedClass 期望类别（可选）
 * @return 检测结果列表
 * 
 * 后处理流程：
 * 1. 解析模型输出张量
 * 2. 提取候选框和类别分数
 * 3. 应用置信度阈值过滤
 * 4. 将坐标映射回原始图像
 * 5. 应用NMS（非极大值抑制）
 * 6. 如果没有检测结果，尝试降低阈值重新过滤
 */
std::vector<DetectionResult> YOLOProcessorORT::PostProcess(const std::vector<Ort::Value>& outputs, const cv::Size& frameSize, const std::string& imagePath, const std::string& expectedClass)
{
    std::vector<DetectionResult> results;
    
    // 检查输出是否为空
    if (outputs.empty()) 
    {
        return results;
    }
    
    // 获取第一个输出的维度信息
    const Ort::Value& output = outputs[0];
    auto output_info = output.GetTypeInfo();
    auto tensor_info = output_info.GetTensorTypeAndShapeInfo();
    auto output_shape = tensor_info.GetShape();
    
    // 对于YOLOv8输出，形状应该是 [1, 10, 8400]
    // 检查是否为3维张量
    if (output_shape.size() != 3) 
    {
        return results;
    }
    
    // 获取输出数据指针
    const float* output_data = output.GetTensorData<float>();
    
    // 按照输出结构 (1,10,N) -> squeeze -> (10,N)
    // 每列代表一个候选框: [cx, cy, w, h, cls0, cls1, ..., clsN]
    int rows = static_cast<int>(output_shape[1]);
    int cols = static_cast<int>(output_shape[2]);
    const int NUM_CLASSES = static_cast<int>(classLabels_.size());

    // 辅助函数：按 (row, col) 读取值，layout: [1, rows, cols] -> index = row*cols + col
    auto get_val = [&](int r, int c)->float {
        if (r < 0 || r >= rows || c < 0 || c >= cols) 
        {
            return 0.0f;
        }
        return output_data[r * cols + c];
    };

    std::vector<cv::Rect> cand_boxes;      // 候选框列表
    std::vector<float> cand_scores;        // 候选框分数列表
    std::vector<int> cand_class_ids;       // 候选框类别ID列表
    std::vector<std::tuple<int, float, float, int>> top_candidates; // 候选框信息元组：(索引, 最大sigmoid分数, 原始分数, 类别ID)

    // 遍历所有候选框 - 使用OpenMP并行化，优化线程竞争
    #pragma omp parallel
    {
        // 每个线程使用局部向量收集结果，减少锁竞争
        vector<cv::Rect> local_boxes;
        vector<float> local_scores;
        vector<int> local_class_ids;
        
        #pragma omp for
        for (int c = 0; c < cols; ++c) 
        {
            // 提取中心点坐标和宽高
            float cx = get_val(0, c);
            float cy = get_val(1, c);
            float w = get_val(2, c);
            float h = get_val(3, c);

            // 找到最大的类别分数
            float max_raw = -std::numeric_limits<float>::infinity();
            int best_cls = 0;
            for (int k = 0; k < NUM_CLASSES; ++k) 
            {
                float raw = get_val(4 + k, c);
                if (raw > max_raw) 
                {
                    max_raw = raw;
                    best_cls = k;
                }
            }

            // 按要求将类别分数乘以100用于筛选
            float scaled_conf = max_raw * 100.0f;
            
            // 应用置信度阈值过滤
            if (scaled_conf < confThreshold_) 
            {
                continue; // 置信度阈值由调用方设置（请注意阈值单位：scaled）
            }

            // 将 640-scale 像素映射回原图
            double x_orig = (static_cast<double>(cx) - letterbox_dw_) / letterbox_r_;
            double y_orig = (static_cast<double>(cy) - letterbox_dh_) / letterbox_r_;
            double w_orig = static_cast<double>(w) / letterbox_r_;
            double h_orig = static_cast<double>(h) / letterbox_r_;

            // 计算边界框的左上角和右下角坐标
            double x1d = x_orig - w_orig / 2.0;
            double y1d = y_orig - h_orig / 2.0;
            double x2d = x_orig + w_orig / 2.0;
            double y2d = y_orig + h_orig / 2.0;

            // 确保坐标在图像范围内
            int x1 = static_cast<int>(std::max(0.0, std::min(x1d, static_cast<double>(frameSize.width - 1))));
            int y1 = static_cast<int>(std::max(0.0, std::min(y1d, static_cast<double>(frameSize.height - 1))));
            int x2 = static_cast<int>(std::max(0.0, std::min(x2d, static_cast<double>(frameSize.width - 1))));
            int y2 = static_cast<int>(std::max(0.0, std::min(y2d, static_cast<double>(frameSize.height - 1))));

            // 计算边界框的宽度和高度
            int bw = x2 - x1;
            int bh = y2 - y1;
            
            // 过滤掉无效的边界框
            if (bw <= 0 || bh <= 0) 
            {
                continue;
            }

            // 添加到局部候选列表
            local_boxes.emplace_back(x1, y1, bw, bh);
            local_scores.push_back(scaled_conf);
            local_class_ids.push_back(best_cls);
        }
        
        // 合并局部结果到全局列表
        #pragma omp critical
        {
            cand_boxes.insert(cand_boxes.end(), local_boxes.begin(), local_boxes.end());
            cand_scores.insert(cand_scores.end(), local_scores.begin(), local_scores.end());
            cand_class_ids.insert(cand_class_ids.end(), local_class_ids.begin(), local_class_ids.end());
        }
    }

    // 应用NMS（非极大值抑制）
    if (!cand_boxes.empty()) 
    {
        std::vector<int> indices;
        cv::dnn::NMSBoxes(cand_boxes, cand_scores, confThreshold_, nmsThreshold_, indices);
        
        // 遍历NMS后的索引，生成检测结果
        for (int idx : indices) 
        {
            DetectionResult dr;
            dr.boundingBox = cand_boxes[idx];
            dr.classId = cand_class_ids[idx];
            dr.confidence = cand_scores[idx];
            
            // 设置类别名称
            if (dr.classId >= 0 && dr.classId < (int)classLabels_.size()) 
            {
                dr.className = classLabels_[dr.classId];
            }
            else 
            {
                dr.className = "";
            }
            
            results.push_back(dr);
        }
    }
    
    // 如果没有检测结果，尝试降低阈值并重新过滤
    if (results.empty() && confThreshold_ > 0.1f) 
    {
        float lowerThreshold = 0.1f;
        std::vector<DetectionResult> lowThresholdResults;

        // num_attributes / num_boxes 对应当前输出布局
        int num_attributes = rows; // 例如：10
        int num_boxes = cols;      // 例如：8400

        // 使用 get_val(row, col) 访问以避免错误的索引计算
        for (int c = 0; c < num_boxes; ++c) 
        {
            // 提取中心点坐标和宽高
            float x_center = get_val(0, c);
            float y_center = get_val(1, c);
            float width = get_val(2, c);
            float height = get_val(3, c);

            // 找到最大的类别分数
            float max_raw_score = -std::numeric_limits<float>::infinity();
            int max_class_id = -1;
            for (int k = 0; k < NUM_CLASSES; ++k) 
            {
                float raw_score = get_val(4 + k, c);
                if (raw_score > max_raw_score) 
                {
                    max_raw_score = raw_score;
                    max_class_id = k;
                }
            }

            // 应用较低的置信度阈值
            if (max_raw_score > lowerThreshold && max_class_id >= 0 && max_class_id < NUM_CLASSES) 
            {
                float input_w = inputSize_.width;
                float input_h = inputSize_.height;

                // 将归一化坐标转换为像素坐标
                float x_center_pix = x_center * input_w;
                float y_center_pix = y_center * input_h;
                float width_pix = width * input_w;
                float height_pix = height * input_h;

                // 将坐标映射回原始图像
                float real_x_center = (x_center_pix - letterbox_dw_) / letterbox_r_;
                float real_y_center = (y_center_pix - letterbox_dh_) / letterbox_r_;
                float real_width = width_pix / letterbox_r_;
                float real_height = height_pix / letterbox_r_;

                // 计算边界框的左上角和右下角坐标
                float x1 = real_x_center - real_width / 2.0f;
                float y1 = real_y_center - real_height / 2.0f;
                float x2 = real_x_center + real_width / 2.0f;
                float y2 = real_y_center + real_height / 2.0f;

                // 确保坐标在图像范围内
                x1 = std::max(0.0f, std::min(x1, static_cast<float>(frameSize.width)));
                y1 = std::max(0.0f, std::min(y1, static_cast<float>(frameSize.height)));
                x2 = std::max(0.0f, std::min(x2, static_cast<float>(frameSize.width)));
                y2 = std::max(0.0f, std::min(y2, static_cast<float>(frameSize.height)));

                // 过滤掉无效的边界框
                if (x2 > x1 + 1.0f && y2 > y1 + 1.0f) 
                {
                    DetectionResult result;
                    result.boundingBox = cv::Rect2i(static_cast<int>(x1), static_cast<int>(y1), static_cast<int>(x2 - x1), static_cast<int>(y2 - y1));
                    result.classId = max_class_id;
                    result.confidence = max_raw_score;

                    // 设置类别名称
                    if (max_class_id < classLabels_.size()) 
                    {
                        result.className = classLabels_[max_class_id];
                    }

                    lowThresholdResults.push_back(result);
                }
            }
        }

        // 如果有低阈值结果，应用NMS
        if (!lowThresholdResults.empty()) 
        {
            // 对低阈值结果应用NMS
            std::vector<int> indices;
            std::vector<float> confidences;
            std::vector<cv::Rect> boxes;

            // 提取边界框和置信度
            for (const auto& result : lowThresholdResults) 
            {
                boxes.push_back(result.boundingBox);
                confidences.push_back(result.confidence);
            }

            // 应用NMS
            cv::dnn::NMSBoxes(boxes, confidences, lowerThreshold, nmsThreshold_, indices);

            // 生成最终结果
            std::vector<DetectionResult> filteredResults;
            for (int idx : indices) 
            {
                filteredResults.push_back(lowThresholdResults[idx]);
            }

            results = filteredResults;
        }
    }
    
    return results;
}

/**
 * @brief 在图像上绘制检测结果
 * @param frame 输入输出图像，将在其上绘制检测结果
 * @param results 检测结果列表
 * 
 * 绘制流程：
 * 1. 为每个类别生成不同的颜色
 * 2. 遍历所有检测结果
 * 3. 绘制边界框
 * 4. 绘制标签文本和背景
 */
void YOLOProcessorORT::DrawDetectionResults(cv::Mat& frame, const std::vector<DetectionResult>& results)
{
    // 为每个类别生成不同颜色，根据标签数量动态扩展
    std::vector<cv::Scalar> colors = {
        cv::Scalar(255, 0, 0),    // 蓝色 - class 0
        cv::Scalar(0, 255, 0),    // 绿色 - class 1
        cv::Scalar(0, 0, 255),    // 红色 - class 2
        cv::Scalar(255, 255, 0),  // 青色 - class 3
        cv::Scalar(255, 0, 255),  // 品红 - class 4
        cv::Scalar(0, 255, 255),  // 黄色 - class 5
        cv::Scalar(128, 0, 0),    // 深蓝色 - class 6
        cv::Scalar(0, 128, 0),    // 深绿色 - class 7
        cv::Scalar(0, 0, 128),    // 深红色 - class 8
        cv::Scalar(128, 128, 0),  // 深青色 - class 9
        cv::Scalar(128, 0, 128),  // 深品红 - class 10
        cv::Scalar(0, 128, 128)   // 深黄色 - class 11
    };
    
    // 如果标签数量超过预设颜色数量，动态扩展颜色列表
    while (colors.size() < classLabels_.size()) 
    {
        int idx = colors.size();
        colors.push_back(cv::Scalar(
            (idx * 137) % 255,  // 简单的哈希函数生成不同颜色
            (idx * 271) % 255,
            (idx * 383) % 255
        ));
    }
    
    // 遍历所有检测结果
    for (const auto& result : results) 
    {
        // 确保类别ID在有效范围内
        int color_idx = result.classId;
        if (color_idx < 0 || color_idx >= static_cast<int>(colors.size())) 
        {
            color_idx = 0;  // 默认使用第一个颜色
        }
        
        // 绘制边界框，使用更粗的线宽以确保可见
        cv::Scalar color = colors[color_idx];
        cv::rectangle(frame, result.boundingBox, color, 3);  // 线宽从2增加到3
        
        // 准备标签文本
        std::string label;
        if (!result.className.empty()) 
        {
            // 有用户标签，使用用户提供的标签
            label = result.className + " " + cv::format("%.2f", result.confidence);
        } 
        else 
        {
            // 没有用户标签，使用数字类别ID和置信度
            label = std::to_string(result.classId) + " " + cv::format("%.2f", result.confidence);
        }
        
        // 确保标签大小和位置正确
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseLine);  // 增加字体大小和粗细
        
        // 确保标签不会超出图像边界
        int top = result.boundingBox.y - labelSize.height - 5;
        if (top < 0) 
        {
            top = result.boundingBox.y + 5;
        }
        
        // 计算标签背景位置
        cv::Point bottomLeft(result.boundingBox.x, top + labelSize.height + baseLine + 5);
        cv::Point topRight(result.boundingBox.x + labelSize.width + 10, top - 5);
        
        // 确保标签背景不会超出图像边界
        if (bottomLeft.x < 0) 
        {
            bottomLeft.x = 0;
        }
        if (bottomLeft.y > frame.rows) 
        {
            bottomLeft.y = frame.rows;
        }
        if (topRight.x > frame.cols) 
        {
            topRight.x = frame.cols;
        }
        if (topRight.y < 0) 
        {
            topRight.y = 0;
        }
        
        // 绘制标签背景，使用半透明填充
        cv::rectangle(frame, bottomLeft, topRight, color, cv::FILLED);
        
        // 绘制文本，使用更大的字体和更粗的线宽
        cv::putText(frame, label, cv::Point(result.boundingBox.x + 5, top + labelSize.height + baseLine), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);  // 使用白色文本以便更好地可见
    }
}

/**
 * @brief 获取最新的延时统计
 * @return 延时统计结构体
 */
YOLOTimingStats YOLOProcessorORT::GetTimingStats() const
{
    return timingStats_;
}

/**
 * @brief 启用/禁用详细调试输出
 * @param enable 是否启用
 */
void YOLOProcessorORT::SetDebugOutput(bool enable)
{
    enableDebug_ = enable;
}

/**
 * @brief 输出调试信息
 * @param debugInfo 调试信息结构体，包含需要输出的各种调试数据
 * 
 * 集中处理所有调试输出，仅保留必要的计时信息，避免调试输出影响核心处理逻辑的性能
 */
void YOLOProcessorORT::DebugOutput(const YOLODebugInfo& debugInfo)
{
    if (!enableDebug_)
    {
        return; // 调试输出已禁用，直接返回
    }

    // 仅输出必要的处理时间统计信息
    qDebug() << "YOLO Processing Timing:";
    qDebug() << "  Preprocess: " << debugInfo.preprocessTime << " ms";
    qDebug() << "  Inference: " << debugInfo.inferenceTime << " ms";
    qDebug() << "  Postprocess: " << debugInfo.postprocessTime << " ms";
    qDebug() << "  Total: " << debugInfo.totalTime << " ms";
    qDebug() << "  FPS: " << debugInfo.fps;
}
