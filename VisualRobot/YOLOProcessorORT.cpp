#include "YOLOProcessorORT.h"
#include <QDebug>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <tuple>

using namespace std;

YOLOProcessorORT::YOLOProcessorORT(QObject* parent)
    : QObject(parent)
    , env_(ORT_LOGGING_LEVEL_WARNING, "YOLOORT")
    , session_(nullptr)
    , inputSize_(640,640)
    , confThreshold_(0.25f)
    , nmsThreshold_(0.45f)
    , scaleFactor_(1.0/255.0)
    , swapRB_(true)
    , letterbox_r_(1.0)
    , letterbox_dw_(0.0)
    , letterbox_dh_(0.0)
{
    sessionOptions_.SetIntraOpNumThreads(1);
    sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
}

YOLOProcessorORT::~YOLOProcessorORT()
{
    session_.reset();
}

bool YOLOProcessorORT::InitModel(const string& modelPath, bool useCUDA)
{
    try {
        // sessionOptions_ can be extended to enable CUDA if needed (user must provide proper provider and libs)
        session_ = std::make_unique<Ort::Session>(env_, modelPath.c_str(), sessionOptions_);
        return true;
    } catch (const Ort::Exception& e) {
        QString msg = QString("ORT InitModel error: %1").arg(e.what());
        qDebug() << msg;
        emit errorOccurred(msg);
        return false;
    }
}

void YOLOProcessorORT::SetInputSize(const Size& size)
{
    inputSize_ = size;
}

void YOLOProcessorORT::SetThresholds(float conf, float nms)
{
    confThreshold_ = conf;
    nmsThreshold_ = nms;
}

void YOLOProcessorORT::SetClassLabels(const vector<string>& labels)
{
    classLabels_ = labels;
}

// Convert Ort outputs into vector<Mat> with shape info preserved
vector<Mat> YOLOProcessorORT::OrtOutputToMats(const std::vector<Ort::Value>& outputs)
{
    vector<Mat> mats;
    Ort::AllocatorWithDefaultOptions allocator;
    for (const auto& out : outputs) {
        auto type_info = out.GetTensorTypeAndShapeInfo();
        vector<int64_t> shape = type_info.GetShape();
        size_t elemCount = 1;
        for (auto d : shape) elemCount *= (d>0? d: 1);

        const float* data = out.GetTensorData<float>();

        // Handle common case: [1, C, H, W] -> convert to (H*W) x C
        if (shape.size() == 4 && shape[0] == 1) {
            int C = (int)shape[1];
            int H = (int)shape[2];
            int W = (int)shape[3];
            int rows = H * W;
            Mat m(rows, C, CV_32F);
            // data layout is N(=1),C,H,W -> index = c*H*W + y*W + x
            for (int c = 0; c < C; ++c) {
                const float* plane = data + (size_t)c * H * W;
                for (int y = 0; y < H; ++y) {
                    for (int x = 0; x < W; ++x) {
                        int rowIdx = y * W + x;
                        m.at<float>(rowIdx, c) = plane[y * W + x];
                    }
                }
            }
            mats.push_back(m);
        }
        // 3D tensors: common patterns:
        // - [1, C, L] -> interpret as L rows, C cols (e.g., (1,8,8400) -> 8400 x 8)
        // - [A,B,C] (other) -> fallback to A x (B*C)
        else if (shape.size() == 3) {
            int A = (int)shape[0];
            int B = (int)shape[1];
            int C = (int)shape[2];
            if (A == 1) {
                int rows = C;
                int cols = B;
                Mat m(rows, cols, CV_32F);
                // data layout N(=1), B, C -> index = b*C + c
                for (int b = 0; b < B; ++b) {
                    const float* plane = data + (size_t)b * C;
                    for (int cidx = 0; cidx < C; ++cidx) {
                        m.at<float>(cidx, b) = plane[cidx];
                    }
                }
                mats.push_back(m);
            } else {
                Mat m(A, B*C, CV_32F);
                memcpy(m.ptr<float>(), data, elemCount * sizeof(float));
                mats.push_back(m);
            }
        } else if (shape.size() == 2) {
            int R = (int)shape[0];
            int C = (int)shape[1];
            Mat m(R, C, CV_32F);
            memcpy(m.ptr<float>(), data, elemCount * sizeof(float));
            mats.push_back(m);
        } else {
            Mat m(1, (int)elemCount, CV_32F);
            memcpy(m.ptr<float>(), data, elemCount * sizeof(float));
            mats.push_back(m);
        }
    }
    return mats;
}

bool YOLOProcessorORT::DetectObjects(const Mat& frame, vector<DetectionResult>& results)
{
    results.clear();
    if (!session_) {
        emit errorOccurred("ORT session not initialized");
        return false;
    }
    if (frame.empty()) {
        emit errorOccurred("Empty frame");
        return false;
    }

    try {
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

        Mat resized;
        if (orig_w != new_unpad_w || orig_h != new_unpad_h)
            cv::resize(frame, resized, Size(new_unpad_w, new_unpad_h));
        else
            resized = frame.clone();

        // pad with 114 (common YOLO) to reach target size
        Mat padded;
        copyMakeBorder(resized, padded, top, bottom, left, right, BORDER_CONSTANT, Scalar(114,114,114));

        // store letterbox params for postprocess
        // store dw/dh as float half-padding (before integer rounding) to match Python letterbox
        letterbox_r_ = r;
        letterbox_dw_ = dw / 2.0; // half padding as float
        letterbox_dh_ = dh / 2.0;
        qDebug() << "letterbox r, dw, dh:" << letterbox_r_ << letterbox_dw_ << letterbox_dh_;

        Mat rgb;
        cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
        rgb.convertTo(rgb, CV_32F, (float)scaleFactor_);

        // HWC -> CHW
        vector<int64_t> inputShape = {1, 3, target_h, target_w};
        size_t inputTensorSize = 1 * 3 * target_h * target_w;
        vector<float> inputTensorValues(inputTensorSize);
        int idx = 0;
        for (int c = 0; c < 3; ++c) {
            for (int y = 0; y < target_h; ++y) {
                for (int x = 0; x < target_w; ++x) {
                    Vec3f v = rgb.at<Vec3f>(y, x);
                    inputTensorValues[idx++] = v[c];
                }
            }
        }

        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(mem_info, inputTensorValues.data(), inputTensorSize, inputShape.data(), inputShape.size());

        // prepare input and output name arrays (use GetInputNames/GetOutputNames to get std::vector<string>)
        auto inputNamesVec = session_->GetInputNames();
        vector<const char*> inputNames;
        inputNames.reserve(inputNamesVec.size());
        for (auto &s : inputNamesVec) inputNames.push_back(s.c_str());

        auto outputNamesVec = session_->GetOutputNames();
        vector<const char*> outputNames;
        outputNames.reserve(outputNamesVec.size());
        for (auto &s : outputNamesVec) outputNames.push_back(s.c_str());

        // run (assume single input tensor)
        size_t numOutputNodes = outputNames.size();
        auto outputTensors = session_->Run(Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor, 1, outputNames.data(), numOutputNodes);

        // debug: print output shapes (useful when user reports mismatch)
        qDebug() << "ORT returned" << (int)outputTensors.size() << "output(s)";
        for (size_t oi=0; oi<outputTensors.size(); ++oi) {
            auto info = outputTensors[oi].GetTensorTypeAndShapeInfo();
            auto shape = info.GetShape();
            QString s = "shape:";
            for (auto d : shape) s += QString::number(d) + ",";
            qDebug() << "  output" << oi << s;
        }

        // convert FIRST output to mat (mimic TestOnnx.py behavior)
        std::vector<Ort::Value> singleOuts;
        // Ort::Value is move-only; avoid copy by moving the element into the new vector
        singleOuts.emplace_back(std::move(outputTensors[0]));
        auto mats = OrtOutputToMats(singleOuts);

        // debug: print first few values of first mat
        if (!mats.empty()) {
            Mat &m0 = mats[0];
            qDebug() << "first mat rows,cols:" << m0.rows << m0.cols;
            int rshow = std::min(3, m0.rows);
            for (int ri=0; ri<rshow; ++ri) {
                QString rowvals;
                for (int ci=0; ci<std::min(6, m0.cols); ++ci) rowvals += QString::number(m0.at<float>(ri,ci),'g',3)+",";
                qDebug() << "  row" << ri << rowvals;
            }
        }

        // Use only the first output's mat (this matches TestOnnx.py / run_yolo_ort.py)
        Mat dets;
        if (!mats.empty()) dets = mats[0];

        // Debug: print column-wise stats and confidence distribution
        if (!dets.empty()) {
            int R = dets.rows;
            int C = dets.cols;
            qDebug() << "dets rows,cols:" << R << C;
            // per-column min/max
            for (int c = 0; c < C; ++c) {
                double minv, maxv;
                cv::minMaxLoc(dets.col(c), &minv, &maxv);
                qDebug() << "  col" << c << "min" << minv << "max" << maxv;
            }

            // compute per-row max class score and product with obj
            vector<float> confidences;
            confidences.reserve(R);
            int cnt_gt_01 = 0, cnt_gt_05 = 0, cnt_gt_001 = 0;
            for (int r = 0; r < R; ++r) {
                const float* row = dets.ptr<float>(r);
                // assume first 4 are xywh, index 4 is obj, remaining are class scores
                float obj = (C > 4) ? row[4] : 1.0f;
                float maxCls = 0.0f;
                for (int cc = 5; cc < C; ++cc) {
                    if (row[cc] > maxCls) maxCls = row[cc];
                }
                float conf = obj * maxCls;
                confidences.push_back(conf);
                if (maxCls > 0.1f) cnt_gt_01++;
                if (maxCls > 0.5f) cnt_gt_05++;
                if (maxCls > 0.001f) cnt_gt_001++;
            }
            // top 5 confidences
            vector<float> sorted = confidences;
            sort(sorted.begin(), sorted.end(), greater<float>());
            int showN = min((int)sorted.size(), 10);
            QString topstr = "top confidences:";
            for (int i = 0; i < showN; ++i) topstr += QString::number(sorted[i],'g',6)+",";
            qDebug() << topstr;
            qDebug() << "counts: >0.001=" << cnt_gt_001 << " >0.1=" << cnt_gt_01 << " >0.5=" << cnt_gt_05;
        }

        // now call PostProcess with the correct parameters
    // 临时添加默认的图像路径和期望类别，实际使用时应该从外部传入
    string imagePath = "test_image.jpg";
    string expectedClass = "unknown";
    vector<DetectionResult> dr = PostProcess(singleOuts, frame.size(), imagePath, expectedClass);
    results = dr;

        return true;
    } catch (const Ort::Exception& e) {
        QString msg = QString("ORT DetectObjects error: %1").arg(e.what());
        emit errorOccurred(msg);
        return false;
    } catch (const cv::Exception& e) {
        QString msg = QString("CV error: %1").arg(e.what());
        emit errorOccurred(msg);
        return false;
    }
}

// 修改后的PostProcess方法，适配6类训练集，与Python实现相符
std::vector<DetectionResult> YOLOProcessorORT::PostProcess(const std::vector<Ort::Value>& outputs, const cv::Size& frameSize, const std::string& imagePath, const std::string& expectedClass)
{
    std::vector<DetectionResult> results;
    
    // 输出调试信息
    std::cout << "图像路径: " << imagePath << std::endl;
    std::cout << "期望类别: " << expectedClass << std::endl;
    std::cout << "Original image size: " << frameSize.width << "x" << frameSize.height << std::endl;
    
    // 输出letterbox结果信息
    std::cout << "Letterbox result: (" << inputSize_.width << ", " << inputSize_.height << ", 3), ratio: " << letterbox_r_ 
              << ", padding: (" << letterbox_dw_ << ", " << letterbox_dh_ << ")" << std::endl;
    
    if (outputs.empty()) {
        std::cout << "Empty outputs received" << std::endl;
        return results;
    }
    
    // 获取第一个输出的维度信息
    const Ort::Value& output = outputs[0];
    auto output_info = output.GetTypeInfo();
    auto tensor_info = output_info.GetTensorTypeAndShapeInfo();
    auto output_shape = tensor_info.GetShape();
    
    // 输出原始输出形状
    std::cout << "Raw output shape: (" << output_shape[0] << ", " << output_shape[1] << ", " << output_shape[2] << ")" << std::endl;
    
    // 对于YOLOv8输出，形状应该是 [1, 10, 8400]
    // 检查是否为3维张量
    if (output_shape.size() != 3) {
        std::cout << "Unexpected output shape dimension: " << output_shape.size() << std::endl;
        return results;
    }
    
    // 获取输出数据
    const float* output_data = output.GetTensorData<float>();
    
    // 计算输出数据的最小值和最大值
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    int total_elements = output_shape[0] * output_shape[1] * output_shape[2];
    for (int i = 0; i < total_elements; i++) {
        min_val = std::min(min_val, output_data[i]);
        max_val = std::max(max_val, output_data[i]);
    }
    std::cout << "Raw output min/max: " << min_val << "/" << max_val << std::endl;
    std::cout << "Processed output shape: (" << 8400 << ", " << 10 << ")" << std::endl << std::endl;
    
    // 按照用户提供的输出结构 (1,10,N) -> squeeze -> (10,N)
    // 每列代表一个候选框: [cx, cy, w, h, cls0, cls1, ..., cls5]
    int rows = static_cast<int>(output_shape[1]);
    int cols = static_cast<int>(output_shape[2]);
    const int NUM_CLASSES = 6;

    // 辅助函数：按 (row, col) 读取值，layout: [1, rows, cols] -> index = row*cols + col
    auto get_val = [&](int r, int c)->float {
        if (r < 0 || r >= rows || c < 0 || c >= cols) return 0.0f;
        return output_data[r * cols + c];
    };

    std::vector<cv::Rect> cand_boxes;
    std::vector<float> cand_scores;
    std::vector<int> cand_class_ids;
    std::vector<std::tuple<int, float, float, int>> top_candidates; // (index, max_sigmoid_score, raw_score, class_id)

    for (int c = 0; c < cols; ++c) {
        float cx = get_val(0, c);
        float cy = get_val(1, c);
        float w = get_val(2, c);
        float h = get_val(3, c);

        // class scores rows 4..9
        float max_raw = -std::numeric_limits<float>::infinity();
        int best_cls = 0;
        for (int k = 0; k < NUM_CLASSES; ++k) {
            float raw = get_val(4 + k, c);
            if (raw > max_raw) {
                max_raw = raw;
                best_cls = k;
            }
        }

        // 按要求将类别分数乘以10000用于筛选
        float scaled_conf = max_raw * 10000.0f;
        top_candidates.emplace_back(c, 0.0f, max_raw, best_cls); // 保存以便后续分析（sigmoid 未计算）

        if (scaled_conf < confThreshold_) continue; // 置信度阈值由调用方设置（请注意阈值单位：scaled）

        // 将 640-scale 像素映射回原图
        double x_orig = (static_cast<double>(cx) - letterbox_dw_) / letterbox_r_;
        double y_orig = (static_cast<double>(cy) - letterbox_dh_) / letterbox_r_;
        double w_orig = static_cast<double>(w) / letterbox_r_;
        double h_orig = static_cast<double>(h) / letterbox_r_;

        double x1d = x_orig - w_orig / 2.0;
        double y1d = y_orig - h_orig / 2.0;
        double x2d = x_orig + w_orig / 2.0;
        double y2d = y_orig + h_orig / 2.0;

        int x1 = static_cast<int>(std::max(0.0, std::min(x1d, static_cast<double>(frameSize.width - 1))));
        int y1 = static_cast<int>(std::max(0.0, std::min(y1d, static_cast<double>(frameSize.height - 1))));
        int x2 = static_cast<int>(std::max(0.0, std::min(x2d, static_cast<double>(frameSize.width - 1))));
        int y2 = static_cast<int>(std::max(0.0, std::min(y2d, static_cast<double>(frameSize.height - 1))));

        int bw = x2 - x1;
        int bh = y2 - y1;
        if (bw <= 0 || bh <= 0) continue;

        cand_boxes.emplace_back(x1, y1, bw, bh);
        cand_scores.push_back(scaled_conf);
        cand_class_ids.push_back(best_cls);
    }

    // 应用NMS
    if (!cand_boxes.empty()) {
        std::vector<int> indices;
        cv::dnn::NMSBoxes(cand_boxes, cand_scores, confThreshold_, nmsThreshold_, indices);
        for (int idx : indices) {
            DetectionResult dr;
            dr.boundingBox = cand_boxes[idx];
            dr.classId = cand_class_ids[idx];
            dr.confidence = cand_scores[idx];
            if (!classLabels_.empty() && dr.classId >=0 && dr.classId < (int)classLabels_.size()) dr.className = classLabels_[dr.classId];
            else dr.className = "";
            results.push_back(dr);
        }
    }
    
    // 对top候选进行排序并输出
    std::sort(top_candidates.begin(), top_candidates.end(), 
              [](const auto& a, const auto& b) { return std::get<1>(a) > std::get<1>(b); });
    
    // 输出max sigmoid scores top10
    std::cout << "\nAnalyzing output structure (using class scores as confidence)..." << std::endl;
    std::cout << "max sigmoid scores top10: [";
    for (int i = 0; i < std::min(10, static_cast<int>(top_candidates.size())); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << '"' << std::fixed << std::setprecision(6) << std::get<1>(top_candidates[i]) << '"';
    }
    std::cout << "]" << std::endl;
    
    // 输出对应的class IDs
    std::cout << "corresponding class IDs: [";
    for (int i = 0; i < std::min(10, static_cast<int>(top_candidates.size())); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << std::get<3>(top_candidates[i]);
    }
    std::cout << "]" << std::endl;
    
    // 输出max raw scores top10
    std::cout << "max raw scores top10: [";
    for (int i = 0; i < std::min(10, static_cast<int>(top_candidates.size())); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << '"' << std::fixed << std::setprecision(6) << std::get<2>(top_candidates[i]) << '"';
    }
    std::cout << "]" << std::endl;
    
    // 计算不同阈值的计数
    int count_gt_05_raw = 0, count_gt_01_raw = 0, count_gt_005_raw = 0;
    int count_gt_05_sigmoid = 0, count_gt_01_sigmoid = 0, count_gt_005_sigmoid = 0;
    
    for (const auto& candidate : top_candidates) {
        float sigmoid_score = std::get<1>(candidate);
        float raw_score = std::get<2>(candidate);
        
        if (sigmoid_score > 0.5f) count_gt_05_sigmoid++;
        if (sigmoid_score > 0.1f) count_gt_01_sigmoid++;
        if (sigmoid_score > 0.05f) count_gt_005_sigmoid++;
        
        if (raw_score > 0.5f) count_gt_05_raw++;
        if (raw_score > 0.1f) count_gt_01_raw++;
        if (raw_score > 0.05f) count_gt_005_raw++;
    }
    
    std::cout << "max sigmoid scores count > 0.5: " << count_gt_05_sigmoid << std::endl;
    std::cout << "max sigmoid scores count > 0.1: " << count_gt_01_sigmoid << std::endl;
    std::cout << "max sigmoid scores count > 0.05: " << count_gt_005_sigmoid << std::endl;
    std::cout << "max raw scores count > 0.5: " << count_gt_05_raw << std::endl;
    std::cout << "max raw scores count > 0.1: " << count_gt_01_raw << std::endl;
    std::cout << "max raw scores count > 0.05: " << count_gt_005_raw << std::endl;
    
    // 输出top候选详细信息
    std::cout << "\nTop candidates (index, max_sigmoid_score, raw_score, class_id):" << std::endl;
    for (int i = 0; i < std::min(10, static_cast<int>(top_candidates.size())); i++) {
        int idx = std::get<0>(top_candidates[i]);
        float sigmoid_score = std::get<1>(top_candidates[i]);
        float raw_score = std::get<2>(top_candidates[i]);
        int class_id = std::get<3>(top_candidates[i]);
        
        std::string class_name = "";
        if (!classLabels_.empty() && class_id >= 0 && class_id < classLabels_.size()) {
            class_name = classLabels_[class_id];
        }
        
        std::cout << idx << ": max_sigmoid_score=" << std::fixed << std::setprecision(7) << sigmoid_score 
                  << ", raw_score=" << std::fixed << std::setprecision(7) << raw_score 
                  << ", class_id=" << class_id;
        if (!class_name.empty()) {
            std::cout << ", class_name=" << class_name;
        }
        std::cout << std::endl;
    }
    
    // 输出NMS前的检测数量
    std::cout << "Found " << results.size() << " detections before NMS with raw score > " << confThreshold_ << std::endl;
    
    // 如果没有检测结果，尝试降低阈值并重新过滤
    if (results.empty() && confThreshold_ > 0.1f) {
        std::cout << "  No detections found. Trying with lower threshold (0.1f)..." << std::endl;
        float lowerThreshold = 0.1f;
        std::vector<DetectionResult> lowThresholdResults;

        // num_attributes / num_boxes 对应当前输出布局
        int num_attributes = rows; // e.g., 10
        int num_boxes = cols;      // e.g., 8400

        // 使用 get_val(row, col) 访问以避免错误的索引计算
        for (int c = 0; c < num_boxes; ++c) {
            float x_center = get_val(0, c);
            float y_center = get_val(1, c);
            float width = get_val(2, c);
            float height = get_val(3, c);

            float max_raw_score = -std::numeric_limits<float>::infinity();
            int max_class_id = -1;
            for (int k = 0; k < NUM_CLASSES; ++k) {
                float raw_score = get_val(4 + k, c);
                if (raw_score > max_raw_score) {
                    max_raw_score = raw_score;
                    max_class_id = k;
                }
            }

            if (max_raw_score > lowerThreshold && max_class_id >= 0 && max_class_id < NUM_CLASSES) {
                float input_w = inputSize_.width;
                float input_h = inputSize_.height;

                float x_center_pix = x_center * input_w;
                float y_center_pix = y_center * input_h;
                float width_pix = width * input_w;
                float height_pix = height * input_h;

                float real_x_center = (x_center_pix - letterbox_dw_) / letterbox_r_;
                float real_y_center = (y_center_pix - letterbox_dh_) / letterbox_r_;
                float real_width = width_pix / letterbox_r_;
                float real_height = height_pix / letterbox_r_;

                float x1 = real_x_center - real_width / 2.0f;
                float y1 = real_y_center - real_height / 2.0f;
                float x2 = real_x_center + real_width / 2.0f;
                float y2 = real_y_center + real_height / 2.0f;

                x1 = std::max(0.0f, std::min(x1, static_cast<float>(frameSize.width)));
                y1 = std::max(0.0f, std::min(y1, static_cast<float>(frameSize.height)));
                x2 = std::max(0.0f, std::min(x2, static_cast<float>(frameSize.width)));
                y2 = std::max(0.0f, std::min(y2, static_cast<float>(frameSize.height)));

                if (x2 > x1 + 1.0f && y2 > y1 + 1.0f) {
                    DetectionResult result;
                    result.boundingBox = cv::Rect2i(static_cast<int>(x1), static_cast<int>(y1),
                                                  static_cast<int>(x2 - x1), static_cast<int>(y2 - y1));
                    result.classId = max_class_id;
                    result.confidence = max_raw_score;

                    if (!classLabels_.empty() && max_class_id < classLabels_.size()) {
                        result.className = classLabels_[max_class_id];
                    }

                    lowThresholdResults.push_back(result);
                }
            }
        }

        if (!lowThresholdResults.empty()) {
            std::cout << "  Found " << lowThresholdResults.size() << " detections with threshold " << lowerThreshold << std::endl;

            // 对低阈值结果应用NMS
            std::vector<int> indices;
            std::vector<float> confidences;
            std::vector<cv::Rect> boxes;

            for (const auto& result : lowThresholdResults) {
                boxes.push_back(result.boundingBox);
                confidences.push_back(result.confidence);
            }

            cv::dnn::NMSBoxes(boxes, confidences, lowerThreshold, nmsThreshold_, indices);

            std::vector<DetectionResult> filteredResults;
            for (int idx : indices) {
                filteredResults.push_back(lowThresholdResults[idx]);
            }

            std::cout << "  After NMS with low threshold: " << filteredResults.size() << " detections" << std::endl;
            results = filteredResults;
        }
    }
    
    return results;
}

void YOLOProcessorORT::DrawDetectionResults(cv::Mat& frame, const std::vector<DetectionResult>& results)
{
    // 输出绘制信息
    std::cout << "Drawing " << results.size() << " detection results on frame" << std::endl;
    
    // 为每个类别生成不同颜色，适配6类训练集
    std::vector<cv::Scalar> colors;
    // 使用固定的颜色方案，确保一致性
    colors.push_back(cv::Scalar(255, 0, 0));    // 蓝色 - class 0
    colors.push_back(cv::Scalar(0, 255, 0));    // 绿色 - class 1
    colors.push_back(cv::Scalar(0, 0, 255));    // 红色 - class 2
    colors.push_back(cv::Scalar(255, 255, 0));  // 青色 - class 3
    colors.push_back(cv::Scalar(255, 0, 255));  // 品红 - class 4
    colors.push_back(cv::Scalar(0, 255, 255));  // 黄色 - class 5
    
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& result = results[i];
        
        // 输出当前绘制的检测结果信息
        std::cout << "  Drawing detection " << i << ": class_id=" << result.classId 
                  << ", confidence=" << result.confidence 
                  << ", box=([" << result.boundingBox.x << "," << result.boundingBox.y 
                  << "," << result.boundingBox.width << "," << result.boundingBox.height << "]" << std::endl;
        
        // 确保类别ID在有效范围内
        int color_idx = result.classId;
        if (color_idx < 0 || color_idx >= static_cast<int>(colors.size())) {
            color_idx = 0;  // 默认使用第一个颜色
        }
        
        // 绘制边界框，使用更粗的线宽以确保可见
        cv::Scalar color = colors[color_idx];
        cv::rectangle(frame, result.boundingBox, color, 3);  // 线宽从2增加到3
        
        // 准备标签文本
        std::string label;
        if (!result.className.empty()) {
            // 有用户标签，使用用户提供的标签
            label = result.className + " " + cv::format("%.2f", result.confidence);
        } else {
            // 没有用户标签，使用数字类别ID和置信度
            // 直接使用数字而不是前缀"class_"，简化显示
            label = std::to_string(result.classId) + " " + cv::format("%.2f", result.confidence);
        }
        
        // 确保标签大小和位置正确
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseLine);  // 增加字体大小和粗细
        
        // 确保标签不会超出图像边界
        int top = result.boundingBox.y - labelSize.height - 5;
        if (top < 0) {
            top = result.boundingBox.y + 5;
        }
        
        // 计算标签背景位置
        cv::Point bottomLeft(result.boundingBox.x, top + labelSize.height + baseLine + 5);
        cv::Point topRight(result.boundingBox.x + labelSize.width + 10, top - 5);
        
        // 确保标签背景不会超出图像边界
        if (bottomLeft.x < 0) bottomLeft.x = 0;
        if (bottomLeft.y > frame.rows) bottomLeft.y = frame.rows;
        if (topRight.x > frame.cols) topRight.x = frame.cols;
        if (topRight.y < 0) topRight.y = 0;
        
        // 绘制标签背景，使用半透明填充
        cv::rectangle(frame, bottomLeft, topRight, color, cv::FILLED);
        
        // 绘制文本，使用更大的字体和更粗的线宽
        cv::putText(frame, label, 
                   cv::Point(result.boundingBox.x + 5, top + labelSize.height + baseLine),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);  // 使用白色文本以便更好地可见
    }
}