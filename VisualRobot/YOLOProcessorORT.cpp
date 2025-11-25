#include "YOLOProcessorORT.h"
#include <QDebug>

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
        letterbox_r_ = r;
        letterbox_dw_ = left;
        letterbox_dh_ = top;
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

        // convert outputs to mats (each mat: rows = H*W, cols = C)
        auto mats = OrtOutputToMats(outputTensors);

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

        // merge mats into dets: vertical concat (rows are candidates, cols=C)
        Mat dets;
        if (!mats.empty()) {
            dets = mats[0];
            for (size_t i = 1; i < mats.size(); ++i) {
                if (mats[i].cols == dets.cols) {
                    vconcat(dets, mats[i], dets);
                } else {
                    // fallback: try to reshape mats[i] to have same cols if possible
                    vconcat(dets, mats[i], dets);
                }
            }
        }

        // now call PostProcess expecting dets where each row is candidate vector
        vector<DetectionResult> dr = PostProcess(frame, dets);
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

// reuse PostProcess logic similar to previous YOLOProcessor
vector<DetectionResult> YOLOProcessorORT::PostProcess(const Mat& frame, const Mat& dets)
{
    vector<DetectionResult> detections;
    if (dets.empty()) return detections;

    Mat dets2 = dets;
    // if dets is single row with long vector, try to reshape
    if (dets2.dims == 2 && dets2.cols < 6 && dets2.rows == 1) {
        return detections;
    }

    int nc = dets2.cols - 5;
    if (nc <= 0) return detections;

    vector<Rect> boxes;
    vector<float> scores;
    vector<int> classIds;

    // use letterbox params to map from model space -> original image
    double r = letterbox_r_;
    double dw = letterbox_dw_;
    double dh = letterbox_dh_;

    for (int i = 0; i < dets2.rows; ++i) {
        const float* row = dets2.ptr<float>(i);
        float cx = row[0];
        float cy = row[1];
        float w  = row[2];
        float h  = row[3];
        float obj = row[4];
        // find class
        double maxClassScore = 0; int cls = 0;
        for (int c=0;c<nc;++c) {
            if (row[5+c] > maxClassScore) { maxClassScore = row[5+c]; cls = c; }
        }
        double conf = obj * maxClassScore;
        if (conf < confThreshold_) continue;

        // remove padding, then scale by 1/r to original image
        double x = (cx - dw) / r;
        double y = (cy - dh) / r;
        double ww = w / r;
        double hh = h / r;

        int x1 = int(x - ww/2.0);
        int y1 = int(y - hh/2.0);
        int boxw = int(ww);
        int boxh = int(hh);

        boxes.emplace_back(x1, y1, boxw, boxh);
        scores.emplace_back((float)conf);
        classIds.emplace_back(cls);
    }

    vector<int> idxs;
    dnn::NMSBoxes(boxes, scores, confThreshold_, nmsThreshold_, idxs);
    for (int id : idxs) {
        DetectionResult dr;
        dr.classId = classIds[id];
        dr.confidence = scores[id];
        if (dr.classId >=0 && dr.classId < (int)classLabels_.size()) dr.className = classLabels_[dr.classId];
        // clip box to image
        Rect clipped = boxes[id] & Rect(0,0,frame.cols, frame.rows);
        dr.boundingBox = clipped;
        detections.push_back(dr);
    }

    return detections;
}

void YOLOProcessorORT::DrawDetectionResults(Mat& frame, const vector<DetectionResult>& results)
{
    for (const auto& r : results) {
        Scalar color = (r.classId==0)? Scalar(0,255,0) : Scalar(0,0,255);
        rectangle(frame, r.boundingBox, color, 2);
        string lbl = r.className + ":" + to_string((int)(r.confidence*100)) + "%";
        int baseLine=0;
        Size tsize = getTextSize(lbl, FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);
        int tx = max(r.boundingBox.x, 0);
        int ty = max(r.boundingBox.y - tsize.height - 4, 0);
        rectangle(frame, Point(tx, ty), Point(tx+tsize.width, ty+tsize.height+4), color, FILLED);
        putText(frame, lbl, Point(tx, ty+tsize.height), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255,255,255), 1);
    }
}
