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

        // convert to cv::Mat according to dims
        if (shape.size() == 4) {
            int N = (int)shape[0];
            int C = (int)shape[1];
            int H = (int)shape[2];
            int W = (int)shape[3];
            // create Mat with size [N, C, H, W] flattened to N x (C*H*W)
            Mat m(N, C*H*W, CV_32F);
            memcpy(m.ptr<float>(), data, elemCount * sizeof(float));
            // store as is; PostProcess will handle reshape
            mats.push_back(m);
        } else if (shape.size() == 3) {
            int A = (int)shape[0];
            int B = (int)shape[1];
            int C = (int)shape[2];
            Mat m(A, B*C, CV_32F);
            memcpy(m.ptr<float>(), data, elemCount * sizeof(float));
            mats.push_back(m);
        } else if (shape.size() == 2) {
            int R = (int)shape[0];
            int C = (int)shape[1];
            Mat m(R, C, CV_32F);
            memcpy(m.ptr<float>(), data, elemCount * sizeof(float));
            mats.push_back(m);
        } else {
            // flatten
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
        // Preprocess: resize, BGR->RGB, float32, /255, NCHW
        Mat resized;
        cv::resize(frame, resized, inputSize_);
        Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
        rgb.convertTo(rgb, CV_32F, (float)scaleFactor_);

        // HWC -> CHW
        vector<int64_t> inputShape = {1, 3, inputSize_.height, inputSize_.width};
        size_t inputTensorSize = 1 * 3 * inputSize_.height * inputSize_.width;
        vector<float> inputTensorValues(inputTensorSize);
        int idx = 0;
        for (int c = 0; c < 3; ++c) {
            for (int y = 0; y < inputSize_.height; ++y) {
                for (int x = 0; x < inputSize_.width; ++x) {
                    inputTensorValues[idx++] = rgb.at<Vec3f>(y, x)[c];
                }
            }
        }

        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(mem_info, inputTensorValues.data(), inputTensorSize, inputShape.data(), inputShape.size());

        // prepare input names
        size_t numInputNodes = session_->GetInputCount();
        vector<const char*> inputNames(numInputNodes);
        for (size_t i=0;i<numInputNodes;++i) {
            char* name = session_->GetInputName(i, Ort::AllocatorWithDefaultOptions());
            inputNames[i] = name;
        }

        // prepare outputs
        size_t numOutputNodes = session_->GetOutputCount();
        vector<const char*> outputNames(numOutputNodes);
        for (size_t i=0;i<numOutputNodes;++i) {
            char* oname = session_->GetOutputName(i, Ort::AllocatorWithDefaultOptions());
            outputNames[i] = oname;
        }

        // run
        auto outputTensors = session_->Run(Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor, 1, outputNames.data(), numOutputNodes);

        // convert outputs to mats
        auto mats = OrtOutputToMats(outputTensors);

        // merge mats into dets similar to earlier logic
        Mat dets;
        if (!mats.empty()) {
            // try to convert each mat to rows x cols where cols = channels (e.g., C)
            vector<Mat> rowMats;
            for (auto &m : mats) {
                if (m.cols > 1 && (m.rows == 1 || m.rows > m.cols)) {
                    // assume each row is a candidate
                    rowMats.push_back(m);
                } else {
                    rowMats.push_back(m);
                }
            }
            dets = rowMats[0].clone();
            for (size_t i=1;i<rowMats.size();++i) {
                Mat tmp;
                if (rowMats[i].cols == dets.cols) vconcat(dets, rowMats[i], tmp);
                else vconcat(dets, rowMats[i], tmp);
                dets = tmp;
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

    float gain_x = (float)frame.cols / inputSize_.width;
    float gain_y = (float)frame.rows / inputSize_.height;

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

        int x1 = int((cx - w/2.0) * gain_x);
        int y1 = int((cy - h/2.0) * gain_y);
        int boxw = int(w * gain_x);
        int boxh = int(h * gain_y);

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
        dr.boundingBox = boxes[id];
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
