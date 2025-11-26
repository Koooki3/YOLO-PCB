#include "YOLOProcessor.h"
#include <QDebug>

using namespace cv;
using namespace std;

YOLOProcessor::YOLOProcessor(QObject* parent)
    : QObject(parent)
    , isModelLoaded_(false)
    , inputSize_(640, 640)
    , confThreshold_(0.25f)
    , nmsThreshold_(0.45f)
    , scaleFactor_(1.0/255.0)
    , meanValues_(0.0,0.0,0.0)
    , swapRB_(true)
{
}

YOLOProcessor::~YOLOProcessor()
{
}

bool YOLOProcessor::InitModel(const string& modelPath, bool useCUDA)
{
    try {
        net_ = dnn::readNet(modelPath);
        if (useCUDA) {
            net_.setPreferableBackend(dnn::DNN_BACKEND_CUDA);
            net_.setPreferableTarget(dnn::DNN_TARGET_CUDA_FP16);
        } else {
            net_.setPreferableBackend(dnn::DNN_BACKEND_DEFAULT);
            net_.setPreferableTarget(dnn::DNN_TARGET_CPU);
        }
        isModelLoaded_ = true;
        return true;
    } catch (const cv::Exception& e) {
        isModelLoaded_ = false;
        QString msg = QString("YOLO InitModel error: %1").arg(e.what());
        qDebug() << msg;
        emit errorOccurred(msg);
        return false;
    }
}

void YOLOProcessor::SetInputSize(const Size& size)
{
    inputSize_ = size;
}

void YOLOProcessor::SetThresholds(float conf, float nms)
{
    confThreshold_ = conf;
    nmsThreshold_ = nms;
}

void YOLOProcessor::SetClassLabels(const vector<string>& labels)
{
    classLabels_ = labels;
}

bool YOLOProcessor::DetectObjects(const Mat& frame, vector<DetectionResult>& results)
{
    results.clear();
    if (!isModelLoaded_) {
        emit errorOccurred("YOLO model not loaded");
        return false;
    }

    if (frame.empty()) {
        emit errorOccurred("Empty frame");
        return false;
    }

    Mat blob;
    try {
        // Preprocess using letterbox to keep aspect ratio (match Python behavior)
        int orig_w = frame.cols;
        int orig_h = frame.rows;
        int target_w = inputSize_.width;
        int target_h = inputSize_.height;
        double r = std::min((double)target_w / orig_w, (double)target_h / orig_h);
        int new_unpad_w = (int)round(orig_w * r);
        int new_unpad_h = (int)round(orig_h * r);
        int dw = target_w - new_unpad_w;
        int dh = target_h - new_unpad_h;
        int top = int(round(dh / 2.0));
        int bottom = dh - top;
        int left = int(round(dw / 2.0));
        int right = dw - left;
        Mat resized;
        if (orig_w != new_unpad_w || orig_h != new_unpad_h) cv::resize(frame, resized, Size(new_unpad_w, new_unpad_h));
        else resized = frame.clone();
        Mat padded;
        copyMakeBorder(resized, padded, top, bottom, left, right, BORDER_CONSTANT, Scalar(114,114,114));
        // store letterbox params
        letterbox_r_ = r;
        letterbox_dw_ = dw / 2.0;
        letterbox_dh_ = dh / 2.0;

        dnn::blobFromImage(padded, blob, scaleFactor_, inputSize_, meanValues_, swapRB_, false);
        net_.setInput(blob);
        vector<Mat> outs;
        net_.forward(outs, net_.getUnconnectedOutLayersNames());

        // Normalize different output layouts into a single 2D Mat where each row is a candidate vector
        // Handle common cases:
        // - output already NxC
        // - multiple feature maps [1,C,H,W] -> convert each to (H*W) x C and vconcat
        vector<Mat> rowMats;
        for (size_t k = 0; k < outs.size(); ++k) {
            Mat &m = outs[k];
            if (m.dims == 2) {
                // already (N x C)
                rowMats.push_back(m.reshape(1, m.rows));
            } else if (m.dims == 3) {
                // treat as (1 x N x C) or (N x C x 1) -> try to flatten first dim if possible
                if (m.size[0] == 1) {
                    Mat r = m.reshape(1, m.size[1]);
                    rowMats.push_back(r);
                } else {
                    // fallback: flatten
                    Mat r = m.reshape(1, m.total() / m.channels());
                    rowMats.push_back(r);
                }
            } else if (m.dims == 4) {
                // common ONNX feature map: [1, C, H, W]
                int N = (int)m.size[0];
                int C = (int)m.size[1];
                int H = (int)m.size[2];
                int W = (int)m.size[3];
                if (N != 1) {
                    // unexpected batch>1 but handle only first batch
                }
                // create Mat of size (H*W) x C
                Mat matHW(C, H*W, CV_32F);
                const float* dataPtr = reinterpret_cast<const float*>(m.data);
                // assume memory layout is NCHW
                for (int c = 0; c < C; ++c) {
                    int planeOffset = c * H * W;
                    for (int i = 0; i < H*W; ++i) {
                        matHW.at<float>(c, i) = dataPtr[planeOffset + i];
                    }
                }
                Mat matHW_t;
                transpose(matHW, matHW_t); // now (H*W) x C
                rowMats.push_back(matHW_t);
            } else {
                // other dims: flatten
                Mat r = m.reshape(1, 1);
                rowMats.push_back(r);
            }
        }

        Mat dets;
        if (!rowMats.empty()) {
            dets = rowMats[0];
            for (size_t i=1;i<rowMats.size();++i) {
                Mat tmp;
                // ensure same cols; if not, try to reshape
                if (rowMats[i].cols != dets.cols) {
                    // try transpose if rows match
                    if (rowMats[i].rows == dets.cols) {
                        Mat t;
                        transpose(rowMats[i], t);
                        vconcat(dets, t, tmp);
                    } else {
                        vconcat(dets, rowMats[i], tmp);
                    }
                } else {
                    vconcat(dets, rowMats[i], tmp);
                }
                dets = tmp;
            }
        }

        results = PostProcess(frame, dets);
        return true;
    } catch (const cv::Exception& e) {
        QString msg = QString("YOLO DetectObjects error: %1").arg(e.what());
        emit errorOccurred(msg);
        return false;
    }
}

vector<DetectionResult> YOLOProcessor::PostProcess(const Mat& frame, const Mat& outputs)
{
    vector<DetectionResult> detections;

    // Support common YOLO export formats: outputs shape [1, N, 85] or [N,85] where 85 = 4+1+num_classes
    Mat dets = outputs;
    if (dets.dims == 3) {
        // shape 1xNxC -> NxC
        dets = dets.reshape(1, dets.size[1]);
    }

    int nc = dets.cols - 5; // number of classes
    if (nc <= 0) return detections;

    vector<Rect> boxes;
    vector<float> scores;
    vector<int> classIds;

    // map from model space (letterbox padded input) back to original image using stored letterbox params
    double r = letterbox_r_;
    double dw = letterbox_dw_;
    double dh = letterbox_dh_;

    // Adaptive decode: compute both sigmoid and softmax based confidences globally, then choose best
    int R = dets.rows;
    vector<double> conf_sig_all(R, 0.0), conf_soft_all(R, 0.0);
    vector<int> cls_sig_all(R, 0), cls_soft_all(R, 0);
    for (int i = 0; i < R; ++i) {
        const float* row = dets.ptr<float>(i);
        double obj = (double)row[4];
        vector<double> class_raw(nc);
        for (int c=0;c<nc;++c) class_raw[c] = (double)row[5+c];
        // sigmoid best
        double max_sig = -1.0; int cls_sig = 0;
        for (int c=0;c<nc;++c) {
            double p = 1.0 / (1.0 + exp(-class_raw[c]));
            if (p > max_sig) { max_sig = p; cls_sig = c; }
        }
        // softmax best
        double ssum = 0.0;
        for (int c=0;c<nc;++c) ssum += exp(class_raw[c]);
        double max_soft = -1.0; int cls_soft = 0;
        if (ssum > 0) {
            for (int c=0;c<nc;++c) {
                double p = exp(class_raw[c]) / ssum;
                if (p > max_soft) { max_soft = p; cls_soft = c; }
            }
        } else { max_soft = max_sig; cls_soft = cls_sig; }
        double obj_sig = 1.0 / (1.0 + exp(-obj));
        conf_sig_all[i] = obj_sig * max_sig;
        conf_soft_all[i] = obj_sig * max_soft;
        cls_sig_all[i] = cls_sig;
        cls_soft_all[i] = cls_soft;
    }
    // choose by comparing avg of top-k
    int topN = min(50, R);
    auto avg_top = [&](vector<double>& arr){
        vector<double> s = arr; sort(s.begin(), s.end(), greater<double>());
        double sum = 0; for (int k=0;k<topN && k<(int)s.size(); ++k) sum += s[k];
        return topN>0 ? sum/topN : 0.0;
    };
    double avg_sig_top = avg_top(conf_sig_all);
    double avg_soft_top = avg_top(conf_soft_all);
    bool use_soft = (avg_soft_top > avg_sig_top * 1.05);
    qDebug() << "YOLOProcessor decode choose softmax=" << use_soft << " avg_sig=" << avg_sig_top << " avg_soft=" << avg_soft_top;

    // Second pass: filter using chosen method and build boxes
    for (int i = 0; i < R; ++i) {
        const float* row = dets.ptr<float>(i);
        float cx = row[0];
        float cy = row[1];
        float w  = row[2];
        float h  = row[3];

        double conf = use_soft ? conf_soft_all[i] : conf_sig_all[i];
        if (conf < confThreshold_) continue;

        int chosen_cls = use_soft ? cls_soft_all[i] : cls_sig_all[i];

        // convert center x,y,w,h (in model input coords) -> corner coords in original image
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
        classIds.emplace_back(chosen_cls);
    }

    // NMS
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

void YOLOProcessor::DrawDetectionResults(Mat& frame, const vector<DetectionResult>& results)
{
    for (const auto& r : results) {
        Scalar color = (r.classId==0)? Scalar(0,255,0) : Scalar(0,0,255);
        rectangle(frame, r.boundingBox, color, 2);
        // show confidence as decimal with two digits, fallback to numeric class id if name missing
        std::ostringstream oss;
        oss.setf(std::ios::fixed); oss.precision(2);
        string cname = r.className.empty() ? string("class_") + std::to_string(r.classId) : r.className;
        oss << cname << ":" << r.confidence;
        string lbl = oss.str();
        int baseLine=0;
        Size tsize = getTextSize(lbl, FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);
        int tx = max(r.boundingBox.x, 0);
        int ty = max(r.boundingBox.y - tsize.height - 4, 0);
        rectangle(frame, Point(tx, ty), Point(tx+tsize.width, ty+tsize.height+4), color, FILLED);
        putText(frame, lbl, Point(tx, ty+tsize.height), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255,255,255), 1);
    }
}
