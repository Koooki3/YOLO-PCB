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
        dnn::blobFromImage(frame, blob, scaleFactor_, inputSize_, meanValues_, swapRB_, false);
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

    float gain_x = (float)frame.cols / inputSize_.width;
    float gain_y = (float)frame.rows / inputSize_.height;

    for (int i = 0; i < dets.rows; ++i) {
        const float* row = dets.ptr<float>(i);
        float cx = row[0];
        float cy = row[1];
        float w  = row[2];
        float h  = row[3];
        float obj = row[4];
        // find class with max score
        Mat scoresRow(1, nc, CV_32F);
        for (int c=0;c<nc;++c) scoresRow.at<float>(0,c) = row[5+c];
        Point classIdPoint;
        double maxClassScore;
        minMaxLoc(scoresRow, 0, &maxClassScore, 0, &classIdPoint);
        double conf = obj * maxClassScore;
        if (conf < confThreshold_) continue;

        int cls = classIdPoint.x;

        // Convert xywh (center) -> x1y1w h in original image scale
        int x1 = int((cx - w/2.0) * gain_x);
        int y1 = int((cy - h/2.0) * gain_y);
        int boxw = int(w * gain_x);
        int boxh = int(h * gain_y);

        boxes.emplace_back(x1, y1, boxw, boxh);
        scores.emplace_back((float)conf);
        classIds.emplace_back(cls);
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
        string lbl = r.className + ":" + to_string((int)(r.confidence*100)) + "%";
        int baseLine=0;
        Size tsize = getTextSize(lbl, FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);
        int tx = max(r.boundingBox.x, 0);
        int ty = max(r.boundingBox.y - tsize.height - 4, 0);
        rectangle(frame, Point(tx, ty), Point(tx+tsize.width, ty+tsize.height+4), color, FILLED);
        putText(frame, lbl, Point(tx, ty+tsize.height), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255,255,255), 1);
    }
}
