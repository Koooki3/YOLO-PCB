#ifndef DATAPROCESSOR_H
#define DATAPROCESSOR_H

#include <opencv2/opencv.hpp>
#include <QObject>
#include <random>

using namespace cv;
using namespace std;

// 特征提取器类型枚举
enum class FeatureType
{
    SIFT,
    ORB,
    AKAZE
};

class DataProcessor : public QObject
{
    Q_OBJECT

public:
    explicit DataProcessor(QObject *parent = nullptr);
    
    // 设置特征提取器类型
    void SetFeatureType(FeatureType type);
    
    // 获取当前特征提取器类型
    FeatureType GetFeatureType() const;
    
    // 预处理方法
    Mat NormalizeImage(const Mat& input, double targetMean, double targetStd);
    Mat StandardizeImage(const Mat& input);
    Mat ResizeWithAspectRatio(const Mat& input, int targetSize);
    
    // 通用预处理方法（YOLO 专用预处理已迁移到独立模块）
    // YOLO 专用预处理方法（由 DLProcessor/YOLO 代码调用）
    Mat PreprocessForYOLO(const Mat& input, Size targetSize);
    Mat Letterbox(const Mat& input, Size targetSize, Scalar color = Scalar(114,114,114));
    Mat NormalizeYOLOInput(const Mat& input);

    // 获取YOLO预处理参数
    vector<float> GetYoloScaleRatio() const;
    vector<float> GetYoloFillPadding() const;
    Size GetYoloInputSize() const;
    
    // 特征提取方法
    Mat ExtractHOGFeatures(const Mat& input);
    vector<KeyPoint> DetectKeypoints(const Mat& input, Mat& descriptors);
    
    // 数据增强方法
    Mat AdjustBrightness(const Mat& input, double alpha);
    Mat AdjustContrast(const Mat& input, double beta);
    Mat AddNoise(const Mat& input, double mean, double stddev);
    Mat RandomRotate(const Mat& input, double maxAngle);
    Mat RandomFlip(const Mat& input);
    Mat RandomCrop(const Mat& input, double scale);
    
    // 批量增强
    vector<Mat> ApplyAugmentation(const Mat& input, int numAugmentations);

private:
    mt19937 rng_;
    Ptr<SIFT> siftDetector_;
    Ptr<ORB> orbDetector_;
    Ptr<AKAZE> akazeDetector_;
    FeatureType currentFeatureType_;
    
    // YOLO 相关参数已移除（由 YOLOProcessor 管理）
    // YOLO 相关成员（在.cpp中使用并初始化）
    Size yoloInputSize_;
    vector<float> ratio_;    // 缩放比例 x,y
    vector<float> dw_dh_;    // 填充宽高 (dw, dh)
    
    // 初始化特征检测器
    void InitializeDetectors();
    
    // 辅助函数
    double RandomDouble(double min, double max);
    int RandomInt(int min, int max);
};

#endif // DATAPROCESSOR_H
