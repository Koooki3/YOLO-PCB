#ifndef DATAPROCESSOR_H
#define DATAPROCESSOR_H

#include <opencv2/opencv.hpp>
#include <QObject>
#include <random>

using namespace cv;
using namespace std;

class DataProcessor : public QObject
{
    Q_OBJECT

public:
    explicit DataProcessor(QObject *parent = nullptr);
    
    // 预处理方法
    Mat normalizeImage(const Mat& input, double targetMean, double targetStd);
    Mat standardizeImage(const Mat& input);
    Mat resizeWithAspectRatio(const Mat& input, int targetSize);
    
    // 特征提取方法
    Mat extractHOGFeatures(const Mat& input);
    vector<KeyPoint> detectKeypoints(const Mat& input, Mat& descriptors);
    
    // 数据增强方法
    Mat adjustBrightness(const Mat& input, double alpha);
    Mat adjustContrast(const Mat& input, double beta);
    Mat addNoise(const Mat& input, double mean, double stddev);
    Mat randomRotate(const Mat& input, double maxAngle);
    Mat randomFlip(const Mat& input);
    Mat randomCrop(const Mat& input, double scale);
    
    // 批量增强
    vector<Mat> applyAugmentation(const Mat& input, int numAugmentations);

private:
    mt19937 rng_;
    Ptr<SIFT> siftDetector_;
    
    // 辅助函数
    double randomDouble(double min, double max);
    int randomInt(int min, int max);
};

#endif // DATAPROCESSOR_H
