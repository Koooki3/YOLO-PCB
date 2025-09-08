#ifndef DATAPROCESSOR_H
#define DATAPROCESSOR_H

#include <opencv2/opencv.hpp>
#include <QObject>
#include <random>

class DataProcessor : public QObject
{
    Q_OBJECT

public:
    explicit DataProcessor(QObject *parent = nullptr);
    
    // 预处理方法
    cv::Mat normalizeImage(const cv::Mat& input, double targetMean = 0.0, double targetStd = 1.0);
    cv::Mat standardizeImage(const cv::Mat& input);
    cv::Mat resizeWithAspectRatio(const cv::Mat& input, int targetSize);
    
    // 特征提取方法
    cv::Mat extractHOGFeatures(const cv::Mat& input);
    std::vector<cv::KeyPoint> detectKeypoints(const cv::Mat& input, cv::Mat& descriptors);
    
    // 数据增强方法
    cv::Mat adjustBrightness(const cv::Mat& input, double alpha);
    cv::Mat adjustContrast(const cv::Mat& input, double beta);
    cv::Mat addNoise(const cv::Mat& input, double mean = 0, double stddev = 25);
    cv::Mat randomRotate(const cv::Mat& input, double maxAngle = 30);
    cv::Mat randomFlip(const cv::Mat& input);
    cv::Mat randomCrop(const cv::Mat& input, double scale = 0.8);
    
    // 批量增强
    std::vector<cv::Mat> applyAugmentation(const cv::Mat& input, int numAugmentations = 5);

private:
    std::mt19937 rng_;
    cv::Ptr<cv::SIFT> siftDetector_;
    
    // 辅助函数
    double randomDouble(double min, double max);
    int randomInt(int min, int max);
};

#endif // DATAPROCESSOR_H
