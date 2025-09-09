#include "DataProcessor.h"
#include <QDebug>

DataProcessor::DataProcessor(QObject *parent)
    : QObject(parent)
    , rng_(std::random_device{}())
{
    siftDetector_ = cv::SIFT::create();
}

cv::Mat DataProcessor::normalizeImage(const cv::Mat& input, double targetMean, double targetStd)
{
    cv::Mat normalized;
    input.convertTo(normalized, CV_32F);
    
    // 计算当前均值和标准差
    cv::Scalar mean, stddev;
    cv::meanStdDev(normalized, mean, stddev);
    
    // 标准化处理
    normalized = (normalized - mean[0]) / stddev[0];
    
    // 调整到目标均值和标准差
    normalized = normalized * targetStd + targetMean;
    
    return normalized;
}

cv::Mat DataProcessor::standardizeImage(const cv::Mat& input)
{
    double maxVal = 0;
    double minVal = 0;
    cv::minMaxLoc(input, &minVal, &maxVal);
    cv::Mat standardized;
    input.convertTo(standardized, CV_8U, 255.0 / (maxVal - minVal), 0);
    return standardized;
}

cv::Mat DataProcessor::resizeWithAspectRatio(const cv::Mat& input, int targetSize)
{
    cv::Mat resized;
    double ratio = static_cast<double>(targetSize) / std::max(input.rows, input.cols);
    cv::resize(input, resized, cv::Size(), ratio, ratio, cv::INTER_AREA);
    return resized;
}

cv::Mat DataProcessor::extractHOGFeatures(const cv::Mat& input)
{
    // 转换为灰度图
    cv::Mat gray;
    if (input.channels() == 3)
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    else
        gray = input.clone();
        
    // 调整大小为标准尺寸
    cv::Mat resized;
    cv::resize(gray, resized, cv::Size(64, 128));
    
    // 计算HOG特征
    cv::HOGDescriptor hog;
    std::vector<float> descriptors;
    hog.compute(resized, descriptors);
    
    // 转换为Mat格式
    cv::Mat hogFeatures(descriptors, true);
    return hogFeatures;
}

std::vector<cv::KeyPoint> DataProcessor::detectKeypoints(const cv::Mat& input, cv::Mat& descriptors)
{
    std::vector<cv::KeyPoint> keypoints;
    siftDetector_->detectAndCompute(input, cv::Mat(), keypoints, descriptors);
    return keypoints;
}

cv::Mat DataProcessor::adjustBrightness(const cv::Mat& input, double alpha)
{
    cv::Mat adjusted = input.clone();
    adjusted.convertTo(adjusted, -1, 1, alpha);
    return adjusted;
}

cv::Mat DataProcessor::adjustContrast(const cv::Mat& input, double beta)
{
    cv::Mat adjusted = input.clone();
    adjusted.convertTo(adjusted, -1, beta, 0);
    return adjusted;
}

cv::Mat DataProcessor::addNoise(const cv::Mat& input, double mean = 0, double stddev = 25)
{
    cv::Mat noise = cv::Mat::zeros(input.size(), CV_32F);
    cv::randn(noise, mean, stddev);
    
    cv::Mat noisy;
    input.convertTo(noisy, CV_32F);
    noisy += noise;
    
    // 确保像素值在有效范围内
    cv::normalize(noisy, noisy, 0, 255, cv::NORM_MINMAX);
    noisy.convertTo(noisy, input.type());
    
    return noisy;
}

cv::Mat DataProcessor::randomRotate(const cv::Mat& input, double maxAngle = 30)
{
    double angle = randomDouble(-maxAngle, maxAngle);
    cv::Point2f center(input.cols/2.0f, input.rows/2.0f);
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, angle, 1.0);
    
    cv::Mat rotated;
    cv::warpAffine(input, rotated, rotationMatrix, input.size());
    return rotated;
}

cv::Mat DataProcessor::randomFlip(const cv::Mat& input)
{
    cv::Mat flipped;
    if (randomInt(0, 1) == 0)
        cv::flip(input, flipped, 1);  // 水平翻转
    else
        cv::flip(input, flipped, 0);  // 垂直翻转
    return flipped;
}

cv::Mat DataProcessor::randomCrop(const cv::Mat& input, double scale = 0.8)
{
    int width = static_cast<int>(input.cols * scale);
    int height = static_cast<int>(input.rows * scale);
    
    int x = randomInt(0, input.cols - width);
    int y = randomInt(0, input.rows - height);
    
    cv::Rect roi(x, y, width, height);
    cv::Mat cropped = input(roi).clone();
    
    cv::resize(cropped, cropped, input.size());
    return cropped;
}

std::vector<cv::Mat> DataProcessor::applyAugmentation(const cv::Mat& input, int numAugmentations = 5)
{
    std::vector<cv::Mat> augmented;
    augmented.reserve(numAugmentations);
    
    for (int i = 0; i < numAugmentations; ++i) {
        cv::Mat current = input.clone();
        
        // 随机应用数据增强
        if (randomDouble(0, 1) > 0.5)
            current = adjustBrightness(current, randomDouble(-50, 50));
            
        if (randomDouble(0, 1) > 0.5)
            current = adjustContrast(current, randomDouble(0.5, 1.5));
            
        if (randomDouble(0, 1) > 0.5)
            current = addNoise(current);
            
        if (randomDouble(0, 1) > 0.5)
            current = randomRotate(current);
            
        if (randomDouble(0, 1) > 0.5)
            current = randomFlip(current);
            
        if (randomDouble(0, 1) > 0.5)
            current = randomCrop(current);
            
        augmented.push_back(current);
    }
    
    return augmented;
}

double DataProcessor::randomDouble(double min, double max)
{
    std::uniform_real_distribution<double> dist(min, max);
    return dist(rng_);
}

int DataProcessor::randomInt(int min, int max)
{
    std::uniform_int_distribution<int> dist(min, max);
    return dist(rng_);
}
