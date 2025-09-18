#include "DataProcessor.h"
#include <QDebug>

using namespace std;
using namespace cv;

DataProcessor::DataProcessor(QObject *parent)
    : QObject(parent)
    , rng_(random_device{}())
{
    siftDetector_ = SIFT::create();
}

Mat DataProcessor::normalizeImage(const Mat& input, double targetMean, double targetStd)
{
    Mat normalized;
    input.convertTo(normalized, CV_32F);
    
    // 计算当前均值和标准差
    Scalar mean, stddev;
    meanStdDev(normalized, mean, stddev);
    
    // 标准化处理
    normalized = (normalized - mean[0]) / stddev[0];
    
    // 调整到目标均值和标准差
    normalized = normalized * targetStd + targetMean;
    
    return normalized;
}

Mat DataProcessor::standardizeImage(const Mat& input)
{
    double maxVal = 0;
    double minVal = 0;
    minMaxLoc(input, &minVal, &maxVal);
    Mat standardized;
    input.convertTo(standardized, CV_8U, 255.0 / (maxVal - minVal), 0);
    return standardized;
}

Mat DataProcessor::resizeWithAspectRatio(const Mat& input, int targetSize)
{
    Mat resized;
    double ratio = static_cast<double>(targetSize) / max(input.rows, input.cols);
    resize(input, resized, Size(), ratio, ratio, INTER_AREA);
    return resized;
}

Mat DataProcessor::extractHOGFeatures(const Mat& input)
{
    // 转换为灰度图
    Mat gray;
    if (input.channels() == 3)
    {
        cvtColor(input, gray, COLOR_BGR2GRAY);
    }
    else
    {
        gray = input.clone();
    }
        
    // 调整大小为标准尺寸
    Mat resized;
    resize(gray, resized, Size(64, 128));
    
    // 计算HOG特征
    HOGDescriptor hog;
    vector<float> descriptors;
    hog.compute(resized, descriptors);
    
    // 转换为Mat格式
    Mat hogFeatures(descriptors, true);
    return hogFeatures;
}

vector<KeyPoint> DataProcessor::detectKeypoints(const Mat& input, Mat& descriptors)
{
    vector<KeyPoint> keypoints;
    siftDetector_->detectAndCompute(input, Mat(), keypoints, descriptors);
    return keypoints;
}

Mat DataProcessor::adjustBrightness(const Mat& input, double alpha)
{
    Mat adjusted = input.clone();
    adjusted.convertTo(adjusted, -1, 1, alpha);
    return adjusted;
}

Mat DataProcessor::adjustContrast(const Mat& input, double beta)
{
    Mat adjusted = input.clone();
    adjusted.convertTo(adjusted, -1, beta, 0);
    return adjusted;
}

Mat DataProcessor::addNoise(const Mat& input, double mean = 0, double stddev = 25)
{
    Mat noise = Mat::zeros(input.size(), CV_32F);
    randn(noise, mean, stddev);
    
    Mat noisy;
    input.convertTo(noisy, CV_32F);
    noisy += noise;
    
    // 确保像素值在有效范围内
    normalize(noisy, noisy, 0, 255, NORM_MINMAX);
    noisy.convertTo(noisy, input.type());
    
    return noisy;
}

Mat DataProcessor::randomRotate(const Mat& input, double maxAngle = 30)
{
    double angle = randomDouble(-maxAngle, maxAngle);
    Point2f center(input.cols/2.0f, input.rows/2.0f);
    Mat rotationMatrix = getRotationMatrix2D(center, angle, 1.0);
    
    Mat rotated;
    warpAffine(input, rotated, rotationMatrix, input.size());
    return rotated;
}

Mat DataProcessor::randomFlip(const Mat& input)
{
    Mat flipped;
    if (randomInt(0, 1) == 0)
    {
        flip(input, flipped, 1);  // 水平翻转
    }
    else
    {
        flip(input, flipped, 0);  // 垂直翻转
    }
    return flipped;
}

Mat DataProcessor::randomCrop(const Mat& input, double scale = 0.8)
{
    int width = static_cast<int>(input.cols * scale);
    int height = static_cast<int>(input.rows * scale);
    
    int x = randomInt(0, input.cols - width);
    int y = randomInt(0, input.rows - height);
    
    Rect roi(x, y, width, height);
    Mat cropped = input(roi).clone();
    
    resize(cropped, cropped, input.size());
    return cropped;
}

vector<Mat> DataProcessor::applyAugmentation(const Mat& input, int numAugmentations = 5)
{
    vector<Mat> augmented;
    augmented.reserve(numAugmentations);
    
    for (int i = 0; i < numAugmentations; ++i) 
    {
        Mat current = input.clone();
        
        // 随机应用数据增强
        if (randomDouble(0, 1) > 0.5)
        {
            current = adjustBrightness(current, randomDouble(-50, 50));
        }
            
        if (randomDouble(0, 1) > 0.5)
        {
            current = adjustContrast(current, randomDouble(0.5, 1.5));
        }
            
        if (randomDouble(0, 1) > 0.5)
        {
            current = addNoise(current);
        }
            
        if (randomDouble(0, 1) > 0.5)
        {
            current = randomRotate(current);
        }
            
        if (randomDouble(0, 1) > 0.5)
        {
            current = randomFlip(current);
        }
            
        if (randomDouble(0, 1) > 0.5)
        {
            current = randomCrop(current);
        }
            
        augmented.push_back(current);
    }
    
    return augmented;
}

double DataProcessor::randomDouble(double min, double max)
{
    uniform_real_distribution<double> dist(min, max);
    return dist(rng_);
}

int DataProcessor::randomInt(int min, int max)
{
    uniform_int_distribution<int> dist(min, max);
    return dist(rng_);
}
