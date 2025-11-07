#include "DataProcessor.h"
#include <QDebug>

using namespace std;
using namespace cv;

DataProcessor::DataProcessor(QObject *parent)
    : QObject(parent)
    , rng_(random_device{}())
    , currentFeatureType_(FeatureType::SIFT)
{
    InitializeDetectors();
}

void DataProcessor::InitializeDetectors()
{
    try {
        siftDetector_ = SIFT::create();
    } catch (const cv::Exception& e) {
        qDebug() << "警告: 无法初始化SIFT检测器:" << e.what();
    }
    

    
    try {
        orbDetector_ = ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);
    } catch (const cv::Exception& e) {
        qDebug() << "警告: 无法初始化ORB检测器:" << e.what();
    }
    
    try {
        akazeDetector_ = AKAZE::create();
    } catch (const cv::Exception& e) {
        qDebug() << "警告: 无法初始化AKAZE检测器:" << e.what();
    }
}

void DataProcessor::SetFeatureType(FeatureType type)
{
    currentFeatureType_ = type;
}

FeatureType DataProcessor::GetFeatureType() const
{
    return currentFeatureType_;
}

Mat DataProcessor::NormalizeImage(const Mat& input, double targetMean, double targetStd)
{
    // 定义局部变量
    Mat normalized;         // 标准化后的图像
    Scalar mean;            // 均值
    Scalar stddev;          // 标准差

    // 转换为浮点型
    input.convertTo(normalized, CV_32F);
    
    // 计算当前均值和标准差
    meanStdDev(normalized, mean, stddev);
    
    // 标准化处理
    normalized = (normalized - mean[0]) / stddev[0];
    
    // 调整到目标均值和标准差
    normalized = normalized * targetStd + targetMean;
    
    return normalized;
}

Mat DataProcessor::StandardizeImage(const Mat& input)
{
    // 定义局部变量
    double maxVal = 0;      // 图像最大像素值
    double minVal = 0;      // 图像最小像素值
    Mat standardized;       // 标准化后的图像

    // 计算图像的最小和最大像素值
    minMaxLoc(input, &minVal, &maxVal);
    
    // 标准化处理: 将像素值映射到0-255范围
    input.convertTo(standardized, CV_8U, 255.0 / (maxVal - minVal), 0);
    
    return standardized;
}

Mat DataProcessor::ResizeWithAspectRatio(const Mat& input, int targetSize)
{
    // 定义局部变量
    Mat resized;        // 调整大小后的图像
    double ratio;       // 缩放比例

    // 计算缩放比例，保持宽高比
    ratio = static_cast<double>(targetSize) / max(input.rows, input.cols);
    
    // 执行图像缩放
    resize(input, resized, Size(), ratio, ratio, INTER_AREA);
    
    return resized;
}

Mat DataProcessor::ExtractHOGFeatures(const Mat& input)
{
    // 定义局部变量
    Mat gray;                   // 灰度图像
    Mat resized;                // 调整大小后的图像
    HOGDescriptor hog;          // HOG描述符对象
    vector<float> descriptors;  // HOG特征描述符向量
    Mat hogFeatures;            // 最终HOG特征矩阵

    // 转换为灰度图
    if (input.channels() == 3)
    {
        cvtColor(input, gray, COLOR_BGR2GRAY);
    }
    else
    {
        gray = input.clone();
    }
        
    // 调整大小为标准尺寸
    resize(gray, resized, Size(64, 128));
    
    // 计算HOG特征
    hog.compute(resized, descriptors);
    
    // 转换为Mat格式
    hogFeatures = Mat(descriptors, true);
    return hogFeatures;
}

vector<KeyPoint> DataProcessor::DetectKeypoints(const Mat& input, Mat& descriptors)
{
    // 定义局部变量
    vector<KeyPoint> keypoints;  // 检测到的关键点集合

    try {
        // 根据当前选择的特征提取器类型进行检测
        if (currentFeatureType_ == FeatureType::ORB)
        {
            if (orbDetector_) {
                // 使用ORB检测器检测关键点并计算描述符
                orbDetector_->detectAndCompute(input, Mat(), keypoints, descriptors);
            } else {
                qDebug() << "错误: ORB检测器未初始化";
            }
        }
        else if (currentFeatureType_ == FeatureType::AKAZE)
        {
            if (akazeDetector_) {
                // 使用AKAZE检测器检测关键点并计算描述符
                akazeDetector_->detectAndCompute(input, Mat(), keypoints, descriptors);
            } else {
                qDebug() << "错误: AKAZE检测器未初始化";
            }
        }
        else
        {
            if (siftDetector_) {
                // 默认使用SIFT检测器
                siftDetector_->detectAndCompute(input, Mat(), keypoints, descriptors);
            } else {
                qDebug() << "错误: SIFT检测器未初始化";
            }
        }
    } catch (const cv::Exception& e) {
        qDebug() << "特征提取过程中发生异常:" << e.what();
        // 清空返回值，避免使用无效数据
        keypoints.clear();
        descriptors = Mat();
    }
    
    return keypoints;
}

Mat DataProcessor::AdjustBrightness(const Mat& input, double alpha)
{
    // 定义局部变量
    Mat adjusted;   // 调整亮度后的图像

    // 克隆原始图像
    adjusted = input.clone();
    
    // 调整亮度: alpha控制亮度偏移量
    adjusted.convertTo(adjusted, -1, 1, alpha);
    
    return adjusted;
}

Mat DataProcessor::AdjustContrast(const Mat& input, double beta)
{
    // 定义局部变量
    Mat adjusted;   // 调整对比度后的图像

    // 克隆原始图像
    adjusted = input.clone();
    
    // 调整对比度: beta控制对比度比例
    adjusted.convertTo(adjusted, -1, beta, 0);
    
    return adjusted;
}

Mat DataProcessor::AddNoise(const Mat& input, double mean = 0, double stddev = 25)
{
    // 定义局部变量
    Mat noise;      // 噪声矩阵
    Mat noisy;      // 添加噪声后的图像

    // 创建与输入图像相同大小的噪声矩阵
    noise = Mat::zeros(input.size(), CV_32F);
    
    // 生成高斯噪声
    randn(noise, mean, stddev);
    
    // 将输入图像转换为浮点型并添加噪声
    input.convertTo(noisy, CV_32F);
    noisy += noise;
    
    // 确保像素值在有效范围内
    normalize(noisy, noisy, 0, 255, NORM_MINMAX);
    noisy.convertTo(noisy, input.type());
    
    return noisy;
}

Mat DataProcessor::RandomRotate(const Mat& input, double maxAngle = 30)
{
    // 定义局部变量
    double angle;               // 旋转角度
    Point2f center;             // 旋转中心
    Mat rotationMatrix;         // 旋转矩阵
    Mat rotated;                // 旋转后的图像

    // 生成随机旋转角度
    angle = RandomDouble(-maxAngle, maxAngle);
    
    // 计算图像中心点
    center = Point2f(input.cols/2.0f, input.rows/2.0f);
    
    // 生成旋转矩阵
    rotationMatrix = getRotationMatrix2D(center, angle, 1.0);
    
    // 执行图像旋转
    warpAffine(input, rotated, rotationMatrix, input.size());
    
    return rotated;
}

Mat DataProcessor::RandomFlip(const Mat& input)
{
    // 定义局部变量
    Mat flipped;        // 翻转后的图像
    int flipCode;       // 翻转方向代码

    // 随机选择翻转方向
    if (RandomInt(0, 1) == 0)
    {
        flipCode = 1;  // 水平翻转
    }
    else
    {
        flipCode = 0;  // 垂直翻转
    }
    
    // 执行图像翻转
    flip(input, flipped, flipCode);
    
    return flipped;
}

Mat DataProcessor::RandomCrop(const Mat& input, double scale = 0.8)
{
    // 定义局部变量
    int width;              // 裁剪宽度
    int height;             // 裁剪高度
    int x;                  // 裁剪起始X坐标
    int y;                  // 裁剪起始Y坐标
    Rect roi;               // 感兴趣区域
    Mat cropped;            // 裁剪后的图像

    // 计算裁剪尺寸
    width = static_cast<int>(input.cols * scale);
    height = static_cast<int>(input.rows * scale);
    
    // 随机生成裁剪起始位置
    x = RandomInt(0, input.cols - width);
    y = RandomInt(0, input.rows - height);
    
    // 定义裁剪区域
    roi = Rect(x, y, width, height);
    
    // 执行裁剪并克隆图像
    cropped = input(roi).clone();
    
    // 调整裁剪后的图像到原始尺寸
    resize(cropped, cropped, input.size());
    
    return cropped;
}

vector<Mat> DataProcessor::ApplyAugmentation(const Mat& input, int numAugmentations = 5)
{
    // 定义局部变量
    vector<Mat> augmented;  // 增强后的图像集合
    Mat current;            // 当前处理的图像

    // 预分配内存空间
    augmented.reserve(numAugmentations);
    
    // 生成多个增强版本
    for (int i = 0; i < numAugmentations; ++i) 
    {
        // 克隆原始图像
        current = input.clone();
        
        // 随机应用数据增强操作
        // 亮度调整 (50%概率) 
        if (RandomDouble(0, 1) > 0.5)
        {
            current = AdjustBrightness(current, RandomDouble(-50, 50));
        }
            
        // 对比度调整 (50%概率) 
        if (RandomDouble(0, 1) > 0.5)
        {
            current = AdjustContrast(current, RandomDouble(0.5, 1.5));
        }
            
        // 添加噪声 (50%概率) 
        if (RandomDouble(0, 1) > 0.5)
        {
            current = AddNoise(current);
        }
            
        // 随机旋转 (50%概率) 
        if (RandomDouble(0, 1) > 0.5)
        {
            current = RandomRotate(current);
        }
            
        // 随机翻转 (50%概率) 
        if (RandomDouble(0, 1) > 0.5)
        {
            current = RandomFlip(current);
        }
            
        // 随机裁剪 (50%概率) 
        if (RandomDouble(0, 1) > 0.5)
        {
            current = RandomCrop(current);
        }
            
        // 添加到结果集合
        augmented.push_back(current);
    }
    
    return augmented;
}

double DataProcessor::RandomDouble(double min, double max)
{
    // 定义局部变量
    uniform_real_distribution<double> dist(min, max);  // 均匀实数分布对象

    // 生成随机浮点数
    return dist(rng_);
}

int DataProcessor::RandomInt(int min, int max)
{
    // 定义局部变量
    uniform_int_distribution<int> dist(min, max);  // 均匀整数分布对象

    // 生成随机整数
    return dist(rng_);
}
