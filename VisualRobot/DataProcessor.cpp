#include "DataProcessor.h"
#include <QDebug>

using namespace std;
using namespace cv;

/**
 * @brief DataProcessor构造函数
 * @param parent 父对象指针
 * 
 * 初始化随机数生成器和SIFT特征检测器
 */
DataProcessor::DataProcessor(QObject *parent)
    : QObject(parent)
    , rng_(random_device{}())
{
    siftDetector_ = SIFT::create();
}

/**
 * @brief 图像归一化处理
 * @param input 输入图像
 * @param targetMean 目标均值
 * @param targetStd 目标标准差
 * @return 归一化后的图像
 * 
 * 将图像像素值归一化到指定的均值和标准差
 * 处理步骤：
 * 1. 转换为浮点型
 * 2. 计算当前均值和标准差
 * 3. 标准化处理
 * 4. 调整到目标均值和标准差
 */
Mat DataProcessor::NormalizeImage(const Mat& input, double targetMean, double targetStd)
{
    // 定义局部变量
    Mat normalized;         ///< 标准化后的图像
    Scalar mean;            ///< 均值
    Scalar stddev;          ///< 标准差

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

/**
 * @brief 图像标准化处理
 * @param input 输入图像
 * @return 标准化后的图像（像素值映射到0-255范围）
 * 
 * 将图像像素值线性映射到0-255范围
 * 处理步骤：
 * 1. 计算图像的最小和最大像素值
 * 2. 线性映射到0-255范围
 */
Mat DataProcessor::StandardizeImage(const Mat& input)
{
    // 定义局部变量
    double maxVal = 0;      ///< 图像最大像素值
    double minVal = 0;      ///< 图像最小像素值
    Mat standardized;       ///< 标准化后的图像

    // 计算图像的最小和最大像素值
    minMaxLoc(input, &minVal, &maxVal);
    
    // 标准化处理: 将像素值映射到0-255范围
    input.convertTo(standardized, CV_8U, 255.0 / (maxVal - minVal), 0);
    
    return standardized;
}

/**
 * @brief 保持宽高比的图像缩放
 * @param input 输入图像
 * @param targetSize 目标尺寸（最长边）
 * @return 缩放后的图像
 * 
 * 保持图像宽高比进行缩放，最长边缩放到目标尺寸
 * 使用INTER_AREA插值方法，适合缩小图像
 */
Mat DataProcessor::ResizeWithAspectRatio(const Mat& input, int targetSize)
{
    // 定义局部变量
    Mat resized;        ///< 调整大小后的图像
    double ratio;       ///< 缩放比例

    // 计算缩放比例，保持宽高比
    ratio = static_cast<double>(targetSize) / max(input.rows, input.cols);
    
    // 执行图像缩放
    resize(input, resized, Size(), ratio, ratio, INTER_AREA);
    
    return resized;
}

/**
 * @brief 提取HOG特征
 * @param input 输入图像
 * @return HOG特征矩阵
 * 
 * 提取图像的HOG（方向梯度直方图）特征
 * 处理步骤：
 * 1. 转换为灰度图
 * 2. 调整大小到标准尺寸64x128
 * 3. 计算HOG特征
 * 4. 转换为Mat格式返回
 */
Mat DataProcessor::ExtractHOGFeatures(const Mat& input)
{
    // 定义局部变量
    Mat gray;                   ///< 灰度图像
    Mat resized;                ///< 调整大小后的图像
    HOGDescriptor hog;          ///< HOG描述符对象
    vector<float> descriptors;  ///< HOG特征描述符向量
    Mat hogFeatures;            ///< 最终HOG特征矩阵

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

/**
 * @brief 检测关键点并计算描述符
 * @param input 输入图像
 * @param descriptors 输出描述符矩阵
 * @return 检测到的关键点集合
 * 
 * 使用SIFT算法检测图像中的关键点并计算描述符
 * 关键点包含位置、尺度、方向等信息
 * 描述符用于特征匹配和识别
 */
vector<KeyPoint> DataProcessor::DetectKeypoints(const Mat& input, Mat& descriptors)
{
    // 定义局部变量
    vector<KeyPoint> keypoints;  ///< 检测到的关键点集合

    // 使用SIFT检测器检测关键点并计算描述符
    siftDetector_->detectAndCompute(input, Mat(), keypoints, descriptors);
    
    return keypoints;
}

/**
 * @brief 调整图像亮度
 * @param input 输入图像
 * @param alpha 亮度偏移量
 * @return 调整亮度后的图像
 * 
 * 通过像素值偏移调整图像亮度
 * 正值增加亮度，负值降低亮度
 */
Mat DataProcessor::AdjustBrightness(const Mat& input, double alpha)
{
    // 定义局部变量
    Mat adjusted;   ///< 调整亮度后的图像

    // 克隆原始图像
    adjusted = input.clone();
    
    // 调整亮度: alpha控制亮度偏移量
    adjusted.convertTo(adjusted, -1, 1, alpha);
    
    return adjusted;
}

/**
 * @brief 调整图像对比度
 * @param input 输入图像
 * @param beta 对比度比例因子
 * @return 调整对比度后的图像
 * 
 * 通过像素值缩放调整图像对比度
 * 大于1的值增加对比度，小于1的值降低对比度
 */
Mat DataProcessor::AdjustContrast(const Mat& input, double beta)
{
    // 定义局部变量
    Mat adjusted;   ///< 调整对比度后的图像

    // 克隆原始图像
    adjusted = input.clone();
    
    // 调整对比度: beta控制对比度比例
    adjusted.convertTo(adjusted, -1, beta, 0);
    
    return adjusted;
}

/**
 * @brief 添加高斯噪声
 * @param input 输入图像
 * @param mean 噪声均值
 * @param stddev 噪声标准差
 * @return 添加噪声后的图像
 * 
 * 向图像添加高斯噪声，模拟真实环境中的噪声
 * 处理步骤：
 * 1. 创建高斯噪声矩阵
 * 2. 将图像转换为浮点型
 * 3. 添加噪声
 * 4. 确保像素值在有效范围内
 */
Mat DataProcessor::AddNoise(const Mat& input, double mean = 0, double stddev = 25)
{
    // 定义局部变量
    Mat noise;      ///< 噪声矩阵
    Mat noisy;      ///< 添加噪声后的图像

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

/**
 * @brief 随机旋转图像
 * @param input 输入图像
 * @param maxAngle 最大旋转角度（度）
 * @return 旋转后的图像
 * 
 * 随机旋转图像，角度范围在[-maxAngle, maxAngle]之间
 * 处理步骤：
 * 1. 生成随机旋转角度
 * 2. 计算图像中心点
 * 3. 生成旋转矩阵
 * 4. 执行仿射变换
 */
Mat DataProcessor::RandomRotate(const Mat& input, double maxAngle = 30)
{
    // 定义局部变量
    double angle;               ///< 旋转角度
    Point2f center;             ///< 旋转中心
    Mat rotationMatrix;         ///< 旋转矩阵
    Mat rotated;                ///< 旋转后的图像

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

/**
 * @brief 随机翻转图像
 * @param input 输入图像
 * @return 翻转后的图像
 * 
 * 随机选择水平或垂直方向翻转图像
 * 水平翻转：左右镜像
 * 垂直翻转：上下镜像
 */
Mat DataProcessor::RandomFlip(const Mat& input)
{
    // 定义局部变量
    Mat flipped;        ///< 翻转后的图像
    int flipCode;       ///< 翻转方向代码

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

/**
 * @brief 随机裁剪图像
 * @param input 输入图像
 * @param scale 裁剪比例（0-1之间）
 * @return 裁剪并缩放后的图像
 * 
 * 随机裁剪图像的一部分区域，然后缩放到原始尺寸
 * 处理步骤：
 * 1. 计算裁剪尺寸
 * 2. 随机生成裁剪起始位置
 * 3. 定义裁剪区域
 * 4. 执行裁剪并缩放到原始尺寸
 */
Mat DataProcessor::RandomCrop(const Mat& input, double scale = 0.8)
{
    // 定义局部变量
    int width;              ///< 裁剪宽度
    int height;             ///< 裁剪高度
    int x;                  ///< 裁剪起始X坐标
    int y;                  ///< 裁剪起始Y坐标
    Rect roi;               ///< 感兴趣区域
    Mat cropped;            ///< 裁剪后的图像

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

/**
 * @brief 应用多种数据增强操作
 * @param input 输入图像
 * @param numAugmentations 增强图像数量
 * @return 增强后的图像集合
 * 
 * 对输入图像应用多种随机数据增强操作，生成多个增强版本
 * 每个增强版本随机应用以下操作（50%概率）：
 * - 亮度调整
 * - 对比度调整
 * - 添加噪声
 * - 随机旋转
 * - 随机翻转
 * - 随机裁剪
 */
vector<Mat> DataProcessor::ApplyAugmentation(const Mat& input, int numAugmentations = 5)
{
    // 定义局部变量
    vector<Mat> augmented;  ///< 增强后的图像集合
    Mat current;            ///< 当前处理的图像

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

/**
 * @brief 生成指定范围内的随机浮点数
 * @param min 最小值
 * @param max 最大值
 * @return 随机浮点数
 * 
 * 使用均匀分布生成[min, max]范围内的随机浮点数
 */
double DataProcessor::RandomDouble(double min, double max)
{
    // 定义局部变量
    uniform_real_distribution<double> dist(min, max);  ///< 均匀实数分布对象

    // 生成随机浮点数
    return dist(rng_);
}

/**
 * @brief 生成指定范围内的随机整数
 * @param min 最小值
 * @param max 最大值
 * @return 随机整数
 * 
 * 使用均匀分布生成[min, max]范围内的随机整数
 */
int DataProcessor::RandomInt(int min, int max)
{
    // 定义局部变量
    uniform_int_distribution<int> dist(min, max);  ///< 均匀整数分布对象

    // 生成随机整数
    return dist(rng_);
}
