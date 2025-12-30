/**
 * @file DataProcessor.h
 * @brief 数据处理器模块头文件
 * 
 * 该模块提供图像预处理、特征提取和数据增强功能，支持多种特征提取算法
 * 
 * @author VisualRobot Team
 * @date 2025-12-29
 * @version 1.0
 * @see DataProcessor
 */

#ifndef DATAPROCESSOR_H
#define DATAPROCESSOR_H

#include <opencv2/opencv.hpp>
#include <QObject>
#include <random>

using namespace cv;
using namespace std;

/**
 * @brief 特征提取器类型枚举
 * 
 * 定义支持的特征提取算法类型
 */
enum class FeatureType
{
    SIFT,   ///< SIFT特征提取器，适用于高精度特征匹配
    ORB,    ///< ORB特征提取器，适用于实时应用
    AKAZE   ///< AKAZE特征提取器，适用于非线性尺度空间
};

/**
 * @brief 数据处理器类
 * 
 * 提供图像预处理、特征提取和数据增强功能，支持多种特征提取算法
 * 
 * 该类封装了OpenCV的图像处理功能，包括：
 * - 图像归一化和标准化
 * - HOG特征提取
 * - 多种特征点检测（SIFT、ORB、AKAZE）
 * - 图像亮度、对比度调整
 * - 高斯噪声添加
 * - 随机数据增强（旋转、翻转、裁剪）
 * 
 * @note 所有图像处理方法都支持多通道图像输入
 * @see FeatureType
 */
class DataProcessor : public QObject
{
    Q_OBJECT

public:
    /**
     * @brief 构造函数
     * 
     * 初始化数据处理器，设置默认特征提取器类型为SIFT，并初始化所有特征检测器
     * 
     * @param parent 父对象指针，默认为nullptr
     * @note 初始化过程会创建SIFT、ORB和AKAZE检测器实例
     * @warning 可能因OpenCV版本或依赖库缺失导致部分检测器初始化失败
     */
    explicit DataProcessor(QObject *parent = nullptr);
    
    /**
     * @brief 设置特征提取器类型
     * 
     * 切换当前使用的特征提取算法
     * 
     * @param type 特征提取器类型（SIFT、ORB或AKAZE）
     * @see GetFeatureType()
     * @see DetectKeypoints()
     */
    void SetFeatureType(FeatureType type);
    
    /**
     * @brief 获取当前特征提取器类型
     * 
     * 查询当前激活的特征提取算法
     * 
     * @return FeatureType 当前使用的特征提取器类型
     * @see SetFeatureType()
     */
    FeatureType GetFeatureType() const;
    
    /**
     * @brief 图像归一化处理
     * 
     * 将图像像素值归一化到指定的均值和标准差
     * 
     * @param input 输入图像（支持多通道）
     * @param targetMean 目标均值
     * @param targetStd 目标标准差
     * @return Mat 归一化后的图像（CV_32F类型）
     * @note 处理流程：先计算当前均值标准差 → 标准化 → 调整到目标分布
     * @see StandardizeImage()
     */
    Mat NormalizeImage(const Mat& input, double targetMean, double targetStd);
    
    /**
     * @brief 图像标准化处理
     * 
     * 将图像像素值线性映射到0-255范围
     * 
     * @param input 输入图像
     * @return Mat 标准化后的图像（CV_8U类型）
     * @note 使用公式：output = 255 * (input - min) / (max - min)
     * @see NormalizeImage()
     */
    Mat StandardizeImage(const Mat& input);
    
    /**
     * @brief 保持宽高比调整图像大小
     * 
     * 按照指定的目标大小，保持图像原始宽高比进行等比例缩放
     * 
     * @param input 输入图像
     * @param targetSize 目标最大尺寸（宽度或高度中的较大值）
     * @return Mat 调整大小后的图像
     * @note 使用INTER_AREA插值方法，适合下采样
     * @warning 图像尺寸可能小于targetSize，但不会超过
     */
    Mat ResizeWithAspectRatio(const Mat& input, int targetSize);
    
    /**
     * @brief 提取HOG特征
     * 
     * 从输入图像中提取HOG（方向梯度直方图）特征
     * 
     * @param input 输入图像（自动转换为灰度）
     * @return Mat HOG特征向量（一维矩阵）
     * @note 处理流程：灰度转换 → 调整为64x128 → HOG特征计算
     * @warning 输出特征为一维向量，需要根据实际应用进行reshape
     */
    Mat ExtractHOGFeatures(const Mat& input);
    
    /**
     * @brief 检测特征点并计算描述符
     * 
     * 根据当前设置的特征提取器类型，检测图像中的特征点并计算相应的描述符
     * 
     * @param input 输入图像（单通道灰度图效果最佳）
     * @param descriptors 输出的特征描述符矩阵
     * @return vector<KeyPoint> 检测到的特征点集合
     * @note 支持SIFT、ORB、AKAZE三种特征提取器
     * @see SetFeatureType()
     * @see GetFeatureType()
     */
    vector<KeyPoint> DetectKeypoints(const Mat& input, Mat& descriptors);
    
    /**
     * @brief 调整图像亮度
     * 
     * 线性调整图像的亮度值
     * 
     * @param input 输入图像
     * @param alpha 亮度调整值，正值增加亮度，负值降低亮度
     * @return Mat 调整亮度后的图像
     * @note 使用公式：output = input + alpha
     * @see AdjustContrast()
     */
    Mat AdjustBrightness(const Mat& input, double alpha);
    
    /**
     * @brief 调整图像对比度
     * 
     * 线性调整图像的对比度
     * 
     * @param input 输入图像
     * @param beta 对比度调整值，大于1增加对比度，小于1降低对比度
     * @return Mat 调整对比度后的图像
     * @note 使用公式：output = input * beta
     * @see AdjustBrightness()
     */
    Mat AdjustContrast(const Mat& input, double beta);
    
    /**
     * @brief 为图像添加高斯噪声
     * 
     * 在图像上叠加高斯分布的随机噪声
     * 
     * @param input 输入图像
     * @param mean 噪声均值，默认为0
     * @param stddev 噪声标准差，默认为25
     * @return Mat 添加噪声后的图像
     * @note 噪声添加后会进行归一化处理确保像素值在0-255范围
     * @see RandomRotate()
     */
    Mat AddNoise(const Mat& input, double mean = 0, double stddev = 25);
    
    /**
     * @brief 随机旋转图像
     * 
     * 在指定角度范围内随机旋转图像
     * 
     * @param input 输入图像
     * @param maxAngle 最大旋转角度（正负），默认为30度
     * @return Mat 旋转后的图像
     * @note 旋转角度在[-maxAngle, maxAngle]范围内随机选择
     * @see RandomFlip()
     */
    Mat RandomRotate(const Mat& input, double maxAngle = 30);
    
    /**
     * @brief 随机翻转图像
     * 
     * 随机选择水平翻转或垂直翻转
     * 
     * @param input 输入图像
     * @return Mat 翻转后的图像
     * @note 50%概率水平翻转，50%概率垂直翻转
     * @see RandomRotate()
     */
    Mat RandomFlip(const Mat& input);
    
    /**
     * @brief 随机裁剪图像
     * 
     * 按指定比例随机裁剪图像并调整回原始尺寸
     * 
     * @param input 输入图像
     * @param scale 裁剪比例，默认为0.8
     * @return Mat 裁剪并调整大小后的图像
     * @note 裁剪区域在图像内部随机选择
     * @see ApplyAugmentation()
     */
    Mat RandomCrop(const Mat& input, double scale = 0.8);
    
    /**
     * @brief 批量应用数据增强
     * 
     * 对输入图像应用多种随机数据增强操作，生成多个增强版本
     * 
     * @param input 输入图像
     * @param numAugmentations 增强版本数量，默认为5
     * @return vector<Mat> 增强后的图像集合
     * @note 每个增强版本随机应用亮度、对比度调整和翻转操作
     * @see RandomCrop()
     */
    vector<Mat> ApplyAugmentation(const Mat& input, int numAugmentations = 5);

private:
    mt19937 rng_;                      ///< 随机数生成器（Mersenne Twister算法）
    Ptr<SIFT> siftDetector_;            ///< SIFT特征检测器实例
    Ptr<ORB> orbDetector_;              ///< ORB特征检测器实例
    Ptr<AKAZE> akazeDetector_;          ///< AKAZE特征检测器实例
    FeatureType currentFeatureType_;    ///< 当前特征提取器类型
    
    /**
     * @brief 初始化特征检测器
     * 
     * 创建并初始化所有支持的特征检测器实例
     * 
     * @note 会尝试初始化SIFT、ORB和AKAZE检测器
     * @warning 可能因OpenCV版本或依赖库问题导致部分检测器初始化失败
     * @see DataProcessor()
     */
    void InitializeDetectors();
    
    /**
     * @brief 生成指定范围内的随机浮点数
     * 
     * 使用均匀分布生成指定范围内的随机浮点数
     * 
     * @param min 最小值
     * @param max 最大值
     * @return double 随机浮点数
     * @note 使用std::uniform_real_distribution
     * @see RandomInt()
     */
    double RandomDouble(double min, double max);
    
    /**
     * @brief 生成指定范围内的随机整数
     * 
     * 使用均匀分布生成指定范围内的随机整数
     * 
     * @param min 最小值（包含）
     * @param max 最大值（包含）
     * @return int 随机整数
     * @note 使用std::uniform_int_distribution
     * @see RandomDouble()
     */
    int RandomInt(int min, int max);
};

#endif // DATAPROCESSOR_H
