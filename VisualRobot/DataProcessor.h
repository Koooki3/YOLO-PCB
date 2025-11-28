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
    SIFT,   // SIFT特征提取器，适用于高精度特征匹配
    ORB,    // ORB特征提取器，适用于实时应用
    AKAZE   // AKAZE特征提取器，适用于非线性尺度空间
};

/**
 * @brief 数据处理器类
 * 
 * 提供图像预处理、特征提取和数据增强功能，支持多种特征提取算法
 */
class DataProcessor : public QObject
{
    Q_OBJECT

public:
    /**
     * @brief 构造函数
     * 
     * @param parent 父对象指针
     */
    explicit DataProcessor(QObject *parent = nullptr);
    
    /**
     * @brief 设置特征提取器类型
     * 
     * @param type 特征提取器类型
     */
    void SetFeatureType(FeatureType type);
    
    /**
     * @brief 获取当前特征提取器类型
     * 
     * @return 当前使用的特征提取器类型
     */
    FeatureType GetFeatureType() const;
    
    /**
     * @brief 图像归一化处理
     * 
     * @param input 输入图像
     * @param targetMean 目标均值
     * @param targetStd 目标标准差
     * @return 归一化后的图像
     */
    Mat NormalizeImage(const Mat& input, double targetMean, double targetStd);
    
    /**
     * @brief 图像标准化处理
     * 
     * 将图像像素值映射到0-255范围
     * @param input 输入图像
     * @return 标准化后的图像
     */
    Mat StandardizeImage(const Mat& input);
    
    /**
     * @brief 保持宽高比调整图像大小
     * 
     * @param input 输入图像
     * @param targetSize 目标最大尺寸
     * @return 调整大小后的图像
     */
    Mat ResizeWithAspectRatio(const Mat& input, int targetSize);
    
    /**
     * @brief 提取HOG特征
     * 
     * @param input 输入图像
     * @return HOG特征矩阵
     */
    Mat ExtractHOGFeatures(const Mat& input);
    
    /**
     * @brief 检测特征点并计算描述符
     * 
     * @param input 输入图像
     * @param descriptors 输出的特征描述符
     * @return 检测到的特征点
     */
    vector<KeyPoint> DetectKeypoints(const Mat& input, Mat& descriptors);
    
    /**
     * @brief 调整图像亮度
     * 
     * @param input 输入图像
     * @param alpha 亮度调整值
     * @return 调整亮度后的图像
     */
    Mat AdjustBrightness(const Mat& input, double alpha);
    
    /**
     * @brief 调整图像对比度
     * 
     * @param input 输入图像
     * @param beta 对比度调整值
     * @return 调整对比度后的图像
     */
    Mat AdjustContrast(const Mat& input, double beta);
    
    /**
     * @brief 为图像添加高斯噪声
     * 
     * @param input 输入图像
     * @param mean 噪声均值
     * @param stddev 噪声标准差
     * @return 添加噪声后的图像
     */
    Mat AddNoise(const Mat& input, double mean, double stddev);
    
    /**
     * @brief 随机旋转图像
     * 
     * @param input 输入图像
     * @param maxAngle 最大旋转角度（正负）
     * @return 旋转后的图像
     */
    Mat RandomRotate(const Mat& input, double maxAngle);
    
    /**
     * @brief 随机翻转图像
     * 
     * @param input 输入图像
     * @return 翻转后的图像
     */
    Mat RandomFlip(const Mat& input);
    
    /**
     * @brief 随机裁剪图像
     * 
     * @param input 输入图像
     * @param scale 裁剪比例
     * @return 裁剪后的图像
     */
    Mat RandomCrop(const Mat& input, double scale);
    
    /**
     * @brief 批量应用数据增强
     * 
     * @param input 输入图像
     * @param numAugmentations 增强版本数量
     * @return 增强后的图像集合
     */
    vector<Mat> ApplyAugmentation(const Mat& input, int numAugmentations);

private:
    mt19937 rng_;                      // 随机数生成器
    Ptr<SIFT> siftDetector_;            // SIFT特征检测器
    Ptr<ORB> orbDetector_;              // ORB特征检测器
    Ptr<AKAZE> akazeDetector_;          // AKAZE特征检测器
    FeatureType currentFeatureType_;    // 当前特征提取器类型
    
    /**
     * @brief 初始化特征检测器
     */
    void InitializeDetectors();
    
    /**
     * @brief 生成指定范围内的随机浮点数
     * 
     * @param min 最小值
     * @param max 最大值
     * @return 随机浮点数
     */
    double RandomDouble(double min, double max);
    
    /**
     * @brief 生成指定范围内的随机整数
     * 
     * @param min 最小值
     * @param max 最大值
     * @return 随机整数
     */
    int RandomInt(int min, int max);
};

#endif // DATAPROCESSOR_H
