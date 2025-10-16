/**
 * @file DataProcessor.h
 * @brief 数据处理器类定义
 * 
 * 提供图像预处理、特征提取和数据增强等功能
 * 支持归一化、标准化、HOG特征提取、SIFT关键点检测等操作
 */

#ifndef DATAPROCESSOR_H
#define DATAPROCESSOR_H

#include <opencv2/opencv.hpp>
#include <QObject>
#include <random>

using namespace cv;
using namespace std;

/**
 * @class DataProcessor
 * @brief 数据处理器类，提供图像数据处理功能
 * 
 * 该类封装了常用的图像处理操作，包括：
 * - 图像预处理（归一化、标准化、尺寸调整）
 * - 特征提取（HOG、SIFT）
 * - 数据增强（亮度调整、对比度调整、噪声添加、旋转、翻转、裁剪）
 */
class DataProcessor : public QObject
{
    Q_OBJECT

public:
    /**
     * @brief 构造函数
     * @param parent 父对象指针
     */
    explicit DataProcessor(QObject *parent = nullptr);
    
    // ==================== 预处理方法 ====================
    
    /**
     * @brief 图像归一化处理
     * @param input 输入图像
     * @param targetMean 目标均值
     * @param targetStd 目标标准差
     * @return 归一化后的图像
     */
    Mat NormalizeImage(const Mat& input, double targetMean, double targetStd);
    
    /**
     * @brief 图像标准化处理
     * @param input 输入图像
     * @return 标准化后的图像（像素值映射到0-255范围）
     */
    Mat StandardizeImage(const Mat& input);
    
    /**
     * @brief 保持宽高比的图像缩放
     * @param input 输入图像
     * @param targetSize 目标尺寸（最长边）
     * @return 缩放后的图像
     */
    Mat ResizeWithAspectRatio(const Mat& input, int targetSize);
    
    // ==================== 特征提取方法 ====================
    
    /**
     * @brief 提取HOG特征
     * @param input 输入图像
     * @return HOG特征矩阵
     */
    Mat ExtractHOGFeatures(const Mat& input);
    
    /**
     * @brief 检测关键点并计算描述符
     * @param input 输入图像
     * @param descriptors 输出描述符矩阵
     * @return 检测到的关键点集合
     */
    vector<KeyPoint> DetectKeypoints(const Mat& input, Mat& descriptors);
    
    // ==================== 数据增强方法 ====================
    
    /**
     * @brief 调整图像亮度
     * @param input 输入图像
     * @param alpha 亮度偏移量
     * @return 调整亮度后的图像
     */
    Mat AdjustBrightness(const Mat& input, double alpha);
    
    /**
     * @brief 调整图像对比度
     * @param input 输入图像
     * @param beta 对比度比例因子
     * @return 调整对比度后的图像
     */
    Mat AdjustContrast(const Mat& input, double beta);
    
    /**
     * @brief 添加高斯噪声
     * @param input 输入图像
     * @param mean 噪声均值
     * @param stddev 噪声标准差
     * @return 添加噪声后的图像
     */
    Mat AddNoise(const Mat& input, double mean, double stddev);
    
    /**
     * @brief 随机旋转图像
     * @param input 输入图像
     * @param maxAngle 最大旋转角度（度）
     * @return 旋转后的图像
     */
    Mat RandomRotate(const Mat& input, double maxAngle);
    
    /**
     * @brief 随机翻转图像
     * @param input 输入图像
     * @return 翻转后的图像
     */
    Mat RandomFlip(const Mat& input);
    
    /**
     * @brief 随机裁剪图像
     * @param input 输入图像
     * @param scale 裁剪比例（0-1之间）
     * @return 裁剪并缩放后的图像
     */
    Mat RandomCrop(const Mat& input, double scale);
    
    // ==================== 批量增强方法 ====================
    
    /**
     * @brief 应用多种数据增强操作
     * @param input 输入图像
     * @param numAugmentations 增强图像数量
     * @return 增强后的图像集合
     */
    vector<Mat> ApplyAugmentation(const Mat& input, int numAugmentations);

private:
    mt19937 rng_;           ///< 随机数生成器
    Ptr<SIFT> siftDetector_; ///< SIFT特征检测器
    
    // ==================== 辅助函数 ====================
    
    /**
     * @brief 生成指定范围内的随机浮点数
     * @param min 最小值
     * @param max 最大值
     * @return 随机浮点数
     */
    double RandomDouble(double min, double max);
    
    /**
     * @brief 生成指定范围内的随机整数
     * @param min 最小值
     * @param max 最大值
     * @return 随机整数
     */
    int RandomInt(int min, int max);
};

#endif // DATAPROCESSOR_H
