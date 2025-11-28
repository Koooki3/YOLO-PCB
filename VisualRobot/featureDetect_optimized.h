#ifndef FEATUREDETECT_OPTIMIZED_H
#define FEATUREDETECT_OPTIMIZED_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <QString>
#include <QDebug>
#include <thread>
#include <future>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include "DataProcessor.h" // 包含FeatureType枚举

/**
 * @brief 特征识别参数结构体（优化版本）
 * 
 * 存储优化版本特征检测和匹配的各种参数，支持并行处理配置
 */
struct FeatureParams_optimize
{
    float ratioThresh = 0.85f;        // 特征匹配比率阈值
    float responseThresh = 0.1f;      // 特征点响应值阈值
    float ransacReprojThresh = 3.0f;  // RANSAC重投影阈值
    int minInliers = 10;             // 最小内点数量
    bool useRansac = true;           // 是否使用RANSAC验证
    int numThreads = 4;              // 并行线程数
    bool enableParallel = true;      // 是否启用并行处理
    FeatureType featureType = FeatureType::SIFT; // 特征提取器类型，可选择SIFT、ORB或AKAZE
};

/**
 * @brief 并行处理结果结构体
 * 
 * 存储并行特征检测的结果，包括匹配点、特征点坐标、处理状态等
 */
struct ParallelResult
{
    std::vector<cv::DMatch> matches;  // 匹配结果
    std::vector<cv::Point2f> points1; // 图像1的特征点
    std::vector<cv::Point2f> points2; // 图像2的特征点
    bool success = false;             // 是否成功
    FeatureType featureType = FeatureType::SIFT; // 使用的特征提取器类型（SIFT、SURF或ORB）
    qint64 processingTime = 0;        // 处理时间（毫秒）
};

/**
 * @brief 优化的特征检测器类
 * 
 * 提供并行优化的特征检测、匹配和几何验证功能，支持多线程处理
 * 相比原始版本，该类通过并行化处理提高了特征检测的效率
 */
class FeatureDetectorOptimized
{
public:
    /**
     * @brief 构造函数
     */
    FeatureDetectorOptimized();
    
    /**
     * @brief 析构函数
     */
    ~FeatureDetectorOptimized();

    /**
     * @brief 特征匹配和几何验证函数 - 并行优化版本
     * @param keypoints1 第一张图像的特征点
     * @param keypoints2 第二张图像的特征点
     * @param knnMatches KNN匹配结果
     * @param points1 输出的第一张图像的匹配点坐标
     * @param points2 输出的第二张图像的匹配点坐标
     * @param params 特征检测参数
     * @return 筛选后的匹配点对
     * 
     * 对特征匹配进行多阶段并行筛选，包括响应值筛选、比率测试和RANSAC几何验证
     */
    static std::vector<cv::DMatch> FilterMatchesParallel(
        const std::vector<cv::KeyPoint>& keypoints1,
        const std::vector<cv::KeyPoint>& keypoints2,
        const std::vector<std::vector<cv::DMatch>>& knnMatches,
        std::vector<cv::Point2f>& points1,
        std::vector<cv::Point2f>& points2,
        const FeatureParams_optimize& params);

    /**
     * @brief 并行特征检测测试函数
     * @param imagePath1 第一张图像的路径
     * @param imagePath2 第二张图像的路径
     * 
     * 测试不同特征提取器的并行处理性能，包括SIFT、ORB和AKAZE
     */
    static void TestFeatureDetectionParallel(const QString& imagePath1, const QString& imagePath2);

    /**
     * @brief 批量特征检测 - 并行处理多对图像
     * @param imagePairs 图像对列表
     * @param params 特征检测参数
     * @return 批量处理结果列表
     * 
     * 并行处理多对图像，提高批量处理效率
     */
    std::vector<ParallelResult> BatchFeatureDetection(
        const std::vector<std::pair<QString, QString>>& imagePairs,
        const FeatureParams_optimize& params);

    /**
     * @brief 异步特征检测 - 非阻塞版本
     * @param imagePath1 第一张图像的路径
     * @param imagePath2 第二张图像的路径
     * @param params 特征检测参数
     * @return 异步处理结果的future对象
     * 
     * 异步处理特征检测，不阻塞主线程
     */
    std::future<ParallelResult> AsyncFeatureDetection(
        const QString& imagePath1, 
        const QString& imagePath2,
        const FeatureParams_optimize& params);

private:
    /**
     * @brief 并行处理函数
     * @param imagePath1 第一张图像的路径
     * @param imagePath2 第二张图像的路径
     * @param params 特征检测参数
     * @return 处理结果
     * 
     * 处理单对图像的特征检测和匹配
     */
    static ParallelResult ProcessImagePair(
        const QString& imagePath1,
        const QString& imagePath2,
        const FeatureParams_optimize& params);

    /**
     * @brief 并行特征点筛选
     * @param keypoints 特征点列表
     * @param validFlags 输出的有效特征点标志
     * @param responseThresh 响应值阈值
     * @param startIdx 起始索引
     * @param endIdx 结束索引
     * 
     * 并行筛选符合响应值阈值的特征点
     */
    static void ParallelKeypointFilter(
        const std::vector<cv::KeyPoint>& keypoints,
        std::vector<bool>& validFlags,
        float responseThresh,
        size_t startIdx,
        size_t endIdx);

    /**
     * @brief 并行比率测试
     * @param knnMatches KNN匹配结果
     * @param validKeypoints1 第一张图像的有效特征点标志
     * @param validKeypoints2 第二张图像的有效特征点标志
     * @param goodMatches 输出的优质匹配
     * @param points1 输出的第一张图像的匹配点坐标
     * @param points2 输出的第二张图像的匹配点坐标
     * @param keypoints1 第一张图像的特征点
     * @param keypoints2 第二张图像的特征点
     * @param ratioThresh 比率阈值
     * @param startIdx 起始索引
     * @param endIdx 结束索引
     * 
     * 并行进行比率测试，筛选优质匹配
     */
    static void ParallelRatioTest(
        const std::vector<std::vector<cv::DMatch>>& knnMatches,
        const std::vector<bool>& validKeypoints1,
        const std::vector<bool>& validKeypoints2,
        std::vector<cv::DMatch>& goodMatches,
        std::vector<cv::Point2f>& points1,
        std::vector<cv::Point2f>& points2,
        const std::vector<cv::KeyPoint>& keypoints1,
        const std::vector<cv::KeyPoint>& keypoints2,
        float ratioThresh,
        size_t startIdx,
        size_t endIdx);

    /**
     * @brief 并行RANSAC验证
     * @param goodMatches 优质匹配
     * @param points1 第一张图像的匹配点坐标
     * @param points2 第二张图像的匹配点坐标
     * @param params 特征检测参数
     * @return RANSAC筛选后的匹配
     * 
     * 并行进行RANSAC几何验证，提高验证效率
     */
    static std::vector<cv::DMatch> ParallelRANSAC(
        const std::vector<cv::DMatch>& goodMatches,
        std::vector<cv::Point2f>& points1,
        std::vector<cv::Point2f>& points2,
        const FeatureParams_optimize& params);

    /**
     * @brief 初始化线程池
     * @param numThreads 线程数量
     * 
     * 初始化并行处理所需的线程池
     */
    static void InitializeThreadPool(int numThreads);
    
    /**
     * @brief 关闭线程池
     * 
     * 关闭并释放线程池资源
     */
    static void ShutdownThreadPool();

    /**
     * @brief 记录性能指标
     * @param operation 操作名称
     * @param elapsedMs 耗时（毫秒）
     * 
     * 记录并输出性能指标
     */
    static void LogPerformanceMetrics(const QString& operation, qint64 elapsedMs);
};

#endif // FEATUREDETECT_OPTIMIZED_H
