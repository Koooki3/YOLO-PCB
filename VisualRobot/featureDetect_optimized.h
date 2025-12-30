/**
 * @file featureDetect_optimized.h
 * @brief 优化的特征检测模块头文件
 * 
 * 该文件定义了FeatureDetectorOptimized类和相关结构体，提供并行优化的特征检测、
 * 匹配和几何验证功能，支持多线程处理，相比原始版本提高了特征检测的效率。
 * 
 * @author VisualRobot Team
 * @date 2025-12-30
 * @version 1.0
 */

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
 * 
 * @note 默认参数适用于大多数场景，可根据具体需求调整
 * @see FeatureDetectorOptimized::FilterMatchesParallel()
 */
struct FeatureParams_optimize
{
    float ratioThresh = 0.85f;        ///< 特征匹配比率阈值，用于KNN匹配筛选
    float responseThresh = 0.1f;      ///< 特征点响应值阈值，过滤低质量特征点
    float ransacReprojThresh = 3.0f;  ///< RANSAC重投影阈值，控制几何验证严格度
    int minInliers = 10;              ///< 最小内点数量，对齐成功的最低要求
    bool useRansac = true;            ///< 是否使用RANSAC进行几何验证
    int numThreads = 4;               ///< 并行线程数，建议设置为CPU核心数
    bool enableParallel = true;       ///< 是否启用并行处理
    FeatureType featureType = FeatureType::SIFT; ///< 特征提取器类型（SIFT、ORB或AKAZE）
};

/**
 * @brief 并行处理结果结构体
 * 
 * 存储并行特征检测的结果，包括匹配点、特征点坐标、处理状态等
 * 
 * @note 当success为true时，其他字段才有效
 * @see FeatureDetectorOptimized::ProcessImagePair()
 */
struct ParallelResult
{
    std::vector<cv::DMatch> matches;  ///< 匹配结果，经过几何验证的优质匹配
    std::vector<cv::Point2f> points1; ///< 图像1的特征点坐标
    std::vector<cv::Point2f> points2; ///< 图像2的特征点坐标
    bool success = false;             ///< 是否成功完成检测
    FeatureType featureType = FeatureType::SIFT; ///< 使用的特征提取器类型
    qint64 processingTime = 0;        ///< 处理时间（毫秒）
};

/**
 * @brief 优化的特征检测器类
 * 
 * 提供并行优化的特征检测、匹配和几何验证功能，支持多线程处理
 * 相比原始版本，该类通过并行化处理提高了特征检测的效率
 * 
 * @note 该类使用C++11线程库和OpenCV特征检测框架
 * @see FeatureParams_optimize, ParallelResult
 */
class FeatureDetectorOptimized
{
public:
    /**
     * @brief 构造函数
     * 
     * 初始化优化版特征检测器，默认创建4线程的线程池
     * 
     * @note 初始化步骤：
     *       - 调用InitializeThreadPool()创建线程池
     *       - 默认使用4个线程
     * @see InitializeThreadPool()
     */
    FeatureDetectorOptimized();
    
    /**
     * @brief 析构函数
     * 
     * 关闭线程池，释放资源
     * 
     * @note 调用ShutdownThreadPool()清理线程池
     * @see ShutdownThreadPool()
     */
    ~FeatureDetectorOptimized();

    /**
     * @brief 特征匹配和几何验证函数 - 并行优化版本
     * 
     * 对特征匹配进行多阶段并行筛选，包括响应值筛选、比率测试和RANSAC几何验证
     * 
     * @param keypoints1 第一张图像的特征点
     * @param keypoints2 第二张图像的特征点
     * @param knnMatches KNN匹配结果
     * @param points1 输出的第一张图像的匹配点坐标
     * @param points2 输出的第二张图像的匹配点坐标
     * @param params 特征检测参数
     * @return std::vector<cv::DMatch> 筛选后的匹配点对
     * 
     * @note 处理流程：
     *       1. 根据响应值并行筛选特征点
     *       2. 并行比率测试筛选优质匹配
     *       3. 并行RANSAC几何验证
     * @note 如果禁用并行处理，会回退到串行算法
     * @see ParallelKeypointFilter(), ParallelRatioTest(), ParallelRANSAC()
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
     * 
     * 测试不同特征提取器的并行处理性能，包括SIFT、ORB和AKAZE
     * 
     * @param imagePath1 第一张图像的路径
     * @param imagePath2 第二张图像的路径
     * 
     * @note 输出内容：
     *       - 各特征提取器的处理时间
     *       - 匹配数量和内点数量
     *       - 相对性能提升倍数
     * @see ProcessImagePair()
     */
    static void TestFeatureDetectionParallel(const QString& imagePath1, const QString& imagePath2);

    /**
     * @brief 批量特征检测 - 并行处理多对图像
     * 
     * 并行处理多对图像，提高批量处理效率
     * 
     * @param imagePairs 图像对列表，每个元素为(路径1, 路径2)
     * @param params 特征检测参数
     * @return std::vector<ParallelResult> 批量处理结果列表
     * 
     * @note 使用std::async并行处理所有图像对
     * @see ProcessImagePair()
     */
    std::vector<ParallelResult> BatchFeatureDetection(
        const std::vector<std::pair<QString, QString>>& imagePairs,
        const FeatureParams_optimize& params);

    /**
     * @brief 异步特征检测 - 非阻塞版本
     * 
     * 异步处理特征检测，不阻塞主线程
     * 
     * @param imagePath1 第一张图像的路径
     * @param imagePath2 第二张图像的路径
     * @param params 特征检测参数
     * @return std::future<ParallelResult> 异步处理结果的future对象
     * 
     * @note 可以通过future.get()获取结果，或使用wait_for()检查完成状态
     * @see ProcessImagePair()
     */
    std::future<ParallelResult> AsyncFeatureDetection(
        const QString& imagePath1, 
        const QString& imagePath2,
        const FeatureParams_optimize& params);

private:
    /**
     * @brief 并行处理函数
     * 
     * 处理单对图像的特征检测和匹配
     * 
     * @param imagePath1 第一张图像的路径
     * @param imagePath2 第二张图像的路径
     * @param params 特征检测参数
     * @return ParallelResult 处理结果
     * 
     * @note 处理流程：
     *       1. 图像读取和预处理（并行）
     *       2. 特征点检测（并行）
     *       3. 特征匹配
     *       4. 匹配筛选（并行）
     * @see FilterMatchesParallel()
     */
    static ParallelResult ProcessImagePair(
        const QString& imagePath1,
        const QString& imagePath2,
        const FeatureParams_optimize& params);

    /**
     * @brief 并行特征点筛选
     * 
     * 并行筛选符合响应值阈值的特征点
     * 
     * @param keypoints 特征点列表
     * @param validFlags 输出的有效特征点标志
     * @param responseThresh 响应值阈值
     * @param startIdx 起始索引
     * @param endIdx 结束索引
     * 
     * @note 该函数由多个线程并行执行，处理特征点列表的不同段
     * @see FilterMatchesParallel()
     */
    static void ParallelKeypointFilter(
        const std::vector<cv::KeyPoint>& keypoints,
        std::vector<bool>& validFlags,
        float responseThresh,
        size_t startIdx,
        size_t endIdx);

    /**
     * @brief 并行比率测试
     * 
     * 并行进行比率测试，筛选优质匹配
     * 
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
     * @note 该函数由多个线程并行执行，处理匹配列表的不同段
     * @see FilterMatchesParallel()
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
     * 
     * 并行进行多次RANSAC，选择最佳结果
     * 
     * @param goodMatches 优质匹配
     * @param points1 第一张图像的匹配点坐标
     * @param points2 第二张图像的匹配点坐标
     * @param params 特征检测参数
     * @return std::vector<cv::DMatch> RANSAC筛选后的匹配
     * 
     * @note 由于RANSAC的随机性，采用多次并行执行，选择内点最多的结果
     * @see FilterMatchesParallel()
     */
    static std::vector<cv::DMatch> ParallelRANSAC(
        const std::vector<cv::DMatch>& goodMatches,
        std::vector<cv::Point2f>& points1,
        std::vector<cv::Point2f>& points2,
        const FeatureParams_optimize& params);

    /**
     * @brief 初始化线程池
     * 
     * 初始化并行处理所需的线程池
     * 
     * @param numThreads 线程数量
     * 
     * @note 线程池使用全局静态变量管理，支持多实例共享
     * @see ShutdownThreadPool()
     */
    static void InitializeThreadPool(int numThreads);
    
    /**
     * @brief 关闭线程池
     * 
     * 关闭并释放线程池资源
     * 
     * @note 会等待所有线程结束，清理全局资源
     * @see InitializeThreadPool()
     */
    static void ShutdownThreadPool();

    /**
     * @brief 记录性能指标
     * 
     * 记录并输出性能指标
     * 
     * @param operation 操作名称
     * @param elapsedMs 耗时（毫秒）
     * 
     * @note 使用qDebug输出到控制台
     */
    static void LogPerformanceMetrics(const QString& operation, qint64 elapsedMs);
};

#endif // FEATUREDETECT_OPTIMIZED_H
