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

// 特征识别参数结构体
struct FeatureParams_optimize
{
    float ratioThresh = 0.85f;        // SIFT匹配比率阈值
    float responseThresh = 0.1f;      // 特征点响应值阈值
    float ransacReprojThresh = 3.0f;  // RANSAC重投影阈值
    int minInliers = 10;             // 最小内点数量
    bool useRansac = true;           // 是否使用RANSAC验证
    int numThreads = 4;              // 并行线程数
    bool enableParallel = true;      // 启用并行计算
};

// 并行处理结果结构体
struct ParallelResult
{
    std::vector<cv::DMatch> matches;
    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;
    bool success = false;
};

class FeatureDetectorOptimized
{
public:
    FeatureDetectorOptimized();
    ~FeatureDetectorOptimized();

    // 特征匹配和几何验证函数 - 并行优化版本
    static std::vector<cv::DMatch> FilterMatchesParallel(
        const std::vector<cv::KeyPoint>& keypoints1,
        const std::vector<cv::KeyPoint>& keypoints2,
        const std::vector<std::vector<cv::DMatch>>& knnMatches,
        std::vector<cv::Point2f>& points1,
        std::vector<cv::Point2f>& points2,
        const FeatureParams_optimize& params);

    // 并行特征检测测试函数
    static void TestFeatureDetectionParallel(const QString& imagePath1, const QString& imagePath2);

    // 批量特征检测 - 并行处理多对图像
    static std::vector<ParallelResult> BatchFeatureDetection(
        const std::vector<std::pair<QString, QString>>& imagePairs,
        const FeatureParams_optimize& params);

    // 异步特征检测 - 非阻塞版本
    static std::future<ParallelResult> AsyncFeatureDetection(
        const QString& imagePath1, 
        const QString& imagePath2,
        const FeatureParams_optimize& params);

private:
    // 并行处理函数
    static ParallelResult ProcessImagePair(
        const QString& imagePath1,
        const QString& imagePath2,
        const FeatureParams_optimize& params);

    // 并行特征点筛选
    static void ParallelKeypointFilter(
        const std::vector<cv::KeyPoint>& keypoints,
        std::vector<bool>& validFlags,
        float responseThresh,
        size_t startIdx,
        size_t endIdx);

    // 并行比率测试
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

    // 并行RANSAC验证
    static std::vector<cv::DMatch> ParallelRANSAC(
        const std::vector<cv::DMatch>& goodMatches,
        std::vector<cv::Point2f>& points1,
        std::vector<cv::Point2f>& points2,
        const FeatureParams_optimize& params);

    // 线程池管理
    static void InitializeThreadPool(int numThreads);
    static void ShutdownThreadPool();

    // 性能监控
    static void LogPerformanceMetrics(const QString& operation, qint64 elapsedMs);
};

#endif // FEATUREDETECT_OPTIMIZED_H
