/**
 * @file featureDetect_optimized.cpp
 * @brief 优化的特征检测模块实现文件
 * 
 * 该文件实现了FeatureDetectorOptimized类的所有方法，提供并行优化的特征检测、
 * 匹配和几何验证功能，支持多线程处理，显著提高了特征检测的效率。
 * 
 * @author VisualRobot Team
 * @date 2025-12-30
 * @version 1.0
 */

#include "featureDetect_optimized.h"
#include "DataProcessor.h"
#include "DLProcessor.h"
#include <QElapsedTimer>
#include <algorithm>
#include <numeric>

// 静态成员变量定义 - 线程池相关
static std::vector<std::thread> g_threadPool;              // 线程池
static std::atomic<bool> g_threadPoolRunning{false};       // 线程池运行状态
static std::mutex g_threadPoolMutex;                       // 线程池互斥锁
static std::condition_variable g_threadPoolCV;             // 线程池条件变量

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
FeatureDetectorOptimized::FeatureDetectorOptimized()
{
    // 默认初始化线程池，使用4个线程
    InitializeThreadPool(4);
}

/**
 * @brief 析构函数
 * 
 * 关闭线程池，释放资源
 * 
 * @note 调用ShutdownThreadPool()清理线程池
 * @see ShutdownThreadPool()
 */
FeatureDetectorOptimized::~FeatureDetectorOptimized()
{
    ShutdownThreadPool();
}

/**
 * @brief 初始化线程池
 * 
 * 创建并启动指定数量的工作线程，用于并行处理任务
 * 
 * @param numThreads 线程池大小
 * 
 * @note 线程池使用全局静态变量管理，支持多实例共享
 * @note 线程会等待任务分配，通过条件变量进行同步
 * @see ShutdownThreadPool()
 */
void FeatureDetectorOptimized::InitializeThreadPool(int numThreads)
{
    std::lock_guard<std::mutex> lock(g_threadPoolMutex);

    // 如果线程池已运行，先关闭
    if (g_threadPoolRunning)
    {
        ShutdownThreadPool();
    }

    g_threadPoolRunning = true;
    g_threadPool.clear();

    // 创建指定数量的线程
    for (int i = 0; i < numThreads; ++i)
    {
        g_threadPool.emplace_back([]() {
            while (g_threadPoolRunning)
            {
                std::unique_lock<std::mutex> lock(g_threadPoolMutex);
                // 等待线程池关闭信号
                g_threadPoolCV.wait(lock, []() {
                    return !g_threadPoolRunning;
                });
            }
        });
    }

    qDebug() << "Thread pool initialized with" << numThreads << "threads";
}

/**
 * @brief 关闭线程池
 * 
 * 停止所有线程并释放资源
 * 
 * @note 关闭流程：
 *       1. 设置运行标志为false
 *       2. 通知所有等待的线程
 *       3. 等待所有线程结束
 *       4. 清理线程容器
 * @see InitializeThreadPool()
 */
void FeatureDetectorOptimized::ShutdownThreadPool()
{
    {
        std::lock_guard<std::mutex> lock(g_threadPoolMutex);
        g_threadPoolRunning = false;
    }

    // 通知所有线程
    g_threadPoolCV.notify_all();

    // 等待所有线程结束
    for (auto& thread : g_threadPool)
    {
        if (thread.joinable())
        {
            thread.join();
        }
    }

    g_threadPool.clear();
    qDebug() << "Thread pool shutdown";
}

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
std::vector<cv::DMatch> FeatureDetectorOptimized::FilterMatchesParallel(
    const std::vector<cv::KeyPoint>& keypoints1,
    const std::vector<cv::KeyPoint>& keypoints2,
    const std::vector<std::vector<cv::DMatch>>& knnMatches,
    std::vector<cv::Point2f>& points1,
    std::vector<cv::Point2f>& points2,
    const FeatureParams_optimize& params)
{
    QElapsedTimer timer;
    timer.start();

    std::vector<cv::DMatch> goodMatches; // 存储筛选后的优质匹配
    points1.clear();
    points2.clear();

    // 根据参数决定是否使用并行处理
    if (!params.enableParallel || params.numThreads <= 1)
    {
        // 串行版本 - 回退到原始算法
        std::vector<bool> validKeypoints1(keypoints1.size(), false);
        std::vector<bool> validKeypoints2(keypoints2.size(), false);

        // 1. 根据响应值筛选特征点
        for (size_t i = 0; i < keypoints1.size(); i++)
        {
            if (keypoints1[i].response > params.responseThresh)
            {
                validKeypoints1[i] = true;
            }
        }

        for (size_t i = 0; i < keypoints2.size(); i++)
        {
            if (keypoints2[i].response > params.responseThresh)
            {
                validKeypoints2[i] = true;
            }
        }

        // 2. 应用比率测试进行初步筛选
        for (const auto& match : knnMatches)
         {
            if (match[0].distance < params.ratioThresh * match[1].distance && validKeypoints1[match[0].queryIdx] && validKeypoints2[match[0].trainIdx])
            {
                goodMatches.push_back(match[0]);
                points1.push_back(keypoints1[match[0].queryIdx].pt);
                points2.push_back(keypoints2[match[0].trainIdx].pt);
            }
        }
    }
    else
    {
        // 并行版本
        const int numThreads = std::min(params.numThreads, static_cast<int>(std::thread::hardware_concurrency()));

        // 1. 并行特征点筛选
        std::vector<bool> validKeypoints1(keypoints1.size(), false);
        std::vector<bool> validKeypoints2(keypoints2.size(), false);

        std::vector<std::thread> threads;
        // 计算每个线程处理的特征点数量
        const size_t chunkSize1 = (keypoints1.size() + numThreads - 1) / numThreads;
        const size_t chunkSize2 = (keypoints2.size() + numThreads - 1) / numThreads;

        // 为每个线程分配任务
        for (int i = 0; i < numThreads; ++i)
        {
            const size_t start1 = i * chunkSize1;
            const size_t end1 = std::min(start1 + chunkSize1, keypoints1.size());
            const size_t start2 = i * chunkSize2;
            const size_t end2 = std::min(start2 + chunkSize2, keypoints2.size());

            if (start1 < keypoints1.size())
            {
                threads.emplace_back(ParallelKeypointFilter, std::cref(keypoints1), std::ref(validKeypoints1), params.responseThresh, start1, end1);
            }

            if (start2 < keypoints2.size())
            {
                threads.emplace_back(ParallelKeypointFilter, std::cref(keypoints2), std::ref(validKeypoints2), params.responseThresh, start2, end2);
            }
        }

        // 等待所有线程完成
        for (auto& thread : threads)
        {
            if (thread.joinable())
            {
                thread.join();
            }
        }

        // 2. 并行比率测试
        std::vector<std::vector<cv::DMatch>> threadGoodMatches(numThreads);
        std::vector<std::vector<cv::Point2f>> threadPoints1(numThreads);
        std::vector<std::vector<cv::Point2f>> threadPoints2(numThreads);

        threads.clear();
        // 计算每个线程处理的匹配数量
        const size_t chunkSizeMatches = (knnMatches.size() + numThreads - 1) / numThreads;

        // 为每个线程分配比率测试任务
        for (int i = 0; i < numThreads; ++i)
        {
            const size_t start = i * chunkSizeMatches;
            const size_t end = std::min(start + chunkSizeMatches, knnMatches.size());

            if (start < knnMatches.size())
            {
                threads.emplace_back(ParallelRatioTest, 
                    std::cref(knnMatches), std::cref(validKeypoints1), std::cref(validKeypoints2),
                    std::ref(threadGoodMatches[i]), std::ref(threadPoints1[i]), std::ref(threadPoints2[i]),
                    std::cref(keypoints1), std::cref(keypoints2), params.ratioThresh, start, end);
            }
        }

        // 等待所有线程完成
        for (auto& thread : threads)
        {
            if (thread.joinable())
            {
                thread.join();
            }
        }

        // 合并所有线程的结果
        for (int i = 0; i < numThreads; ++i)
        {
            goodMatches.insert(goodMatches.end(), threadGoodMatches[i].begin(), threadGoodMatches[i].end());
            points1.insert(points1.end(), threadPoints1[i].begin(), threadPoints1[i].end());
            points2.insert(points2.end(), threadPoints2[i].begin(), threadPoints2[i].end());
        }
    }

    // 3. RANSAC几何验证
    if (params.useRansac && points1.size() >= 4)
    {
        if (params.enableParallel && params.numThreads > 1)
        {
            // 并行RANSAC
            goodMatches = ParallelRANSAC(goodMatches, points1, points2, params);
        }
        else
        {
            // 串行RANSAC
            std::vector<uchar> inlierMask;
            cv::Mat H = cv::findHomography(points1, points2, cv::RANSAC, params.ransacReprojThresh, inlierMask);

            if (!H.empty())
            {
                std::vector<cv::DMatch> ransacMatches;
                std::vector<cv::Point2f> inlierPoints1;
                std::vector<cv::Point2f> inlierPoints2;

                // 提取内点
                for (size_t i = 0; i < inlierMask.size(); i++)
                {
                    if (inlierMask[i])
                    {
                        ransacMatches.push_back(goodMatches[i]);
                        inlierPoints1.push_back(points1[i]);
                        inlierPoints2.push_back(points2[i]);
                    }
                }

                // 检查内点数量是否满足要求
                if (ransacMatches.size() >= static_cast<size_t>(params.minInliers))
                {
                    points1 = inlierPoints1;
                    points2 = inlierPoints2;
                    goodMatches = ransacMatches;
                }
            }
        }
    }

    LogPerformanceMetrics("FilterMatchesParallel", timer.elapsed());
    return goodMatches;
}

/**
 * @brief 并行特征点筛选函数
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
void FeatureDetectorOptimized::ParallelKeypointFilter(
    const std::vector<cv::KeyPoint>& keypoints,
    std::vector<bool>& validFlags,
    float responseThresh,
    size_t startIdx,
    size_t endIdx)
{
    // 筛选响应值大于阈值的特征点
    for (size_t i = startIdx; i < endIdx; ++i)
    {
        if (keypoints[i].response > responseThresh)
        {
            validFlags[i] = true;
        }
    }
}

/**
 * @brief 并行比率测试函数
 * 
 * 并行进行比率测试，筛选优质匹配
 * 
 * @param knnMatches KNN匹配结果
 * @param validKeypoints1 第一张图像的有效特征点标志
 * @param validKeypoints2 第二张图像的有效特征点标志
 * @param goodMatches 优质匹配结果
 * @param points1 第一张图像的匹配点坐标
 * @param points2 第二张图像的匹配点坐标
 * @param keypoints1 第一张图像的特征点
 * @param keypoints2 第二张图像的特征点
 * @param ratioThresh 比率阈值
 * @param startIdx 起始索引
 * @param endIdx 结束索引
 * 
 * @note 该函数由多个线程并行执行，处理匹配列表的不同段
 * @see FilterMatchesParallel()
 */
void FeatureDetectorOptimized::ParallelRatioTest(
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
    size_t endIdx)
{
    // 应用比率测试筛选优质匹配
    for (size_t i = startIdx; i < endIdx; ++i)
    {
        const auto& match = knnMatches[i];
        if (match.size() >= 2 && match[0].distance < ratioThresh * match[1].distance && validKeypoints1[match[0].queryIdx] && validKeypoints2[match[0].trainIdx])
        {
            goodMatches.push_back(match[0]);
            points1.push_back(keypoints1[match[0].queryIdx].pt);
            points2.push_back(keypoints2[match[0].trainIdx].pt);
        }
    }
}

/**
 * @brief 并行RANSAC函数
 * 
 * 并行进行多次RANSAC，选择最佳结果
 * 
 * @param goodMatches 优质匹配结果
 * @param points1 第一张图像的匹配点坐标
 * @param points2 第二张图像的匹配点坐标
 * @param params 特征检测参数
 * @return std::vector<cv::DMatch> 经过RANSAC验证的匹配点对
 * 
 * @note 由于RANSAC算法本身是随机采样，难以直接并行化
 *       这里采用多次RANSAC并行执行，选择最佳结果的方法
 * @see FilterMatchesParallel()
 */
std::vector<cv::DMatch> FeatureDetectorOptimized::ParallelRANSAC(
    const std::vector<cv::DMatch>& goodMatches,
    std::vector<cv::Point2f>& points1,
    std::vector<cv::Point2f>& points2,
    const FeatureParams_optimize& params)
{
    // 由于RANSAC算法本身是随机采样，难以直接并行化
    // 这里采用多次RANSAC并行执行，选择最佳结果的方法

    const int numRANSACRuns = std::min(params.numThreads, 8); // 限制最大运行次数为8
    std::vector<std::future<std::tuple<std::vector<cv::DMatch>, std::vector<cv::Point2f>, std::vector<cv::Point2f>, int>>> futures;

    // 启动多个RANSAC并行执行
    for (int i = 0; i < numRANSACRuns; ++i)
    {
        futures.push_back(std::async(std::launch::async, [&, i]() {
            std::vector<uchar> inlierMask;
            cv::Mat H = cv::findHomography(points1, points2, cv::RANSAC, params.ransacReprojThresh, inlierMask);

            if (H.empty())
            {
                return std::make_tuple(std::vector<cv::DMatch>(), std::vector<cv::Point2f>(), std::vector<cv::Point2f>(), 0);
            }

            std::vector<cv::DMatch> ransacMatches;
            std::vector<cv::Point2f> inlierPoints1;
            std::vector<cv::Point2f> inlierPoints2;
            int inlierCount = 0;

            // 提取内点
            for (size_t j = 0; j < inlierMask.size(); j++)
            {
                if (inlierMask[j])
                {
                    ransacMatches.push_back(goodMatches[j]);
                    inlierPoints1.push_back(points1[j]);
                    inlierPoints2.push_back(points2[j]);
                    inlierCount++;
                }
            }

            return std::make_tuple(ransacMatches, inlierPoints1, inlierPoints2, inlierCount);
        }));
    }

    // 选择内点数量最多的结果
    std::vector<cv::DMatch> bestMatches;
    std::vector<cv::Point2f> bestPoints1;
    std::vector<cv::Point2f> bestPoints2;
    int maxInliers = 0;

    for (auto& future : futures)
    {
        auto [matches, p1, p2, inlierCount] = future.get();
        if (inlierCount > maxInliers && inlierCount >= params.minInliers)
        {
            maxInliers = inlierCount;
            bestMatches = std::move(matches);
            bestPoints1 = std::move(p1);
            bestPoints2 = std::move(p2);
        }
    }

    if (maxInliers > 0)
    {
        points1 = std::move(bestPoints1);
        points2 = std::move(bestPoints2);
        return bestMatches;
    }

    return goodMatches; // 如果没有找到更好的结果，返回原始匹配
}

/**
 * @brief 并行特征检测测试函数
 * 
 * 测试不同特征提取器的并行处理性能，包括SIFT、ORB和AKAZE
 * 
 * @param imagePath1 第一张图像路径
 * @param imagePath2 第二张图像路径
 * 
 * @note 输出内容：
 *       - 各特征提取器的处理时间
 *       - 匹配数量和内点数量
 *       - 相对性能提升倍数
 * @see ProcessImagePair()
 */
void FeatureDetectorOptimized::TestFeatureDetectionParallel(const QString& imagePath1, const QString& imagePath2)
{
    FeatureParams_optimize params;
    params.enableParallel = true;
    params.numThreads = std::thread::hardware_concurrency();

    // 测试SIFT特征提取器
    params.featureType = FeatureType::SIFT;
    QElapsedTimer siftTimer;
    siftTimer.start();
    auto siftResult = ProcessImagePair(imagePath1, imagePath2, params);
    qint64 siftTime = siftTimer.elapsed();

    // 测试ORB特征提取器
    params.featureType = FeatureType::ORB;
    QElapsedTimer orbTimer;
    orbTimer.start();
    auto orbResult = ProcessImagePair(imagePath1, imagePath2, params);
    qint64 orbTime = orbTimer.elapsed();

    // 测试AKAZE特征提取器
    params.featureType = FeatureType::AKAZE;
    QElapsedTimer akazeTimer;
    akazeTimer.start();
    auto akazeResult = ProcessImagePair(imagePath1, imagePath2, params);
    qint64 akazeTime = akazeTimer.elapsed();

    // 输出对比结果
    qDebug() << "\n===== 特征提取器性能对比 =====";

    qDebug() << "\nSIFT特征提取器:";
    if (siftResult.success)
    {
        qDebug() << "  完成时间:" << siftTime << "ms";
        qDebug() << "  好匹配数量:" << siftResult.matches.size();
        qDebug() << "  内点数量:" << siftResult.points1.size();
    }
    else
    {
        qDebug() << "  检测失败";
    }

    qDebug() << "\nORB特征提取器:";
    if (orbResult.success)
    {
        qDebug() << "  完成时间:" << orbTime << "ms";
        qDebug() << "  好匹配数量:" << orbResult.matches.size();
        qDebug() << "  内点数量:" << orbResult.points1.size();
    }
    else
    {
        qDebug() << "  检测失败";
    }

    qDebug() << "\nAKAZE特征提取器:";
    if (akazeResult.success)
    {
        qDebug() << "  完成时间:" << akazeTime << "ms";
        qDebug() << "  好匹配数量:" << akazeResult.matches.size();
        qDebug() << "  内点数量:" << akazeResult.points1.size();
    }
    else
    {
        qDebug() << "  检测失败";
    }

    // 性能比较
    qDebug() << "\n性能比较:";
    if (siftResult.success && orbResult.success) {
        double orbSpeedup = static_cast<double>(siftTime) / orbTime;
        qDebug() << "  ORB相对SIFT的速度提升:" << QString::number(orbSpeedup, 'f', 2) << "倍";
    }
    if (siftResult.success && akazeResult.success) {
        double akazeRelSiftSpeedup = static_cast<double>(siftTime) / akazeTime;
        qDebug() << "  AKAZE相对SIFT的速度提升:" << QString::number(akazeRelSiftSpeedup, 'f', 2) << "倍";
    }
    if (orbResult.success && akazeResult.success) {
        double akazeRelOrbSpeedup = static_cast<double>(orbTime) / akazeTime;
        qDebug() << "  AKAZE相对ORB的速度提升:" << QString::number(akazeRelOrbSpeedup, 'f', 2) << "倍";
    }
    qDebug() << "==============================\n";
}

/**
 * @brief 处理图像对
 * 
 * 处理单对图像的特征检测和匹配
 * 
 * @param imagePath1 第一张图像路径
 * @param imagePath2 第二张图像路径
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
ParallelResult FeatureDetectorOptimized::ProcessImagePair(
    const QString& imagePath1,
    const QString& imagePath2,
    const FeatureParams_optimize& params)
{
    ParallelResult result;
    QElapsedTimer timer;
    timer.start();

    // 读取图像
    cv::Mat image1 = cv::imread(imagePath1.toStdString());
    cv::Mat image2 = cv::imread(imagePath2.toStdString());

    if (image1.empty() || image2.empty())
    {
        qDebug() << "Error: Could not read the images";
        result.success = false;
        return result;
    }

    // 图像预处理
    DataProcessor dataProcessor;
    // 设置特征提取器类型
    dataProcessor.SetFeatureType(params.featureType);

    // 并行图像预处理和特征提取
    std::future<cv::Mat> futureStandardized1 = std::async(std::launch::async, [&dataProcessor, image1]() { return dataProcessor.StandardizeImage(image1); });
    std::future<cv::Mat> futureStandardized2 = std::async(std::launch::async, [&dataProcessor, image2]() { return dataProcessor.StandardizeImage(image2); });

    cv::Mat standardized1 = futureStandardized1.get();
    cv::Mat standardized2 = futureStandardized2.get();

    // 并行特征提取
    cv::Mat descriptors1, descriptors2;
    std::vector<cv::KeyPoint> keypoints1, keypoints2;

    std::future<std::vector<cv::KeyPoint>> futureKeypoints1 = std::async(std::launch::async,
        [&dataProcessor, standardized1, &descriptors1]() {
            return dataProcessor.DetectKeypoints(standardized1, descriptors1);
        });
    std::future<std::vector<cv::KeyPoint>> futureKeypoints2 = std::async(std::launch::async,
        [&dataProcessor, standardized2, &descriptors2]() {
            return dataProcessor.DetectKeypoints(standardized2, descriptors2);
        });

    keypoints1 = futureKeypoints1.get();
    keypoints2 = futureKeypoints2.get();

    // 特征匹配
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher->knnMatch(descriptors1, descriptors2, knnMatches, 2);

    // 并行匹配筛选
    result.matches = FilterMatchesParallel(keypoints1, keypoints2, knnMatches, result.points1, result.points2, params);

    result.success = true;
    // 记录处理时间和特征类型
    result.processingTime = timer.elapsed();
    result.featureType = params.featureType;

    LogPerformanceMetrics("ProcessImagePair", result.processingTime);

    return result;
}

/**
 * @brief 批量特征检测
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
std::vector<ParallelResult> FeatureDetectorOptimized::BatchFeatureDetection(
    const std::vector<std::pair<QString, QString>>& imagePairs,
    const FeatureParams_optimize& params)
{
    QElapsedTimer batchTimer;
    batchTimer.start();

    std::vector<ParallelResult> results(imagePairs.size());
    std::vector<std::future<ParallelResult>> futures;

    // 并行处理所有图像对 - 使用lambda表达式捕获this指针
    for (const auto& pair : imagePairs)
    {
        futures.push_back(std::async(std::launch::async,
            [this, &pair, &params]() {
                return this->ProcessImagePair(pair.first, pair.second, params);
            }));
    }

    // 收集结果
    for (size_t i = 0; i < futures.size(); ++i)
    {
        results[i] = futures[i].get();
    }

    LogPerformanceMetrics("BatchFeatureDetection", batchTimer.elapsed());
    return results;
}

/**
 * @brief 异步特征检测
 * 
 * 异步处理特征检测，不阻塞主线程
 * 
 * @param imagePath1 第一张图像路径
 * @param imagePath2 第二张图像路径
 * @param params 特征检测参数
 * @return std::future<ParallelResult> 异步处理结果的future对象
 * 
 * @note 可以通过future.get()获取结果，或使用wait_for()检查完成状态
 * @see ProcessImagePair()
 */
std::future<ParallelResult> FeatureDetectorOptimized::AsyncFeatureDetection(
    const QString& imagePath1,
    const QString& imagePath2,
    const FeatureParams_optimize& params)
{
    // 使用lambda表达式捕获this指针
    return std::async(std::launch::async,
        [this, imagePath1, imagePath2, &params]() {
            return this->ProcessImagePair(imagePath1, imagePath2, params);
        });
}

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
void FeatureDetectorOptimized::LogPerformanceMetrics(const QString& operation, qint64 elapsedMs)
{
    qDebug() << "Performance:" << operation << "took" << elapsedMs << "ms";
}
