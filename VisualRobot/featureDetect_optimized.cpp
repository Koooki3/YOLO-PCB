#include "featureDetect_optimized.h"
#include "DataProcessor.h"
#include "DLProcessor.h"
#include <QElapsedTimer>
#include <algorithm>
#include <numeric>

// 静态成员变量定义
static std::vector<std::thread> g_threadPool;
static std::atomic<bool> g_threadPoolRunning{false};
static std::mutex g_threadPoolMutex;
static std::condition_variable g_threadPoolCV;

FeatureDetectorOptimized::FeatureDetectorOptimized()
{
    // 默认初始化线程池
    InitializeThreadPool(4);
}

FeatureDetectorOptimized::~FeatureDetectorOptimized()
{
    ShutdownThreadPool();
}

void FeatureDetectorOptimized::InitializeThreadPool(int numThreads)
{
    std::lock_guard<std::mutex> lock(g_threadPoolMutex);

    if (g_threadPoolRunning) 
    {
        ShutdownThreadPool();
    }

    g_threadPoolRunning = true;
    g_threadPool.clear();

    for (int i = 0; i < numThreads; ++i) 
    {
        g_threadPool.emplace_back([]() {
            while (g_threadPoolRunning) 
            {
                std::unique_lock<std::mutex> lock(g_threadPoolMutex);
                g_threadPoolCV.wait(lock, []() {
                    return !g_threadPoolRunning;
                });
            }
        });
    }

    qDebug() << "Thread pool initialized with" << numThreads << "threads";
}

void FeatureDetectorOptimized::ShutdownThreadPool()
{
    {
        std::lock_guard<std::mutex> lock(g_threadPoolMutex);
        g_threadPoolRunning = false;
    }

    g_threadPoolCV.notify_all();

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

    std::vector<cv::DMatch> goodMatches;
    points1.clear();
    points2.clear();

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
        const size_t chunkSize1 = (keypoints1.size() + numThreads - 1) / numThreads;
        const size_t chunkSize2 = (keypoints2.size() + numThreads - 1) / numThreads;

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
        const size_t chunkSizeMatches = (knnMatches.size() + numThreads - 1) / numThreads;

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

        for (auto& thread : threads) 
        {
            if (thread.joinable()) 
            {
                thread.join();
            }
        }

        // 合并结果
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

                for (size_t i = 0; i < inlierMask.size(); i++) 
                {
                    if (inlierMask[i]) 
                    {
                        ransacMatches.push_back(goodMatches[i]);
                        inlierPoints1.push_back(points1[i]);
                        inlierPoints2.push_back(points2[i]);
                    }
                }

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

void FeatureDetectorOptimized::ParallelKeypointFilter(
    const std::vector<cv::KeyPoint>& keypoints,
    std::vector<bool>& validFlags,
    float responseThresh,
    size_t startIdx,
    size_t endIdx)
{
    for (size_t i = startIdx; i < endIdx; ++i) 
    {
        if (keypoints[i].response > responseThresh) 
        {
            validFlags[i] = true;
        }
    }
}

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

std::vector<cv::DMatch> FeatureDetectorOptimized::ParallelRANSAC(
    const std::vector<cv::DMatch>& goodMatches,
    std::vector<cv::Point2f>& points1,
    std::vector<cv::Point2f>& points2,
    const FeatureParams_optimize& params)
{
    // 由于RANSAC算法本身是随机采样，难以直接并行化
    // 这里采用多次RANSAC并行执行，选择最佳结果的方法

    const int numRANSACRuns = std::min(params.numThreads, 8); // 限制最大运行次数
    std::vector<std::future<std::tuple<std::vector<cv::DMatch>, std::vector<cv::Point2f>, std::vector<cv::Point2f>, int>>> futures;

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

void FeatureDetectorOptimized::TestFeatureDetectionParallel(const QString& imagePath1, const QString& imagePath2)
{
    QElapsedTimer totalTimer;
    totalTimer.start();

    FeatureParams_optimize params;
    params.enableParallel = true;
    params.numThreads = std::thread::hardware_concurrency();

    auto result = ProcessImagePair(imagePath1, imagePath2, params);

    if (result.success) 
    {
        qDebug() << "Parallel Feature Detection Completed in" << totalTimer.elapsed() << "ms";
        qDebug() << "Good matches:" << result.matches.size();
        qDebug() << "Inlier points:" << result.points1.size();
    } 
    else 
    {
        qDebug() << "Parallel Feature Detection Failed";
    }
}

// 修复：移除类作用域限定
ParallelResult FeatureDetectorOptimized::ProcessImagePair(
    const QString& imagePath1,
    const QString& imagePath2,
    const FeatureParams_optimize& params)
{
    ParallelResult result;

    DataProcessor dataProcessor;
    cv::Mat image1 = cv::imread(imagePath1.toStdString());
    cv::Mat image2 = cv::imread(imagePath2.toStdString());

    if (image1.empty() || image2.empty()) 
    {
        qDebug() << "Error: Could not read the images";
        return result;
    }

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
    return result;
}

// 修复：移除类作用域限定
std::vector<ParallelResult> FeatureDetectorOptimized::BatchFeatureDetection(
    const std::vector<std::pair<QString, QString>>& imagePairs,
    const FeatureParams_optimize& params)
{
    QElapsedTimer batchTimer;
    batchTimer.start();

    std::vector<ParallelResult> results(imagePairs.size());
    std::vector<std::future<ParallelResult>> futures;

    // 并行处理所有图像对
    for (const auto& pair : imagePairs) 
    {
        futures.push_back(std::async(std::launch::async, ProcessImagePair, pair.first, pair.second, params));
    }

    // 收集结果
    for (size_t i = 0; i < futures.size(); ++i) 
    {
        results[i] = futures[i].get();
    }

    LogPerformanceMetrics("BatchFeatureDetection", batchTimer.elapsed());
    return results;
}

// 修复：移除类作用域限定
std::future<ParallelResult> FeatureDetectorOptimized::AsyncFeatureDetection(
    const QString& imagePath1,
    const QString& imagePath2,
    const FeatureParams_optimize& params)
{
    return std::async(std::launch::async, ProcessImagePair, imagePath1, imagePath2, params);
}

void FeatureDetectorOptimized::LogPerformanceMetrics(const QString& operation, qint64 elapsedMs)
{
    qDebug() << "Performance:" << operation << "took" << elapsedMs << "ms";
}
