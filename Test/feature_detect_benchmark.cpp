#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <QString>
#include "featureDetect.h"
#include "featureDetect_optimized.h"
#include "DataProcessor.h"

class FeatureDetectBenchmark {
public:
    static void RunBenchmark(const std::string& imagePath1, const std::string& imagePath2) {
        // 运行特征提取算法比较测试
        TestAllFeatureExtractors(imagePath1, imagePath2);
        
        std::cout << "\n" << std::endl;
        std::cout << "=== 其他基准测试 ===" << std::endl;
        std::cout << "=== Feature Detection Benchmark ===" << std::endl;
        std::cout << "Image 1: " << imagePath1 << std::endl;
        std::cout << "Image 2: " << imagePath2 << std::endl;
        std::cout << "===================================" << std::endl;

        // 测试原始版本
        std::cout << "\n--- Original Algorithm ---" << std::endl;
        auto originalTime = TestOriginalAlgorithm(imagePath1, imagePath2);

        // 测试并行版本 - 不同线程数
        std::vector<int> threadCounts = {1, 2, 4, 8};
        for (int threads : threadCounts) {
            std::cout << "\n--- Parallel Algorithm (" << threads << " threads) ---" << std::endl;
            auto parallelTime = TestParallelAlgorithm(imagePath1, imagePath2, threads);

            double speedup = static_cast<double>(originalTime) / parallelTime;
            std::cout << "Speedup: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
        }

        // 测试批量处理
        std::cout << "\n--- Batch Processing Test ---" << std::endl;
        TestBatchProcessing();

        std::cout << "\n=== Benchmark Completed ===" << std::endl;
    }

private:
    static long long TestOriginalAlgorithm(const std::string& imagePath1, const std::string& imagePath2) {
        auto start = std::chrono::high_resolution_clock::now();

        // 调用原始算法
        featureDetector::TestFeatureDetection(
            QString::fromStdString(imagePath1),
            QString::fromStdString(imagePath2)
        );

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Execution Time: " << duration.count() << " ms" << std::endl;
        return duration.count();
    }

    static long long TestParallelAlgorithm(const std::string& imagePath1, const std::string& imagePath2, int numThreads) {
        auto start = std::chrono::high_resolution_clock::now();

        // 配置并行参数 - 使用优化版本的特征参数
        FeatureParams_optimize params;  // 这会使用 featureDetect_optimized.h 中的定义
        params.enableParallel = true;
        params.numThreads = numThreads;

        // 调用并行算法
        FeatureDetectorOptimized::TestFeatureDetectionParallel(
            QString::fromStdString(imagePath1),
            QString::fromStdString(imagePath2)
        );

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Execution Time: " << duration.count() << " ms" << std::endl;
        return duration.count();
    }

    static void TestAllFeatureExtractors(const std::string& imagePath1, const std::string& imagePath2) {
        std::cout << "=== 所有特征提取算法比较测试 ===" << std::endl;
        std::cout << "图像1: " << imagePath1 << std::endl;
        std::cout << "图像2: " << imagePath2 << std::endl;
        std::cout << "===================================" << std::endl;
        
        // 测试SIFT特征提取器
        std::cout << "\n--- SIFT特征提取器 ---" << std::endl;
        TestFeatureExtractor(imagePath1, imagePath2, FeatureType::SIFT);
        
    
        
        // 测试ORB特征提取器
        std::cout << "\n--- ORB特征提取器 ---" << std::endl;
        TestFeatureExtractor(imagePath1, imagePath2, FeatureType::ORB);
        
        // 测试AKAZE特征提取器
        std::cout << "\n--- AKAZE特征提取器 ---" << std::endl;
        TestFeatureExtractor(imagePath1, imagePath2, FeatureType::AKAZE);
        
        std::cout << "\n=== 特征提取算法比较完成 ===" << std::endl;
    }
    
    static void TestFeatureExtractor(const std::string& imagePath1, const std::string& imagePath2, FeatureType featureType) {
        
        // 设置特征检测参数
        FeatureParams params;
        params.featureType = featureType;
        params.useRansac = true;
        params.minInliers = 10;
        params.ratioThresh = 0.7f;  // 匹配比率阈值
        
        // 创建DataProcessor实例
        DataProcessor dataProcessor;
        dataProcessor.SetFeatureType(featureType);
        
        // 读取图像
        cv::Mat image1 = cv::imread(imagePath1);
        cv::Mat image2 = cv::imread(imagePath2);
        
        if (image1.empty() || image2.empty()) {
            std::cout << "错误: 无法读取图像文件" << std::endl;
            return;
        }
        
        // 记录开始时间
        auto start = std::chrono::high_resolution_clock::now();
        
        // 图像预处理
        cv::Mat standardized1 = dataProcessor.StandardizeImage(image1);
        cv::Mat standardized2 = dataProcessor.StandardizeImage(image2);
        
        // 提取特征
        std::vector<cv::KeyPoint> keypoints1;
        std::vector<cv::KeyPoint> keypoints2;
        cv::Mat descriptors1;
        cv::Mat descriptors2;
        std::vector<std::vector<cv::DMatch>> knnMatches;
        std::vector<cv::Point2f> points1;
        std::vector<cv::Point2f> points2;
        std::vector<cv::DMatch> goodMatches;
        
        try {
            keypoints1 = dataProcessor.DetectKeypoints(standardized1, descriptors1);
            keypoints2 = dataProcessor.DetectKeypoints(standardized2, descriptors2);
            
            if (keypoints1.empty() || keypoints2.empty() || descriptors1.empty() || descriptors2.empty()) {
                std::cout << "错误: 无法提取足够的特征点或描述符" << std::endl;
                return;
            }
            
            // 特征匹配
            cv::Ptr<cv::DescriptorMatcher> matcher;
            
            // 根据特征类型选择合适的匹配器
            if (featureType == FeatureType::ORB || featureType == FeatureType::AKAZE) {
                matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
            } else {
                matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
            }
            
            matcher->knnMatch(descriptors1, descriptors2, knnMatches, 2);
            
            // 筛选匹配点
            goodMatches = featureDetector::FilterMatches(
                keypoints1, keypoints2, knnMatches, points1, points2, params);
            
            // 记录结束时间
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            // 输出结果
            std::cout << "处理时间: " << duration.count() << " ms" << std::endl;
            std::cout << "特征点数量1: " << keypoints1.size() << std::endl;
            std::cout << "特征点数量2: " << keypoints2.size() << std::endl;
            std::cout << "匹配点数量: " << goodMatches.size() << std::endl;
            std::cout << "内点数量: " << points1.size() << std::endl;
            
            // 获取特征提取器名称
            std::string featureName;
            if (featureType == FeatureType::SIFT) featureName = "SIFT";
            else if (featureType == FeatureType::SURF) featureName = "SURF";
            else if (featureType == FeatureType::ORB) featureName = "ORB";
            else if (featureType == FeatureType::AKAZE) featureName = "AKAZE";
            
            // 绘制匹配结果
            cv::Mat matchImage;
            cv::drawMatches(
                image1, keypoints1,
                image2, keypoints2,
                goodMatches,
                matchImage,
                cv::Scalar::all(-1), cv::Scalar::all(-1),
                std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
            );
            
            // 保存匹配结果图像
            std::string outputPath = "../feature_matches_" + featureName + ".jpg";
            if (cv::imwrite(outputPath, matchImage)) {
                std::cout << "特征点连线图像已保存至: " << outputPath << std::endl;
            } else {
                std::cout << "警告: 无法保存特征点连线图像" << std::endl;
            }
            
            // 如果有内点，绘制内点连线
            if (!points1.empty()) {
                cv::Mat inlierImage;
                // 由于FilterMatches已经返回内点的匹配，我们可以直接使用goodMatches
                std::vector<cv::DMatch> inlierMatches = goodMatches;
                
                cv::drawMatches(
                    image1, keypoints1,
                    image2, keypoints2,
                    inlierMatches,
                    inlierImage,
                    cv::Scalar::all(-1), cv::Scalar(0, 255, 0),
                    std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
                );
                
                std::string inlierPath = "../feature_inliers_" + featureName + ".jpg";
                if (cv::imwrite(inlierPath, inlierImage)) {
                    std::cout << "内点连线图像已保存至: " << inlierPath << std::endl;
                } else {
                    std::cout << "警告: 无法保存内点连线图像" << std::endl;
                }
            }
        } catch (const cv::Exception& e) {
            std::cout << "错误: 特征提取失败 - " << e.what() << std::endl;
        } catch (const std::exception& e) {
            std::cout << "错误: 处理异常 - " << e.what() << std::endl;
        }
    }
    
    static void TestBatchProcessing() {
        // 创建测试图像对列表
        std::vector<std::pair<QString, QString>> imagePairs = {
            {"../Img/test1.jpg", "../Img/test2.jpg"},
            {"../Img/test3.jpg", "../Img/test4.jpg"},
            // 可以根据需要添加更多测试图像对
        };

        // 配置并行参数 - 使用优化版本的特征参数
        FeatureParams_optimize params;  // 这会使用 featureDetect_optimized.h 中的定义
        params.enableParallel = true;
        params.numThreads = 4;

        auto start = std::chrono::high_resolution_clock::now();

        // 执行批量处理
        auto results = FeatureDetectorOptimized::BatchFeatureDetection(imagePairs, params);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Batch processed " << imagePairs.size() << " image pairs" << std::endl;
        std::cout << "Total Time: " << duration.count() << " ms" << std::endl;
        std::cout << "Average Time per Pair: " << duration.count() / imagePairs.size() << " ms" << std::endl;

        // 输出结果统计
        int successfulPairs = 0;
        int totalMatches = 0;
        for (const auto& result : results) {
            if (result.success) {
                successfulPairs++;
                totalMatches += result.matches.size();
            }
        }

        std::cout << "Successful Pairs: " << successfulPairs << "/" << imagePairs.size() << std::endl;
        std::cout << "Total Matches Found: " << totalMatches << std::endl;
        if (successfulPairs > 0) {
            std::cout << "Average Matches per Pair: " << totalMatches / successfulPairs << std::endl;
        }
    }
};

// 异步处理测试函数
void TestAsyncProcessing() {
    std::cout << "\n--- 异步处理测试 ---" << std::endl;

    // 使用优化版本的特征参数
    FeatureParams_optimize params;
    params.enableParallel = true;
    params.numThreads = 4;

    // 启动多个异步任务
    std::vector<std::future<ParallelResult>> futures;

    std::vector<std::pair<QString, QString>> imagePairs = {
        {"../Img/test1.jpg", "../Img/test2.jpg"},
        {"../Img/test3.jpg", "../Img/test4.jpg"}
    };

    auto start = std::chrono::high_resolution_clock::now();

    for (const auto& pair : imagePairs) {
        futures.push_back(FeatureDetectorOptimized::AsyncFeatureDetection(
            pair.first, pair.second, params
        ));
    }

    std::cout << "Async tasks launched, waiting for completion..." << std::endl;

    // 等待所有任务完成
    for (auto& future : futures) {
        auto result = future.get();
        if (result.success) {
            std::cout << "Async task completed with " << result.matches.size() << " matches" << std::endl;
        } else {
            std::cout << "Async task failed" << std::endl;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "All async tasks completed in " << duration.count() << " ms" << std::endl;
}

int main() {
    std::cout << "特征提取算法测试工具" << std::endl;
    std::cout << "=======================================" << std::endl;
    std::cout << "本工具将对指定的两幅图像使用SIFT、ORB和AKAZE三种特征提取算法进行测试" << std::endl;
    std::cout << "并生成每种算法的特征点连线图像和处理性能数据" << std::endl;
    std::cout << "=======================================" << std::endl;

    // 获取用户输入的图像路径
    std::string image1, image2;
    
    std::cout << "请输入第一幅图像的路径 (默认为 ../Img/test1.jpg): ";
    std::getline(std::cin, image1);
    if (image1.empty()) {
        image1 = "../Img/test1.jpg";
    }
    
    std::cout << "请输入第二幅图像的路径 (默认为 ../Img/test2.jpg): ";
    std::getline(std::cin, image2);
    if (image2.empty()) {
        image2 = "../Img/test2.jpg";
    }
    
    // 验证图像文件是否存在
    if (!cv::imread(image1).data) {
        std::cerr << "错误: 无法加载第一幅图像: " << image1 << std::endl;
        return 1;
    }
    
    if (!cv::imread(image2).data) {
        std::cerr << "错误: 无法加载第二幅图像: " << image2 << std::endl;
        return 1;
    }
    
    std::cout << "\n图像加载成功，开始测试...\n" << std::endl;

    // 运行基准测试
    FeatureDetectBenchmark::RunBenchmark(image1, image2);

    // 测试异步处理
    TestAsyncProcessing();
    
    std::cout << "\n=======================================" << std::endl;
    std::cout << "测试完成!" << std::endl;
    std::cout << "特征点连线图像已保存至项目根目录，命名格式为 feature_matches_<算法名>.jpg" << std::endl;
    std::cout << "内点连线图像已保存至项目根目录，命名格式为 feature_inliers_<算法名>.jpg" << std::endl;

    return 0;
}
