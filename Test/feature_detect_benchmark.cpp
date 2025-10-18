#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include "featureDetect.h"
#include "featureDetect_optimized.h"

class FeatureDetectBenchmark {
public:
    static void RunBenchmark(const std::string& imagePath1, const std::string& imagePath2) {
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
    std::cout << "\n--- Async Processing Test ---" << std::endl;

    // 使用优化版本的特征参数
    FeatureParams_optimize params;
    params.enableParallel = true;
    params.numThreads = 4;

    // 启动多个异步任务
    std::vector<std::future<ParallelResult>> futures;

    std::vector<std::pair<QString, QString>> imagePairs = {
        {"../Img/test5.jpg", "../Img/test6.jpg"},
        {"../Img/test1.jpg", "../Img/test2.jpg"}
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
    std::cout << "Feature Detection Performance Benchmark" << std::endl;
    std::cout << "=======================================" << std::endl;

    // 使用项目中的测试图像
    std::string image1 = "../Img/test5.jpg";
    std::string image2 = "../Img/test6.jpg";

    // 运行基准测试
    FeatureDetectBenchmark::RunBenchmark(image1, image2);

    // 测试异步处理
    TestAsyncProcessing();

    return 0;
}
