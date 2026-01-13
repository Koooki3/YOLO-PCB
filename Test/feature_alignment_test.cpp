#include <iostream>
#include <opencv2/opencv.hpp>
#include "../VisualRobot/FeatureAlignment.h"
#include <filesystem>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <future>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <chrono>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

struct SSIMResult {
    double ssimValue;
    string imagePath;
    string imageName;
};

// 线程池类
class ThreadPool {
public:
    ThreadPool(size_t numThreads) : stop(false) {
        for(size_t i = 0; i < numThreads; ++i) {
            workers.emplace_back([this] {
                for(;;) {
                    function<void()> task;
                    {
                        unique_lock<mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this]{ return this->stop || !this->tasks.empty(); });
                        if(this->stop && this->tasks.empty())
                            return;
                        task = move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    template<class F>
    future<typename result_of<F()>::type> enqueue(F f) {
        using return_type = typename result_of<F()>::type;
        auto task = make_shared<packaged_task<return_type()>>(move(f));
        future<return_type> res = task->get_future();
        {
            unique_lock<mutex> lock(queue_mutex);
            if(stop)
                throw runtime_error("enqueue on stopped ThreadPool");
            tasks.emplace([task](){ (*task)(); });
        }
        condition.notify_one();
        return res;
    }

    ~ThreadPool() {
        {
            unique_lock<mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for(thread &worker: workers)
            worker.join();
    }

private:
    vector<thread> workers;
    queue<function<void()>> tasks;
    mutex queue_mutex;
    condition_variable condition;
    bool stop;
};

// 并行SSIM计算函数（基于OpenCV并行框架）
class ParallelSSIM {
public:
    static double CalculateSSIMParallel(const Mat& img1, const Mat& img2, int numThreads = 4) {
        // 设置OpenCV并行线程数
        cv::setNumThreads(numThreads);
        
        Mat I1, I2;
        if (img1.channels() == 3) {
            cvtColor(img1, I1, COLOR_BGR2GRAY);
        } else {
            I1 = img1.clone();
        }
        
        if (img2.channels() == 3) {
            cvtColor(img2, I2, COLOR_BGR2GRAY);
        } else {
            I2 = img2.clone();
        }
        
        I1.convertTo(I1, CV_32F);
        I2.convertTo(I2, CV_32F);
        
        const double C1 = 6.5025, C2 = 58.5225;
        
        Mat I1_2 = I1.mul(I1);
        Mat I2_2 = I2.mul(I2);
        Mat I1_I2 = I1.mul(I2);
        
        Mat mu1, mu2;
        GaussianBlur(I1, mu1, Size(11, 11), 1.5);
        GaussianBlur(I2, mu2, Size(11, 11), 1.5);
        
        Mat mu1_2 = mu1.mul(mu1);
        Mat mu2_2 = mu2.mul(mu2);
        Mat mu1_mu2 = mu1.mul(mu2);
        
        Mat sigma1_2, sigma2_2, sigma12;
        GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
        sigma1_2 -= mu1_2;
        
        GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
        sigma2_2 -= mu2_2;
        
        GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
        sigma12 -= mu1_mu2;
        
        Mat t1, t2, t3;
        t1 = 2 * mu1_mu2 + C1;
        t2 = 2 * sigma12 + C2;
        t3 = t1.mul(t2);
        
        t1 = mu1_2 + mu2_2 + C1;
        t2 = sigma1_2 + sigma2_2 + C2;
        t1 = t1.mul(t2);
        
        Mat ssim_map;
        divide(t3, t1, ssim_map);
        
        Scalar mssim = mean(ssim_map);
        return mssim[0];
    }
};

// 单张SSIM计算函数（兼容性保留）
double CalculateSSIM(const Mat& img1, const Mat& img2) {
    return ParallelSSIM::CalculateSSIMParallel(img1, img2, 1);
}

// 单线程批量SSIM计算（用于对比）
vector<SSIMResult> CalculateSSIMWithFolderSingleThread(const Mat& testImage, const string& folderPath) {
    vector<SSIMResult> results;
    
    if (!fs::exists(folderPath)) {
        cout << "文件夹不存在: " << folderPath << endl;
        return results;
    }
    
    cout << "\n开始单线程批量SSIM计算..." << endl;
    cout << "测试图像尺寸: " << testImage.cols << "x" << testImage.rows << endl;
    cout << "==================" << endl;
    
    auto startTime = chrono::high_resolution_clock::now();
    
    int fileCount = 0;
    for (const auto& entry : fs::directory_iterator(folderPath)) {
        if (entry.is_regular_file()) {
            string ext = entry.path().extension().string();
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                string imagePath = entry.path().string();
                string imageName = entry.path().filename().string();
                
                Mat folderImg = imread(imagePath);
                if (folderImg.empty()) {
                    cout << "无法读取图像: " << imageName << endl;
                    continue;
                }
                
                Mat resizedFolderImg;
                if (folderImg.size() != testImage.size()) {
                    resize(folderImg, resizedFolderImg, testImage.size());
                } else {
                    resizedFolderImg = folderImg;
                }
                
                double ssim = CalculateSSIM(testImage, resizedFolderImg);
                
                SSIMResult result;
                result.ssimValue = ssim;
                result.imagePath = imagePath;
                result.imageName = imageName;
                results.push_back(result);
                
                cout << "[" << setw(3) << results.size() << "] " << setw(30) << left << imageName 
                     << " SSIM: " << fixed << setprecision(4) << ssim << endl;
                
                fileCount++;
            }
        }
    }
    
    auto endTime = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime);
    
    cout << "==================" << endl;
    cout << "共处理 " << fileCount << " 张图像" << endl;
    cout << "单线程处理时间: " << duration.count() << " ms" << endl;
    
    return results;
}

// 多线程批量SSIM计算（基于std::async）
vector<SSIMResult> CalculateSSIMWithFolderAsync(const Mat& testImage, const string& folderPath, int numThreads = 4) {
    vector<SSIMResult> results;
    vector<string> imagePaths;
    vector<string> imageNames;
    
    if (!fs::exists(folderPath)) {
        cout << "文件夹不存在: " << folderPath << endl;
        return results;
    }
    
    // 首先收集所有图像路径
    for (const auto& entry : fs::directory_iterator(folderPath)) {
        if (entry.is_regular_file()) {
            string ext = entry.path().extension().string();
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                imagePaths.push_back(entry.path().string());
                imageNames.push_back(entry.path().filename().string());
            }
        }
    }
    
    if (imagePaths.empty()) {
        cout << "文件夹中没有找到图像文件" << endl;
        return results;
    }
    
    cout << "\n开始多线程批量SSIM计算（std::async）..." << endl;
    cout << "测试图像尺寸: " << testImage.cols << "x" << testImage.rows << endl;
    cout << "使用线程数: " << numThreads << endl;
    cout << "==================" << endl;
    
    auto startTime = chrono::high_resolution_clock::now();
    
    // 创建异步任务
    vector<future<SSIMResult>> futures;
    for (size_t i = 0; i < imagePaths.size(); ++i) {
        futures.push_back(async(launch::async, [&testImage, &imagePaths, &imageNames, i]() {
            Mat folderImg = imread(imagePaths[i]);
            if (folderImg.empty()) {
                SSIMResult empty;
                empty.ssimValue = -1.0;
                empty.imageName = imageNames[i];
                return empty;
            }
            
            Mat resizedFolderImg;
            if (folderImg.size() != testImage.size()) {
                resize(folderImg, resizedFolderImg, testImage.size());
            } else {
                resizedFolderImg = folderImg;
            }
            
            double ssim = ParallelSSIM::CalculateSSIMParallel(testImage, resizedFolderImg, 2);
            
            SSIMResult result;
            result.ssimValue = ssim;
            result.imagePath = imagePaths[i];
            result.imageName = imageNames[i];
            return result;
        }));
    }
    
    // 收集结果
    results.resize(futures.size());
    for (size_t i = 0; i < futures.size(); ++i) {
        results[i] = futures[i].get();
        if (results[i].ssimValue >= 0) {
            cout << "[" << setw(3) << (i+1) << "] " << setw(30) << left << results[i].imageName 
                 << " SSIM: " << fixed << setprecision(4) << results[i].ssimValue << endl;
        }
    }
    
    auto endTime = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime);
    
    cout << "==================" << endl;
    cout << "共处理 " << imagePaths.size() << " 张图像" << endl;
    cout << "多线程（std::async）处理时间: " << duration.count() << " ms" << endl;
    
    return results;
}

// 线程池批量SSIM计算
vector<SSIMResult> CalculateSSIMWithFolderThreadPool(const Mat& testImage, const string& folderPath, int numThreads = 4) {
    vector<SSIMResult> results;
    vector<string> imagePaths;
    vector<string> imageNames;
    
    if (!fs::exists(folderPath)) {
        cout << "文件夹不存在: " << folderPath << endl;
        return results;
    }
    
    // 首先收集所有图像路径
    for (const auto& entry : fs::directory_iterator(folderPath)) {
        if (entry.is_regular_file()) {
            string ext = entry.path().extension().string();
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                imagePaths.push_back(entry.path().string());
                imageNames.push_back(entry.path().filename().string());
            }
        }
    }
    
    if (imagePaths.empty()) {
        cout << "文件夹中没有找到图像文件" << endl;
        return results;
    }
    
    cout << "\n开始线程池批量SSIM计算..." << endl;
    cout << "测试图像尺寸: " << testImage.cols << "x" << testImage.rows << endl;
    cout << "使用线程数: " << numThreads << endl;
    cout << "==================" << endl;
    
    auto startTime = chrono::high_resolution_clock::now();
    
    ThreadPool pool(numThreads);
    vector<future<SSIMResult>> futures;
    
    // 提交任务到线程池
    for (size_t i = 0; i < imagePaths.size(); ++i) {
        futures.push_back(pool.enqueue([&testImage, &imagePaths, &imageNames, i]() {
            Mat folderImg = imread(imagePaths[i]);
            if (folderImg.empty()) {
                SSIMResult empty;
                empty.ssimValue = -1.0;
                empty.imageName = imageNames[i];
                return empty;
            }
            
            Mat resizedFolderImg;
            if (folderImg.size() != testImage.size()) {
                resize(folderImg, resizedFolderImg, testImage.size());
            } else {
                resizedFolderImg = folderImg;
            }
            
            double ssim = ParallelSSIM::CalculateSSIMParallel(testImage, resizedFolderImg, 2);
            
            SSIMResult result;
            result.ssimValue = ssim;
            result.imagePath = imagePaths[i];
            result.imageName = imageNames[i];
            return result;
        }));
    }
    
    // 收集结果
    results.resize(futures.size());
    for (size_t i = 0; i < futures.size(); ++i) {
        results[i] = futures[i].get();
        if (results[i].ssimValue >= 0) {
            cout << "[" << setw(3) << (i+1) << "] " << setw(30) << left << results[i].imageName 
                 << " SSIM: " << fixed << setprecision(4) << results[i].ssimValue << endl;
        }
    }
    
    auto endTime = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime);
    
    cout << "==================" << endl;
    cout << "共处理 " << imagePaths.size() << " 张图像" << endl;
    cout << "线程池处理时间: " << duration.count() << " ms" << endl;
    
    return results;
}

// 兼容性函数（默认使用线程池）
vector<SSIMResult> CalculateSSIMWithFolder(const Mat& testImage, const string& folderPath) {
    return CalculateSSIMWithFolderThreadPool(testImage, folderPath, 4);
}

SSIMResult FindBestMatch(const vector<SSIMResult>& results) {
    if (results.empty()) {
        SSIMResult empty;
        empty.ssimValue = -1.0;
        return empty;
    }
    
    auto bestIt = max_element(results.begin(), results.end(), 
        [](const SSIMResult& a, const SSIMResult& b) {
            return a.ssimValue < b.ssimValue;
        });
    
    return *bestIt;
}

int main() {
    cout << "特征对齐功能测试（增强版 - 多线程优化）" << endl;
    cout << "========================================" << endl;
    
    string testImagePath, templateFolderPath;
    int threadMode;
    
    cout << "请输入待测试图像路径: ";
    getline(cin, testImagePath);
    
    cout << "请输入模板图像文件夹路径: ";
    getline(cin, templateFolderPath);
    
    cout << "\n请选择线程模式:" << endl;
    cout << "1. 单线程（基准测试）" << endl;
    cout << "2. 多线程（std::async）" << endl;
    cout << "3. 线程池（推荐）" << endl;
    cout << "4. 全部模式对比测试" << endl;
    cout << "请输入选项 (1-4): ";
    cin >> threadMode;
    cin.ignore();
    
    Mat testImage = imread(testImagePath);
    if (testImage.empty()) {
        cout << "无法读取测试图像，请检查文件路径" << endl;
        return -1;
    }
    
    cout << "\n测试图像尺寸: " << testImage.cols << "x" << testImage.rows << endl;
    cout << "测试图像通道数: " << testImage.channels() << endl;
    
    vector<SSIMResult> ssimResults;
    
    if (threadMode == 1) {
        ssimResults = CalculateSSIMWithFolderSingleThread(testImage, templateFolderPath);
    } else if (threadMode == 2) {
        ssimResults = CalculateSSIMWithFolderAsync(testImage, templateFolderPath, 4);
    } else if (threadMode == 3) {
        ssimResults = CalculateSSIMWithFolderThreadPool(testImage, templateFolderPath, 4);
    } else if (threadMode == 4) {
        cout << "\n========== 性能对比测试 ==========" << endl;
        
        auto totalStart = chrono::high_resolution_clock::now();
        
        auto singleStart = chrono::high_resolution_clock::now();
        vector<SSIMResult> singleResults = CalculateSSIMWithFolderSingleThread(testImage, templateFolderPath);
        auto singleEnd = chrono::high_resolution_clock::now();
        auto singleDuration = chrono::duration_cast<chrono::milliseconds>(singleEnd - singleStart);
        
        auto asyncStart = chrono::high_resolution_clock::now();
        vector<SSIMResult> asyncResults = CalculateSSIMWithFolderAsync(testImage, templateFolderPath, 4);
        auto asyncEnd = chrono::high_resolution_clock::now();
        auto asyncDuration = chrono::duration_cast<chrono::milliseconds>(asyncEnd - asyncStart);
        
        auto poolStart = chrono::high_resolution_clock::now();
        vector<SSIMResult> poolResults = CalculateSSIMWithFolderThreadPool(testImage, templateFolderPath, 4);
        auto poolEnd = chrono::high_resolution_clock::now();
        auto poolDuration = chrono::duration_cast<chrono::milliseconds>(poolEnd - poolStart);
        
        auto totalEnd = chrono::high_resolution_clock::now();
        auto totalDuration = chrono::duration_cast<chrono::milliseconds>(totalEnd - totalStart);
        
        cout << "\n========== 性能对比结果 ==========" << endl;
        cout << "单线程耗时: " << singleDuration.count() << " ms" << endl;
        cout << "std::async耗时: " << asyncDuration.count() << " ms" << endl;
        cout << "线程池耗时: " << poolDuration.count() << " ms" << endl;
        cout << "总测试时间: " << totalDuration.count() << " ms" << endl;
        
        double speedupAsync = (double)singleDuration.count() / asyncDuration.count();
        double speedupPool = (double)singleDuration.count() / poolDuration.count();
        
        cout << "\n加速比:" << endl;
        cout << "std::async加速比: " << fixed << setprecision(2) << speedupAsync << "x" << endl;
        cout << "线程池加速比: " << fixed << setprecision(2) << speedupPool << "x" << endl;
        cout << "==================================" << endl;
        
        ssimResults = poolResults;
    }
    
    if (ssimResults.empty()) {
        cout << "未找到任何有效的模板图像" << endl;
        return -1;
    }
    
    SSIMResult bestMatch = FindBestMatch(ssimResults);
    
    cout << "\n最佳匹配结果:" << endl;
    cout << "==================" << endl;
    cout << "图像名称: " << bestMatch.imageName << endl;
    cout << "图像路径: " << bestMatch.imagePath << endl;
    cout << "SSIM值: " << fixed << setprecision(4) << bestMatch.ssimValue << endl;
    cout << "==================" << endl;
    
    Mat bestTemplateImg = imread(bestMatch.imagePath);
    if (bestTemplateImg.empty()) {
        cout << "无法读取最佳匹配图像" << endl;
        return -1;
    }
    
    FeatureAlignment alignment;
    
    AlignmentParams params;
    params.minInliers = 10;
    params.enableParallel = true;
    params.numThreads = 4;
    
    cout << "\n开始特征对齐..." << endl;
    
    AlignmentResult result = alignment.FastAlignImages(testImage, bestTemplateImg, params);
    
    if (result.success) {
        cout << "特征对齐成功!" << endl;
        cout << "内点数量: " << result.inlierCount << endl;
        cout << "重投影误差: " << fixed << setprecision(4) << result.reprojectionError << endl;
        
        Mat alignedImage = alignment.WarpImage(testImage, result.transformMatrix, bestTemplateImg.size());
        
        if (!alignedImage.empty()) {
            cout << "图像重构成功，尺寸: " << alignedImage.cols << "x" << alignedImage.rows << endl;
            
            string outputDir = "../Img/Results/";
            fs::create_directories(outputDir);
            
            string alignedPath = outputDir + "aligned_result.jpg";
            imwrite(alignedPath, alignedImage);
            cout << "对齐结果已保存到: " << alignedPath << endl;
            
            string bestMatchCopyPath = outputDir + "best_match_template.jpg";
            imwrite(bestMatchCopyPath, bestTemplateImg);
            cout << "最佳匹配模板已保存到: " << bestMatchCopyPath << endl;
            
            string testImageCopyPath = outputDir + "test_image.jpg";
            imwrite(testImageCopyPath, testImage);
            cout << "测试图像已保存到: " << testImageCopyPath << endl;
        } else {
            cout << "图像重构失败" << endl;
        }
    } else {
        cout << "特征对齐失败" << endl;
        cout << "内点数量: " << result.inlierCount << endl;
    }
    
    cout << "\n测试完成" << endl;
    return 0;
}
