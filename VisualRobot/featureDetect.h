#ifndef FEATUREDETECT_H
#define FEATUREDETECT_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <QString>
#include <QDebug>
#include "DataProcessor.h"

/**
 * @brief 特征识别参数结构体
 * 
 * 存储特征检测和匹配的各种参数，用于配置特征检测算法的行为
 */
struct FeatureParams
{
    float ratioThresh = 0.7f;        // 匹配比率阈值，用于筛选匹配点对
    float responseThresh = 0.0f;      // 特征点响应值阈值，用于筛选强特征点
    float ransacReprojThresh = 3.0f;  // RANSAC重投影阈值，用于几何验证
    int minInliers = 10;             // 最小内点数量，用于判断匹配是否有效
    bool useRansac = true;           // 是否使用RANSAC验证匹配结果
    FeatureType featureType = FeatureType::SIFT; // 特征提取器类型，可选择SIFT、ORB或AKAZE
};

/**
 * @brief 特征检测器类
 * 
 * 提供特征检测、匹配和几何验证功能，支持多种特征提取算法
 */
class featureDetector
{
public:
    /**
     * @brief 构造函数
     */
    featureDetector();
    
    /**
     * @brief 析构函数
     */
    ~featureDetector();

    /**
     * @brief 特征匹配和几何验证函数
     * @param keypoints1 第一张图像的特征点
     * @param keypoints2 第二张图像的特征点
     * @param knnMatches KNN匹配结果
     * @param points1 输出的第一张图像的匹配点坐标
     * @param points2 输出的第二张图像的匹配点坐标
     * @param params 特征检测参数
     * @return 筛选后的匹配点对
     * 
     * 对特征匹配进行多阶段筛选，包括响应值筛选、比率测试和RANSAC几何验证
     */
    static std::vector<cv::DMatch> FilterMatches(
        const std::vector<cv::KeyPoint>& keypoints1,
        const std::vector<cv::KeyPoint>& keypoints2,
        const std::vector<std::vector<cv::DMatch>>& knnMatches,
        std::vector<cv::Point2f>& points1,
        std::vector<cv::Point2f>& points2,
        const FeatureParams& params);

    /**
     * @brief 特征识别测试函数
     * @param imagePath1 第一张图像的路径
     * @param imagePath2 第二张图像的路径
     * 
     * 完整的特征检测流程测试，包括图像读取、预处理、特征提取、匹配和结果可视化
     */
    static void TestFeatureDetection(const QString& imagePath1, const QString& imagePath2);

private:
    // 可以在这里添加类的私有成员
};

#endif // FEATUREDETECT_H
