#ifndef FEATUREDETECT_H
#define FEATUREDETECT_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <QString>
#include <QDebug>

// 特征识别参数结构体
struct FeatureParams
{
    float ratioThresh = 0.7f;        // SIFT匹配比率阈值
    float responseThresh = 0.0f;      // 特征点响应值阈值
    float ransacReprojThresh = 3.0f;  // RANSAC重投影阈值
    int minInliers = 10;             // 最小内点数量
    bool useRansac = true;           // 是否使用RANSAC验证
};

class FeatureDetector
{
public:
    FeatureDetector();
    ~FeatureDetector();

    // 特征匹配和几何验证函数
    static std::vector<cv::DMatch> filterMatches(
        const std::vector<cv::KeyPoint>& keypoints1,
        const std::vector<cv::KeyPoint>& keypoints2,
        const std::vector<std::vector<cv::DMatch>>& knnMatches,
        std::vector<cv::Point2f>& points1,
        std::vector<cv::Point2f>& points2,
        const FeatureParams& params);

    // 特征识别测试函数
    static void testFeatureDetection(const QString& imagePath1, const QString& imagePath2);

private:
    // 可以在这里添加类的私有成员
};

#endif // FEATUREDETECT_H
