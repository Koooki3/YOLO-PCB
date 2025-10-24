#ifndef FEATUREALIGNMENT_H
#define FEATUREALIGNMENT_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <QString>
#include <QDebug>
#include "featureDetect_optimized.h"

// 特征对齐参数结构体
struct AlignmentParams
{
    int minInliers = 10;              // 最小内点数量（匹配到10个即可停止）
    float ransacReprojThresh = 3.0f;  // RANSAC重投影阈值
    bool enableParallel = true;       // 启用并行计算
    int numThreads = 4;               // 并行线程数
    int maxIterations = 1000;         // 最大迭代次数
    double confidence = 0.99;         // 置信度
};

// 对齐结果结构体
struct AlignmentResult
{
    cv::Mat transformMatrix;          // 变换矩阵（单应性矩阵）
    std::vector<cv::DMatch> matches;  // 匹配结果
    std::vector<cv::Point2f> srcPoints; // 源图像特征点
    std::vector<cv::Point2f> dstPoints; // 目标图像特征点
    bool success = false;             // 是否成功
    int inlierCount = 0;              // 内点数量
    double reprojectionError = 0.0;   // 重投影误差
};

class FeatureAlignment
{
public:
    FeatureAlignment();
    ~FeatureAlignment();

    // 使用特征匹配对齐两幅图像
    AlignmentResult AlignImages(const cv::Mat& srcImage, const cv::Mat& dstImage, const AlignmentParams& params = AlignmentParams());

    // 使用特征匹配对齐两幅图像（灰度图版本）
    AlignmentResult AlignImagesGray(const cv::Mat& srcGray, const cv::Mat& dstGray, const AlignmentParams& params = AlignmentParams());

    // 重构图像：根据变换矩阵将源图像变换到目标图像坐标系
    cv::Mat WarpImage(const cv::Mat& srcImage, const cv::Mat& transformMatrix, const cv::Size& dstSize);

    // 快速对齐：当匹配到足够内点时立即停止
    AlignmentResult FastAlignImages(const cv::Mat& srcImage, const cv::Mat& dstImage, const AlignmentParams& params = AlignmentParams());

    // 获取对齐状态信息
    QString GetAlignmentInfo(const AlignmentResult& result) const;

private:
    // 特征检测和描述符提取
    bool ExtractFeatures(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

    // 特征匹配
    bool MatchFeatures(const cv::Mat& descriptors1, const cv::Mat& descriptors2, std::vector<cv::DMatch>& matches);

    // 几何验证和变换矩阵计算
    bool GeometricVerification(const std::vector<cv::KeyPoint>& keypoints1, const std::vector<cv::KeyPoint>& keypoints2, const std::vector<cv::DMatch>& matches, cv::Mat& transformMatrix, std::vector<cv::DMatch>& inlierMatches, const AlignmentParams& params);

    // 计算重投影误差
    double ComputeReprojectionError(const std::vector<cv::Point2f>& points1, const std::vector<cv::Point2f>& points2, const cv::Mat& transformMatrix);

    // 特征检测器
    cv::Ptr<cv::Feature2D> m_featureDetector;
    // 特征匹配器
    cv::Ptr<cv::DescriptorMatcher> m_featureMatcher;
};

#endif // FEATUREALIGNMENT_H
