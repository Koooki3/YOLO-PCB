#include "FeatureAlignment.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <QElapsedTimer>

FeatureAlignment::FeatureAlignment()
{
    // 初始化SIFT特征检测器
    m_featureDetector = cv::SIFT::create();
    
    // 初始化FLANN特征匹配器
    m_featureMatcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
}

FeatureAlignment::~FeatureAlignment()
{
    // 清理资源
}

AlignmentResult FeatureAlignment::AlignImages(
    const cv::Mat& srcImage,
    const cv::Mat& dstImage,
    const AlignmentParams& params)
{
    AlignmentResult result;
    QElapsedTimer timer;
    timer.start();

    // 转换为灰度图
    cv::Mat srcGray, dstGray;
    if (srcImage.channels() == 3) 
    {
        cv::cvtColor(srcImage, srcGray, cv::COLOR_BGR2GRAY);
    } 
    else 
    {
        srcGray = srcImage.clone();
    }
    
    if (dstImage.channels() == 3) 
    {
        cv::cvtColor(dstImage, dstGray, cv::COLOR_BGR2GRAY);
    } 
    else 
    {
        dstGray = dstImage.clone();
    }

    // 调用灰度图版本
    result = AlignImagesGray(srcGray, dstGray, params);
    
    qDebug() << "特征对齐耗时:" << timer.elapsed() << "ms";
    return result;
}

AlignmentResult FeatureAlignment::AlignImagesGray(
    const cv::Mat& srcGray, 
    const cv::Mat& dstGray,
    const AlignmentParams& params)
{
    AlignmentResult result;
    
    if (srcGray.empty() || dstGray.empty()) 
    {
        qDebug() << "输入图像为空";
        return result;
    }

    // 1. 特征检测和描述符提取
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    
    if (!ExtractFeatures(srcGray, keypoints1, descriptors1) || !ExtractFeatures(dstGray, keypoints2, descriptors2)) 
    {
        qDebug() << "特征提取失败";
        return result;
    }

    qDebug() << "特征点数量 - 源图像:" << keypoints1.size() << "目标图像:" << keypoints2.size();

    // 2. 特征匹配
    std::vector<cv::DMatch> matches;
    if (!MatchFeatures(descriptors1, descriptors2, matches)) 
    {
        qDebug() << "特征匹配失败";
        return result;
    }

    qDebug() << "初始匹配数量:" << matches.size();

    // 3. 几何验证和变换矩阵计算
    std::vector<cv::DMatch> inlierMatches;
    cv::Mat transformMatrix;
    
    if (!GeometricVerification(keypoints1, keypoints2, matches, transformMatrix, inlierMatches, params)) 
    {
        qDebug() << "几何验证失败";
        return result;
    }

    // 4. 填充结果
    result.transformMatrix = transformMatrix;
    result.matches = inlierMatches;
    result.success = true;
    result.inlierCount = static_cast<int>(inlierMatches.size());
    
    // 提取内点对应的特征点坐标
    result.srcPoints.reserve(inlierMatches.size());
    result.dstPoints.reserve(inlierMatches.size());
    for (const auto& match : inlierMatches) 
    {
        result.srcPoints.push_back(keypoints1[match.queryIdx].pt);
        result.dstPoints.push_back(keypoints2[match.trainIdx].pt);
    }
    
    // 计算重投影误差
    result.reprojectionError = ComputeReprojectionError(result.srcPoints, result.dstPoints, transformMatrix);

    qDebug() << "对齐成功 - 内点数量:" << result.inlierCount  << "重投影误差:" << result.reprojectionError;

    return result;
}

AlignmentResult FeatureAlignment::FastAlignImages(
    const cv::Mat& srcImage, 
    const cv::Mat& dstImage,
    const AlignmentParams& params)
{
    AlignmentResult result;
    QElapsedTimer timer;
    timer.start();

    // 转换为灰度图
    cv::Mat srcGray, dstGray;
    if (srcImage.channels() == 3) 
    {
        cv::cvtColor(srcImage, srcGray, cv::COLOR_BGR2GRAY);
    } 
    else 
    {
        srcGray = srcImage.clone();
    }
    
    if (dstImage.channels() == 3) 
    {
        cv::cvtColor(dstImage, dstGray, cv::COLOR_BGR2GRAY);
    } 
    else 
    {
        dstGray = dstImage.clone();
    }

    if (srcGray.empty() || dstGray.empty()) 
    {
        qDebug() << "输入图像为空";
        return result;
    }

    // 1. 特征检测和描述符提取
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    
    if (!ExtractFeatures(srcGray, keypoints1, descriptors1) || !ExtractFeatures(dstGray, keypoints2, descriptors2)) 
    {
        qDebug() << "特征提取失败";
        return result;
    }

    // 2. 特征匹配
    std::vector<cv::DMatch> matches;
    if (!MatchFeatures(descriptors1, descriptors2, matches)) 
    {
        qDebug() << "特征匹配失败";
        return result;
    }

    // 3. 快速几何验证 - 当匹配到足够内点时立即停止
    std::vector<cv::DMatch> inlierMatches;
    cv::Mat transformMatrix;
    
    if (matches.size() >= 4) 
    {
        std::vector<cv::Point2f> points1, points2;
        points1.reserve(matches.size());
        points2.reserve(matches.size());
        
        for (const auto& match : matches) 
        {
            points1.push_back(keypoints1[match.queryIdx].pt);
            points2.push_back(keypoints2[match.trainIdx].pt);
        }
        
        // 使用RANSAC计算单应性矩阵
        std::vector<uchar> inlierMask;
        transformMatrix = cv::findHomography(points1, points2, cv::RANSAC, params.ransacReprojThresh, inlierMask, params.maxIterations, params.confidence);
        
        if (!transformMatrix.empty()) 
        {
            // 提取内点
            for (size_t i = 0; i < inlierMask.size(); ++i) 
            {
                if (inlierMask[i]) 
                {
                    inlierMatches.push_back(matches[i]);
                }
            }
            
            // 检查是否达到最小内点要求
            if (inlierMatches.size() >= static_cast<size_t>(params.minInliers)) 
            {
                result.transformMatrix = transformMatrix;
                result.matches = inlierMatches;
                result.success = true;
                result.inlierCount = static_cast<int>(inlierMatches.size());
                
                // 提取内点对应的特征点坐标
                result.srcPoints.reserve(inlierMatches.size());
                result.dstPoints.reserve(inlierMatches.size());
                for (const auto& match : inlierMatches) 
                {
                    result.srcPoints.push_back(keypoints1[match.queryIdx].pt);
                    result.dstPoints.push_back(keypoints2[match.trainIdx].pt);
                }
                
                result.reprojectionError = ComputeReprojectionError(result.srcPoints, result.dstPoints, transformMatrix);
            }
        }
    }

    qDebug() << "快速对齐耗时:" << timer.elapsed() << "ms" << "内点数量:" << result.inlierCount;
    
    return result;
}

cv::Mat FeatureAlignment::WarpImage(
    const cv::Mat& srcImage, 
    const cv::Mat& transformMatrix,
    const cv::Size& dstSize)
{
    if (srcImage.empty() || transformMatrix.empty()) 
    {
        qDebug() << "输入图像或变换矩阵为空";
        return cv::Mat();
    }
    
    cv::Mat warpedImage;
    try 
    {
        // 使用白色背景填充空白区域
        cv::warpPerspective(srcImage, warpedImage, transformMatrix, dstSize, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
    } 
    catch (const cv::Exception& e) 
    {
        qDebug() << "图像变换失败:" << e.what();
        return cv::Mat();
    }
    
    return warpedImage;
}

QString FeatureAlignment::GetAlignmentInfo(const AlignmentResult& result) const
{
    if (!result.success) 
    {
        return "对齐失败";
    }
    
    return QString("对齐成功 - 内点数量: %1, 重投影误差: %2").arg(result.inlierCount).arg(result.reprojectionError, 0, 'f', 3);
}

bool FeatureAlignment::ExtractFeatures(
    const cv::Mat& image,
    std::vector<cv::KeyPoint>& keypoints,
    cv::Mat& descriptors)
{
    if (image.empty()) 
    {
        return false;
    }
    
    try 
    {
        m_featureDetector->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
        return !keypoints.empty() && !descriptors.empty();
    } 
    catch (const cv::Exception& e) 
    {
        qDebug() << "特征提取异常:" << e.what();
        return false;
    }
}

bool FeatureAlignment::MatchFeatures(
    const cv::Mat& descriptors1,
    const cv::Mat& descriptors2,
    std::vector<cv::DMatch>& matches)
{
    if (descriptors1.empty() || descriptors2.empty()) 
    {
        return false;
    }
    
    try 
    {
        // 使用KNN匹配
        std::vector<std::vector<cv::DMatch>> knnMatches;
        m_featureMatcher->knnMatch(descriptors1, descriptors2, knnMatches, 2);
        
        // 应用比率测试筛选优质匹配
        const float ratio_thresh = 0.7f;
        for (const auto& knnMatch : knnMatches) 
        {
            if (knnMatch.size() == 2 && knnMatch[0].distance < ratio_thresh * knnMatch[1].distance) 
            {
                matches.push_back(knnMatch[0]);
            }
        }
        
        return !matches.empty();
    } 
    catch (const cv::Exception& e) 
    {
        qDebug() << "特征匹配异常:" << e.what();
        return false;
    }
}

bool FeatureAlignment::GeometricVerification(
    const std::vector<cv::KeyPoint>& keypoints1,
    const std::vector<cv::KeyPoint>& keypoints2,
    const std::vector<cv::DMatch>& matches,
    cv::Mat& transformMatrix,
    std::vector<cv::DMatch>& inlierMatches,
    const AlignmentParams& params)
{
    if (matches.size() < 4) 
    {
        qDebug() << "匹配点数量不足，需要至少4个点";
        return false;
    }
    
    // 提取匹配点坐标
    std::vector<cv::Point2f> points1, points2;
    points1.reserve(matches.size());
    points2.reserve(matches.size());
    
    for (const auto& match : matches) 
    {
        points1.push_back(keypoints1[match.queryIdx].pt);
        points2.push_back(keypoints2[match.trainIdx].pt);
    }
    
    // 使用RANSAC计算单应性矩阵
    std::vector<uchar> inlierMask;
    transformMatrix = cv::findHomography(points1, points2, cv::RANSAC, params.ransacReprojThresh, inlierMask, params.maxIterations, params.confidence);
    
    if (transformMatrix.empty()) 
    {
        qDebug() << "单应性矩阵计算失败";
        return false;
    }
    
    // 提取内点
    for (size_t i = 0; i < inlierMask.size(); ++i) 
    {
        if (inlierMask[i]) 
        {
            inlierMatches.push_back(matches[i]);
        }
    }
    
    // 检查内点数量是否足够
    if (inlierMatches.size() < static_cast<size_t>(params.minInliers)) 
    {
        qDebug() << "内点数量不足，需要至少" << params.minInliers << "个，实际" << inlierMatches.size();
        return false;
    }
    
    return true;
}

double FeatureAlignment::ComputeReprojectionError(
    const std::vector<cv::Point2f>& points1,
    const std::vector<cv::Point2f>& points2,
    const cv::Mat& transformMatrix)
{
    if (points1.empty() || points2.empty() || transformMatrix.empty()) 
    {
        return 0.0;
    }
    
    std::vector<cv::Point2f> transformedPoints;
    cv::perspectiveTransform(points1, transformedPoints, transformMatrix);
    
    double totalError = 0.0;
    for (size_t i = 0; i < points2.size(); ++i) 
    {
        double dx = points2[i].x - transformedPoints[i].x;
        double dy = points2[i].y - transformedPoints[i].y;
        totalError += std::sqrt(dx * dx + dy * dy);
    }
    
    return totalError / points2.size();
}
