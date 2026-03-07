/**
 * @file FeatureAlignment.cpp
 * @brief 特征对齐模块实现文件
 * 
 * 该文件实现了FeatureAlignment类的所有方法，提供基于特征匹配的图像对齐功能，
 * 包括特征检测、匹配、几何验证、变换矩阵计算和图像变换等完整流程。
 * 
 * @author VisualRobot Team
 * @date 2025-12-30
 * @version 1.0
 */

#include "FeatureAlignment.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <QElapsedTimer>

/**
 * @brief 构造函数
 * 
 * 初始化各类特征检测器和匹配器，默认使用SIFT特征提取器
 * 
 * @note 初始化步骤：
 *       - 创建SIFT特征检测器
 *       - 创建ORB特征检测器（参数：500特征点，1.2缩放因子，8层金字塔）
 *       - 创建AKAZE特征检测器
 *       - 设置默认特征类型为SIFT
 *       - 调用UpdateFeatureDetector()初始化检测器和匹配器
 * @see UpdateFeatureDetector()
 */
FeatureAlignment::FeatureAlignment()
{
    // 初始化SIFT特征检测器
    m_siftDetector = cv::SIFT::create();
    
    // 初始化ORB特征检测器，参数说明：
    // 500: 最大特征点数
    // 1.2f: 金字塔缩放因子
    // 8: 金字塔层数
    // 31: 边缘阈值
    // 0: 第一个通道的像素值偏移
    // 2: 方向数量
    // cv::ORB::HARRIS_SCORE: 角点检测算法
    // 31: 描述符长度
    // 20: 匹配距离阈值
    m_orbDetector = cv::ORB::create(500, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);
    
    // 初始化AKAZE特征检测器
    m_akazeDetector = cv::AKAZE::create();
    
    // 默认使用SIFT特征提取器
    m_featureType = FeatureType::SIFT;
    
    // 初始化并更新特征检测器和匹配器
    UpdateFeatureDetector();
}

/**
 * @brief 设置特征提取器类型
 * 
 * @param type 特征提取器类型（SIFT、ORB或AKAZE）
 * 
 * @note 如果类型发生变化，会自动更新检测器和匹配器
 * @see GetFeatureType(), UpdateFeatureDetector()
 */
void FeatureAlignment::SetFeatureType(FeatureType type)
{
    if (m_featureType != type)
    {
        m_featureType = type;
        UpdateFeatureDetector();
    }
}

/**
 * @brief 获取当前特征提取器类型
 * 
 * @return 当前使用的特征提取器类型
 * @see SetFeatureType()
 */
FeatureType FeatureAlignment::GetFeatureType() const
{
    return m_featureType;
}

/**
 * @brief 更新特征检测器和匹配器
 * 
 * 根据当前设置的特征类型，更新内部使用的特征检测器和匹配器
 * 
 * @note 匹配器选择规则：
 *       - SIFT：FLANN匹配器（适用于浮点型描述符）
 *       - ORB/AKAZE：汉明距离匹配器（适用于二进制描述符）
 * @see SetFeatureType()
 */
void FeatureAlignment::UpdateFeatureDetector()
{
    if (m_featureType == FeatureType::SIFT)
    {
        m_featureDetector = m_siftDetector;
        // SIFT使用FLANN匹配器，适用于浮点型描述符
        m_featureMatcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }
    else if (m_featureType == FeatureType::ORB)
    {
        m_featureDetector = m_orbDetector;
        // ORB使用汉明距离匹配器，适用于二进制描述符
        m_featureMatcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
    }
    else if (m_featureType == FeatureType::AKAZE)
    {
        m_featureDetector = m_akazeDetector;
        // AKAZE使用汉明距离匹配器，适用于二进制描述符
        m_featureMatcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
    }
    else
    {
        // 默认使用SIFT特征提取器
        m_featureDetector = m_siftDetector;
        m_featureMatcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }
}

/**
 * @brief 析构函数
 * 
 * 清理资源
 * 
 * @note 由于使用OpenCV智能指针，资源会自动释放
 */
FeatureAlignment::~FeatureAlignment()
{
    // 清理资源，OpenCV智能指针会自动释放内存
}

/**
 * @brief 使用特征匹配对齐两幅图像
 * 
 * @param srcImage 源图像（支持彩色或灰度）
 * @param dstImage 目标图像（支持彩色或灰度）
 * @param params 对齐参数
 * @return AlignmentResult 对齐结果，包含变换矩阵和匹配信息
 * 
 * @note 处理流程：
 *       1. 转换为灰度图像
 *       2. 调用AlignImagesGray()进行核心对齐处理
 *       3. 记录处理时间
 * @see AlignImagesGray()
 */
AlignmentResult FeatureAlignment::AlignImages(
    const cv::Mat& srcImage,
    const cv::Mat& dstImage,
    const AlignmentParams& params)
{
    AlignmentResult result;
    QElapsedTimer timer;
    timer.start();

    // 转换为灰度图，特征提取通常在灰度图上进行
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

    // 调用灰度图版本的对齐函数
    result = AlignImagesGray(srcGray, dstGray, params);
    
    qDebug() << "特征对齐耗时:" << timer.elapsed() << "ms";
    return result;
}

/**
 * @brief 使用特征匹配对齐两幅灰度图像
 * 
 * @param srcGray 源灰度图像
 * @param dstGray 目标灰度图像
 * @param params 对齐参数
 * @return AlignmentResult 对齐结果，包含变换矩阵和匹配信息
 * 
 * @note 核心对齐流程：
 *       1. 特征检测和描述符提取
 *       2. 特征匹配
 *       3. 几何验证和变换矩阵计算
 *       4. 提取内点和计算重投影误差
 * @see ExtractFeatures(), MatchFeatures(), GeometricVerification()
 */
AlignmentResult FeatureAlignment::AlignImagesGray(
    const cv::Mat& srcGray, 
    const cv::Mat& dstGray,
    const AlignmentParams& params)
{
    AlignmentResult result;
    
    // 检查输入图像是否为空
    if (srcGray.empty() || dstGray.empty()) 
    {
        qDebug() << "输入图像为空";
        return result;
    }

    // 保存使用的特征提取器类型
    result.featureType = params.featureType;
    
    // 临时保存当前特征类型，然后设置为参数中指定的类型
    FeatureType originalType = m_featureType;
    SetFeatureType(params.featureType);

    // 1. 特征检测和描述符提取
    std::vector<cv::KeyPoint> keypoints1, keypoints2; // 存储源图像和目标图像的特征点
    cv::Mat descriptors1, descriptors2; // 存储源图像和目标图像的特征描述符
    
    // 提取源图像和目标图像的特征
    if (!ExtractFeatures(srcGray, keypoints1, descriptors1) || !ExtractFeatures(dstGray, keypoints2, descriptors2)) 
    {
        qDebug() << "特征提取失败";
        // 恢复原始特征类型
        SetFeatureType(originalType);
        return result;
    }

    qDebug() << "特征点数量 - 源图像:" << keypoints1.size() << "目标图像:" << keypoints2.size();

    // 2. 特征匹配
    std::vector<cv::DMatch> matches; // 存储初始匹配结果
    if (!MatchFeatures(descriptors1, descriptors2, matches)) 
    {
        qDebug() << "特征匹配失败";
        // 恢复原始特征类型
        SetFeatureType(originalType);
        return result;
    }

    qDebug() << "初始匹配数量:" << matches.size();

    // 3. 几何验证和变换矩阵计算
    std::vector<cv::DMatch> inlierMatches; // 存储经过几何验证的内点匹配
    cv::Mat transformMatrix; // 存储计算得到的变换矩阵
    
    if (!GeometricVerification(keypoints1, keypoints2, matches, transformMatrix, inlierMatches, params)) 
    {
        qDebug() << "几何验证失败";
        // 恢复原始特征类型
        SetFeatureType(originalType);
        return result;
    }

    // 4. 填充结果
    result.transformMatrix = transformMatrix; // 保存变换矩阵
    result.matches = inlierMatches; // 保存内点匹配结果
    result.success = true; // 标记对齐成功
    result.inlierCount = static_cast<int>(inlierMatches.size()); // 保存内点数量
    
    // 提取内点对应的特征点坐标
    result.srcPoints.reserve(inlierMatches.size());
    result.dstPoints.reserve(inlierMatches.size());
    for (const auto& match : inlierMatches) 
    {
        result.srcPoints.push_back(keypoints1[match.queryIdx].pt); // 源图像特征点坐标
        result.dstPoints.push_back(keypoints2[match.trainIdx].pt); // 目标图像特征点坐标
    }
    
    // 计算重投影误差，衡量对齐精度
    result.reprojectionError = ComputeReprojectionError(result.srcPoints, result.dstPoints, transformMatrix);

    qDebug() << "对齐成功 - 内点数量:" << result.inlierCount  << "重投影误差:" << result.reprojectionError;
    
    // 恢复原始特征类型
    SetFeatureType(originalType);

    return result;
}

/**
 * @brief 快速对齐：当匹配到足够内点时立即停止
 * 
 * @param srcImage 源图像
 * @param dstImage 目标图像
 * @param params 对齐参数
 * @return AlignmentResult 对齐结果，包含变换矩阵和匹配信息
 * 
 * @note 与普通对齐相比，找到足够内点后立即返回，不进行额外计算
 * @see AlignImages()
 */
AlignmentResult FeatureAlignment::FastAlignImages(
    const cv::Mat& srcImage, 
    const cv::Mat& dstImage,
    const AlignmentParams& params)
{
    AlignmentResult result;
    QElapsedTimer timer;
    timer.start();

    // 保存使用的特征提取器类型
    result.featureType = params.featureType;
    
    // 临时保存当前特征类型，然后设置为参数中指定的类型
    FeatureType originalType = m_featureType;
    SetFeatureType(params.featureType);

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

    // 检查输入图像是否为空
    if (srcGray.empty() || dstGray.empty()) 
    {
        qDebug() << "输入图像为空";
        // 恢复原始特征类型
        SetFeatureType(originalType);
        return result;
    }

    // 1. 特征检测和描述符提取
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    
    if (!ExtractFeatures(srcGray, keypoints1, descriptors1) || !ExtractFeatures(dstGray, keypoints2, descriptors2)) 
    {
        qDebug() << "特征提取失败";
        // 恢复原始特征类型
        SetFeatureType(originalType);
        return result;
    }

    // 2. 特征匹配
    std::vector<cv::DMatch> matches;
    if (!MatchFeatures(descriptors1, descriptors2, matches)) 
    {
        qDebug() << "特征匹配失败";
        // 恢复原始特征类型
        SetFeatureType(originalType);
        return result;
    }

    // 3. 快速几何验证 - 当匹配到足够内点时立即停止
    std::vector<cv::DMatch> inlierMatches;
    cv::Mat transformMatrix;
    
    // 至少需要4个点来计算单应性矩阵
    if (matches.size() >= 4) 
    {
        // 提取匹配点坐标
        std::vector<cv::Point2f> points1, points2;
        points1.reserve(matches.size());
        points2.reserve(matches.size());
        
        for (const auto& match : matches) 
        {
            points1.push_back(keypoints1[match.queryIdx].pt);
            points2.push_back(keypoints2[match.trainIdx].pt);
        }
        
        // 使用RANSAC算法计算单应性矩阵
        std::vector<uchar> inlierMask; // 内点掩码
        transformMatrix = cv::findHomography(points1, points2, cv::RANSAC, params.ransacReprojThresh, inlierMask, params.maxIterations, params.confidence);
        
        // 检查单应性矩阵是否有效
        if (!transformMatrix.empty()) 
        {
            // 提取内点匹配
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
                // 填充对齐结果
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
            }
        }
    }

    // 恢复原始特征类型
    SetFeatureType(originalType);
    
    qDebug() << "快速对齐耗时:" << timer.elapsed() << "ms" << "内点数量:" << result.inlierCount;
    
    return result;
}

/**
 * @brief 根据变换矩阵将源图像变换到目标图像坐标系
 * 
 * @param srcImage 源图像
 * @param transformMatrix 变换矩阵（单应性矩阵）
 * @param dstSize 目标图像尺寸
 * @return cv::Mat 变换后的图像
 * 
 * @note 使用透视变换，边界填充为白色
 * @see AlignImages()
 */
cv::Mat FeatureAlignment::WarpImage(
    const cv::Mat& srcImage, 
    const cv::Mat& transformMatrix,
    const cv::Size& dstSize)
{
    // 检查输入参数是否有效
    if (srcImage.empty() || transformMatrix.empty()) 
    {
        qDebug() << "输入图像或变换矩阵为空";
        return cv::Mat();
    }
    
    cv::Mat warpedImage; // 存储变换后的图像
    try 
    {
        // 使用透视变换将源图像映射到目标图像坐标系
        // 参数说明：
        // srcImage: 源图像
        // warpedImage: 输出图像
        // transformMatrix: 变换矩阵
        // dstSize: 输出图像尺寸
        // cv::INTER_LINEAR: 线性插值方法
        // cv::BORDER_CONSTANT: 边界填充方式
        // cv::Scalar(255, 255, 255): 边界填充颜色（白色）
        cv::warpPerspective(srcImage, warpedImage, transformMatrix, dstSize, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
    } 
    catch (const cv::Exception& e) 
    {
        qDebug() << "图像变换失败:" << e.what();
        return cv::Mat();
    }
    
    return warpedImage;
}

/**
 * @brief 获取对齐状态信息
 * 
 * @param result 对齐结果
 * @return QString 格式化的对齐状态字符串
 * 
 * @note 返回格式："对齐成功 - 内点数量: X, 重投影误差: Y" 或 "对齐失败"
 * @see AlignmentResult
 */
QString FeatureAlignment::GetAlignmentInfo(const AlignmentResult& result) const
{
    if (!result.success) 
    {
        return "对齐失败";
    }
    
    // 格式化输出对齐成功信息，包含内点数量和重投影误差
    return QString("对齐成功 - 内点数量: %1, 重投影误差: %2").arg(result.inlierCount).arg(result.reprojectionError, 0, 'f', 3);
}

/**
 * @brief 特征检测和描述符提取
 * 
 * @param image 输入图像（灰度）
 * @param keypoints 输出的特征点
 * @param descriptors 输出的特征描述符
 * @return bool 是否成功提取特征
 * 
 * @note 使用当前设置的特征检测器
 * @see UpdateFeatureDetector()
 */
bool FeatureAlignment::ExtractFeatures(
    const cv::Mat& image,
    std::vector<cv::KeyPoint>& keypoints,
    cv::Mat& descriptors)
{
    // 检查输入图像是否为空
    if (image.empty()) 
    {
        return false;
    }
    
    try 
    {
        // 使用当前特征检测器检测特征点并计算描述符
        // cv::noArray(): 不使用掩码
        m_featureDetector->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
        // 检查是否成功提取到特征点和描述符
        return !keypoints.empty() && !descriptors.empty();
    } 
    catch (const cv::Exception& e) 
    {
        qDebug() << "特征提取异常:" << e.what();
        return false;
    }
}

/**
 * @brief 特征匹配
 * 
 * @param descriptors1 第一幅图像的特征描述符
 * @param descriptors2 第二幅图像的特征描述符
 * @param matches 输出的匹配结果
 * @return bool 是否成功匹配特征
 * 
 * @note 使用KNN匹配和比率测试筛选优质匹配
 * @see GeometricVerification()
 */
bool FeatureAlignment::MatchFeatures(
    const cv::Mat& descriptors1,
    const cv::Mat& descriptors2,
    std::vector<cv::DMatch>& matches)
{
    // 检查输入描述符是否为空
    if (descriptors1.empty() || descriptors2.empty()) 
    {
        return false;
    }
    
    try 
    {
        // 使用KNN匹配算法，为每个特征点找到最近的2个匹配
        std::vector<std::vector<cv::DMatch>> knnMatches;
        m_featureMatcher->knnMatch(descriptors1, descriptors2, knnMatches, 2);
        
        // 应用比率测试筛选优质匹配
        // 原理：如果最佳匹配与次佳匹配的距离比率小于阈值，则认为是优质匹配
        const float ratio_thresh = 0.7f;
        for (const auto& knnMatch : knnMatches) 
        {
            if (knnMatch.size() == 2 && knnMatch[0].distance < ratio_thresh * knnMatch[1].distance) 
            {
                matches.push_back(knnMatch[0]);
            }
        }
        
        // 检查是否成功匹配到特征点
        return !matches.empty();
    } 
    catch (const cv::Exception& e) 
    {
        qDebug() << "特征匹配异常:" << e.what();
        return false;
    }
}

/**
 * @brief 几何验证和变换矩阵计算
 * 
 * 使用RANSAC算法进行几何验证，计算单应性矩阵，并提取内点匹配
 * 
 * @param keypoints1 第一幅图像的特征点
 * @param keypoints2 第二幅图像的特征点
 * @param matches 初始匹配结果
 * @param transformMatrix 输出的变换矩阵
 * @param inlierMatches 输出的内点匹配结果
 * @param params 对齐参数
 * @return bool 是否成功计算变换矩阵
 * 
 * @note 使用RANSAC算法计算单应性矩阵
 * @see ComputeReprojectionError()
 */
bool FeatureAlignment::GeometricVerification(
    const std::vector<cv::KeyPoint>& keypoints1,
    const std::vector<cv::KeyPoint>& keypoints2,
    const std::vector<cv::DMatch>& matches,
    cv::Mat& transformMatrix,
    std::vector<cv::DMatch>& inlierMatches,
    const AlignmentParams& params)
{
    // 至少需要4个点来计算单应性矩阵
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
    
    // 使用RANSAC算法计算单应性矩阵
    std::vector<uchar> inlierMask; // 内点掩码
    transformMatrix = cv::findHomography(points1, points2, cv::RANSAC, params.ransacReprojThresh, inlierMask, params.maxIterations, params.confidence);
    
    // 检查单应性矩阵是否有效
    if (transformMatrix.empty()) 
    {
        qDebug() << "单应性矩阵计算失败";
        return false;
    }
    
    // 提取内点匹配
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

/**
 * @brief 计算重投影误差
 * 
 * 重投影误差是衡量对齐精度的重要指标，表示将源图像特征点通过变换矩阵映射到目标图像后，与目标图像特征点的平均距离
 * 
 * @param points1 第一组特征点（源图像）
 * @param points2 第二组特征点（目标图像）
 * @param transformMatrix 变换矩阵
 * @return double 平均重投影误差
 * 
 * @note 重投影误差越小，对齐精度越高
 * @see GeometricVerification()
 */
double FeatureAlignment::ComputeReprojectionError(
    const std::vector<cv::Point2f>& points1,
    const std::vector<cv::Point2f>& points2,
    const cv::Mat& transformMatrix)
{
    // 检查输入参数是否有效
    if (points1.empty() || points2.empty() || transformMatrix.empty()) 
    {
        return 0.0;
    }
    
    // 将源图像特征点通过变换矩阵映射到目标图像坐标系
    std::vector<cv::Point2f> transformedPoints;
    cv::perspectiveTransform(points1, transformedPoints, transformMatrix);
    
    // 计算总重投影误差
    double totalError = 0.0;
    for (size_t i = 0; i < points2.size(); ++i) 
    {
        // 计算每个点的欧氏距离
        double dx = points2[i].x - transformedPoints[i].x;
        double dy = points2[i].y - transformedPoints[i].y;
        totalError += std::sqrt(dx * dx + dy * dy);
    }
    
    // 返回平均重投影误差
    return totalError / points2.size();
}
