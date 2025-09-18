#include "featureDetect.h"
#include "DataProcessor.h"
#include "DLProcessor.h"

FeatureDetector::FeatureDetector()
{

}

FeatureDetector::~FeatureDetector()
{

}

std::vector<cv::DMatch> FeatureDetector::filterMatches(
    const std::vector<cv::KeyPoint>& keypoints1,
    const std::vector<cv::KeyPoint>& keypoints2,
    const std::vector<std::vector<cv::DMatch>>& knnMatches,
    std::vector<cv::Point2f>& points1,
    std::vector<cv::Point2f>& points2,
    const FeatureParams& params)
{
    std::vector<cv::DMatch> goodMatches;
    points1.clear();
    points2.clear();

    // 1. 根据响应值筛选特征点
    std::vector<bool> validKeypoints1(keypoints1.size(), false);
    std::vector<bool> validKeypoints2(keypoints2.size(), false);

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
        if (match[0].distance < params.ratioThresh * match[1].distance &&
            validKeypoints1[match[0].queryIdx] &&
            validKeypoints2[match[0].trainIdx])
        {
            goodMatches.push_back(match[0]);
            points1.push_back(keypoints1[match[0].queryIdx].pt);
            points2.push_back(keypoints2[match[0].trainIdx].pt);
        }
    }

    // 3. 使用RANSAC进行几何验证
    if (params.useRansac && points1.size() >= 4)
    {
        std::vector<uchar> inlierMask;
        cv::Mat H = cv::findHomography(points1, points2, cv::RANSAC, params.ransacReprojThresh, inlierMask);

        if (!H.empty())
        {
            std::vector<cv::DMatch> ransacMatches;
            std::vector<cv::Point2f> inlierPoints1, inlierPoints2;

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
                return ransacMatches;
            }
        }
    }

    return goodMatches;
}

void FeatureDetector::testFeatureDetection(const QString& imagePath1, const QString& imagePath2)
{
    // 创建数据处理器和深度学习处理器实例
    DataProcessor dataProcessor;
    DLProcessor dlProcessor;

    // 读取测试图像
    cv::Mat image1 = cv::imread(imagePath1.toStdString());
    cv::Mat image2 = cv::imread(imagePath2.toStdString());

    if (image1.empty() || image2.empty())
    {
        qDebug() << "Error: Could not read the images";
        return;
    }

    // 1. 图像预处理
    cv::Mat standardized1 = dataProcessor.standardizeImage(image1);
    cv::Mat standardized2 = dataProcessor.standardizeImage(image2);

    // 2. 提取特征
    cv::Mat descriptors1, descriptors2;
    std::vector<cv::KeyPoint> keypoints1 = dataProcessor.detectKeypoints(standardized1, descriptors1);
    std::vector<cv::KeyPoint> keypoints2 = dataProcessor.detectKeypoints(standardized2, descriptors2);

    // 设置特征识别参数
    FeatureParams params;
    params.ratioThresh = 0.7f;         // SIFT匹配比率阈值
    params.responseThresh = 0.01f;      // 特征点响应值阈值
    params.ransacReprojThresh = 3.0f;   // RANSAC重投影阈值
    params.minInliers = 10;             // 最小内点数量
    params.useRansac = true;            // 启用RANSAC验证

    // 3. 特征匹配
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher->knnMatch(descriptors1, descriptors2, knnMatches, 2);

    // 4. 特征点筛选和几何验证
    std::vector<cv::Point2f> points1, points2;
    std::vector<cv::DMatch> goodMatches = filterMatches(keypoints1, keypoints2, knnMatches, points1, points2, params);

    // 5. 绘制匹配结果
    cv::Mat imgMatches;
    cv::drawMatches(image1, keypoints1, image2, keypoints2, goodMatches, imgMatches,
                cv::Scalar::all(-1), cv::Scalar::all(-1),
                std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // 6. 如果有足够的匹配点，绘制变换关系
    if (points1.size() >= 4)
    {
        // 计算单应性矩阵
        cv::Mat H = cv::findHomography(points1, points2, cv::RANSAC);

        // 绘制目标区域边界
        std::vector<cv::Point2f> objCorners(4);
        objCorners[0] = cv::Point2f(0, 0);
        objCorners[1] = cv::Point2f(image1.cols, 0);
        objCorners[2] = cv::Point2f(image1.cols, image1.rows);
        objCorners[3] = cv::Point2f(0, image1.rows);

        std::vector<cv::Point2f> sceneCorners(4);
        cv::perspectiveTransform(objCorners, sceneCorners, H);

        // 在匹配图像上绘制边界
        cv::line(imgMatches, sceneCorners[0] + cv::Point2f(image1.cols, 0),
             sceneCorners[1] + cv::Point2f(image1.cols, 0), cv::Scalar(0, 255, 0), 2);
        cv::line(imgMatches, sceneCorners[1] + cv::Point2f(image1.cols, 0),
             sceneCorners[2] + cv::Point2f(image1.cols, 0), cv::Scalar(0, 255, 0), 2);
        cv::line(imgMatches, sceneCorners[2] + cv::Point2f(image1.cols, 0),
             sceneCorners[3] + cv::Point2f(image1.cols, 0), cv::Scalar(0, 255, 0), 2);
        cv::line(imgMatches, sceneCorners[3] + cv::Point2f(image1.cols, 0),
             sceneCorners[0] + cv::Point2f(image1.cols, 0), cv::Scalar(0, 255, 0), 2);
    }

    // 7. 保存结果
    cv::imwrite("../feature_matches.jpg", imgMatches);

    // 8. 输出匹配信息
    qDebug() << "Total keypoints in image1:" << keypoints1.size();
    qDebug() << "Total keypoints in image2:" << keypoints2.size();
    qDebug() << "Initial matches:" << knnMatches.size();
    qDebug() << "Good matches after filtering:" << goodMatches.size();
    qDebug() << "Inlier points:" << points1.size();
}
