#include "featureDetect.h"
#include "DataProcessor.h"
#include "DLProcessor.h"

featureDetector::featureDetector()
{

}

featureDetector::~featureDetector()
{

}

std::vector<cv::DMatch> featureDetector::FilterMatches(
    const std::vector<cv::KeyPoint>& keypoints1,
    const std::vector<cv::KeyPoint>& keypoints2,
    const std::vector<std::vector<cv::DMatch>>& knnMatches,
    std::vector<cv::Point2f>& points1,
    std::vector<cv::Point2f>& points2,
    const FeatureParams& params)
{
    // 定义局部变量
    std::vector<cv::DMatch> goodMatches; // 存储筛选后的优质匹配
    std::vector<bool> validKeypoints1;   // 图像1中有效的关键点标志
    std::vector<bool> validKeypoints2;   // 图像2中有效的关键点标志
    
    points1.clear();
    points2.clear();

    // 1. 根据响应值筛选特征点
    validKeypoints1.resize(keypoints1.size(), false);
    validKeypoints2.resize(keypoints2.size(), false);

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
        std::vector<uchar> inlierMask;    // RANSAC内点掩码
        cv::Mat H;                        // 单应性矩阵
        
        H = cv::findHomography(points1, points2, cv::RANSAC, params.ransacReprojThresh, inlierMask);

        if (!H.empty())
        {
            std::vector<cv::DMatch> ransacMatches;      // RANSAC筛选后的匹配
            std::vector<cv::Point2f> inlierPoints1;     // 图像1的内点点集
            std::vector<cv::Point2f> inlierPoints2;     // 图像2的内点点集

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

void featureDetector::TestFeatureDetection(const QString& imagePath1, const QString& imagePath2)
{
    // 定义局部变量
    DataProcessor dataProcessor;                     // 数据处理器实例
    DLProcessor dlProcessor;                         // 深度学习处理器实例
    cv::Mat image1;                                  // 第一幅输入图像
    cv::Mat image2;                                  // 第二幅输入图像
    cv::Mat standardized1;                           // 标准化后的图像1
    cv::Mat standardized2;                           // 标准化后的图像2
    cv::Mat descriptors1;                            // 图像1的特征描述符
    cv::Mat descriptors2;                            // 图像2的特征描述符
    std::vector<cv::KeyPoint> keypoints1;            // 图像1的关键点
    std::vector<cv::KeyPoint> keypoints2;            // 图像2的关键点
    FeatureParams params;                            // 特征识别参数
    cv::Ptr<cv::DescriptorMatcher> matcher;          // 特征匹配器
    std::vector<std::vector<cv::DMatch>> knnMatches; // K近邻匹配结果
    std::vector<cv::Point2f> points1;                // 图像1的匹配点
    std::vector<cv::Point2f> points2;                // 图像2的匹配点
    std::vector<cv::DMatch> goodMatches;             // 筛选后的优质匹配
    cv::Mat imgMatches;                              // 匹配结果图像

    // 读取测试图像
    image1 = cv::imread(imagePath1.toStdString());
    image2 = cv::imread(imagePath2.toStdString());

    if (image1.empty() || image2.empty())
    {
        qDebug() << "Error: Could not read the images";
        return;
    }

    // 1. 图像预处理
    standardized1 = dataProcessor.StandardizeImage(image1);
    standardized2 = dataProcessor.StandardizeImage(image2);

    // 2. 提取特征
    keypoints1 = dataProcessor.DetectKeypoints(standardized1, descriptors1);
    keypoints2 = dataProcessor.DetectKeypoints(standardized2, descriptors2);

    // 设置特征识别参数
    params.ratioThresh = 0.7f;          // SIFT匹配比率阈值
    params.responseThresh = 0.01f;      // 特征点响应值阈值
    params.ransacReprojThresh = 3.0f;   // RANSAC重投影阈值
    params.minInliers = 10;             // 最小内点数量
    params.useRansac = true;            // 启用RANSAC验证

    // 3. 特征匹配
    matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    matcher->knnMatch(descriptors1, descriptors2, knnMatches, 2);

    // 4. 特征点筛选和几何验证
    goodMatches = FilterMatches(keypoints1, keypoints2, knnMatches, points1, points2, params);

    // 5. 绘制匹配结果
    cv::drawMatches(image1, keypoints1, image2, keypoints2, goodMatches, imgMatches,
                cv::Scalar::all(-1), cv::Scalar::all(-1),
                std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // 6. 如果有足够的匹配点，绘制变换关系
    if (points1.size() >= 4)
    {
        cv::Mat H; // 单应性矩阵
        std::vector<cv::Point2f> objCorners(4);   // 对象角点
        std::vector<cv::Point2f> sceneCorners(4); // 场景角点

        // 计算单应性矩阵
        H = cv::findHomography(points1, points2, cv::RANSAC);

        // 设置对象角点 (原始图像的四个角) 
        objCorners[0] = cv::Point2f(0, 0);
        objCorners[1] = cv::Point2f(image1.cols, 0);
        objCorners[2] = cv::Point2f(image1.cols, image1.rows);
        objCorners[3] = cv::Point2f(0, image1.rows);

        // 进行透视变换得到场景角点
        cv::perspectiveTransform(objCorners, sceneCorners, H);

        // 在匹配图像上绘制边界框
        cv::line(imgMatches, sceneCorners[0] + cv::Point2f(image1.cols, 0),
             sceneCorners[1] + cv::Point2f(image1.cols, 0), cv::Scalar(0, 255, 0), 2);
        cv::line(imgMatches, sceneCorners[1] + cv::Point2f(image1.cols, 0),
             sceneCorners[2] + cv::Point2f(image1.cols, 0), cv::Scalar(0, 255, 0), 2);
        cv::line(imgMatches, sceneCorners[2] + cv::Point2f(image1.cols, 0),
             sceneCorners[3] + cv::Point2f(image1.cols, 0), cv::Scalar(0, 255, 0), 2);
        cv::line(imgMatches, sceneCorners[3] + cv::Point2f(image1.cols, 0),
             sceneCorners[0] + cv::Point2f(image1.cols, 0), cv::Scalar(0, 255, 0), 2);
    }

    // 7. 保存结果图像
    cv::imwrite("../feature_matches.jpg", imgMatches);

    // 8. 输出匹配信息
    qDebug() << "Total keypoints in image1:" << keypoints1.size();
    qDebug() << "Total keypoints in image2:" << keypoints2.size();
    qDebug() << "Initial matches:" << knnMatches.size();
    qDebug() << "Good matches after filtering:" << goodMatches.size();
    qDebug() << "Inlier points:" << points1.size();
}
