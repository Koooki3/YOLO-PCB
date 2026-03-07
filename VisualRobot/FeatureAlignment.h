/**
 * @file FeatureAlignment.h
 * @brief 特征对齐模块头文件
 * 
 * 该文件定义了FeatureAlignment类和相关结构体，提供基于特征匹配的图像对齐功能，
 * 支持多种特征提取算法（SIFT、ORB、AKAZE），包含完整的特征检测、匹配、
 * 几何验证和图像变换功能。
 * 
 * @author VisualRobot Team
 * @date 2025-12-30
 * @version 1.0
 */

#ifndef FEATUREALIGNMENT_H
#define FEATUREALIGNMENT_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <QString>
#include <QDebug>
#include "featureDetect_optimized.h"
#include "DataProcessor.h" // 包含FeatureType枚举

/**
 * @brief 特征对齐参数结构体
 * 
 * 用于配置特征对齐算法的各项参数，包括RANSAC阈值、并行计算设置、特征提取器类型等
 * 
 * @note 默认参数适用于大多数场景，可根据具体需求调整
 * @see FeatureAlignment::AlignImages()
 */
struct AlignmentParams
{
    int minInliers = 10;              ///< 最小内点数量，匹配到该数量即可认为对齐成功
    float ransacReprojThresh = 3.0f;  ///< RANSAC重投影阈值，控制匹配点的几何一致性
    bool enableParallel = true;       ///< 是否启用并行计算，提高处理速度
    int numThreads = 4;               ///< 并行计算使用的线程数
    int maxIterations = 1000;         ///< RANSAC最大迭代次数
    double confidence = 0.99;         ///< RANSAC置信度，控制结果可靠性
    FeatureType featureType = FeatureType::SIFT; ///< 特征提取器类型（SIFT、ORB或AKAZE）
};

/**
 * @brief 对齐结果结构体
 * 
 * 存储特征对齐算法的输出结果，包括变换矩阵、匹配点对、内点数量等
 * 
 * @note 当success为true时，其他字段才有效
 * @see FeatureAlignment::AlignImages()
 */
struct AlignmentResult
{
    cv::Mat transformMatrix;          ///< 变换矩阵（单应性矩阵），用于将源图像映射到目标图像
    std::vector<cv::DMatch> matches;  ///< 经过几何验证的匹配结果
    std::vector<cv::Point2f> srcPoints; ///< 源图像中匹配的特征点坐标
    std::vector<cv::Point2f> dstPoints; ///< 目标图像中匹配的特征点坐标
    bool success = false;             ///< 对齐是否成功
    int inlierCount = 0;              ///< 内点数量，即符合几何约束的匹配点数量
    double reprojectionError = 0.0;   ///< 重投影误差，衡量对齐精度
    FeatureType featureType = FeatureType::SIFT; ///< 使用的特征提取器类型
};

/**
 * @brief 特征对齐类
 * 
 * 实现基于特征匹配的图像对齐功能，支持多种特征提取算法（SIFT、ORB、AKAZE）
 * 提供完整的特征检测、匹配、几何验证和图像变换功能
 * 
 * @note 该类使用OpenCV的特征检测和匹配框架
 * @see AlignmentParams, AlignmentResult
 */
class FeatureAlignment
{
public:
    /**
     * @brief 构造函数
     * 
     * 初始化各类特征检测器和匹配器，默认使用SIFT特征提取器
     * 
     * @note 初始化步骤：
     *       - 创建SIFT、ORB、AKAZE特征检测器
     *       - 设置默认特征类型为SIFT
     *       - 调用UpdateFeatureDetector()初始化检测器和匹配器
     * @see UpdateFeatureDetector()
     */
    FeatureAlignment();
    
    /**
     * @brief 析构函数
     * 
     * 清理资源
     * 
     * @note 由于使用OpenCV智能指针，资源会自动释放
     */
    ~FeatureAlignment();

    /**
     * @brief 设置特征提取器类型
     * 
     * @param type 特征提取器类型（SIFT、ORB或AKAZE）
     * 
     * @note 会自动更新内部检测器和匹配器
     * @see GetFeatureType(), UpdateFeatureDetector()
     */
    void SetFeatureType(FeatureType type);
    
    /**
     * @brief 获取当前特征提取器类型
     * 
     * @return 当前使用的特征提取器类型
     * @see SetFeatureType()
     */
    FeatureType GetFeatureType() const;

    /**
     * @brief 使用特征匹配对齐两幅图像
     * 
     * @param srcImage 源图像（支持彩色或灰度）
     * @param dstImage 目标图像（支持彩色或灰度）
     * @param params 对齐参数，默认使用AlignmentParams()
     * @return AlignmentResult 对齐结果，包含变换矩阵和匹配信息
     * 
     * @note 处理流程：
     *       1. 转换为灰度图像
     *       2. 特征检测和描述符提取
     *       3. 特征匹配
     *       4. 几何验证和变换矩阵计算
     *       5. 计算重投影误差
     * @see AlignImagesGray(), FastAlignImages()
     */
    AlignmentResult AlignImages(const cv::Mat& srcImage, const cv::Mat& dstImage, const AlignmentParams& params = AlignmentParams());

    /**
     * @brief 使用特征匹配对齐两幅灰度图像
     * 
     * @param srcGray 源灰度图像
     * @param dstGray 目标灰度图像
     * @param params 对齐参数
     * @return AlignmentResult 对齐结果，包含变换矩阵和匹配信息
     * 
     * @note 这是核心对齐方法，直接处理灰度图像
     * @see AlignImages()
     */
    AlignmentResult AlignImagesGray(const cv::Mat& srcGray, const cv::Mat& dstGray, const AlignmentParams& params = AlignmentParams());

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
    cv::Mat WarpImage(const cv::Mat& srcImage, const cv::Mat& transformMatrix, const cv::Size& dstSize);

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
    AlignmentResult FastAlignImages(const cv::Mat& srcImage, const cv::Mat& dstImage, const AlignmentParams& params = AlignmentParams());

    /**
     * @brief 获取对齐状态信息
     * 
     * @param result 对齐结果
     * @return QString 格式化的对齐状态字符串
     * 
     * @note 返回格式："对齐成功 - 内点数量: X, 重投影误差: Y" 或 "对齐失败"
     * @see AlignmentResult
     */
    QString GetAlignmentInfo(const AlignmentResult& result) const;

private:
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
    bool ExtractFeatures(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

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
    bool MatchFeatures(const cv::Mat& descriptors1, const cv::Mat& descriptors2, std::vector<cv::DMatch>& matches);

    /**
     * @brief 几何验证和变换矩阵计算
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
    bool GeometricVerification(const std::vector<cv::KeyPoint>& keypoints1, const std::vector<cv::KeyPoint>& keypoints2, const std::vector<cv::DMatch>& matches, cv::Mat& transformMatrix, std::vector<cv::DMatch>& inlierMatches, const AlignmentParams& params);

    /**
     * @brief 计算重投影误差
     * 
     * @param points1 第一组特征点（源图像）
     * @param points2 第二组特征点（目标图像）
     * @param transformMatrix 变换矩阵
     * @return double 平均重投影误差
     * 
     * @note 重投影误差是衡量对齐精度的重要指标
     * @see GeometricVerification()
     */
    double ComputeReprojectionError(const std::vector<cv::Point2f>& points1, const std::vector<cv::Point2f>& points2, const cv::Mat& transformMatrix);
    
    /**
     * @brief 更新特征检测器和匹配器
     * 
     * 根据当前设置的特征类型，更新内部使用的特征检测器和匹配器
     * 
     * @note 匹配器选择规则：
     *       - SIFT：FLANN匹配器（浮点型描述符）
     *       - ORB/AKAZE：汉明距离匹配器（二进制描述符）
     * @see SetFeatureType()
     */
    void UpdateFeatureDetector();

    // 成员变量
    cv::Ptr<cv::Feature2D> m_featureDetector; ///< 当前使用的特征检测器
    cv::Ptr<cv::DescriptorMatcher> m_featureMatcher; ///< 当前使用的特征匹配器
    FeatureType m_featureType; ///< 当前特征提取器类型
    cv::Ptr<cv::SIFT> m_siftDetector; ///< SIFT特征检测器实例
    cv::Ptr<cv::ORB> m_orbDetector; ///< ORB特征检测器实例
    cv::Ptr<cv::AKAZE> m_akazeDetector; ///< AKAZE特征检测器实例
};

#endif // FEATUREALIGNMENT_H
