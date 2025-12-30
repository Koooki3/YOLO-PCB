/**
 * @file DefectDetection.h
 * @brief 缺陷检测模块头文件
 * 
 * 该模块实现了一套面向嵌入式/工业视觉的轻量化缺陷二分类流水线。
 * 主要功能包括：
 * - 使用多色域统计特征（BGR/HSV/Lab）和纹理特征（LBP）作为轻量特征描述子
 * - 通过PCA进行降维以减少在线计算量
 * - 使用OpenCV的SVM做二分类（适合小样本场景）
 * - 提供模型的保存/加载接口以支持脱机训练、在线推理分离
 * - 模板法缺陷检测和特征对齐功能
 * 
 * 主要流程：
 * 1) Preprocess: 图像预处理（色彩校正/直方图均衡化/去噪）
 * 2) ExtractFeatures: 多色域统计特征 + LBP 纹理直方图 -> 得到定长特征向量
 * 3) FitPCA/ProjectPCA: 基于样本的 PCA 降维与投影
 * 4) TrainSVM/Predict: 在（可选）PCA 子空间上训练/预测 SVM
 * 5) SaveModel/LoadModel: 将 SVM 与 PCA 参数持久化到文件
 * 
 * @note 本类适用于图像/ROI 级别的缺陷判别（即给出整张图或局部 ROI 的缺陷/正常标签），
 *       并不提供像素级分割或缺陷定位功能
 * @author VisualRobot Team
 * @date 2025-12-29
 * @version 1.0
 */

#ifndef DEFECTDETECTION_H
#define DEFECTDETECTION_H

#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <string>
#include <vector>
#include <thread>
#include <future>
#include <mutex>
#include <atomic>
#include "FeatureAlignment.h"

using namespace cv;
using namespace cv::ml;
using namespace std;

/**
 * @brief 缺陷检测类
 * 
 * 提供完整的缺陷检测流水线，包括图像预处理、特征提取、PCA降维、SVM训练/预测、
 * 模板法缺陷检测和特征对齐等功能。
 * 
 * @note 特征和参数（如直方图 bins、LBP 配置、PCA 保留维度、SVM 超参）可根据数据集调整
 */
class DefectDetection {
public:
    /**
     * @brief 构造函数
     * 
     * 初始化默认 SVM 及 PCA 维度（可通过 SetPCADim 调整）
     * 
     * @note 初始化内容：
     *       - SVM: C_SVC类型，RBF核函数，终止准则为1000次迭代或1e-6精度
     *       - PCA: 默认保留32维
     *       - 特征对齐对象: 创建FeatureAlignment实例
     */
    DefectDetection();

    // ==================== 模板法缺陷检测相关功能 ====================

    /**
     * @brief 从当前帧设置模板
     * 
     * @param currentFrame 当前帧图像（支持BGR或灰度）
     * @return bool 设置成功返回true，失败返回false
     * @note 会同时保存灰度模板和BGR模板（用于特征对齐）
     * @see SetTemplateFromFile()
     */
    bool SetTemplateFromCurrent(const cv::Mat& currentFrame);
    
    /**
     * @brief 从文件设置模板
     * 
     * @param filePath 模板图像文件路径
     * @return bool 设置成功返回true，失败返回false
     * @note 会自动转换为灰度并进行预处理
     * @see SetTemplateFromCurrent()
     */
    bool SetTemplateFromFile(const std::string& filePath);
    
    /**
     * @brief 计算单应性矩阵（模板 <- 当前）
     * 
     * @param currentGray 当前图像的灰度图
     * @param homography 输出的单应性矩阵
     * @param debugMatches 可选的调试匹配点对（用于可视化）
     * @return bool 计算成功返回true，失败返回false
     * @note 使用ORB特征和RANSAC算法
     * @see ComputeHomographyWithFeatureAlignment()
     */
    bool ComputeHomography(const cv::Mat& currentGray, cv::Mat& homography, std::vector<cv::DMatch>* debugMatches = nullptr);
    
    /**
     * @brief 缺陷检测：根据配准后差异得到缺陷外接框
     * 
     * @param currentBGR 当前BGR图像
     * @param homography 单应性矩阵
     * @param debugMask 可选的调试掩码（差异二值图）
     * @return std::vector<cv::Rect> 缺陷外接框集合（当前图像坐标系）
     * @note 缺陷框坐标在当前图像坐标系下
     * @see DetectDefects()
     */
    std::vector<cv::Rect> DetectDefects(const cv::Mat& currentBGR, const cv::Mat& homography, cv::Mat* debugMask = nullptr);
    
    /**
     * @brief 设置模板差异阈值
     * 
     * @param threshold 差异二值化阈值
     * @note 默认25.0，小于等于0时使用Otsu自动阈值
     */
    void SetTemplateDiffThreshold(double threshold) { m_templateDiffThresh = threshold; }
    
    /**
     * @brief 设置最小缺陷面积
     * 
     * @param area 最小面积（像素数）
     * @note 默认1200
     */
    void SetMinDefectArea(double area) { m_minDefectArea = area; }
    
    /**
     * @brief 设置ORB特征点数量
     * 
     * @param count 特征点数量
     * @note 默认1500
     */
    void SetORBFeaturesCount(int count) { m_orbFeatures = count; }
    
    /**
     * @brief 获取模板差异阈值
     */
    double GetTemplateDiffThreshold() const { return m_templateDiffThresh; }
    
    /**
     * @brief 获取最小缺陷面积
     */
    double GetMinDefectArea() const { return m_minDefectArea; }
    
    /**
     * @brief 获取ORB特征点数量
     */
    int GetORBFeaturesCount() const { return m_orbFeatures; }
    
    /**
     * @brief 检查是否已设置模板
     */
    bool HasTemplate() const { return m_hasTemplate; }

    // ==================== 特征对齐相关配置 ====================

    /**
     * @brief 设置是否使用特征对齐
     * 
     * @param use 是否使用特征对齐
     * @note 默认启用
     */
    void SetUseFeatureAlignment(bool use) { m_useFeatureAlignment = use; }
    
    /**
     * @brief 设置特征对齐所需最小内点数量
     * 
     * @param minInliers 最小内点数
     * @note 默认10
     */
    void SetMinInliersForAlignment(int minInliers) { m_minInliersForAlignment = minInliers; }
    
    /**
     * @brief 获取是否使用特征对齐
     */
    bool GetUseFeatureAlignment() const { return m_useFeatureAlignment; }
    
    /**
     * @brief 获取特征对齐所需最小内点数量
     */
    int GetMinInliersForAlignment() const { return m_minInliersForAlignment; }
    
    /**
     * @brief 使用特征对齐进行图像配准
     * 
     * @param currentBGR 当前BGR图像
     * @param homography 输出的单应性矩阵
     * @return bool 配准成功返回true，失败返回false
     * @note 使用FeatureAlignment类进行快速对齐
     * @see ComputeHomography()
     */
    bool ComputeHomographyWithFeatureAlignment(const cv::Mat& currentBGR, cv::Mat& homography);
    
    /**
     * @brief 使用特征对齐重构图像
     * 
     * @param currentBGR 当前BGR图像
     * @return cv::Mat 重构后的图像
     * @note 先计算变换矩阵，然后进行图像变换
     * @see ComputeHomographyWithFeatureAlignment()
     */
    cv::Mat AlignAndWarpImage(const cv::Mat& currentBGR);

    // ==================== 缺陷分类相关功能 ====================

    /**
     * @brief 加载模板库
     * 
     * @param templateDir 模板目录路径
     * @return bool 加载成功返回true，失败返回false
     * @note 支持jpg、jpeg、png、bmp格式的图像
     * @see ClassifyDefect()
     */
    bool LoadTemplateLibrary(const std::string& templateDir);
    
    /**
     * @brief 对缺陷区域进行分类
     * 
     * @param defectROI 缺陷区域图像
     * @return std::string 分类结果（模板名称或"Unknown"）
     * @note 使用模板匹配算法，相似度超过阈值才返回模板名称
     * @see LoadTemplateLibrary()
     */
    std::string ClassifyDefect(const cv::Mat& defectROI) const;
    
    /**
     * @brief 获取模板库信息
     * 
     * @return std::vector<std::string> 模板名称列表
     */
    std::vector<std::string> GetTemplateNames() const;
    
    /**
     * @brief 设置模板匹配阈值
     * 
     * @param threshold 匹配阈值（0-1）
     * @note 默认0.6
     */
    void SetTemplateMatchThreshold(double threshold) { m_templateMatchThresh = threshold; }
    
    /**
     * @brief 获取模板匹配阈值
     */
    double GetTemplateMatchThreshold() const { return m_templateMatchThresh; }

    // ==================== 图像预处理与特征提取 ====================

    /**
     * @brief 图像预处理（色彩校正与去噪）
     * 
     * @param src 输入BGR彩色图像（CV_8UC3或可转换到该类型）
     * @param dst 预处理后输出图像（BGR，CV_8UC3）
     * @note 实现内容：
     *       - Lab空间L通道直方图均衡化（改善亮度分布）
     *       - 轻度高斯模糊（减少噪声对特征提取的影响）
     * @note 该方法为轻量级色彩恒常性近似，并非完整Retinex实现
     */
    void Preprocess(const Mat& src, Mat& dst) const;

    /**
     * @brief 提取特征向量（行向量，类型CV_32F）
     * 
     * @param src 经过Preprocess的BGR图像
     * @return Mat 1xN的特征向量（CV_32F），包含多色域统计特征与LBP纹理直方图
     * @note 输出长度为固定值（依赖于直方图bins与LBP bins），适合直接构建训练矩阵
     * @see ComputeLBPHist()
     */
    Mat ExtractFeatures(const Mat& src) const;

    /**
     * @brief 计算灰度图的LBP直方图（默认256维）
     * 
     * @param gray 单通道灰度图（CV_8U）
     * @return Mat LBP直方图（1x256，CV_32F）
     * @note 使用多线程并行计算LBP
     * @see ExtractFeatures()
     */
    Mat ComputeLBPHist(const Mat& gray) const;

    // ==================== PCA相关功能 ====================

    /**
     * @brief PCA训练：对输入样本做PCA降维
     * 
     * @param samples 每行一个样本（CV_32F）
     * @param retainedComponents 保留的主成分数量，默认32
     * @note 使用PCA::DATA_AS_ROW模式
     * @see FitPCAByVariance()
     */
    void FitPCA(const Mat& samples, int retainedComponents = 32);

    /**
     * @brief 基于方差阈值自动选择主成分数并训练PCA
     * 
     * @param samples 每行一个样本（CV_32F）
     * @param varianceThreshold 方差阈值（0-1），例如0.95表示选择最小主成分数以覆盖>=95%的方差
     * @note 会生成pca_explained_variance.csv文件记录各主成分解释方差
     * @see FitPCA()
     */
    void FitPCAByVariance(const Mat& samples, double varianceThreshold = 0.95);

    /**
     * @brief 将单样本投影到已训练的PCA子空间
     * 
     * @param sample 1xN行向量（CV_32F）
     * @return Mat 投影后的行向量（CV_32F），如果未训练PCA则返回原始样本
     * @note 如果已设置特征标准化参数，会先进行标准化
     * @see FitPCA()
     */
    Mat ProjectPCA(const Mat& sample) const;

    // ==================== SVM训练与预测 ====================

    /**
     * @brief SVM训练
     * 
     * @param samples 行样本（CV_32F）
     * @param labels 标签（CV_32S单列）
     * @return bool 训练成功返回true，失败返回false
     * @note 若已FitPCA，则自动在PCA子空间训练
     * @note 会自动计算并保存特征标准化参数（均值和标准差）
     * @see TrainSVMAuto()
     */
    bool TrainSVM(const Mat& samples, const Mat& labels);

    /**
     * @brief 对单样本进行预测
     * 
     * @param sample 1xN行向量（CV_32F或可转换）
     * @return int 预测标签（例如0/1）
     * @note 如果训练时使用了PCA，该方法会对输入先执行PCA投影
     * @see TrainSVM()
     */
    int Predict(const Mat& sample) const;

    // ==================== 模型持久化 ====================

    /**
     * @brief 保存SVM模型和PCA参数到文件
     * 
     * @param basePath 文件名前缀
     * @return bool 保存成功返回true，失败返回false
     * @note 保存文件：
     *       - basePath_svm.yml: SVM模型
     *       - basePath_pca.yml: PCA参数（mean/eigenvectors/eigenvalues）和特征标准化参数
     * @see LoadModel()
     */
    bool SaveModel(const string& basePath) const;

    /**
     * @brief 从文件加载SVM与PCA参数
     * 
     * @param basePath 文件名前缀
     * @return bool 加载成功返回true，失败返回false
     * @see SaveModel()
     */
    bool LoadModel(const string& basePath);

    // ==================== PCA维度访问器 ====================

    /**
     * @brief 获取PCA维度
     */
    int GetPCADim() const { return m_pcaDim; }
    
    /**
     * @brief 设置PCA维度
     */
    void SetPCADim(int d) { m_pcaDim = d; }

    /**
     * @brief 获取每个主成分的解释方差和累积方差
     * 
     * @return std::vector<std::pair<double,double>> (解释方差, 累积方差)对的向量
     * @note 返回向量长度等于当前PCA的分量数，若未训练PCA返回空向量
     */
    std::vector<std::pair<double,double>> GetPCAExplainedVariance() const;

    // ==================== 自动SVM训练 ====================

    /**
     * @brief 使用OpenCV的trainAuto自动搜索SVM超参数并训练
     * 
     * @param samples 行样本（CV_32F）
     * @param labels 标签（CV_32S单列）
     * @return bool 训练成功返回true，失败返回false
     * @note 内部使用交叉验证，会自动计算并保存特征标准化参数
     * @see TrainSVM()
     */
    bool TrainSVMAuto(const Mat& samples, const Mat& labels);

private:
    /**
     * @brief 内部帮助函数：计算通道均值/标准差以及直方图
     * 
     * @param ch 单通道图像
     * @param outFeatures 输出特征向量（追加到末尾）
     * @param histBins 直方图bins数量，默认16
     * @note 计算内容：均值、标准差、归一化直方图
     */
    void ChannelStatsAndHist(const Mat& ch, vector<float>& outFeatures, int histBins = 16) const;

    // ==================== 成员变量 ====================

    Ptr<SVM> m_svm;                    ///< SVM分类器
    PCA m_pca;                         PCA对象
    int m_pcaDim;                      ///< PCA保留维度
    Mat m_featMean;                    ///< 特征标准化均值（1xD CV_64F）
    Mat m_featStd;                     ///< 特征标准化标准差（1xD CV_64F）

    // 模板法缺陷检测相关成员变量
    Mat m_templateGray;                ///< 模板灰度图
    bool m_hasTemplate;                ///< 是否已有模板
    double m_templateDiffThresh;       ///< 差异二值阈值
    double m_minDefectArea;            ///< 最小缺陷面积
    int m_orbFeatures;                 ///< ORB特征点数量

    // 特征对齐相关成员变量
    FeatureAlignment* m_featureAlignment; ///< 特征对齐对象
    Mat m_templateBGR;                 ///< 模板BGR图像（用于特征对齐）
    bool m_useFeatureAlignment;        ///< 是否使用特征对齐
    int m_minInliersForAlignment;      ///< 特征对齐所需最小内点数量

    // 缺陷分类相关成员变量
    std::vector<cv::Mat> m_templateImages;    ///< 模板图像
    std::vector<std::string> m_templateNames; ///< 模板名称（文件名）
    double m_templateMatchThresh;       ///< 模板匹配阈值
    bool m_templateLibraryLoaded;      ///< 模板库是否已加载
};

#endif // DEFECTDETECTION_H
