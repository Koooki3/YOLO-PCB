// DefectDetection
// 此类实现了一套面向嵌入式/工业视觉的轻量化缺陷二分类流水线。
// 设计目标：
// - 使用多色域统计特征（BGR/HSV/Lab）和纹理特征（LBP）作为轻量特征描述子；
// - 通过 PCA 做降维以减少在线计算量；
// - 使用 OpenCV 的 SVM 做二分类（适合小样本场景）；
// - 提供模型的保存/加载接口以支持脱机训练、在线推理分离。
//
// 主要流程：
// 1) Preprocess: 图像预处理（色彩校正/直方图均衡化/去噪）
// 2) ExtractFeatures: 多色域统计特征 + LBP 纹理直方图 -> 得到定长特征向量
// 3) FitPCA/ProjectPCA: 基于样本的 PCA 降维与投影
// 4) TrainSVM/Predict: 在（可选）PCA 子空间上训练/预测 SVM
// 5) SaveModel/LoadModel: 将 SVM 与 PCA 参数持久化到文件
//
// 备注：
// - 本类适用于图像/ROI 级别的缺陷判别（即给出整张图或局部 ROI 的缺陷/正常标签），
//   并不提供像素级分割或缺陷定位功能；
// - 特征和参数（如直方图 bins、LBP 配置、PCA 保留维度、SVM 超参）可根据数据集调整；
// - 方法输入输出约定详见各方法注释。

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

class DefectDetection {
public:
	// 构造函数
	// 初始化默认 SVM 及 PCA 维度（可通过 SetPCADim 调整）
	DefectDetection();

	// 模板法缺陷检测相关功能
	// 设置模板（从当前帧或文件）
	bool SetTemplateFromCurrent(const cv::Mat& currentFrame);
	bool SetTemplateFromFile(const std::string& filePath);
	
	// 计算单应性矩阵（模板 <- 当前）
	bool ComputeHomography(const cv::Mat& currentGray, cv::Mat& homography, std::vector<cv::DMatch>* debugMatches = nullptr);
	
	// 缺陷检测：根据配准后差异，得到在"当前图像坐标系"的缺陷外接框
	std::vector<cv::Rect> DetectDefects(const cv::Mat& currentBGR, const cv::Mat& homography, cv::Mat* debugMask = nullptr);
	
	// 模板法参数配置
	void SetTemplateDiffThreshold(double threshold) { m_templateDiffThresh = threshold; }
	void SetMinDefectArea(double area) { m_minDefectArea = area; }
	void SetORBFeaturesCount(int count) { m_orbFeatures = count; }
	
	double GetTemplateDiffThreshold() const { return m_templateDiffThresh; }
	double GetMinDefectArea() const { return m_minDefectArea; }
	int GetORBFeaturesCount() const { return m_orbFeatures; }
	bool HasTemplate() const { return m_hasTemplate; }

	// 特征对齐相关配置
	void SetUseFeatureAlignment(bool use) { m_useFeatureAlignment = use; }
	void SetMinInliersForAlignment(int minInliers) { m_minInliersForAlignment = minInliers; }
	bool GetUseFeatureAlignment() const { return m_useFeatureAlignment; }
	int GetMinInliersForAlignment() const { return m_minInliersForAlignment; }
	
	// 使用特征对齐进行图像配准
	bool ComputeHomographyWithFeatureAlignment(const cv::Mat& currentBGR, cv::Mat& homography);
	
	// 使用特征对齐重构图像
	cv::Mat AlignAndWarpImage(const cv::Mat& currentBGR);

	// 缺陷分类相关功能
	// 加载模板库
	bool LoadTemplateLibrary(const std::string& templateDir);
	
	// 对缺陷区域进行分类
	std::string ClassifyDefect(const cv::Mat& defectROI) const;
	
	// 获取模板库信息
	std::vector<std::string> GetTemplateNames() const;
	
	// 设置模板匹配阈值
	void SetTemplateMatchThreshold(double threshold) { m_templateMatchThresh = threshold; }
	double GetTemplateMatchThreshold() const { return m_templateMatchThresh; }

	// 图像预处理（色彩校正与去噪）
	// 参数：
	// - src: 输入 BGR 彩色图像（CV_8UC3 或可转换到该类型）
	// - dst: 预处理后输出图像（BGR，CV_8UC3）
	// 说明：该方法实现了简单的色彩恒常性近似（Lab 空间 L 通道直方图均衡化）
	//       以及轻度高斯模糊以减少噪声对特征提取的影响。
	void Preprocess(const Mat& src, Mat& dst) const;

	// 提取特征向量（行向量，类型 CV_32F）
	// - 输入：经过 Preprocess 的 BGR 图像
	// - 输出：1xN 的 Mat（CV_32F），包含多色域统计特征与 LBP 纹理直方图
	// 注意：输出长度为固定值（依赖于直方图 bins 与 LBP bins），适合直接构建训练矩阵
	Mat ExtractFeatures(const Mat& src) const;

	// 计算灰度图的 LBP 直方图（默认 256 维）
	// 输入：单通道灰度图（CV_8U）
	Mat ComputeLBPHist(const Mat& gray) const;

	// PCA 训练：samples 每行为一个样本（CV_32F）
	// retainedComponents：保留的主成分数量，默认 32
	void FitPCA(const Mat& samples, int retainedComponents = 32);

	// 基于方差阈值自动选择主成分数并训练 PCA
	// varianceThreshold: [0,1]，例如 0.95 表示选择最小主成分数以覆盖 >=95% 的方差
	void FitPCAByVariance(const Mat& samples, double varianceThreshold = 0.95);

	// 将单样本投影到已训练的 PCA 子空间（如果未训练 PCA 则返回原始样本）
	Mat ProjectPCA(const Mat& sample) const;

	// SVM 训练与预测
	// - TrainSVM: samples 行样本（CV_32F），labels 为 CV_32S 单列；若已 FitPCA，则自动在 PCA 子空间训练
	// - Predict:  对单个样本预测标签（如果训练时使用 PCA，会在预测前投影）
	bool TrainSVM(const Mat& samples, const Mat& labels);
	int Predict(const Mat& sample) const;

	// 模型持久化（SVM + PCA）
	// basePath 用作文件名前缀，保存为 basePath_svm.yml 与 basePath_pca.yml
	bool SaveModel(const string& basePath) const;
	bool LoadModel(const string& basePath);

	// PCA 维度访问器
	int GetPCADim() const { return m_pcaDim; }
	void SetPCADim(int d) { m_pcaDim = d; }

	// 返回每个主成分的 (explained_variance, cumulative_variance)
	// 返回向量长度等于当前 PCA 的分量数（若未训练 PCA 返回空向量）
	std::vector<std::pair<double,double>> GetPCAExplainedVariance() const;

private:
	// 内部帮助函数：计算通道均值/标准差以及直方图
    void ChannelStatsAndHist(const Mat& ch, vector<float>& outFeatures, int histBins = 16) const;

	// 成员变量
    Ptr<SVM> m_svm;
    PCA m_pca;
	int m_pcaDim;
	// 特征标准化参数（训练时计算并保存）
	Mat m_featMean; // 1 x D CV_64F
	Mat m_featStd;  // 1 x D CV_64F

	// 模板法缺陷检测相关成员变量
	Mat m_templateGray;              // 模板灰度图
	bool m_hasTemplate = false;      // 是否已有模板
	double m_templateDiffThresh = 25.0; // 差异二值阈值
	double m_minDefectArea = 1200;   // 最小缺陷面积
	int m_orbFeatures = 1500;        // ORB特征点数量

	// 特征对齐相关成员变量
	FeatureAlignment* m_featureAlignment; // 特征对齐对象
	Mat m_templateBGR;               // 模板BGR图像（用于特征对齐）
	bool m_useFeatureAlignment = true; // 是否使用特征对齐
	int m_minInliersForAlignment = 10; // 特征对齐所需最小内点数量

	// 缺陷分类相关成员变量
	std::vector<cv::Mat> m_templateImages;    // 模板图像
	std::vector<std::string> m_templateNames; // 模板名称（文件名）
	double m_templateMatchThresh = 0.6;       // 模板匹配阈值
	bool m_templateLibraryLoaded = false;     // 模板库是否已加载

public:
	// 使用 OpenCV 的 trainAuto 自动搜索 SVM 超参数并训练（内部使用交叉验证）
	// 返回训练成功与否
	bool TrainSVMAuto(const Mat& samples, const Mat& labels);
};

#endif // DEFECTDETECTION_H
