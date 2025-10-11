// DefectDetection.h
// 封装缺陷检测相关算法：色彩校正、多色域特征提取、LBP、PCA降维、SVM训练/预测
// 采用帕斯卡命名法，方法和类名均使用 PascalCase

#ifndef DEFECTDETECTION_H
#define DEFECTDETECTION_H

#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <string>
#include <vector>

using namespace cv;
using namespace cv::ml;
using namespace std;

// DefectDetection
// 说明（中文注释）:
// 这个类提供一套用于缺陷分类的管线：
// 1) Preprocess: 简单白平衡（色彩恒常性近似）并可转换颜色空间
// 2) ExtractFeatures: 在RGB/HSV/Lab空间提取均值/方差/直方图和LBP纹理直方图
// 3) FitPCA/ProjectPCA: 对特征进行PCA降维
// 4) TrainSVM/Predict: 训练 SVM 并对单张图片预测标签
// 5) Save/Load: 模型持久化（SVM + PCA）

class DefectDetection {
public:
	// 构造函数
	DefectDetection();

	// 简单白平衡 + 去噪（可选）
	// src: 输入BGR图像
	// dst: 输出已校正BGR图像
    void Preprocess(const Mat& src, Mat& dst) const;

	// 从一张图像提取特征向量（行向量）
	// 输出特征为 CV_32F 单行 Mat
    Mat ExtractFeatures(const Mat& src) const;

	// 计算灰度图的 LBP 直方图（256维）
    Mat ComputeLBPHist(const Mat& gray) const;

	// PCA 训练：输入样本为每行一个样本（CV_32F）
    void FitPCA(const Mat& samples, int retainedComponents = 32);
    // 对单个样本投影到 PCA 子空间（输入行向量 CV_32F）
    Mat ProjectPCA(const Mat& sample) const;

	// SVM 训练（samples 行样本，labels 单列 CV_32S）
	// 若已训练 PCA，会对 samples 先进行投影
    bool TrainSVM(const Mat& samples, const Mat& labels);
	// 预测单样本标签，返回预测的整数标签
    int Predict(const Mat& sample) const;

	// 模型持久化（保存/加载 SVM + PCA）
    bool SaveModel(const string& basePath) const;
    bool LoadModel(const string& basePath);

	// 设置/获取 PCA 保留维度
	int GetPCADim() const { return m_pcaDim; }
	void SetPCADim(int d) { m_pcaDim = d; }

private:
	// 内部帮助函数：计算通道均值/标准差以及直方图
    void ChannelStatsAndHist(const Mat& ch, vector<float>& outFeatures, int histBins = 16) const;

	// 成员变量
    Ptr<SVM> m_svm;
    PCA m_pca;
	int m_pcaDim;
};

#endif // DEFECTDETECTION_H
