// DefectDetection.cpp
// 实现 DefectDetection.h 中声明的功能

#include "DefectDetection.h"
#include <numeric>

using namespace cv;
using namespace cv::ml;

DefectDetection::DefectDetection()
	: m_pcaDim(32)
{
	m_svm = SVM::create();
	// 默认 SVM 参数（可根据需要调整）
	m_svm->setType(SVM::C_SVC);
	m_svm->setKernel(SVM::RBF);
	m_svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 1e-6));
}

void DefectDetection::Preprocess(const Mat& src, Mat& dst) const
{
	// 简单白平衡：将每通道按其平均值缩放到全局平均
	CV_Assert(!src.empty());
	src.copyTo(dst);

	Mat lab;
	cvtColor(dst, lab, COLOR_BGR2Lab);
	// 使用简单直方图均衡化对 L 通道进行调整
	std::vector<Mat> labCh(3);
	split(lab, labCh);
	equalizeHist(labCh[0], labCh[0]);
	merge(labCh, lab);
	cvtColor(lab, dst, COLOR_Lab2BGR);

	// 额外：轻微高斯模糊去噪
	GaussianBlur(dst, dst, Size(3, 3), 0.5);
}

void DefectDetection::ChannelStatsAndHist(const Mat& ch, std::vector<float>& outFeatures, int histBins) const
{
	CV_Assert(ch.type() == CV_8U || ch.type() == CV_32F);
	Mat tmp;
	if (ch.type() == CV_32F) ch.convertTo(tmp, CV_8U);
	else tmp = ch;

	// mean and std
	Scalar meanVal, stddevVal;
	meanStdDev(tmp, meanVal, stddevVal);
	outFeatures.push_back(static_cast<float>(meanVal[0]));
	outFeatures.push_back(static_cast<float>(stddevVal[0]));

	// histogram
	int histSize = histBins;
	float range[] = {0, 256};
	const float* histRange = {range};
	Mat hist;
	calcHist(std::vector<Mat>{tmp}, std::vector<int>{0}, Mat(), hist, std::vector<int>{histSize}, std::vector<float>{0,256});
	// normalize hist and append
	hist /= tmp.total();
	for (int i = 0; i < histSize; ++i)
		outFeatures.push_back(hist.at<float>(i));
}

Mat DefectDetection::ComputeLBPHist(const Mat& gray) const
{
	CV_Assert(gray.channels() == 1);
	Mat lbp = Mat::zeros(gray.size(), CV_8U);
	for (int y = 1; y < gray.rows - 1; ++y) {
		for (int x = 1; x < gray.cols - 1; ++x) {
			uchar center = gray.at<uchar>(y, x);
			unsigned char code = 0;
			code |= (gray.at<uchar>(y-1, x-1) > center) << 7;
			code |= (gray.at<uchar>(y-1, x  ) > center) << 6;
			code |= (gray.at<uchar>(y-1, x+1) > center) << 5;
			code |= (gray.at<uchar>(y,   x+1) > center) << 4;
			code |= (gray.at<uchar>(y+1, x+1) > center) << 3;
			code |= (gray.at<uchar>(y+1, x  ) > center) << 2;
			code |= (gray.at<uchar>(y+1, x-1) > center) << 1;
			code |= (gray.at<uchar>(y,   x-1) > center) << 0;
			lbp.at<uchar>(y, x) = code;
		}
	}

	// histogram 256 bins
	Mat hist;
	int histSize = 256;
	float range[] = {0, 256};
	const float* histRange = {range};
	calcHist(std::vector<Mat>{lbp}, std::vector<int>{0}, Mat(), hist, std::vector<int>{histSize}, std::vector<float>{0,256});
	hist = hist.reshape(1,1);
	hist.convertTo(hist, CV_32F);
	// normalize
	double s = sum(hist)[0];
	if (s > 0) hist /= static_cast<float>(s);
	return hist;
}

Mat DefectDetection::ExtractFeatures(const Mat& src) const
{
	CV_Assert(!src.empty());
	Mat img;
	src.copyTo(img);
	// 确保为 BGR 8U
	if (img.type() != CV_8UC3) img.convertTo(img, CV_8UC3);

	Mat bgr = img;
	Mat hsv, lab;
	cvtColor(bgr, hsv, COLOR_BGR2HSV);
	cvtColor(bgr, lab, COLOR_BGR2Lab);

	std::vector<Mat> bgrCh(3), hsvCh(3), labCh(3);
	split(bgr, bgrCh);
	split(hsv, hsvCh);
	split(lab, labCh);

	std::vector<float> feats;
	// 每个通道均值/方差/直方图（16 bins）
	for (int i = 0; i < 3; ++i) ChannelStatsAndHist(bgrCh[i], feats, 16);
	for (int i = 0; i < 3; ++i) ChannelStatsAndHist(hsvCh[i], feats, 16);
	for (int i = 0; i < 3; ++i) ChannelStatsAndHist(labCh[i], feats, 16);

	// LBP 特征（灰度）
	Mat gray;
	cvtColor(bgr, gray, COLOR_BGR2GRAY);
	Mat lbpHist = ComputeLBPHist(gray); // 1x256 CV_32F
	feats.insert(feats.end(), (float*)lbpHist.datastart, (float*)lbpHist.dataend);

	// 转为 Mat 行向量 CV_32F
	Mat featMat(1, static_cast<int>(feats.size()), CV_32F);
	for (size_t i = 0; i < feats.size(); ++i) featMat.at<float>(0, static_cast<int>(i)) = feats[i];
	return featMat;
}

void DefectDetection::FitPCA(const Mat& samples, int retainedComponents)
{
	CV_Assert(samples.type() == CV_32F);
	int dims = std::min(retainedComponents, samples.cols);
	m_pcaDim = dims;
	m_pca = PCA(samples, Mat(), PCA::DATA_AS_ROW, m_pcaDim);
}

Mat DefectDetection::ProjectPCA(const Mat& sample) const
{
	Mat input = sample;
	if (input.type() != CV_32F) input.convertTo(input, CV_32F);
	Mat projected;
	if (m_pca.eigenvectors.empty()) {
		// 未训练 PCA，直接返回原始
		return input.clone();
	}
	m_pca.project(input, projected);
	return projected;
}

bool DefectDetection::TrainSVM(const Mat& samples, const Mat& labels)
{
	CV_Assert(!samples.empty());
	CV_Assert(samples.rows == labels.rows);

	Mat trainData = samples;
	if (trainData.type() != CV_32F) trainData.convertTo(trainData, CV_32F);

	// 如果有 PCA，则先投影
	if (!m_pca.eigenvectors.empty()) {
		Mat proj;
		m_pca.project(trainData, proj);
		trainData = proj;
	}

	Ptr<TrainData> td = TrainData::create(trainData, ROW_SAMPLE, labels);
	return m_svm->train(td);
}

int DefectDetection::Predict(const Mat& sample) const
{
	Mat x = sample;
	if (x.type() != CV_32F) x.convertTo(x, CV_32F);
	if (!m_pca.eigenvectors.empty()) {
		Mat proj;
		m_pca.project(x, proj);
		x = proj;
	}
	return static_cast<int>(m_svm->predict(x));
}

bool DefectDetection::SaveModel(const std::string& basePath) const
{
	// 保存 SVM
	std::string svmFile = basePath + "_svm.yml";
	std::string pcaFile = basePath + "_pca.yml";
	try {
		m_svm->save(svmFile);
		FileStorage fs(pcaFile, FileStorage::WRITE);
		if (!fs.isOpened()) return false;
		fs << "mean" << m_pca.mean;
		fs << "eigenvectors" << m_pca.eigenvectors;
		fs << "eigenvalues" << m_pca.eigenvalues;
		fs.release();
	} catch (...) {
		return false;
	}
	return true;
}

bool DefectDetection::LoadModel(const std::string& basePath)
{
	std::string svmFile = basePath + "_svm.yml";
	std::string pcaFile = basePath + "_pca.yml";
	try {
		m_svm = Algorithm::load<SVM>(svmFile);
		FileStorage fs(pcaFile, FileStorage::READ);
		if (!fs.isOpened()) return false;
		fs["mean"] >> m_pca.mean;
		fs["eigenvectors"] >> m_pca.eigenvectors;
		fs["eigenvalues"] >> m_pca.eigenvalues;
		fs.release();
	} catch (...) {
		return false;
	}
	return true;
}

