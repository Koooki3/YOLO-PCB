#include "DefectDetection.h"
#include <numeric>
#include <fstream>
#include <filesystem>

using namespace cv;
using namespace cv::ml;
using namespace std;

// ---------------------------------------------------------------------------
// 构造函数：初始化成员变量
// - 初始化默认的 SVM（C_SVC, RBF），并设置终止准则
// - 初始化 PCA 保留维度为 32（可通过 SetPCADim 修改）
// 复杂度：O(1)
// ---------------------------------------------------------------------------
DefectDetection::DefectDetection()
	: m_pcaDim(32)
{
	m_svm = SVM::create();
	// 默认 SVM 参数（可根据数据与需求调整）
	m_svm->setType(SVM::C_SVC);
	m_svm->setKernel(SVM::RBF);
	m_svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 1e-6));
}


// ---------------------------------------------------------------------------
// Preprocess
// 功能：对输入 BGR 图像进行简单的色彩校正与去噪
// 输入：src - 原始 BGR 图像（CV_8UC3 或可转换）
// 输出：dst - 处理后的 BGR 图像
// 说明：采用 Lab 空间对 L 通道进行直方图均衡化以改善亮度分布，随后进行小半径的高斯模糊
// 注意：该方法为轻量级色彩恒常性近似，并非完整 Retinex 实现
// 复杂度：O(N)（N = 像素数）
// ---------------------------------------------------------------------------
void DefectDetection::Preprocess(const Mat& src, Mat& dst) const
{
	CV_Assert(!src.empty());
	src.copyTo(dst);

	Mat lab;
	// 转到 Lab 空间，对 L 通道做直方图均衡化以减少光照影响
	cvtColor(dst, lab, COLOR_BGR2Lab);
	vector<Mat> labCh(3);
	split(lab, labCh);
	equalizeHist(labCh[0], labCh[0]);
	merge(labCh, lab);
	cvtColor(lab, dst, COLOR_Lab2BGR);

	// 轻微模糊以抑制噪声
	GaussianBlur(dst, dst, Size(3, 3), 0.5);
}


// ---------------------------------------------------------------------------
// ChannelStatsAndHist
// 功能：对单通道图像计算均值、标准差及归一化直方图（histBins）
// 输入：ch - 单通道图像（CV_8U 或 CV_32F）
// 输出：outFeatures - 追加均值、方差及直方图值
// 返回：无（结果通过 outFeatures 追加）
// 复杂度：O(N + B)（N = 像素数, B = 直方图 bin 数）
// ---------------------------------------------------------------------------
void DefectDetection::ChannelStatsAndHist(const Mat& ch, vector<float>& outFeatures, int histBins) const
{
	CV_Assert(ch.type() == CV_8U || ch.type() == CV_32F);
	Mat tmp;
	if (ch.type() == CV_32F)
	{
		ch.convertTo(tmp, CV_8U);
	}
	else
	{
		tmp = ch;
	}

	// 计算均值与标准差
	Scalar meanVal, stddevVal;
	meanStdDev(tmp, meanVal, stddevVal);
	outFeatures.push_back(static_cast<float>(meanVal[0]));
	outFeatures.push_back(static_cast<float>(stddevVal[0]));

	// 计算并归一化直方图
	int histSize = histBins;
	Mat hist;
	calcHist(vector<Mat>{tmp}, vector<int>{0}, Mat(), hist, vector<int>{histSize}, vector<float>{0,256});
	hist /= tmp.total();
	for (int i = 0; i < histSize; ++i)
	{
		outFeatures.push_back(hist.at<float>(i));
	}
}


// ---------------------------------------------------------------------------
// ComputeLBPHist
// 功能：基于 8 邻域计算每像素 LBP 编码，并返回 256 维归一化直方图
// 输入：gray - 灰度图（CV_8U）
// 输出：1x256 的 CV_32F 行向量（归一化）
// 说明：实现了基础的原始 LBP 编码，未使用旋转不变或统一模式的优化
// 复杂度：O(N)
// ---------------------------------------------------------------------------
Mat DefectDetection::ComputeLBPHist(const Mat& gray) const
{
	CV_Assert(gray.channels() == 1);
	Mat lbp = Mat::zeros(gray.size(), CV_8U);
	for (int y = 1; y < gray.rows - 1; ++y)
	{
		for (int x = 1; x < gray.cols - 1; ++x)
		{
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

	// 统计直方图并归一化
	Mat hist;
	int histSize = 256;
	calcHist(vector<Mat>{lbp}, vector<int>{0}, Mat(), hist, vector<int>{histSize}, vector<float>{0,256});
	hist = hist.reshape(1,1);
	hist.convertTo(hist, CV_32F);
	double s = sum(hist)[0];
	if (s > 0) hist /= static_cast<float>(s);
	return hist;
}


// ---------------------------------------------------------------------------
// ExtractFeatures
// 功能：在多色域（BGR/HSV/Lab）上对每个通道提取统计量（均值/方差）及直方图，
//       并将灰度 LBP 直方图拼接为最终特征向量
// 输入：src - BGR 彩色图（会在内部保证类型为 CV_8UC3）
// 输出：1xN 的 CV_32F 行向量
// 注意：输出向量长度为固定值（由 3 色域 * 3 通道 * (2 + histBins) + 256 决定）
// 复杂度：O(N_pixels + B)（B 为直方图总 bin 数）
// ---------------------------------------------------------------------------
Mat DefectDetection::ExtractFeatures(const Mat& src) const
{
	CV_Assert(!src.empty());
	Mat img;
	src.copyTo(img);
	if (img.type() != CV_8UC3) img.convertTo(img, CV_8UC3);

	Mat bgr = img, hsv, lab;
	cvtColor(bgr, hsv, COLOR_BGR2HSV);
	cvtColor(bgr, lab, COLOR_BGR2Lab);

	vector<Mat> bgrCh(3), hsvCh(3), labCh(3);
	split(bgr, bgrCh);
	split(hsv, hsvCh);
	split(lab, labCh);

	vector<float> feats;
	// 在每个通道上加入 均值/方差 + 直方图（16 bins）
	for (int i = 0; i < 3; ++i) ChannelStatsAndHist(bgrCh[i], feats, 16);
	for (int i = 0; i < 3; ++i) ChannelStatsAndHist(hsvCh[i], feats, 16);
	for (int i = 0; i < 3; ++i) ChannelStatsAndHist(labCh[i], feats, 16);

	// LBP 纹理直方图（灰度）
	Mat gray; cvtColor(bgr, gray, COLOR_BGR2GRAY);
	Mat lbpHist = ComputeLBPHist(gray); // 1x256 CV_32F
	// 将 lbpHist 数据追加到 feats
	feats.insert(feats.end(), (float*)lbpHist.datastart, (float*)lbpHist.dataend);

	// 转为 CV_32F 单行 Mat
	Mat featMat(1, static_cast<int>(feats.size()), CV_32F);
	for (size_t i = 0; i < feats.size(); ++i) featMat.at<float>(0, static_cast<int>(i)) = feats[i];
	return featMat;
}


// ---------------------------------------------------------------------------
// FitPCA
// 功能：对输入样本做 PCA 训练（DATA_AS_ROW 模式）
// 输入：samples - 每行一个样本，类型 CV_32F
// retainedComponents - 要保留的主成分数量
// 复杂度：O(M*N^2)（视矩阵大小而定，M 为样本数，N 为特征维度）
// ---------------------------------------------------------------------------
void DefectDetection::FitPCA(const Mat& samples, int retainedComponents)
{
	CV_Assert(samples.type() == CV_32F);
	int dims = min(retainedComponents, samples.cols);
	m_pcaDim = dims;
	m_pca = PCA(samples, Mat(), PCA::DATA_AS_ROW, m_pcaDim);
}

// 基于累计解释方差自动选择主成分并训练 PCA
void DefectDetection::FitPCAByVariance(const Mat& samples, double varianceThreshold)
{
	CV_Assert(samples.type() == CV_32F);
	CV_Assert(varianceThreshold > 0.0 && varianceThreshold <= 1.0);

	// 首先做完整版的 PCA（不截断）以获得所有特征的特征值
	PCA fullPca(samples, Mat(), PCA::DATA_AS_ROW, 0);
	Mat eigvals = fullPca.eigenvalues; // may be CV_32F or CV_64F and shape Nx1 or 1xN

	// 将 eigenvalues 转为 double 向量，做稳健读取
	size_t n = static_cast<size_t>(eigvals.total());
	vector<double> vals(n, 0.0);
	if (n > 0) {
		if (eigvals.type() == CV_64F) {
			const double* p = eigvals.ptr<double>();
			for (size_t i = 0; i < n; ++i) vals[i] = std::isfinite(p[i]) && p[i] > 0.0 ? p[i] : 0.0;
		} else if (eigvals.type() == CV_32F) {
			const float* p = eigvals.ptr<float>();
			for (size_t i = 0; i < n; ++i) vals[i] = std::isfinite(p[i]) && p[i] > 0.0f ? static_cast<double>(p[i]) : 0.0;
		} else {
			// Fallback generic read
			for (size_t i = 0; i < n; ++i) {
				double v = 0.0;
				try { v = eigvals.at<double>((int)i, 0); } catch (...) {
					try { v = eigvals.at<float>((int)i, 0); } catch (...) { v = 0.0; }
				}
				vals[i] = std::isfinite(v) && v > 0.0 ? v : 0.0;
			}
		}
	}

	double total = 0.0; for (double v : vals) total += v;
	if (total <= 0.0) {
		// fallback: use original FitPCA with default dim
		FitPCA(samples, m_pcaDim);
		return;
	}

	// 计算累计解释方差并将其写入 CSV 便于可视化
	vector<double> cumRatio; cumRatio.reserve(n);
	double cum = 0.0;
	for (size_t i = 0; i < n; ++i) {
		cum += vals[i];
		cumRatio.push_back(cum / total);
	}

	// 确保 models 目录存在，然后写文件
	try {
		std::filesystem::create_directories("models");
		std::ofstream ofs("models/pca_explained_variance.csv");
		if (ofs.is_open()) {
			ofs << "component,explained_variance,cumulative_variance\n";
			for (size_t i = 0; i < n; ++i) {
				double explained = vals[i] / total;
				ofs << (i+1) << "," << explained << "," << cumRatio[i] << "\n";
			}
			ofs.close();
		}
	} catch (...) {
		// 忽略文件写入错误（非致命）
	}

	int k = 0;
	for (size_t i = 0; i < cumRatio.size(); ++i) {
		if (cumRatio[i] >= varianceThreshold) { k = static_cast<int>(i) + 1; break; }
	}
	if (k <= 0) k = 1;
	k = std::min(k, samples.cols);

	// 重新生成 PCA 并截断到 k 个分量
	m_pcaDim = k;
	m_pca = PCA(samples, Mat(), PCA::DATA_AS_ROW, m_pcaDim);
}


// ---------------------------------------------------------------------------
// ProjectPCA
// 功能：将单样本投影到已训练 PCA 子空间
// 输入：sample - 1xN 行向量（CV_32F）
// 返回：投影后的行向量（CV_32F），如果未训练 PCA 则返回输入的克隆
// ---------------------------------------------------------------------------
Mat DefectDetection::ProjectPCA(const Mat& sample) const
{
	Mat input = sample;
	if (input.type() != CV_32F) input.convertTo(input, CV_32F);
	if (m_pca.eigenvectors.empty()) return input.clone();
	Mat projected; m_pca.project(input, projected); return projected;
}


// ---------------------------------------------------------------------------
// TrainSVM
// 功能：训练 SVM 分类器；若已经 FitPCA 则先投影到 PCA 子空间
// 输入：samples（每行一个样本 CV_32F），labels（单列 CV_32S）
// 返回：训练成功与否
// 复杂度：依赖于 SVM 内部实现（核函数、样本数、维度）
// ---------------------------------------------------------------------------
bool DefectDetection::TrainSVM(const Mat& samples, const Mat& labels)
{
	CV_Assert(!samples.empty());
	CV_Assert(samples.rows == labels.rows);

	Mat trainData = samples;
	if (trainData.type() != CV_32F) trainData.convertTo(trainData, CV_32F);

	if (!m_pca.eigenvectors.empty()) { Mat proj; m_pca.project(trainData, proj); trainData = proj; }

	Ptr<TrainData> td = TrainData::create(trainData, ROW_SAMPLE, labels);
	return m_svm->train(td);
}


// ---------------------------------------------------------------------------
// Predict
// 功能：对单样本进行预测，返回整型标签（例如 0/1）
// 输入：sample - 1xN 行向量（CV_32F 或可转换）
// 注意：如果训练时使用了 PCA，该方法会对输入先执行 PCA 投影
// ---------------------------------------------------------------------------
int DefectDetection::Predict(const Mat& sample) const
{
	Mat x = sample;
	if (x.type() != CV_32F) x.convertTo(x, CV_32F);
	if (!m_pca.eigenvectors.empty()) { Mat proj; m_pca.project(x, proj); x = proj; }
	return static_cast<int>(m_svm->predict(x));
}


// ---------------------------------------------------------------------------
// SaveModel
// 功能：保存 SVM 模型和 PCA 参数到文件
// - SVM 保存为 basePath_svm.yml（OpenCV SVM 自带序列化）
// - PCA 参数以 FileStorage 保存 mean/eigenvectors/eigenvalues
// 返回：保存成功/失败
// ---------------------------------------------------------------------------
bool DefectDetection::SaveModel(const string& basePath) const
{
	string svmFile = basePath + "_svm.yml";
	string pcaFile = basePath + "_pca.yml";
	try {
		m_svm->save(svmFile);
		FileStorage fs(pcaFile, FileStorage::WRITE);
		if (!fs.isOpened()) return false;
		fs << "mean" << m_pca.mean;
		fs << "eigenvectors" << m_pca.eigenvectors;
		fs << "eigenvalues" << m_pca.eigenvalues;
		fs.release();
	} catch (...) { return false; }
	return true;
}


// ---------------------------------------------------------------------------
// LoadModel
// 功能：从文件加载 SVM 与 PCA 参数
// 注意：若文件不存在或读取失败则返回 false
// ---------------------------------------------------------------------------
bool DefectDetection::LoadModel(const string& basePath)
{
	string svmFile = basePath + "_svm.yml";
	string pcaFile = basePath + "_pca.yml";
	try {
		m_svm = Algorithm::load<SVM>(svmFile);
		FileStorage fs(pcaFile, FileStorage::READ);
		if (!fs.isOpened()) return false;
		fs["mean"] >> m_pca.mean;
		fs["eigenvectors"] >> m_pca.eigenvectors;
		fs["eigenvalues"] >> m_pca.eigenvalues;
		fs.release();
	} catch (...) { return false; }
	return true;
}

std::vector<std::pair<double,double>> DefectDetection::GetPCAExplainedVariance() const
{
	std::vector<std::pair<double,double>> res;
	if (m_pca.eigenvalues.empty()) return res;

	Mat eigvals = m_pca.eigenvalues;
	size_t n = static_cast<size_t>(eigvals.total());
	vector<double> vals(n, 0.0);
	if (n > 0) {
		if (eigvals.type() == CV_64F) {
			const double* p = eigvals.ptr<double>();
			for (size_t i = 0; i < n; ++i) vals[i] = std::isfinite(p[i]) && p[i] > 0.0 ? p[i] : 0.0;
		} else if (eigvals.type() == CV_32F) {
			const float* p = eigvals.ptr<float>();
			for (size_t i = 0; i < n; ++i) vals[i] = std::isfinite(p[i]) && p[i] > 0.0f ? static_cast<double>(p[i]) : 0.0;
		} else {
			for (size_t i = 0; i < n; ++i) {
				double v = 0.0;
				try { v = eigvals.at<double>((int)i, 0); } catch (...) {
					try { v = eigvals.at<float>((int)i, 0); } catch (...) { v = 0.0; }
				}
				vals[i] = std::isfinite(v) && v > 0.0 ? v : 0.0;
			}
		}
	}
	double total = 0.0; for (double v : vals) total += v;
	if (total <= 0.0) return res;
	double cum = 0.0;
	for (size_t i = 0; i < n; ++i) {
		double explained = vals[i] / total;
		cum += vals[i];
		double cumul = cum / total;
		res.emplace_back(explained, cumul);
	}
	return res;
}

