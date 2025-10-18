#include "DefectDetection.h"
#include <numeric>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <QDebug>

using namespace cv;
using namespace cv::ml;
using namespace std;

// 计算特征矩阵的列均值和标准差（double 精度），并保存到成员变量
static void computeMeanStd(const Mat& samples, Mat& meanOut, Mat& stdOut)
{
    CV_Assert(samples.type() == CV_32F || samples.type() == CV_64F);
    Mat tmp;
    if (samples.type() == CV_32F)
    {
        samples.convertTo(tmp, CV_64F);
    }
    else
    {
        tmp = samples;
    }
    int cols = tmp.cols;
    meanOut = Mat::zeros(1, cols, CV_64F);
    stdOut = Mat::zeros(1, cols, CV_64F);
    for (int j = 0; j < cols; ++j)
    {
        Scalar mu, sd;
        meanStdDev(tmp.col(j), mu, sd);
        meanOut.at<double>(0,j) = mu[0];
        stdOut.at<double>(0,j) = sd[0] > 1e-12 ? sd[0] : 1.0; // 防止除零
    }
}

// 对样本做列向量标准化（(x-mean)/std），返回 CV_32F
static Mat applyStandardize(const Mat& samples, const Mat& mean, const Mat& stdv)
{
    // Convert input to CV_64F for stable arithmetic, then apply (x-mean)/std per column
    Mat tmp;
    samples.convertTo(tmp, CV_64F);
    int rows = tmp.rows, cols = tmp.cols;
    Mat out(rows, cols, CV_32F);
    for (int j = 0; j < cols; ++j)
    {
        double m = mean.at<double>(0,j);
        double s = stdv.at<double>(0,j);
        for (int i = 0; i < rows; ++i)
        {
            double v = tmp.at<double>(i,j);
            out.at<float>(i,j) = static_cast<float>((v - m) / s);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// 构造函数：初始化成员变量
// - 初始化默认的 SVM（C_SVC, RBF），并设置终止准则
// - 初始化 PCA 保留维度为 32（可通过 SetPCADim 修改）
// - 初始化特征对齐对象
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
    
    // 初始化特征对齐对象
    m_featureAlignment = new FeatureAlignment();
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
// FitPCA
// 功能：对输入样本做 PCA 训练（DATA_AS_ROW 模式）
// 输入：samples - 每行一个样本，类型 CV_32F
// retainedComponents - 要保留的主成分数量
// 复杂度：O(M*N^2)（视矩阵大小而定，M 为样本数，N 为特征维度）
// ---------------------------------------------------------------------------
void DefectDetection::FitPCA(const Mat& samples, int retainedComponents)
{
    CV_Assert(samples.type() == CV_32F);
    int dims = std::min(retainedComponents, samples.cols);
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
    if (n > 0)
    {
        if (eigvals.type() == CV_64F)
        {
            const double* p = eigvals.ptr<double>();
            for (size_t i = 0; i < n; ++i)
            {
                vals[i] = std::isfinite(p[i]) && p[i] > 0.0 ? p[i] : 0.0;
            }
        }
        else if (eigvals.type() == CV_32F)
        {
            const float* p = eigvals.ptr<float>();
            for (size_t i = 0; i < n; ++i)
            {
                vals[i] = std::isfinite(p[i]) && p[i] > 0.0f ? static_cast<double>(p[i]) : 0.0;
            }
        }
        else
        {
            // Fallback generic read
            for (size_t i = 0; i < n; ++i)
            {
                double v = 0.0;
                try
                {
                    v = eigvals.at<double>((int)i, 0);
                }
                catch (...)
                {
                    try
                    {
                        v = eigvals.at<float>((int)i, 0);
                    }
                    catch (...)
                    {
                        v = 0.0;
                    }
                }
                vals[i] = std::isfinite(v) && v > 0.0 ? v : 0.0;
            }
        }
    }

    double total = 0.0;
    for (double v : vals)
    {
        total += v;
    }
    if (total <= 0.0)
    {
        // fallback: use original FitPCA with default dim
        FitPCA(samples, m_pcaDim);
        return;
    }

    // 计算累计解释方差并将其写入 CSV 便于可视化
    vector<double> cumRatio; cumRatio.reserve(n);
    double cum = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        cum += vals[i];
        cumRatio.push_back(cum / total);
    }

    // 确保 models 目录存在，然后写文件
    try
    {
        std::filesystem::create_directories("models");
        std::ofstream ofs("models/pca_explained_variance.csv");
        if (ofs.is_open())
        {
            ofs << "component,explained_variance,cumulative_variance\n";
            for (size_t i = 0; i < n; ++i)
            {
                double explained = vals[i] / total;
                ofs << (i+1) << "," << explained << "," << cumRatio[i] << "\n";
            }
            ofs.close();
        }
    }
    catch (...)
    {
        // 忽略文件写入错误（非致命）
    }

    int k = 0;
    for (size_t i = 0; i < cumRatio.size(); ++i)
    {
        if (cumRatio[i] >= varianceThreshold)
        {
            k = static_cast<int>(i) + 1;
            break;
        }
    }
    if (k <= 0)
    {
        k = 1;
    }
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
    if (input.type() != CV_32F)
    {
        input.convertTo(input, CV_32F);
    }
    // Apply standardization if available
    if (!m_featMean.empty() && !m_featStd.empty())
    {
        input = applyStandardize(input, m_featMean, m_featStd);
    }
    if (m_pca.eigenvectors.empty())
    {
        return input.clone();
    }
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
    if (trainData.type() != CV_32F)
    {
        trainData.convertTo(trainData, CV_32F);
    }

    // Ensure feature standardization params exist; compute if absent
    if (m_featMean.empty() || m_featStd.empty())
    {
        Mat sampF = trainData;
        computeMeanStd(sampF, m_featMean, m_featStd);
    }
    // Apply standardization
    Mat trainStd = applyStandardize(trainData, m_featMean, m_featStd);

    if (!m_pca.eigenvectors.empty())
    {
        Mat proj; m_pca.project(trainStd, proj);
        trainData = proj;
    }
    else
    {
        trainData = trainStd;
    }

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
    if (x.type() != CV_32F)
    {
        x.convertTo(x, CV_32F);
    }
    // Apply standardization if available, then PCA projection
    if (!m_featMean.empty() && !m_featStd.empty())
    {
        x = applyStandardize(x, m_featMean, m_featStd);
    }
    if (!m_pca.eigenvectors.empty())
    {
        Mat proj; m_pca.project(x, proj);
        x = proj;
    }
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
    try
    {
        m_svm->save(svmFile);
        FileStorage fs(pcaFile, FileStorage::WRITE);
        if (!fs.isOpened())
        {
            return false;
        }
        fs << "mean" << m_pca.mean;
        fs << "eigenvectors" << m_pca.eigenvectors;
        fs << "eigenvalues" << m_pca.eigenvalues;
        // 保存特征标准化参数（若存在）
        if (!m_featMean.empty())
        {
            fs << "feat_mean" << m_featMean;
        }
        if (!m_featStd.empty())
        {
            fs << "feat_std" << m_featStd;
        }
        fs.release();
    }
    catch (...)
    {
        return false;
    }
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
    try
    {
        m_svm = Algorithm::load<SVM>(svmFile);
        FileStorage fs(pcaFile, FileStorage::READ);
        if (!fs.isOpened())
        {
            return false;
        }
        fs["mean"] >> m_pca.mean;
        fs["eigenvectors"] >> m_pca.eigenvectors;
        fs["eigenvalues"] >> m_pca.eigenvalues;
        // 读取特征标准化参数（可选）
        fs["feat_mean"] >> m_featMean;
        fs["feat_std"] >> m_featStd;
        fs.release();
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool DefectDetection::TrainSVMAuto(const Mat& samples, const Mat& labels)
{
    CV_Assert(!samples.empty());
    CV_Assert(samples.rows == labels.rows);

    // 先标准化特征并保存均值/方差
    Mat sampF;
    if (samples.type() != CV_32F)
    {
        samples.convertTo(sampF, CV_32F);
    }
    else
    {
        sampF = samples;
    }
    computeMeanStd(sampF, m_featMean, m_featStd);
    Mat trainStd = applyStandardize(sampF, m_featMean, m_featStd);

    // 若已有 PCA，则投影到 PCA 子空间
    Mat trainData = trainStd;
    if (!m_pca.eigenvectors.empty())
    {
        Mat proj; m_pca.project(trainStd, proj); trainData = proj;
    }

    Ptr<TrainData> td = TrainData::create(trainData, ROW_SAMPLE, labels);

    // 使用 OpenCV 的 trainAuto 搜索参数（使用默认的参数网格）
    // 注意：trainAuto 内部会做自动参数搜索并训练最终模型
    bool ok = m_svm->trainAuto(td);
    return ok;
}

// ---------------------------------------------------------------------------
// 模板法缺陷检测相关功能实现
// ---------------------------------------------------------------------------

// 从当前帧设置模板
bool DefectDetection::SetTemplateFromCurrent(const Mat& currentFrame)
{
    if (currentFrame.empty())
    {
        return false;
    }

    Mat gray;
    if (currentFrame.channels() == 3)
    {
        cvtColor(currentFrame, gray, COLOR_BGR2GRAY);
    }
    else
    {
        gray = currentFrame.clone();
    }
    
    GaussianBlur(gray, gray, Size(3,3), 0);
    m_templateGray = gray.clone();
    m_hasTemplate = true;

    return true;
}

// 从文件设置模板
bool DefectDetection::SetTemplateFromFile(const string& filePath)
{
    Mat bgr = imread(filePath, IMREAD_COLOR);
    if (bgr.empty())
    {
        return false;
    }
    
    Mat gray;
    cvtColor(bgr, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, gray, Size(3,3), 0);
    m_templateGray = gray.clone();
    m_hasTemplate = true;

    return true;
}

// 计算单应性矩阵（模板 <- 当前）
bool DefectDetection::ComputeHomography(const Mat& currentGray, Mat& homography, vector<DMatch>* debugMatches)
{
    if (!m_hasTemplate || m_templateGray.empty())
    {
        return false;
    }

    // ORB 特征检测
    Ptr<ORB> orb = ORB::create(m_orbFeatures);
    vector<KeyPoint> keypointsTemplate, keypointsCurrent;
    Mat descriptorsTemplate, descriptorsCurrent;
    orb->detectAndCompute(m_templateGray, noArray(), keypointsTemplate, descriptorsTemplate);
    orb->detectAndCompute(currentGray, noArray(), keypointsCurrent, descriptorsCurrent);

    if (descriptorsTemplate.empty() || descriptorsCurrent.empty())
    {
        return false;
    }

    // 暴力匹配 + 交叉检验
    BFMatcher matcher(NORM_HAMMING, true);
    vector<DMatch> matches;
    matcher.match(descriptorsTemplate, descriptorsCurrent, matches);
    if (matches.size() < 8)
    {
        return false;
    }

    // 根据距离剔除离群点
    double maxDist = 0, minDist = 1e9;
    for (auto& match : matches)
    {
        double d = match.distance;
        maxDist = std::max(maxDist, d);
        minDist = std::min(minDist, d);
    }
    
    vector<DMatch> goodMatches;
    double threshold = std::max(2.0 * minDist, 30.0); // 经验阈值
    for (auto& match : matches)
    {
        if (match.distance <= threshold)
        {
            goodMatches.push_back(match);
        }
    }
    if (goodMatches.size() < 8)
    {
        goodMatches = matches; // 兜底
    }

    if (debugMatches)
    {
        *debugMatches = goodMatches;
    }

    vector<Point2f> pointsTemplate, pointsCurrent;
    pointsTemplate.reserve(goodMatches.size());
    pointsCurrent.reserve(goodMatches.size());
    for (auto& match : goodMatches)
    {
        pointsTemplate.push_back(keypointsTemplate[match.queryIdx].pt);
        pointsCurrent.push_back(keypointsCurrent[match.trainIdx].pt);
    }

    // RANSAC 求单应性矩阵（模板 <- 当前）
    vector<unsigned char> inliers;
    homography = findHomography(pointsCurrent, pointsTemplate, RANSAC, 3.0, inliers);
    if (homography.empty())
    {
        return false;
    }
    
    return true;
}

// 并行处理轮廓的函数
void ProcessContourParallel(const vector<Point>& contour, const Mat& inverseHomography, 
                           const Size& imageSize, double minArea, vector<Rect>& result, mutex& mtx)
{
    double area = contourArea(contour);
    if (area < minArea)
    {
        return;
    }

    Rect boundingRectTemplate = boundingRect(contour); // 模板系下
    
    // 四角坐标
    vector<Point2f> sourcePoints = {
        { (float)boundingRectTemplate.x, (float)boundingRectTemplate.y },
        { (float)(boundingRectTemplate.x + boundingRectTemplate.width), (float)boundingRectTemplate.y },
        { (float)(boundingRectTemplate.x + boundingRectTemplate.width), (float)(boundingRectTemplate.y + boundingRectTemplate.height) },
        { (float)boundingRectTemplate.x, (float)(boundingRectTemplate.y + boundingRectTemplate.height) }
    };
    
    vector<Point2f> targetPoints;
    perspectiveTransform(sourcePoints, targetPoints, inverseHomography);

    // 用回投影后的四角做外接框（当前图像坐标系）
    float minX = imageSize.width, minY = imageSize.height, maxX = 0, maxY = 0;
    for (auto& point : targetPoints)
    {
        minX = std::min(minX, point.x);
        minY = std::min(minY, point.y);
        maxX = std::max(maxX, point.x);
        maxY = std::max(maxY, point.y);
    }
    
    Rect defectBox(Point2f(minX, minY), Point2f(maxX, maxY));
    defectBox &= Rect(0, 0, imageSize.width, imageSize.height); // 裁边
    
    if (defectBox.area() > 0)
    {
        lock_guard<mutex> lock(mtx);
        result.push_back(defectBox);
    }
}

// 并行计算LBP的函数
void ComputeLBPParallel(const Mat& gray, Mat& lbp, int startRow, int endRow)
{
    for (int y = startRow; y < endRow; ++y)
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
}

// 并行计算LBP直方图
Mat DefectDetection::ComputeLBPHist(const Mat& gray) const
{
    CV_Assert(gray.channels() == 1);
    Mat lbp = Mat::zeros(gray.size(), CV_8U);
    
    // 使用多线程并行计算LBP
    int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;
    
    vector<thread> threads;
    int rowsPerThread = gray.rows / numThreads;
    
    for (int i = 0; i < numThreads; ++i)
    {
        int startRow = i * rowsPerThread + 1; // +1 避免边界
        int endRow = (i == numThreads - 1) ? gray.rows - 1 : (i + 1) * rowsPerThread;
        threads.emplace_back(ComputeLBPParallel, std::cref(gray), std::ref(lbp), startRow, endRow);
    }
    
    for (auto& t : threads)
    {
        t.join();
    }

    // 统计直方图并归一化
    Mat hist;
    int histSize = 256;
    calcHist(vector<Mat>{lbp}, vector<int>{0}, Mat(), hist, vector<int>{histSize}, vector<float>{0,256});
    hist = hist.reshape(1,1);
    hist.convertTo(hist, CV_32F);
    double s = sum(hist)[0];
    if (s > 0)
    {
        hist /= static_cast<float>(s);
    }
    return hist;
}

// 并行处理多通道统计特征
void ProcessChannelStatsParallel(const Mat& channel, vector<float>& results, int histBins, mutex& mtx)
{
    Mat tmp;
    if (channel.type() == CV_32F)
    {
        channel.convertTo(tmp, CV_8U);
    }
    else
    {
        tmp = channel;
    }

    // 计算均值与标准差
    Scalar meanVal, stddevVal;
    meanStdDev(tmp, meanVal, stddevVal);
    
    // 计算并归一化直方图
    int histSize = histBins;
    Mat hist;
    calcHist(vector<Mat>{tmp}, vector<int>{0}, Mat(), hist, vector<int>{histSize}, vector<float>{0,256});
    hist /= tmp.total();

    // 合并结果
    vector<float> channelFeatures;
    channelFeatures.push_back(static_cast<float>(meanVal[0]));
    channelFeatures.push_back(static_cast<float>(stddevVal[0]));
    for (int i = 0; i < histSize; ++i)
    {
        channelFeatures.push_back(hist.at<float>(i));
    }

    lock_guard<mutex> lock(mtx);
    results.insert(results.end(), channelFeatures.begin(), channelFeatures.end());
}

// 优化后的特征提取函数
Mat DefectDetection::ExtractFeatures(const Mat& src) const
{
    CV_Assert(!src.empty());
    Mat img;
    src.copyTo(img);
    if (img.type() != CV_8UC3)
    {
        img.convertTo(img, CV_8UC3);
    }

    Mat bgr = img, hsv, lab;
    cvtColor(bgr, hsv, COLOR_BGR2HSV);
    cvtColor(bgr, lab, COLOR_BGR2Lab);

    vector<Mat> bgrCh(3), hsvCh(3), labCh(3);
    split(bgr, bgrCh);
    split(hsv, hsvCh);
    split(lab, labCh);

    vector<float> feats;
    mutex featsMutex;
    vector<thread> threads;

    // 并行处理所有通道的统计特征
    auto processChannels = [&](const vector<Mat>& channels) {
        for (const auto& channel : channels)
        {
            threads.emplace_back(ProcessChannelStatsParallel, std::cref(channel), std::ref(feats), 16, std::ref(featsMutex));
        }
    };

    processChannels(bgrCh);
    processChannels(hsvCh);
    processChannels(labCh);

    for (auto& t : threads)
    {
        t.join();
    }

    // LBP 纹理直方图（灰度）
    Mat gray; cvtColor(bgr, gray, COLOR_BGR2GRAY);
    Mat lbpHist = ComputeLBPHist(gray); // 1x256 CV_32F
    // 将 lbpHist 数据追加到 feats（使用安全的 ptr 访问）
    CV_Assert(lbpHist.isContinuous());
    const float* p = lbpHist.ptr<float>(0);
    int len = lbpHist.cols * lbpHist.rows;
    feats.insert(feats.end(), p, p + len);

    // 转为 CV_32F 单行 Mat
    Mat featMat(1, static_cast<int>(feats.size()), CV_32F);
    for (size_t i = 0; i < feats.size(); ++i)
    {
        featMat.at<float>(0, static_cast<int>(i)) = feats[i];
    }
    return featMat;
}

// 优化后的缺陷检测函数（并行版本）
vector<Rect> DefectDetection::DetectDefects(const Mat& currentBGR, const Mat& homography, Mat* debugMask)
{
    vector<Rect> defectBoxes;

    // 统一到模板坐标系做差异检测
    Mat currentGray;
    cvtColor(currentBGR, currentGray, COLOR_BGR2GRAY);
    GaussianBlur(currentGray, currentGray, Size(3,3), 0);

    Mat warped;
    warpPerspective(currentGray, warped, homography, m_templateGray.size(), INTER_LINEAR);

    // 差异图
    Mat diff;
    absdiff(m_templateGray, warped, diff);
    GaussianBlur(diff, diff, Size(5,5), 0);

    // 二值化
    Mat binary;
    if (m_templateDiffThresh <= 0)
    {
        threshold(diff, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
    }
    else
    {
        threshold(diff, binary, m_templateDiffThresh, 255, THRESH_BINARY);
    }

    // 形态学净化
    morphologyEx(binary, binary, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(5,5)));
    morphologyEx(binary, binary, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(9,9)));

    if (debugMask)
    {
        *debugMask = binary.clone();
    }

    // 找轮廓（在模板坐标系）
    vector<vector<Point>> contours;
    findContours(binary, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if (contours.empty())
    {
        return defectBoxes;
    }

    // 将模板坐标系下的外接框四角回投影到"当前图像坐标系"
    Mat inverseHomography;
    if (!invert(homography, inverseHomography))
    {
        return defectBoxes;
    }

    // 并行处理轮廓
    mutex resultMutex;
    vector<thread> contourThreads;
    
    for (auto& contour : contours)
    {
        contourThreads.emplace_back(ProcessContourParallel, std::cref(contour), 
                                   std::cref(inverseHomography), currentBGR.size(), 
                                   m_minDefectArea, std::ref(defectBoxes), std::ref(resultMutex));
    }

    for (auto& t : contourThreads)
    {
        t.join();
    }

    return defectBoxes;
}

std::vector<std::pair<double,double>> DefectDetection::GetPCAExplainedVariance() const
{
    std::vector<std::pair<double,double>> res;
    if (m_pca.eigenvalues.empty())
    {
        return res;
    }

    Mat eigvals = m_pca.eigenvalues;
    size_t n = static_cast<size_t>(eigvals.total());
    vector<double> vals(n, 0.0);
    if (n > 0)
    {
        if (eigvals.type() == CV_64F)
        {
            const double* p = eigvals.ptr<double>();
            for (size_t i = 0; i < n; ++i)
            {
                vals[i] = std::isfinite(p[i]) && p[i] > 0.0 ? p[i] : 0.0;
            }
        }
        else if (eigvals.type() == CV_32F)
        {
            const float* p = eigvals.ptr<float>();
            for (size_t i = 0; i < n; ++i)
            {
                vals[i] = std::isfinite(p[i]) && p[i] > 0.0f ? static_cast<double>(p[i]) : 0.0;
            }
        }
        else
        {
            for (size_t i = 0; i < n; ++i)
            {
                double v = 0.0;
                try
                {
                    v = eigvals.at<double>((int)i, 0);
                }
                catch (...)
                {
                    try
                    {
                        v = eigvals.at<float>((int)i, 0);
                    }
                    catch (...)
                    {
                        v = 0.0;
                    }
                }
                vals[i] = std::isfinite(v) && v > 0.0 ? v : 0.0;
            }
        }
    }
    double total = 0.0;
    for (double v : vals)
    {
        total += v;
    }

    if (total <= 0.0)
    {
        return res;
    }
    double cum = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        double explained = vals[i] / total;
        cum += vals[i];
        double cumul = cum / total;
        res.emplace_back(explained, cumul);
    }
    return res;
}

// ---------------------------------------------------------------------------
// 缺陷分类相关功能实现
// ---------------------------------------------------------------------------

// 加载模板库
bool DefectDetection::LoadTemplateLibrary(const std::string& templateDir)
{
    m_templateImages.clear();
    m_templateNames.clear();
    
    try {
        // 遍历模板目录中的所有图像文件
        for (const auto& entry : std::filesystem::directory_iterator(templateDir)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                // 检查是否为图像文件
                if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                    // 读取模板图像
                    Mat templateImg = imread(entry.path().string(), IMREAD_GRAYSCALE);
                    if (!templateImg.empty()) {
                        // 预处理模板图像
                        GaussianBlur(templateImg, templateImg, Size(3,3), 0);
                        
                        // 保存模板图像和名称
                        m_templateImages.push_back(templateImg);
                        m_templateNames.push_back(entry.path().stem().string());
                    }
                }
            }
        }
        
        if (m_templateImages.empty()) {
            return false;
        }
        
        m_templateLibraryLoaded = true;
        return true;
        
    } catch (const std::exception& e) {
        return false;
    }
}

// 对缺陷区域进行分类
std::string DefectDetection::ClassifyDefect(const cv::Mat& defectROI) const
{
    if (!m_templateLibraryLoaded || m_templateImages.empty()) {
        return "Unknown";
    }
    
    if (defectROI.empty()) {
        return "Unknown";
    }
    
    // 预处理缺陷区域
    Mat defectGray;
    if (defectROI.channels() == 3) {
        cvtColor(defectROI, defectGray, COLOR_BGR2GRAY);
    } else {
        defectGray = defectROI.clone();
    }
    GaussianBlur(defectGray, defectGray, Size(3,3), 0);
    
    double bestScore = -1.0;
    int bestIndex = -1;
    
    // 与所有模板进行匹配
    for (size_t i = 0; i < m_templateImages.size(); ++i) {
        const Mat& templateImg = m_templateImages[i];
        
        // 调整缺陷区域大小以匹配模板
        Mat defectResized;
        resize(defectGray, defectResized, templateImg.size());
        
        // 使用模板匹配
        Mat result;
        matchTemplate(defectResized, templateImg, result, TM_CCOEFF_NORMED);
        
        double minVal, maxVal;
        Point minLoc, maxLoc;
        minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
        
        // 取最大相似度作为匹配分数
        if (maxVal > bestScore) {
            bestScore = maxVal;
            bestIndex = static_cast<int>(i);
        }
    }
    
    // 检查是否超过阈值
    if (bestIndex >= 0 && bestScore >= m_templateMatchThresh) {
        return m_templateNames[bestIndex];
    } else {
        return "Unknown";
    }
}

// 获取模板库信息
std::vector<std::string> DefectDetection::GetTemplateNames() const
{
    return m_templateNames;
}

// ---------------------------------------------------------------------------
// 特征对齐相关功能实现
// ---------------------------------------------------------------------------

// 使用特征对齐进行图像配准
bool DefectDetection::ComputeHomographyWithFeatureAlignment(const cv::Mat& currentBGR, cv::Mat& homography)
{
    if (!m_hasTemplate || m_templateBGR.empty()) {
        qDebug() << "尚未设置模板或模板为空";
        return false;
    }

    if (currentBGR.empty()) {
        qDebug() << "当前图像为空";
        return false;
    }

    // 设置对齐参数
    AlignmentParams params;
    params.minInliers = m_minInliersForAlignment;
    params.enableParallel = true;
    params.numThreads = 4;

    // 使用快速对齐（匹配到足够内点时立即停止）
    AlignmentResult result = m_featureAlignment->FastAlignImages(currentBGR, m_templateBGR, params);
    
    if (!result.success) {
        qDebug() << "特征对齐失败，内点数量:" << result.inlierCount;
        return false;
    }

    // 返回变换矩阵
    homography = result.transformMatrix;
    
    qDebug() << "特征对齐成功 - 内点数量:" << result.inlierCount 
             << "重投影误差:" << result.reprojectionError;
    
    return true;
}

// 使用特征对齐重构图像
cv::Mat DefectDetection::AlignAndWarpImage(const cv::Mat& currentBGR)
{
    if (!m_hasTemplate || m_templateBGR.empty()) {
        qDebug() << "尚未设置模板或模板为空";
        return cv::Mat();
    }

    if (currentBGR.empty()) {
        qDebug() << "当前图像为空";
        return cv::Mat();
    }

    // 计算变换矩阵
    cv::Mat homography;
    if (!ComputeHomographyWithFeatureAlignment(currentBGR, homography)) {
        qDebug() << "无法计算变换矩阵";
        return cv::Mat();
    }

    // 重构图像
    cv::Mat alignedImage = m_featureAlignment->WarpImage(currentBGR, homography, m_templateBGR.size());
    
    if (alignedImage.empty()) {
        qDebug() << "图像重构失败";
        return cv::Mat();
    }

    qDebug() << "图像重构成功，尺寸:" << alignedImage.cols << "x" << alignedImage.rows;
    
    return alignedImage;
}

// 修改SetTemplateFromCurrent方法，保存BGR模板
bool DefectDetection::SetTemplateFromCurrent(const cv::Mat& currentFrame)
{
    if (currentFrame.empty())
    {
        return false;
    }

    Mat gray;
    if (currentFrame.channels() == 3)
    {
        cvtColor(currentFrame, gray, COLOR_BGR2GRAY);
    }
    else
    {
        gray = currentFrame.clone();
    }
    
    GaussianBlur(gray, gray, Size(3,3), 0);
    m_templateGray = gray.clone();
    
    // 保存BGR模板用于特征对齐
    if (currentFrame.channels() == 3) {
        m_templateBGR = currentFrame.clone();
    } else {
        cvtColor(currentFrame, m_templateBGR, COLOR_GRAY2BGR);
    }
    imwrite("../Img/templateBGR.jpg", m_templateBGR); 
    m_hasTemplate = true;

    return true;
}

// 修改SetTemplateFromFile方法，保存BGR模板
bool DefectDetection::SetTemplateFromFile(const std::string& filePath)
{
    Mat bgr = imread(filePath, IMREAD_COLOR);
    if (bgr.empty())
    {
        return false;
    }
    
    Mat gray;
    cvtColor(bgr, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, gray, Size(3,3), 0);
    m_templateGray = gray.clone();
    
    // 保存BGR模板用于特征对齐
    m_templateBGR = bgr.clone();
    m_hasTemplate = true;
    imwrite("../Img/templateBGR.jpg", m_templateBGR); 

    return true;
}
