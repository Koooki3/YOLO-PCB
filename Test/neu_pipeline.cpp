// neu_pipeline_plain.cpp
// 纯 C++（不依赖 Qt）的 NEU 数据集训练/评估管线
// 使用方式（位置参数）：
//   neu_pipeline_plain <mode> <root> <pcaDim> <outCsv>
// 示例：
//   neu_pipeline_plain train ../Data/NEU 32 neu_predictions.csv
//   neu_pipeline_plain eval  ../Data/NEU 32 neu_predictions.csv
//
// 目录约定（root 参数指向数据集根目录）：
//   <root>/train/<class>/*.png
//   <root>/test/<class>/*.png
//
// 说明：依赖 OpenCV 和项目中的 DefectDetection 类（基于 OpenCV）。

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <algorithm>

#include "DefectDetection.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

static bool hasSuffixIgnoreCase(const string &s, const string &suf) {
    if (s.size() < suf.size()) return false;
    size_t off = s.size() - suf.size();
    for (size_t i = 0; i < suf.size(); ++i) {
        char a = s[off + i];
        char b = suf[i];
        if (std::tolower(static_cast<unsigned char>(a)) != std::tolower(static_cast<unsigned char>(b))) return false;
    }
    return true;
}

static bool isImageExt(const string &name) {
    if (name.empty()) return false;
    static const vector<string> exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"};
    for (const auto &e : exts) if (hasSuffixIgnoreCase(name, e)) return true;
    return false;
}

// load dataset where each subfolder under dirPath is a class
static bool loadDatasetMultiClass(const string &dirPath, DefectDetection &detector,
                                  vector<Mat> &outFeats, vector<int> &outLabels,
                                  vector<string> &classNames, int resizeW = 128, int resizeH = 128)
{
    fs::path root(dirPath);
    if (!fs::exists(root) || !fs::is_directory(root)) return false;

    // iterate subdirectories, sorted by name
    vector<fs::directory_entry> subs;
    for (auto &p : fs::directory_iterator(root)) if (p.is_directory()) subs.push_back(p);
    sort(subs.begin(), subs.end(), [](const fs::directory_entry &a, const fs::directory_entry &b){ return a.path().filename().string() < b.path().filename().string(); });

    for (auto &d : subs) {
        string cname = d.path().filename().string();
        classNames.push_back(cname);
        int label = (int)classNames.size() - 1;
        for (auto &f : fs::directory_iterator(d.path())) {
            if (!f.is_regular_file()) continue;
            string fname = f.path().filename().string();
            if (!isImageExt(fname)) continue;
            Mat img = imread(f.path().string());
            if (img.empty()) continue;
            if (img.cols != resizeW || img.rows != resizeH) cv::resize(img, img, Size(resizeW, resizeH));
            Mat pre; detector.Preprocess(img, pre);
            Mat feat = detector.ExtractFeatures(pre);
            outFeats.push_back(feat);
            outLabels.push_back(label);
        }
    }
    return true;
}

static void savePredictionsCsv(const string &outCsv, const vector<string> &paths,
                               const vector<int> &labels, const vector<int> &preds,
                               const vector<string> &classNames)
{
    ofstream ofs(outCsv);
    if (!ofs.is_open()) return;
    ofs << "path,true_label,true_name,pred_label,pred_name\n";
    for (size_t i = 0; i < paths.size(); ++i) {
        string trueName = (labels[i] >= 0 && labels[i] < (int)classNames.size()) ? classNames[labels[i]] : string("");
        string predName = (preds[i] >= 0 && preds[i] < (int)classNames.size()) ? classNames[preds[i]] : string("");
        ofs << paths[i] << "," << labels[i] << "," << trueName << "," << preds[i] << "," << predName << "\n";
    }
    ofs.close();
}

static vector<string> readClassFile(const string &clsFile) {
    vector<string> classNames;
    ifstream ifs(clsFile);
    if (!ifs.is_open()) return classNames;
    string line;
    while (getline(ifs, line)) {
        if (line.size()) {
            // trim
            line.erase(line.find_last_not_of(" \r\n\t") + 1);
            line.erase(0, line.find_first_not_of(" \r\n\t"));
            if (!line.empty()) classNames.push_back(line);
        }
    }
    ifs.close();
    return classNames;
}

int main(int argc, char *argv[])
{
    string mode = "train";
    string root = "../Data/NEU";
    int pcaDim = 32;
    string outCsv = "neu_predictions.csv";
    double varianceThreshold = 0.95;

    if (argc >= 2) mode = argv[1];
    if (argc >= 3) root = argv[2];
    if (argc >= 4) pcaDim = stoi(argv[3]);
    if (argc >= 5) outCsv = argv[4];
    if (argc >= 6) varianceThreshold = stod(argv[5]);

    cout << "Mode: " << mode << " Root: " << root << " PCA: " << pcaDim << "\n";

    DefectDetection detector;

    if (mode == "train") {
        vector<Mat> feats;
        vector<int> labels;
        vector<string> classNames;
        string trainDir = (fs::path(root) / "train").string();
        cout << "Loading train folder: " << trainDir << "\n";
        if (!loadDatasetMultiClass(trainDir, detector, feats, labels, classNames)) {
            cerr << "无法读取训练目录：" << trainDir << "\n";
            return -1;
        }
        if (feats.empty()) { cerr << "训练样本为空，请检查数据路径与目录结构\n"; return -1; }

        Mat sampleMat((int)feats.size(), feats[0].cols, CV_32F);
        for (size_t i = 0; i < feats.size(); ++i) {
            Mat r = feats[i]; if (r.type() != CV_32F) r.convertTo(r, CV_32F);
            r.copyTo(sampleMat.row((int)i));
        }
        Mat labelMat((int)labels.size(), 1, CV_32S);
        for (size_t i = 0; i < labels.size(); ++i) labelMat.at<int>((int)i, 0) = labels[i];

        if (pcaDim == 0) {
            cout << "PCA 自动选择模式：按累计方差 " << varianceThreshold << " 选择主成分..." << endl;
            detector.FitPCAByVariance(sampleMat, varianceThreshold);
            cout << "自动选择 PCA 维度=" << detector.GetPCADim() << "" << endl;
        } else {
            detector.FitPCA(sampleMat, pcaDim);
        }
        bool ok = detector.TrainSVM(sampleMat, labelMat);
        if (!ok) { cerr << "SVM 训练失败\n"; return -1; }
        cout << "SVM 训练完成。保存模型...\n";
        fs::create_directories("models");
        if (!detector.SaveModel("models/defect_detector")) { cerr << "模型保存失败\n"; return -1; }
        cout << "模型已保存到 models/defect_detector_*\n";

        // 保存类名映射
        string clsFile = string("models/defect_detector_classes.txt");
        ofstream ofs(clsFile);
        if (ofs.is_open()) {
            for (auto &cn : classNames) ofs << cn << "\n";
            ofs.close();
            cout << "类别映射已保存到 " << clsFile << "\n";
        } else {
            cerr << "警告：无法保存类别映射文件：" << clsFile << "\n";
        }
            // 打印前 20 个主成分的 explained / cumulative 比例
            auto ev = detector.GetPCAExplainedVariance();
            if (!ev.empty()) {
                cout << "前 20 个主成分的解释方差与累计解释方差：\n";
                cout << "idx\texplained\tcumulative\n";
                int limit = (int)min<size_t>(20, ev.size());
                for (int i = 0; i < limit; ++i) cout << (i+1) << "\t" << ev[i].first << "\t" << ev[i].second << "\n";
            }
        return 0;
    }
    else if (mode == "eval") {
        string testDir = (fs::path(root) / "test").string();
        cout << "Loading test folder: " << testDir << "\n";
        vector<string> imgPaths;
        vector<int> trueLabels;

        string clsFile = "models/defect_detector_classes.txt";
        vector<string> classNames = readClassFile(clsFile);
        if (classNames.empty()) {
            cerr << "无法打开类别映射文件：" << clsFile << "，将按测试目录子文件夹顺序自动推断类别（可能与训练不一致）\n";
            // fallback
            fs::path td(testDir);
            if (fs::exists(td) && fs::is_directory(td)) {
                vector<fs::directory_entry> subs;
                for (auto &p : fs::directory_iterator(td)) if (p.is_directory()) subs.push_back(p);
                sort(subs.begin(), subs.end(), [](const fs::directory_entry &a, const fs::directory_entry &b){ return a.path().filename().string() < b.path().filename().string(); });
                for (auto &d : subs) classNames.push_back(d.path().filename().string());
            }
        }

        for (size_t lbl = 0; lbl < classNames.size(); ++lbl) {
            string sub = classNames[lbl];
            fs::path subdir = fs::path(testDir) / sub;
            if (!fs::exists(subdir) || !fs::is_directory(subdir)) continue;
            for (auto &f : fs::directory_iterator(subdir)) {
                if (!f.is_regular_file()) continue;
                if (!isImageExt(f.path().filename().string())) continue;
                imgPaths.push_back(f.path().string()); trueLabels.push_back((int)lbl);
            }
        }

        if (imgPaths.empty()) { cerr << "测试集为空，请检查路径\n"; return -1; }

        if (!detector.LoadModel("models/defect_detector")) { cerr << "加载模型失败，请先运行 train 模式或检查 models 路径\n"; return -1; }

        vector<int> preds; preds.reserve(imgPaths.size());
        int numClasses = (int)classNames.size();
        vector<vector<int>> confMat(numClasses, vector<int>(numClasses, 0));

        for (size_t i = 0; i < imgPaths.size(); ++i) {
            Mat img = imread(imgPaths[i]);
            if (img.empty()) { preds.push_back(-1); continue; }
            Mat pre; detector.Preprocess(img, pre);
            Mat feat = detector.ExtractFeatures(pre);
            int p = detector.Predict(feat);
            preds.push_back(p);
            int gt = trueLabels[i];
            if (gt >= 0 && gt < numClasses && p >= 0 && p < numClasses) confMat[gt][p]++;
        }

        vector<int> gtTotals(numClasses,0), predTotals(numClasses,0), correct(numClasses,0);
        int totalSamples = 0; int totalCorrect = 0;
        for (int i = 0; i < numClasses; ++i) {
            for (int j = 0; j < numClasses; ++j) {
                gtTotals[i] += confMat[i][j];
                predTotals[j] += confMat[i][j];
                if (i==j) { correct[i] = confMat[i][j]; totalCorrect += confMat[i][j]; }
                totalSamples += confMat[i][j];
            }
        }

        cout << "混淆矩阵 (行=真实, 列=预测):\n";
        cout << "\t";
        for (int j = 0; j < numClasses; ++j) cout << classNames[j] << "\t";
        cout << "\n";
        for (int i = 0; i < numClasses; ++i) {
            cout << classNames[i] << "\t";
            for (int j = 0; j < numClasses; ++j) cout << confMat[i][j] << "\t";
            cout << "\n";
        }

        cout << "\nPer-class metrics:\n";
        cout << "Class\tPrecision\tRecall\tF1\tSupport\n";
        for (int i = 0; i < numClasses; ++i) {
            int TP = confMat[i][i];
            int FP = 0; for (int g = 0; g < numClasses; ++g) if (g!=i) FP += confMat[g][i];
            int FN = 0; for (int p = 0; p < numClasses; ++p) if (p!=i) FN += confMat[i][p];
            double precision = (TP + FP) ? double(TP) / double(TP + FP) : 0.0;
            double recall = (TP + FN) ? double(TP) / double(TP + FN) : 0.0;
            double f1 = (precision + recall) ? 2.0 * precision * recall / (precision + recall) : 0.0;
            cout << classNames[i] << "\t" << precision << "\t" << recall << "\t" << f1 << "\t" << gtTotals[i] << "\n";
        }

        double acc = (totalSamples>0) ? double(totalCorrect) / double(totalSamples) : 0.0;
        cout << "\nOverall Accuracy=" << acc << " (" << totalCorrect << "/" << totalSamples << ")\n";

        savePredictionsCsv(outCsv, imgPaths, trueLabels, preds, classNames);
        cout << "预测结果已保存到 " << outCsv << "\n";
        return 0;
    }
    else {
        cerr << "未知模式：" << mode << "（支持 train 或 eval）\n";
        return -1;
    }

    return 0;
}
