// neu_pipeline.cpp
// 使用 Qt5 + OpenCV 的控制台程序，针对 NEU 数据集执行：
// - train: 从指定训练目录读取图片（结构见下），提取特征，PCA 降维，训练 SVM，并保存模型
// - eval:  从指定测试目录读取图片，加载模型并预测，输出混淆矩阵与 CSV 预测结果
// 
// 目录约定（root 参数指向数据集根目录）：
//   <root>/train/normal/*.png
//   <root>/train/defect/*.png
//   <root>/test/normal/*.png
//   <root>/test/defect/*.png
// 
// 编译说明见 README_NEU.md

#include <QCoreApplication>
#include <QCommandLineParser>
#include <QDir>
#include <QFile>
#include <QTextStream>
#include <QDateTime>

#include "DefectDetection.h"
#include <opencv2/opencv.hpp>

#include <iostream>
#include <vector>
#include <map>
#include <fstream>

using namespace std;
using namespace cv;

// 多类数据集加载
// 约定：dirPath 下每个子目录为一个类别（子目录名作为类别名），子目录内为图片文件
// 返回：feats/labels 填充完成，classNames 按子目录顺序记录类别名称
static bool loadDatasetMultiClass(const QString &dirPath, DefectDetection &detector,
                                  std::vector<Mat> &outFeats, std::vector<int> &outLabels,
                                  std::vector<std::string> &classNames,
                                  int resizeW = 128, int resizeH = 128)
{
    QDir dir(dirPath);
    if (!dir.exists()) return false;

    // 列出子目录（排除 . 和 ..）
    QStringList subdirs = dir.entryList(QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name);
    for (const QString &sub : subdirs) {
        QString subpath = dir.filePath(sub);
        QDir d(subpath);
        if (!d.exists()) continue;

        // 记录类别名（子目录名）
        classNames.push_back(sub.toStdString());
        int label = static_cast<int>(classNames.size()) - 1;

        QStringList nameFilters; nameFilters << "*.png" << "*.jpg" << "*.bmp" << "*.tif";
        QFileInfoList files = d.entryInfoList(nameFilters, QDir::Files, QDir::Name);
        for (auto fi : files) {
            Mat img = imread(fi.absoluteFilePath().toStdString());
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

static void savePredictionsCsv(const QString &outCsv, const vector<string> &paths,
                               const vector<int> &labels, const vector<int> &preds,
                               const vector<string> &classNames)
{
    QFile f(outCsv);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) return;
    QTextStream out(&f);
    out << "path,true_label,true_name,pred_label,pred_name\n";
    for (size_t i = 0; i < paths.size(); ++i) {
        string trueName = (labels[i] >= 0 && labels[i] < (int)classNames.size()) ? classNames[labels[i]] : string("");
        string predName = (preds[i] >= 0 && preds[i] < (int)classNames.size()) ? classNames[preds[i]] : string("");
        out << paths[i].c_str() << "," << labels[i] << "," << trueName.c_str() << "," << preds[i] << "," << predName.c_str() << "\n";
    }
    f.close();
}

int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);
    QCoreApplication::setApplicationName("NEU Pipeline");

    QCommandLineParser parser;
    parser.setApplicationDescription("NEU dataset training/evaluation pipeline using DefectDetection");
    parser.addHelpOption();
    parser.addVersionOption();

    QCommandLineOption modeOption(QStringList() << "m" << "mode", "Mode: train or eval", "mode", "train");
    QCommandLineOption rootOption(QStringList() << "r" << "root", "Dataset root path", "root", "../Data/NEU");
    QCommandLineOption pcaOption(QStringList() << "p" << "pca", "PCA dimension (default 32)", "pca", "32");
    QCommandLineOption outOption(QStringList() << "o" << "out", "Output CSV for predictions", "out", "neu_predictions.csv");

    parser.addOption(modeOption);
    parser.addOption(rootOption);
    parser.addOption(pcaOption);
    parser.addOption(outOption);
    parser.process(app);

    QString mode = parser.value(modeOption);
    QString root = parser.value(rootOption);
    int pcaDim = parser.value(pcaOption).toInt();
    QString outCsv = parser.value(outOption);

    cout << "Mode: " << mode.toStdString() << " Root: " << root.toStdString() << " PCA: " << pcaDim << "\n";

    DefectDetection detector;

    if (mode == "train") {
        vector<Mat> feats;
        vector<int> labels;
        vector<string> classNames;
        QString trainDir = QDir(root).filePath("train");
        cout << "Loading train folder: " << trainDir.toStdString() << "\n";
        if (!loadDatasetMultiClass(trainDir, detector, feats, labels, classNames)) {
            cerr << "无法读取训练目录：" << trainDir.toStdString() << "\n";
            return -1;
        }
        if (feats.empty()) {
            cerr << "训练样本为空，请检查数据路径与目录结构" << endl;
            return -1;
        }
        Mat sampleMat((int)feats.size(), feats[0].cols, CV_32F);
        for (size_t i = 0; i < feats.size(); ++i) {
            Mat r = feats[i]; if (r.type() != CV_32F) r.convertTo(r, CV_32F);
            r.copyTo(sampleMat.row((int)i));
        }
        Mat labelMat((int)labels.size(), 1, CV_32S);
        for (size_t i = 0; i < labels.size(); ++i) labelMat.at<int>((int)i, 0) = labels[i];

        detector.FitPCA(sampleMat, pcaDim);
        bool ok = detector.TrainSVM(sampleMat, labelMat);
        if (!ok) {
            cerr << "SVM 训练失败" << endl;
            return -1;
        }
        cout << "SVM 训练完成。保存模型...\n";
        QDir().mkpath("models");
        if (!detector.SaveModel("models/defect_detector")) {
            cerr << "模型保存失败" << endl;
            return -1;
        }
        cout << "模型已保存到 models/defect_detector_*" << endl;

        // 保存类名映射（按训练时子目录顺序）
        QString clsFile = QString::fromStdString("models/defect_detector_classes.txt");
        QFile cf(clsFile);
        if (cf.open(QIODevice::WriteOnly | QIODevice::Text)) {
            QTextStream out(&cf);
            for (const auto &cn : classNames) out << cn.c_str() << "\n";
            cf.close();
            cout << "类别映射已保存到 " << clsFile.toStdString() << "\n";
        } else {
            cerr << "警告：无法保存类别映射文件：" << clsFile.toStdString() << "\n";
        }
        return 0;
    }
    else if (mode == "eval") {
        // 评估模式
        QString testDir = QDir(root).filePath("test");
        cout << "Loading test folder: " << testDir.toStdString() << "\n";
        vector<string> imgPaths;
        vector<int> trueLabels;

        // 多类评估：使用训练时保存的类别映射文件来保证 label 对齐
        QString clsFile = "models/defect_detector_classes.txt";
        vector<string> classNames;
        QFile cf(clsFile);
        if (cf.open(QIODevice::ReadOnly | QIODevice::Text)) {
            QTextStream in(&cf);
            while (!in.atEnd()) {
                QString line = in.readLine().trimmed();
                if (!line.isEmpty()) classNames.push_back(line.toStdString());
            }
            cf.close();
        } else {
            cerr << "无法打开类别映射文件：" << clsFile.toStdString() << "，将按测试目录子文件夹顺序自动推断类别（可能与训练不一致）\n";
            // fallback: list subdirs under testDir
            QDir td(testDir);
            QStringList subs = td.entryList(QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name);
            for (const QString &s : subs) classNames.push_back(s.toStdString());
        }

        // 收集测试样本，按照 classNames 的顺序遍历目录
        for (size_t lbl = 0; lbl < classNames.size(); ++lbl) {
            QString sub = QString::fromStdString(classNames[lbl]);
            QString subdir = QDir(testDir).filePath(sub);
            QDir d(subdir);
            if (!d.exists()) continue;
            QStringList nameFilters; nameFilters << "*.png" << "*.jpg" << "*.bmp" << "*.tif";
            QFileInfoList files = d.entryInfoList(nameFilters, QDir::Files, QDir::Name);
            for (auto fi : files) { imgPaths.push_back(fi.absoluteFilePath().toStdString()); trueLabels.push_back(static_cast<int>(lbl)); }
        }

        if (imgPaths.empty()) {
            cerr << "测试集为空，请检查路径" << endl;
            return -1;
        }

        // 加载模型
        if (!detector.LoadModel("models/defect_detector")) {
            cerr << "加载模型失败，请先运行 train 模式或检查 models 路径" << endl;
            return -1;
        }

        vector<int> preds; preds.reserve(imgPaths.size());

        int numClasses = (int)classNames.size();
        vector<vector<int>> confMat(numClasses, vector<int>(numClasses, 0)); // [gt][pred]

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

        // 计算 per-class 指标
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
        // header
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
            double prec = (predTotals[i] > 0) ? double(confMat[i][i]) / double(predTotals[i]) : 0.0; // note: this is not standard; better compute per-class differently
            // standard: precision = TP / (TP+FP) => TP = confMat[i][i], FP = sum over gt!=i confMat[gt][i]
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

        // 保存预测结果 CSV（包含类名）
        savePredictionsCsv(outCsv, imgPaths, trueLabels, preds, classNames);
        cout << "预测结果已保存到 " << outCsv.toStdString() << "\n";
        return 0;
    }
    else {
        cerr << "未知模式：" << mode.toStdString() << "（支持 train 或 eval）" << endl;
        return -1;
    }

    return 0;
}
