// defect_detector_sample.cpp
// 简单训练/测试示例，演示如何使用 DefectDetection 类进行特征提取、PCA训练、SVM训练、保存与预测。
// 说明：此示例为快速演示，使用合成/简单数据构造训练集。真实工程中请用实际图像数据替换。

#include "DefectDetection.h"
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    // 创建示例目录结构和少量合成样本
    // 如果你有真实数据，请把 imagesNormal/ 和 imagesDefect/ 替换为实际文件夹

    DefectDetection detector;

    // 生成合成数据：两类各 10 张小图（纯色与带噪点）
    vector<Mat> samples;
    vector<int> labels;
    for (int i = 0; i < 10; ++i) {
        Mat imgNormal(64,64,CV_8UC3, Scalar(100 + i, 120 + i, 130 + i));
        Mat pre; detector.Preprocess(imgNormal, pre);
        Mat feat = detector.ExtractFeatures(pre);
        samples.push_back(feat);
        labels.push_back(0);
    }
    RNG rng(12345);
    for (int i = 0; i < 10; ++i) {
        Mat imgDef(64,64,CV_8UC3, Scalar(100 + i, 120 + i, 130 + i));
        // 在图像上加入一些随机亮点作为缺陷
        for (int k = 0; k < 30; ++k) {
            int x = rng.uniform(0, imgDef.cols);
            int y = rng.uniform(0, imgDef.rows);
            imgDef.at<Vec3b>(y,x) = Vec3b(0,0,255);
        }
        Mat pre; detector.Preprocess(imgDef, pre);
        Mat feat = detector.ExtractFeatures(pre);
        samples.push_back(feat);
        labels.push_back(1);
    }

    // 将 samples 转为 Mat（每行一个样本）
    Mat sampleMat(static_cast<int>(samples.size()), samples[0].cols, CV_32F);
    for (size_t i = 0; i < samples.size(); ++i) {
        Mat r = samples[i];
        if (r.type() != CV_32F) r.convertTo(r, CV_32F);
        CV_Assert(r.cols == sampleMat.cols);
        r.copyTo(sampleMat.row((int)i));
    }
    Mat labelMat(static_cast<int>(labels.size()), 1, CV_32S);
    for (size_t i = 0; i < labels.size(); ++i) labelMat.at<int>((int)i,0) = labels[i];

    // PCA 降维
    int pcaDim = min(32, sampleMat.cols);
    detector.FitPCA(sampleMat, pcaDim);

    // 训练 SVM
    bool ok = detector.TrainSVM(sampleMat, labelMat);
    if (!ok) {
        cerr << "SVM 训练失败" << endl;
        return -1;
    }
    cout << "SVM 训练完成" << endl;

    // 保存模型
    if (!detector.SaveModel("models/defect_detector"))
        cerr << "模型保存失败" << endl;
    else
        cout << "模型已保存到 models/defect_detector_*" << endl;

    // 测试：用一张合成缺陷图进行预测
    Mat testImg(64,64,CV_8UC3, Scalar(110,130,140));
    for (int k = 0; k < 20; ++k) {
        int x = rng.uniform(0, testImg.cols);
        int y = rng.uniform(0, testImg.rows);
        testImg.at<Vec3b>(y,x) = Vec3b(0,0,255);
    }
    Mat pre; detector.Preprocess(testImg, pre);
    Mat feat = detector.ExtractFeatures(pre);
    int pred = detector.Predict(feat);
    cout << "Predicted label: " << pred << endl;

    // 演示 LoadModel
    DefectDetection loaded;
    if (loaded.LoadModel("models/defect_detector")) {
        Mat feat2 = loaded.ExtractFeatures(pre);
        int pred2 = loaded.Predict(feat2);
        cout << "Loaded model prediction: " << pred2 << endl;
    } else {
        cerr << "加载模型失败（请先确保 models 文件夹存在且有读写权限）" << endl;
    }

    return 0;
}
