// TestBench.cpp
// 交互式测试程序：
// - 生成 100 张合成测试图像（其中随机 5 张为缺陷图像）
// - 将图片保存到相对路径 ../Data/test_images
// - 列出生成的图片并允许用户输入编号选择一张进行预测与可视化
// - 预测返回 0（正常）或 1（有缺陷）并在显示窗口中标注

#include "DefectDetection.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <numeric>
#include <random>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

static void EnsureDir(const fs::path& p)
{
    if (!fs::exists(p))
    {
        fs::create_directories(p);
    }
}

int main(int argc, char** argv)
{
    DefectDetection detector;

    // 1) 生成 100 张测试图像，其中 5 张含缺陷
    const int total = 100;
    const int defects = 5;
    fs::path outDir = fs::path("..") / "Data" / "test_images";
    EnsureDir(outDir);

    RNG rng((unsigned)time(nullptr));
    vector<string> fileList;

    // 随机选择 defects 个索引作为缺陷图
    vector<int> idx(total);
    iota(idx.begin(), idx.end(), 0);
    // shuffle then take first `defects`
    std::shuffle(idx.begin(), idx.end(), std::default_random_engine((unsigned)time(nullptr)));
    std::set<int> defectIdx(idx.begin(), idx.begin() + defects);

    for (int i = 0; i < total; ++i)
    {
        Mat img(128, 128, CV_8UC3);
        // 随机底色
        Vec3b base((uchar)rng.uniform(80, 180), (uchar)rng.uniform(80, 180), (uchar)rng.uniform(80, 180));
        img.setTo(Scalar(base[0], base[1], base[2]));

        bool isDefect = defectIdx.count(i) > 0;
        if (isDefect)
        {
            // 插入若干亮点或者划痕模拟缺陷
            for (int k = 0; k < 50; ++k)
            {
                int x = rng.uniform(0, img.cols);
                int y = rng.uniform(0, img.rows);
                img.at<Vec3b>(y, x) = Vec3b(0, 0, 255);
            }
            // 以及一条随机短线
            Point p1(rng.uniform(0, img.cols), rng.uniform(0, img.rows));
            Point p2 = p1 + Point(rng.uniform(-10,10), rng.uniform(-10,10));
            line(img, p1, p2, Scalar(0,255,255), 1);
        }

        // 保存
        string name = ("test_" + to_string(i) + (isDefect ? string("_def.png") : string(".png")));
        fs::path out = outDir / name;
        imwrite(out.string(), img);
        fileList.push_back(out.string());
    }

    cout << "生成完成，共 " << total << " 张图像，保存路径：" << outDir.string() << "\n";

    // 2) 交互：列出文件并允许用户选择
    for (size_t i = 0; i < fileList.size(); ++i)
    {
        cout << i << ": " << fileList[i] << "\n";
    }

    // 2.1 尝试加载已有模型，如果没有则用生成的数据自动训练一个模型
    fs::path modelDir = fs::path("../models");
    EnsureDir(modelDir);
    bool loaded = detector.LoadModel((modelDir / "defect_detector").string());
    if (!loaded)
    {
        cout << "未找到已保存模型，使用生成的样本自动训练模型..." << endl;
        // 构造训练矩阵
        vector<Mat> feats;
        vector<int> labs;
        for (size_t i = 0; i < fileList.size(); ++i)
        {
            Mat img = imread(fileList[i]);
            if (img.empty())
            {
                continue;
            }
            Mat pre;
            detector.Preprocess(img, pre);
            Mat f = detector.ExtractFeatures(pre);
            feats.push_back(f);
            // 判断文件名中是否包含 "_def" 标注缺陷
            if (fileList[i].find("_def") != string::npos)
            {
                labs.push_back(1);
            }
            else
            {
                labs.push_back(0);
            }
        }
        if (feats.empty())
        {
            cerr << "没有可用的训练样本，无法训练模型" << endl;
        }
        else
        {
            Mat sampleMat(static_cast<int>(feats.size()), feats[0].cols, CV_32F);
            for (size_t i = 0; i < feats.size(); ++i)
            {
                Mat r = feats[i];
                if (r.type() != CV_32F)
                {
                    r.convertTo(r, CV_32F);
                }
                r.copyTo(sampleMat.row((int)i));
            }
            Mat labelMat(static_cast<int>(labs.size()), 1, CV_32S);
            for (size_t i = 0; i < labs.size(); ++i)
            {
                labelMat.at<int>((int)i,0) = labs[i];
            }

            int pcaDim = min(32, sampleMat.cols);
            detector.FitPCA(sampleMat, pcaDim);
            bool ok = detector.TrainSVM(sampleMat, labelMat);
            if (!ok)
            {
                cerr << "自动训练 SVM 失败" << endl;
            }
            else
            {
                cout << "自动训练完成，保存模型..." << endl;
                if (!detector.SaveModel((modelDir / "defect_detector").string()))
                {
                    cerr << "模型保存失败" << endl;
                }
                else
                {
                    cout << "模型已保存到 " << (modelDir / "defect_detector").string() << "_*" << endl;
                }
            }
        }
    }
    else
    {
        cout << "已加载模型：" << (modelDir / "defect_detector").string() << "_*" << endl;
    }

    while (true)
    {
        cout << "输入要测试的图像编号（q 退出）：";
        string line;
        if (!getline(cin, line))
        {
            break;
        }
        if (line.empty())
        {
            continue;
        }
        if (line == "q" || line == "Q")
        {
            break;
        }

        int id = -1;
        try
        {
            id = stoi(line);
        }
        catch (...)
        {
            cout << "非法输入\n";
            continue;
        }
        if (id < 0 || id >= (int)fileList.size())
        {
            cout << "编号越界\n";
            continue;
        }

        Mat img = imread(fileList[id]);
        if (img.empty())
        {
            cout << "读取图片失败：" << fileList[id] << "\n";
            continue;
        }

        Mat pre;
        detector.Preprocess(img, pre);
        Mat feat = detector.ExtractFeatures(pre);
        int pred = detector.Predict(feat);

        // 可视化：在图像上标注预测结果
        Mat vis;
        img.copyTo(vis);
        string txt = (pred == 0) ? "Normal (0)" : "Defect (1)";
        Scalar color = (pred == 0) ? Scalar(0,255,0) : Scalar(0,0,255);
        putText(vis, txt, Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
        imshow("Test Image", vis);
        cout << "预测结果：" << pred << " (0=正常, 1=缺陷)" << endl;

        // 等待用户按键后继续
        int k = waitKey(0);
        destroyWindow("Test Image");
        if (k == 'q' || k == 'Q')
        {
            break;
        }
    }

    cout << "退出测试台。" << endl;
    return 0;
}
