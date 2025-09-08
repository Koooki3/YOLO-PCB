#include "mainwindow.h"
#include "DLProcessor.h"
#include "DataProcessor.h"
#include <QApplication>
#include <QTranslator>
#include <QLocale>
#include <QVector>
#include <QPointF>
#include <QDir>
#include <QCoreApplication>
#include <QDebug>
#include "eigen3/Eigen/Dense"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>

using namespace std;
using namespace Eigen;

// 特征识别测试函数
void testFeatureDetection(const QString& imagePath1, const QString& imagePath2)
{
    // 创建数据处理器和深度学习处理器实例
    DataProcessor dataProcessor;
    DLProcessor dlProcessor;

    // 读取测试图像
    cv::Mat image1 = cv::imread(imagePath1.toStdString());
    cv::Mat image2 = cv::imread(imagePath2.toStdString());

    if (image1.empty() || image2.empty()) {
        qDebug() << "Error: Could not read the images";
        return;
    }

    // 1. 图像预处理
    cv::Mat standardized1 = dataProcessor.standardizeImage(image1);
    cv::Mat standardized2 = dataProcessor.standardizeImage(image2);

    // 2. 提取特征
    cv::Mat descriptors1, descriptors2;
    std::vector<cv::KeyPoint> keypoints1 = dataProcessor.detectKeypoints(standardized1, descriptors1);
    std::vector<cv::KeyPoint> keypoints2 = dataProcessor.detectKeypoints(standardized2, descriptors2);

    // 3. 特征匹配
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher->knnMatch(descriptors1, descriptors2, knnMatches, 2);

    // 4. 应用比率测试进行筛选
    std::vector<cv::DMatch> goodMatches;
    const float ratioThresh = 0.7f;
    for (const auto& match : knnMatches) {
        if (match[0].distance < ratioThresh * match[1].distance) {
            goodMatches.push_back(match[0]);
        }
    }

    // 5. 绘制匹配结果
    cv::Mat imgMatches;
    cv::drawMatches(image1, keypoints1, image2, keypoints2, goodMatches, imgMatches,
                   cv::Scalar::all(-1), cv::Scalar::all(-1),
                   std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // 6. 保存结果
    cv::imwrite("../feature_matches.jpg", imgMatches);
    
    // 7. 输出匹配信息
    qDebug() << "Total keypoints in image1:" << keypoints1.size();
    qDebug() << "Total keypoints in image2:" << keypoints2.size();
    qDebug() << "Number of good matches:" << goodMatches.size();

    // 8. 显示结果图像
    cv::namedWindow("Feature Matches", cv::WINDOW_NORMAL);
    cv::imshow("Feature Matches", imgMatches);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main(int argc, char *argv[])
{
    qputenv("QT_QPA_PLATFORM", "xcb");
    QApplication a(argc, argv);
    QTranslator translator;
    QLocale locale = QLocale::system();

    if( locale.language() == QLocale::Chinese )
    {
        //ch:中文语言环境加载默认设计界面 | en:The Chinese language environment load the default design
    }
    else
    {
        //ch:其他语言环境加载英文界面 | en:Other language environments load the English design
        translator.load(QString("VisualRobot_zh_EN.qm")); //ch:选择翻译文件 | en:Choose the translation file
        a.installTranslator(&translator);
    }

    // 运行特征识别测试
    // 使用工作目录下的测试图像
    QString testImage1 = QDir::currentPath() + "/Img/capture.jpg";
    QString testImage2 = QDir::currentPath() + "/Img/circle_detected.jpg";
    
    if (QFile::exists(testImage1) && QFile::exists(testImage2)) {
        qDebug() << "Running feature detection test...";
        testFeatureDetection(testImage1, testImage2);
    } else {
        qDebug() << "Test images not found at:" << testImage1 << "and" << testImage2;
    }

    // 启动主窗口
    MainWindow w;
    w.setWindowFlags(w.windowFlags() &~ Qt::WindowMaximizeButtonHint); //ch:禁止最大化 | en:prohibit maximization
    w.show();
    return a.exec();    
}
