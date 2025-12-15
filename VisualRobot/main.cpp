//#include "mainwindow.h"
//#include "featureDetect.h"
//#include "configmanager.h"
//#include <QApplication>
//#include <QTranslator>
//#include <QLocale>
//#include <QDebug>
//#include <QFile>
//#include "rknn_api.h"

//int main(int argc, char *argv[])
//{
//    qputenv("QT_QPA_PLATFORM", "xcb");
//    QApplication a(argc, argv);
//    QTranslator translator;
//    QLocale locale = QLocale::system();

//    if( locale.language() == QLocale::Chinese )
//    {
//        //ch:中文语言环境加载默认设计界面 | en:The Chinese language environment load the default design
//    }
//    else
//    {
//        //ch:其他语言环境加载英文界面 | en:Other language environments load the English design
//        translator.load(QString("VisualRobot_zh_EN.qm")); //ch:选择翻译文件 | en:Choose the translation file
//        a.installTranslator(&translator);
//    }

//    // 初始化配置管理器
//    if (!ConfigManager::instance()->init())
//    {
//        qCritical() << "Failed to initialize config manager!";
//        return -1;
//    }
//    qDebug() << "Config manager initialized successfully. Current config:" << ConfigManager::instance()->getCurrentConfigName();

//    // 优先从资源加载样式表（打包情况），若资源不可用则回退到可执行目录下的 style.qss 文件
//    QString resourcePath = ":/style/style.qss";
//    bool loaded = false;
//    QFile qrcStyle(resourcePath);
//    if (qrcStyle.open(QFile::ReadOnly | QFile::Text))
//    {
//        a.setStyleSheet(qrcStyle.readAll());
//        qrcStyle.close();
//        qDebug() << "Loaded style from resource:" << resourcePath;
//        loaded = true;
//    }
//    if (!loaded)
//    {
//        QString stylePath = QDir(QCoreApplication::applicationDirPath()).filePath("style.qss");
//        QFile styleFile(stylePath);
//        if (styleFile.open(QFile::ReadOnly | QFile::Text))
//        {
//            a.setStyleSheet(styleFile.readAll());
//            styleFile.close();
//            qDebug() << "Loaded style from file:" << stylePath;
//            loaded = true;
//        }
//        else
//        {
//            qDebug() << "No custom stylesheet found; using default style.";
//        }
//    }

//    // 启动主窗口
//    MainWindow w;
//    w.setWindowFlags(w.windowFlags() &~ Qt::WindowMaximizeButtonHint); //ch:禁止最大化 | en:prohibit maximization
//    w.show();
//    return a.exec();
//}

// main.cpp 使用示例
#include <QCoreApplication>
#include "YOLOProcessorRKNN.h"
#include <iostream>

int main(int argc, char* argv[])
{
    QCoreApplication a(argc, argv);

//    if (argc < 4) {
//        std::cout << "使用方法: " << argv[0] << " <rknn模型路径> <测试图像路径> <标签文件路径> [输出图像路径]" << std::endl;
//        std::cout << "示例: " << argv[0] << " yolov11n.rknn test.jpg coco_labels.txt result.jpg" << std::endl;
//        return -1;
//    }

    std::string modelPath = "/home/orangepi/Desktop/VisualRobot_Local/models/arcuchi.rknn";
    std::string imagePath = "/home/orangepi/Desktop/VisualRobot_Local/Img/test_arcuchi/0355.jpg";
    std::string labelsPath = "/home/orangepi/Desktop/VisualRobot_Local/labels/class_labels.txt";
    std::string outputPath = (argc > 4) ? argv[4] : "detection_result.jpg";

    // 创建处理器
    YOLOProcessorRKNN processor;

    // 连接信号
    QObject::connect(&processor, &YOLOProcessorRKNN::processingComplete,
                     [](const cv::Mat& resultImage) {
                         std::cout << "图像处理完成!" << std::endl;
                     });

    QObject::connect(&processor, &YOLOProcessorRKNN::errorOccurred,
                     [](const QString& error) {
                         std::cerr << "错误: " << error.toStdString() << std::endl;
                     });

    QObject::connect(&processor, &YOLOProcessorRKNN::detectionResults,
                     [](const std::vector<DetectionResult>& results) {
                         std::cout << "检测到 " << results.size() << " 个目标:" << std::endl;
                         for (const auto& result : results) {
                             std::cout << "  " << result.className
                                       << " (置信度: " << result.confidence
                                       << ")" << std::endl;
                         }
                     });

    // 初始化模型
    std::cout << "正在初始化RKNN模型..." << std::endl;
    if (!processor.InitModel(modelPath, labelsPath)) {
        std::cerr << "模型初始化失败!" << std::endl;
        return -1;
    }

    // 设置参数
    processor.SetThresholds(0.25f, 0.45f);  // 置信度阈值0.25，NMS阈值0.45
    processor.SetDebugOutput(true);         // 启用调试输出

    // 处理图像
    std::cout << "正在处理图像: " << imagePath << std::endl;
    if (processor.ProcessImage(imagePath, outputPath)) {
        std::cout << "处理成功! 结果保存在: " << outputPath << std::endl;

        // 获取性能统计
        RKNNTimingStats stats = processor.GetTimingStats();
        std::cout << "\n性能统计:" << std::endl;
        std::cout << "  预处理时间: " << stats.preprocessTime << " ms" << std::endl;
        std::cout << "  推理时间: " << stats.inferenceTime << " ms" << std::endl;
        std::cout << "  后处理时间: " << stats.postprocessTime << " ms" << std::endl;
        std::cout << "  总时间: " << stats.totalTime << " ms" << std::endl;
        std::cout << "  FPS: " << stats.fps << std::endl;
    } else {
        std::cerr << "处理失败!" << std::endl;
        return -1;
    }

    return 0;
}
