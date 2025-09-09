#include "mainwindow.h"
#include "featureDetect.h"
#include <QApplication>
#include <QTranslator>
#include <QLocale>
#include <QDebug>
#include <QFile>

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

//    // 运行特征识别测试
//    // 使用工作目录下的测试图像
//    QString testImage1 = "../Img/test3.jpg";
//    QString testImage2 = "../Img/test4.jpg";
    
//    if (QFile::exists(testImage1) && QFile::exists(testImage2)) {
//        qDebug() << "Running feature detection test...";
//        FeatureDetector::testFeatureDetection(testImage1, testImage2);
//    } else {
//        qDebug() << "Test images not found at:" << testImage1 << "and" << testImage2;
//    }

    // 启动主窗口
    MainWindow w;
    w.setWindowFlags(w.windowFlags() &~ Qt::WindowMaximizeButtonHint); //ch:禁止最大化 | en:prohibit maximization
    w.show();
    return a.exec();
}

//深度学习二分类示例

//#include "DLExample.h"
//#include <QApplication>

//int main(int argc, char *argv[])
//{
//    QApplication app(argc, argv);

//    DLExample window;
//    window.show();

//    return app.exec();
//}
