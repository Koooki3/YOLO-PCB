#include "mainwindow.h"
#include "featureDetect.h"
#include "configmanager.h"
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

    // 初始化配置管理器
    if (!ConfigManager::instance()->init())
    {
        qCritical() << "Failed to initialize config manager!";
        return -1;
    }
    qDebug() << "Config manager initialized successfully. Current config:" << ConfigManager::instance()->getCurrentConfigName();

    // 优先从资源加载样式表（打包情况），若资源不可用则回退到可执行目录下的 style.qss 文件
    QString resourcePath = ":/style/style.qss";
    bool loaded = false;
    QFile qrcStyle(resourcePath);
    if (qrcStyle.open(QFile::ReadOnly | QFile::Text))
    {
        a.setStyleSheet(qrcStyle.readAll());
        qrcStyle.close();
        qDebug() << "Loaded style from resource:" << resourcePath;
        loaded = true;
    }
    if (!loaded)
    {
        QString stylePath = QDir(QCoreApplication::applicationDirPath()).filePath("style.qss");
        QFile styleFile(stylePath);
        if (styleFile.open(QFile::ReadOnly | QFile::Text))
        {
            a.setStyleSheet(styleFile.readAll());
            styleFile.close();
            qDebug() << "Loaded style from file:" << stylePath;
            loaded = true;
        }
        else
        {
            qDebug() << "No custom stylesheet found; using default style.";
        }
    }

    // 启动主窗口
    MainWindow w;
    w.setWindowFlags(w.windowFlags() &~ Qt::WindowMaximizeButtonHint); //ch:禁止最大化 | en:prohibit maximization
    w.show();
    return a.exec();
}
