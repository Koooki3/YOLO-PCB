#ifndef YOLOEXAMPLE_H
#define YOLOEXAMPLE_H

#include <QWidget>
#include <QLineEdit>
#include <QLabel>
#include <QProgressBar>
#include "YOLOProcessorORT.h"

QT_BEGIN_NAMESPACE
class QPushButton;
class QVBoxLayout;
class QHBoxLayout;
class QComboBox;
QT_END_NAMESPACE

class YOLOExample : public QWidget
{
    Q_OBJECT

public:
    explicit YOLOExample(QWidget *parent = nullptr);
    ~YOLOExample();

private slots:
    void BrowseModel();
    void LoadModel();
    void SelectImage();
    void RunDetect();
    void OnProcessingComplete(const cv::Mat& resultImage);
    void OnDLError(const QString& error);

private:
    void SetupUI();
    void ConnectSignals();

    YOLOProcessorORT* yoloProcessor_;
    QLineEdit* modelPathEdit_;
    QLabel* statusLabel_;
    QLabel* imageLabel_;
    QPushButton* detectBtn_;
    QString currentImagePath_;
};

#endif // YOLOEXAMPLE_H
