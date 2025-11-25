#include "YOLOExample.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QFileDialog>
#include <QPixmap>
#include <QDebug>

using namespace cv;

YOLOExample::YOLOExample(QWidget *parent)
    : QWidget(parent)
    , yoloProcessor_(new YOLOProcessorORT(this))
    , modelPathEdit_(nullptr)
    , statusLabel_(nullptr)
    , imageLabel_(nullptr)
    , detectBtn_(nullptr)
{
    SetupUI();
    ConnectSignals();
}

YOLOExample::~YOLOExample()
{
}

void YOLOExample::SetupUI()
{
    setWindowTitle("深度学习 - YOLO");
    setMinimumSize(800, 600);

    QVBoxLayout* mainLayout = new QVBoxLayout(this);

    QHBoxLayout* modelLayout = new QHBoxLayout();
    modelLayout->addWidget(new QLabel("YOLO ONNX 模型:"));
    modelPathEdit_ = new QLineEdit();
    modelPathEdit_->setPlaceholderText("选择 .onnx 模型文件");
    modelLayout->addWidget(modelPathEdit_);
    QPushButton* browseBtn = new QPushButton("浏览");
    connect(browseBtn, &QPushButton::clicked, this, &YOLOExample::BrowseModel);
    modelLayout->addWidget(browseBtn);
    QPushButton* loadBtn = new QPushButton("加载模型");
    connect(loadBtn, &QPushButton::clicked, this, &YOLOExample::LoadModel);
    modelLayout->addWidget(loadBtn);
    mainLayout->addLayout(modelLayout);

    QHBoxLayout* imgLayout = new QHBoxLayout();
    QPushButton* selImg = new QPushButton("选择图像");
    connect(selImg, &QPushButton::clicked, this, &YOLOExample::SelectImage);
    imgLayout->addWidget(selImg);
    detectBtn_ = new QPushButton("开始检测");
    connect(detectBtn_, &QPushButton::clicked, this, &YOLOExample::RunDetect);
    imgLayout->addWidget(detectBtn_);
    imgLayout->addStretch();
    mainLayout->addLayout(imgLayout);

    imageLabel_ = new QLabel("图像显示区域");
    imageLabel_->setMinimumSize(640, 480);
    imageLabel_->setAlignment(Qt::AlignCenter);
    imageLabel_->setStyleSheet("QLabel { border: 1px solid #ccc; background-color: white; }");
    mainLayout->addWidget(imageLabel_);

    statusLabel_ = new QLabel("状态: 未加载模型");
    mainLayout->addWidget(statusLabel_);
}

void YOLOExample::ConnectSignals()
{
    connect(yoloProcessor_, &YOLOProcessorORT::processingComplete, this, &YOLOExample::OnProcessingComplete);
    connect(yoloProcessor_, &YOLOProcessorORT::errorOccurred, this, &YOLOExample::OnDLError);
}

void YOLOExample::BrowseModel()
{
    QString file = QFileDialog::getOpenFileName(this, "选择 ONNX 模型", ".", "ONNX Files (*.onnx);;All Files (*)");
    if (!file.isEmpty()) modelPathEdit_->setText(file);
}

void YOLOExample::LoadModel()
{
    QString path = modelPathEdit_->text();
    if (path.isEmpty()) {
        statusLabel_->setText("状态: 请先选择模型文件");
        return;
    }
    bool ok = yoloProcessor_->InitModel(path.toStdString(), false);
    if (ok) {
        statusLabel_->setText("状态: 模型已加载");
    } else {
        statusLabel_->setText("状态: 模型加载失败");
    }
}

void YOLOExample::SelectImage()
{
    QString file = QFileDialog::getOpenFileName(this, "选择图像", ".", "Images (*.png *.jpg *.jpeg *.bmp)");
    if (!file.isEmpty()) {
        currentImagePath_ = file;
        QPixmap pm(file);
        if (!pm.isNull()) {
            QPixmap scaled = pm.scaled(imageLabel_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
            imageLabel_->setPixmap(scaled);
        }
    }
}

void YOLOExample::RunDetect()
{
    if (!yoloProcessor_->IsModelLoaded()) {
        statusLabel_->setText("状态: 请先加载模型");
        return;
    }
    if (currentImagePath_.isEmpty()) {
        statusLabel_->setText("状态: 请先选择图像");
        return;
    }
    Mat img = imread(currentImagePath_.toStdString());
    if (img.empty()) {
        statusLabel_->setText("状态: 无法读取图像");
        return;
    }

    vector<DetectionResult> results;
    bool ok = yoloProcessor_->DetectObjects(img, results);
    if (!ok) {
        statusLabel_->setText("状态: 推理失败");
        return;
    }

    // 在图像上绘制
    Mat out = img.clone();
    yoloProcessor_->DrawDetectionResults(out, results);

    // 显示
    QPixmap pm = QPixmap::fromImage(QImage(out.data, out.cols, out.rows, out.step, QImage::Format_BGR888).rgbSwapped());
    if (!pm.isNull()) {
        imageLabel_->setPixmap(pm.scaled(imageLabel_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
    statusLabel_->setText(QString("检测完成: %1 目标").arg(results.size()));
}

void YOLOExample::OnProcessingComplete(const cv::Mat& resultImage)
{
    Mat out = resultImage.clone();
    QPixmap pm = QPixmap::fromImage(QImage(out.data, out.cols, out.rows, out.step, QImage::Format_BGR888).rgbSwapped());
    if (!pm.isNull()) imageLabel_->setPixmap(pm.scaled(imageLabel_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

void YOLOExample::OnDLError(const QString& error)
{
    statusLabel_->setText(QString("错误: %1").arg(error));
}
