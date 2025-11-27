#include "YOLOExample.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QFileDialog>
#include <QPixmap>
#include <QDebug>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonArray>
#include <QJsonObject>

using namespace cv;

// YOLOExample构造函数
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

// YOLOExample析构函数
YOLOExample::~YOLOExample()
{
}

// 设置UI界面
void YOLOExample::SetupUI()
{
    setWindowTitle("深度学习 - YOLO");
    setMinimumSize(800, 600);

    QVBoxLayout* mainLayout = new QVBoxLayout(this);

    // 模型和标签文件选择布局
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
    
    // 标签文件加载
    modelLayout->addWidget(new QLabel("标签文件:"));
    labelPathEdit_ = new QLineEdit();
    labelPathEdit_->setPlaceholderText("Data/Labels/class_labels.txt");
    modelLayout->addWidget(labelPathEdit_);
    QPushButton* browseLabelBtn = new QPushButton("加载标签");
    connect(browseLabelBtn, &QPushButton::clicked, this, &YOLOExample::LoadLabels);
    modelLayout->addWidget(browseLabelBtn);
    mainLayout->addLayout(modelLayout);

    // 图像选择和检测按钮布局
    QHBoxLayout* imgLayout = new QHBoxLayout();
    QPushButton* selImg = new QPushButton("选择图像");
    connect(selImg, &QPushButton::clicked, this, &YOLOExample::SelectImage);
    imgLayout->addWidget(selImg);
    detectBtn_ = new QPushButton("开始检测");
    connect(detectBtn_, &QPushButton::clicked, this, &YOLOExample::RunDetect);
    imgLayout->addWidget(detectBtn_);
    imgLayout->addStretch();
    mainLayout->addLayout(imgLayout);

    // 图像显示区域
    imageLabel_ = new QLabel("图像显示区域");
    imageLabel_->setMinimumSize(640, 480);
    imageLabel_->setAlignment(Qt::AlignCenter);
    imageLabel_->setStyleSheet("QLabel { border: 1px solid #ccc; background-color: white; }");
    mainLayout->addWidget(imageLabel_);

    // 状态标签
    statusLabel_ = new QLabel("状态: 未加载模型");
    mainLayout->addWidget(statusLabel_);
}

// 连接信号和槽
void YOLOExample::ConnectSignals()
{
    connect(yoloProcessor_, &YOLOProcessorORT::processingComplete, this, &YOLOExample::OnProcessingComplete);
    connect(yoloProcessor_, &YOLOProcessorORT::errorOccurred, this, &YOLOExample::OnDLError);
}

// 加载标签文件
void YOLOExample::LoadLabels()
{
    QString file = QFileDialog::getOpenFileName(this, "选择标签文件", ".", "Text Files (*.txt);;All Files (*)");
    if (file.isEmpty()) return;
    
    QFile f(file);
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
        statusLabel_->setText("状态: 无法打开标签文件");
        return;
    }
    
    QStringList labels;
    while (!f.atEnd()) {
        QByteArray line = f.readLine();
        QString s = QString::fromUtf8(line).trimmed();
        if (!s.isEmpty()) labels.append(s);
    }
    f.close();
    
    currentClassLabels_ = labels;
    // 将标签传递给处理器
    vector<string> cls;
    cls.reserve(labels.size());
    for (const QString &qs : labels) cls.push_back(qs.toStdString());
    yoloProcessor_->SetClassLabels(cls);
    
    labelPathEdit_->setText(file);
    statusLabel_->setText(QString("状态: 已加载 %1 类").arg(labels.size()));
}

// 浏览模型文件
void YOLOExample::BrowseModel()
{
    QString file = QFileDialog::getOpenFileName(this, "选择 ONNX 模型", ".", "ONNX Files (*.onnx);;All Files (*)");
    if (!file.isEmpty()) modelPathEdit_->setText(file);
}

// 加载模型
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

// 选择图像
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

// 运行目标检测
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

    // 调用DetectObjects方法，使用引用参数获取结果
    std::vector<DetectionResult> results;
    bool success = yoloProcessor_->DetectObjects(img, results);
    if (!success) {
        qDebug() << "目标检测失败";
        return;
    }

    // 在图像上绘制检测结果
    Mat out = img.clone();
    yoloProcessor_->DrawDetectionResults(out, results);

    // 保存检测结果图片与 JSON（与 test-YOLO.py 行为一致）
    QFileInfo info(currentImagePath_);
    QString baseName = info.completeBaseName();
    QString dir = info.path();
    QString saveImagePath = dir + "/" + baseName + "_detections.jpg";
    QString saveJsonPath = dir + "/" + baseName + "_detections.json";
    
    // 保存图像
    imwrite(saveImagePath.toStdString(), out);

    // 构建JSON结果
    QJsonArray detArray;
    for (const auto &d : results) {
        QJsonObject obj;
        // 包含数字类别ID和备用类别名称（如果未加载标签）
        obj["class_id"] = d.classId;
        QString clsName = d.className.empty() ? QString("class_%1").arg(d.classId) : QString::fromStdString(d.className);
        obj["class"] = clsName;
        obj["confidence"] = d.confidence;
        
        // 边界框坐标
        QJsonArray bbox;
        int x1 = d.boundingBox.x;
        int y1 = d.boundingBox.y;
        int x2 = d.boundingBox.x + d.boundingBox.width;
        int y2 = d.boundingBox.y + d.boundingBox.height;
        bbox.append(x1);
        bbox.append(y1);
        bbox.append(x2);
        bbox.append(y2);
        obj["bbox"] = bbox;
        
        detArray.append(obj);
    }
    
    // 保存JSON结果
    QJsonDocument doc(detArray);
    QFile jsonFile(saveJsonPath);
    if (jsonFile.open(QIODevice::WriteOnly)) {
        jsonFile.write(doc.toJson(QJsonDocument::Compact));
        jsonFile.close();
    }

    // 显示检测结果
    QPixmap pm = QPixmap::fromImage(QImage(out.data, out.cols, out.rows, out.step, QImage::Format_BGR888).rgbSwapped());
    if (!pm.isNull()) {
        imageLabel_->setPixmap(pm.scaled(imageLabel_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
    
    statusLabel_->setText(QString("检测完成: %1 目标").arg(results.size()));
}

// 处理完成信号槽
void YOLOExample::OnProcessingComplete(const cv::Mat& resultImage)
{
    Mat out = resultImage.clone();
    QPixmap pm = QPixmap::fromImage(QImage(out.data, out.cols, out.rows, out.step, QImage::Format_BGR888).rgbSwapped());
    if (!pm.isNull()) imageLabel_->setPixmap(pm.scaled(imageLabel_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

// 错误处理信号槽
void YOLOExample::OnDLError(const QString& error)
{
    statusLabel_->setText(QString("错误: %1").arg(error));
}