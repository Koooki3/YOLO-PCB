#include "DLExample.h"
#include <QDebug>
#include <QFileDialog>
#include <QMessageBox>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QLineEdit>
#include <QProgressBar>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

DLExample::DLExample(QWidget *parent)
    : QWidget(parent)
    , dlProcessor_(new DLProcessor(this))
    , isModelLoaded_(false)
{
    setupUI();
    connectSignals();
}

DLExample::~DLExample()
{

}

void DLExample::setupUI()
{
    setWindowTitle("深度学习二分类示例");
    setMinimumSize(800, 600);
    
    QVBoxLayout* mainLayout = new QVBoxLayout(this);
    
    // 模型加载区域
    QHBoxLayout* modelLayout = new QHBoxLayout();
    modelLayout->addWidget(new QLabel("模型路径:"));
    modelPathEdit_ = new QLineEdit();
    modelPathEdit_->setPlaceholderText("选择ONNX/PB/Caffe模型文件");
    modelLayout->addWidget(modelPathEdit_);
    
    QPushButton* browseModelBtn = new QPushButton("浏览");
    connect(browseModelBtn, &QPushButton::clicked, this, &DLExample::browseModel);
    modelLayout->addWidget(browseModelBtn);
    
    QPushButton* loadModelBtn = new QPushButton("加载模型");
    connect(loadModelBtn, &QPushButton::clicked, this, &DLExample::loadModel);
    modelLayout->addWidget(loadModelBtn);
    
    mainLayout->addLayout(modelLayout);
    
    // 标签文件区域
    QHBoxLayout* labelLayout = new QHBoxLayout();
    labelLayout->addWidget(new QLabel("标签文件:"));
    labelPathEdit_ = new QLineEdit();
    labelPathEdit_->setText("Data/Labels/class_labels.txt");
    labelLayout->addWidget(labelPathEdit_);
    
    QPushButton* browseLabelBtn = new QPushButton("浏览");
    connect(browseLabelBtn, &QPushButton::clicked, this, &DLExample::browseLabels);
    labelLayout->addWidget(browseLabelBtn);
    
    mainLayout->addLayout(labelLayout);
    
    // 参数设置区域
    QHBoxLayout* paramLayout = new QHBoxLayout();
    paramLayout->addWidget(new QLabel("置信度阈值:"));
    confidenceEdit_ = new QLineEdit("0.5");
    confidenceEdit_->setMaximumWidth(80);
    paramLayout->addWidget(confidenceEdit_);
    
    paramLayout->addWidget(new QLabel("输入尺寸:"));
    inputSizeEdit_ = new QLineEdit("224");
    inputSizeEdit_->setMaximumWidth(80);
    paramLayout->addWidget(inputSizeEdit_);
    
    QPushButton* setParamsBtn = new QPushButton("设置参数");
    connect(setParamsBtn, &QPushButton::clicked, this, &DLExample::setParameters);
    paramLayout->addWidget(setParamsBtn);
    
    paramLayout->addStretch();
    mainLayout->addLayout(paramLayout);
    
    // 图像处理区域
    QHBoxLayout* imageLayout = new QHBoxLayout();
    
    QPushButton* selectImageBtn = new QPushButton("选择图像");
    connect(selectImageBtn, &QPushButton::clicked, this, &DLExample::selectImage);
    imageLayout->addWidget(selectImageBtn);
    
    QPushButton* classifyBtn = new QPushButton("开始分类");
    connect(classifyBtn, &QPushButton::clicked, this, &DLExample::classifyImage);
    imageLayout->addWidget(classifyBtn);
    
    QPushButton* batchBtn = new QPushButton("批量分类");
    connect(batchBtn, &QPushButton::clicked, this, &DLExample::batchClassify);
    imageLayout->addWidget(batchBtn);
    
    imageLayout->addStretch();
    mainLayout->addLayout(imageLayout);
    
    // 进度条
    progressBar_ = new QProgressBar();
    progressBar_->setVisible(false);
    mainLayout->addWidget(progressBar_);
    
    // 结果显示区域
    resultLabel_ = new QLabel("等待分类结果...");
    resultLabel_->setStyleSheet("QLabel { background-color: #f0f0f0; padding: 10px; border: 1px solid #ccc; }");
    resultLabel_->setMinimumHeight(100);
    resultLabel_->setWordWrap(true);
    mainLayout->addWidget(resultLabel_);
    
    // 图像显示区域
    imageLabel_ = new QLabel();
    imageLabel_->setMinimumSize(400, 300);
    imageLabel_->setStyleSheet("QLabel { border: 1px solid #ccc; background-color: white; }");
    imageLabel_->setAlignment(Qt::AlignCenter);
    imageLabel_->setText("图像显示区域");
    mainLayout->addWidget(imageLabel_);
    
    // 状态标签
    statusLabel_ = new QLabel("就绪");
    statusLabel_->setStyleSheet("QLabel { color: blue; }");
    mainLayout->addWidget(statusLabel_);
}

void DLExample::connectSignals()
{
    // 连接DLProcessor信号
    connect(dlProcessor_, &DLProcessor::classificationComplete,
            this, &DLExample::onClassificationComplete);
    
    connect(dlProcessor_, &DLProcessor::batchProcessingComplete,
            this, &DLExample::onBatchProcessingComplete);
    
    connect(dlProcessor_, &DLProcessor::errorOccurred,
            this, &DLExample::onDLError);
}

void DLExample::browseModel()
{
    QString fileName = QFileDialog::getOpenFileName(this,
        "选择深度学习模型文件",
        "",
        "模型文件 (*.onnx *.pb *.caffemodel *.weights);;所有文件 (*.*)");
    
    if (!fileName.isEmpty()) 
    {
        modelPathEdit_->setText(fileName);
    }
}

void DLExample::browseLabels()
{
    QString fileName = QFileDialog::getOpenFileName(this,
        "选择类别标签文件",
        "",
        "文本文件 (*.txt);;所有文件 (*.*)");
    
    if (!fileName.isEmpty()) 
    {
        labelPathEdit_->setText(fileName);
        // 立即加载标签文件，无论模型是否已加载
        loadLabels();
    }
}

void DLExample::loadLabels()
{
    QString labelPath = labelPathEdit_->text().trimmed();
    if (labelPath.isEmpty()) 
    {
        return;
    }
    
    if (!QFile::exists(labelPath)) 
    {
        QMessageBox::warning(this, "警告", "标签文件不存在！");
        return;
    }
    
    // 加载标签文件
    if (dlProcessor_->loadClassLabels(labelPath.toStdString())) 
    {
        statusLabel_->setText("标签文件加载成功");
        statusLabel_->setStyleSheet("QLabel { color: green; }");
        qDebug() << "Labels loaded successfully from:" << labelPath;
    } 
    else 
    {
        statusLabel_->setText("标签文件加载失败");
        statusLabel_->setStyleSheet("QLabel { color: red; }");
        qDebug() << "Failed to load labels from:" << labelPath;
    }
}

void DLExample::loadModel()
{
    QString modelPath = modelPathEdit_->text().trimmed();
    if (modelPath.isEmpty()) 
    {
        QMessageBox::warning(this, "警告", "请先选择模型文件！");
        return;
    }
    
    statusLabel_->setText("正在加载模型...");
    statusLabel_->setStyleSheet("QLabel { color: orange; }");
    
    // 检查是否需要配置文件
    QString configPath = "";
    if (modelPath.endsWith(".caffemodel")) 
    {
        QString prototxt = modelPath;
        prototxt.replace(".caffemodel", ".prototxt");
        if (QFile::exists(prototxt)) 
        {
            configPath = prototxt;
        }
    } 
    else if (modelPath.endsWith(".weights")) 
    {
        QString cfg = modelPath;
        cfg.replace(".weights", ".cfg");
        if (QFile::exists(cfg)) 
        {
            configPath = cfg;
        }
    }
    
    // 加载模型
    bool success = dlProcessor_->initModel(modelPath.toStdString(), configPath.toStdString());
    
    if (success) 
    {
        isModelLoaded_ = true;
        statusLabel_->setText("模型加载成功");
        statusLabel_->setStyleSheet("QLabel { color: green; }");
        
        // 加载标签文件
        QString labelPath = labelPathEdit_->text().trimmed();
        if (!labelPath.isEmpty() && QFile::exists(labelPath)) 
        {
            dlProcessor_->loadClassLabels(labelPath.toStdString());
        }
        
        resultLabel_->setText("模型已加载，可以开始分类");
    } 
    else 
    {
        isModelLoaded_ = false;
        statusLabel_->setText("模型加载失败");
        statusLabel_->setStyleSheet("QLabel { color: red; }");
    }
}

void DLExample::setParameters()
{
    if (!isModelLoaded_) 
    {
        QMessageBox::warning(this, "警告", "请先加载模型！");
        return;
    }
    
    bool ok;
    float confidence = confidenceEdit_->text().toFloat(&ok);
    if (!ok || confidence < 0.0f || confidence > 1.0f) 
    {
        QMessageBox::warning(this, "警告", "置信度阈值必须在0.0-1.0之间！");
        return;
    }
    
    int inputSize = inputSizeEdit_->text().toInt(&ok);
    if (!ok || inputSize < 32 || inputSize > 1024) 
    {
        QMessageBox::warning(this, "警告", "输入尺寸必须在32-1024之间！");
        return;
    }
    
    dlProcessor_->setModelParams(confidence, 0.4f);
    dlProcessor_->setInputSize(Size(inputSize, inputSize));
    
    statusLabel_->setText(QString("参数已更新: 置信度=%1, 输入尺寸=%2x%2")
                         .arg(confidence)
                         .arg(inputSize));
    statusLabel_->setStyleSheet("QLabel { color: blue; }");
}

void DLExample::selectImage()
{
    QString fileName = QFileDialog::getOpenFileName(this,
        "选择要分类的图像",
        "",
        "图像文件 (*.jpg *.jpeg *.png *.bmp *.tiff);;所有文件 (*.*)");
    
    if (!fileName.isEmpty()) 
    {
        currentImagePath_ = fileName;
        
        // 显示图像
        QPixmap pixmap(fileName);
        if (!pixmap.isNull()) 
        {
            QPixmap scaledPixmap = pixmap.scaled(imageLabel_->size(), 
                                               Qt::KeepAspectRatio, 
                                               Qt::SmoothTransformation);
            imageLabel_->setPixmap(scaledPixmap);
            
            statusLabel_->setText("图像已加载: " + QFileInfo(fileName).fileName());
            statusLabel_->setStyleSheet("QLabel { color: blue; }");
        }
    }
}

void DLExample::classifyImage()
{
    if (!isModelLoaded_) 
    {
        QMessageBox::warning(this, "警告", "请先加载模型！");
        return;
    }
    
    if (currentImagePath_.isEmpty()) 
    {
        QMessageBox::warning(this, "警告", "请先选择图像！");
        return;
    }
    
    // 加载图像
    Mat image = imread(currentImagePath_.toStdString());
    if (image.empty()) 
    {
        QMessageBox::warning(this, "错误", "无法加载图像文件！");
        return;
    }
    
    statusLabel_->setText("正在分类...");
    statusLabel_->setStyleSheet("QLabel { color: orange; }");
    
    // 执行分类
    ClassificationResult result;
    if (dlProcessor_->classifyImage(image, result)) 
    {
        // 结果会通过信号槽处理
    }
     else 
    {
        statusLabel_->setText("分类失败");
        statusLabel_->setStyleSheet("QLabel { color: red; }");
    }
}

void DLExample::batchClassify()
{
    if (!isModelLoaded_) 
    {
        QMessageBox::warning(this, "警告", "请先加载模型！");
        return;
    }
    
    QStringList fileNames = QFileDialog::getOpenFileNames(this,
        "选择要批量分类的图像",
        "",
        "图像文件 (*.jpg *.jpeg *.png *.bmp *.tiff)");
    
    if (fileNames.isEmpty()) 
    {
        return;
    }
    
    // 加载所有图像
    vector<Mat> images;
    QStringList validFiles;
    
    for (const QString& fileName : fileNames) 
    {
        Mat image = imread(fileName.toStdString());
        if (!image.empty()) 
        {
            images.push_back(image);
            validFiles.append(fileName);
        }
    }
    
    if (images.empty()) 
    {
        QMessageBox::warning(this, "错误", "没有有效的图像文件！");
        return;
    }
    
    // 显示进度条
    progressBar_->setVisible(true);
    progressBar_->setRange(0, images.size());
    progressBar_->setValue(0);
    
    statusLabel_->setText(QString("正在批量分类 %1 张图像...").arg(images.size()));
    statusLabel_->setStyleSheet("QLabel { color: orange; }");
    
    // 保存文件名用于结果显示
    batchFileNames_ = validFiles;
    
    // 执行批量分类
    vector<ClassificationResult> results;
    dlProcessor_->classifyBatch(images, results);
}

void DLExample::onClassificationComplete(const ClassificationResult& result)
{
    QString resultText = QString("分类结果:\n"
                                "类别: %1 (ID: %2)\n"
                                "置信度: %3\n"
                                "状态: %4")
                        .arg(QString::fromStdString(result.className))
                        .arg(result.classId)
                        .arg(result.confidence, 0, 'f', 4)
                        .arg(result.isValid ? "有效" : "无效");
    
    resultLabel_->setText(resultText);
    
    // 根据结果设置状态颜色
    if (result.isValid) 
    {
        statusLabel_->setText("分类完成");
        statusLabel_->setStyleSheet("QLabel { color: green; }");
    } 
    else 
    {
        statusLabel_->setText("分类结果置信度过低");
        statusLabel_->setStyleSheet("QLabel { color: orange; }");
    }
}

void DLExample::onBatchProcessingComplete(const vector<ClassificationResult>& results)
{
    progressBar_->setVisible(false);
    
    QString resultText = QString("批量分类完成，共处理 %1 张图像:\n\n").arg(results.size());
    
    for (size_t i = 0; i < results.size() && i < static_cast<size_t>(batchFileNames_.size()); ++i) 
    {
        QString fileName = QFileInfo(batchFileNames_[i]).fileName();
        const auto& result = results[i];
        
        resultText += QString("%1: %2 (%.3f)\n")
                     .arg(fileName)
                     .arg(QString::fromStdString(result.className))
                     .arg(result.confidence);
    }
    
    resultLabel_->setText(resultText);
    
    statusLabel_->setText("批量分类完成");
    statusLabel_->setStyleSheet("QLabel { color: green; }");
}

void DLExample::onDLError(const QString& error)
{
    QMessageBox::critical(this, "深度学习错误", error);
    
    statusLabel_->setText("错误: " + error);
    statusLabel_->setStyleSheet("QLabel { color: red; }");
    
    progressBar_->setVisible(false);
}
