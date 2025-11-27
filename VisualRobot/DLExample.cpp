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
#include <QComboBox>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

DLExample::DLExample(QWidget *parent)
    : QWidget(parent)
    , dlProcessor_(new DLProcessor(this))
    , isModelLoaded_(false)
    , quantizationType_(nullptr)
    , calibrationImagesBtn_(nullptr)
    , quantizeBtn_(nullptr)
    , quantizationStatusLabel_(nullptr)
    , calibrationImages_()
{
    SetupUI();
    ConnectSignals();
}

DLExample::~DLExample()
{

}

void DLExample::SetupUI()
{
    // 设置窗口标题和最小尺寸
    setWindowTitle("深度学习");
    setMinimumSize(800, 600);
    
    // 主布局
    QVBoxLayout* mainLayout = new QVBoxLayout(this);
    
    // 模型加载区域
    QHBoxLayout* modelLayout = new QHBoxLayout();
    modelLayout->addWidget(new QLabel("模型路径:"));
    modelPathEdit_ = new QLineEdit();
    modelPathEdit_->setPlaceholderText("选择ONNX/PB/Caffe模型文件");
    modelLayout->addWidget(modelPathEdit_);
    
    QPushButton* browseModelBtn = new QPushButton("浏览");
    connect(browseModelBtn, &QPushButton::clicked, this, &DLExample::BrowseModel);
    modelLayout->addWidget(browseModelBtn);
    
    QPushButton* loadModelBtn = new QPushButton("加载模型");
    connect(loadModelBtn, &QPushButton::clicked, this, &DLExample::LoadModel);
    modelLayout->addWidget(loadModelBtn);
    
    mainLayout->addLayout(modelLayout);
    
    // 标签文件区域
    QHBoxLayout* labelLayout = new QHBoxLayout();
    labelLayout->addWidget(new QLabel("标签文件:"));
    labelPathEdit_ = new QLineEdit();
    labelPathEdit_->setText("Data/Labels/class_labels.txt");
    labelLayout->addWidget(labelPathEdit_);
    
    QPushButton* browseLabelBtn = new QPushButton("浏览");
    connect(browseLabelBtn, &QPushButton::clicked, this, &DLExample::BrowseLabels);
    labelLayout->addWidget(browseLabelBtn);
    
    mainLayout->addLayout(labelLayout);
    
    // 参数设置区域
    QHBoxLayout* paramLayout = new QHBoxLayout();
    
    // 任务类型选择
    paramLayout->addWidget(new QLabel("任务类型:"));
    taskTypeComboBox_ = new QComboBox();
    taskTypeComboBox_->addItems({"图像分类"});
    connect(taskTypeComboBox_, static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, &DLExample::OnTaskTypeChanged);
    paramLayout->addWidget(taskTypeComboBox_);
    
    paramLayout->addWidget(new QLabel("置信度阈值:"));
    confidenceEdit_ = new QLineEdit("0.5");
    confidenceEdit_->setMaximumWidth(80);
    paramLayout->addWidget(confidenceEdit_);
    
    paramLayout->addWidget(new QLabel("输入尺寸:"));
    inputSizeEdit_ = new QLineEdit("224");
    inputSizeEdit_->setMaximumWidth(80);
    paramLayout->addWidget(inputSizeEdit_);
    
    QPushButton* setParamsBtn = new QPushButton("设置参数");
    connect(setParamsBtn, &QPushButton::clicked, this, &DLExample::SetParameters);
    paramLayout->addWidget(setParamsBtn);
    
    paramLayout->addStretch();
    mainLayout->addLayout(paramLayout);
    
    // 模型量化设置区域
    QHBoxLayout* quantLayout = new QHBoxLayout();
    quantLayout->addWidget(new QLabel("量化类型:"));
    
    quantizationType_ = new QComboBox();
    quantizationType_->addItems({"None", "FP16", "INT8", "UINT8"});
    quantLayout->addWidget(quantizationType_);
    
    calibrationImagesBtn_ = new QPushButton("选择校准图像");
    connect(calibrationImagesBtn_, &QPushButton::clicked, this, &DLExample::SelectCalibrationImages);
    quantLayout->addWidget(calibrationImagesBtn_);
    
    quantizeBtn_ = new QPushButton("量化模型");
    connect(quantizeBtn_, &QPushButton::clicked, this, &DLExample::QuantizeModel);
    quantLayout->addWidget(quantizeBtn_);
    
    quantLayout->addStretch();
    mainLayout->addLayout(quantLayout);
    
    // 量化状态显示
    quantizationStatusLabel_ = new QLabel("未量化");
    quantizationStatusLabel_->setStyleSheet("QLabel { color: blue; font-style: italic; }");
    mainLayout->addWidget(quantizationStatusLabel_);
    
    // 图像处理区域
    QHBoxLayout* imageLayout = new QHBoxLayout();
    
    QPushButton* selectImageBtn = new QPushButton("选择图像");
    connect(selectImageBtn, &QPushButton::clicked, this, &DLExample::SelectImage);
    imageLayout->addWidget(selectImageBtn);
    
    classifyBtn_ = new QPushButton("开始分类");
    connect(classifyBtn_, &QPushButton::clicked, this, &DLExample::ProcessImage);
    imageLayout->addWidget(classifyBtn_);
    
    batchBtn_ = new QPushButton("批量分类");
    connect(batchBtn_, &QPushButton::clicked, this, &DLExample::BatchProcess);
    imageLayout->addWidget(batchBtn_);
    
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

void DLExample::ConnectSignals()
{
    // 连接DLProcessor信号
    connect(dlProcessor_, &DLProcessor::classificationComplete, this, &DLExample::OnClassificationComplete);
    connect(dlProcessor_, &DLProcessor::batchProcessingComplete, this, &DLExample::OnBatchProcessingComplete);
    connect(dlProcessor_, &DLProcessor::errorOccurred, this, &DLExample::OnDLError);
}

void DLExample::BrowseModel()
{
    // 变量定义
    QString fileName; // 选择的模型文件路径
    
    // 打开文件对话框选择模型文件
    fileName = QFileDialog::getOpenFileName(this, "选择深度学习模型文件", "", "模型文件 (*.onnx *.pb *.caffemodel *.weights);;所有文件 (*.*)");
    
    if (!fileName.isEmpty()) 
    {
        modelPathEdit_->setText(fileName);
    }
}

void DLExample::BrowseLabels()
{
    // 变量定义
    QString fileName; // 选择的标签文件路径
    
    // 打开文件对话框选择标签文件
    fileName = QFileDialog::getOpenFileName(this, "选择类别标签文件", "", "文本文件 (*.txt);;所有文件 (*.*)");
    
    if (!fileName.isEmpty()) 
    {
        labelPathEdit_->setText(fileName);
        // 立即加载标签文件，无论模型是否已加载
        LoadLabels();
    }
}

void DLExample::LoadLabels()
{
    // 变量定义
    QString labelPath; // 标签文件路径
    
    // 获取标签文件路径
    labelPath = labelPathEdit_->text().trimmed();
    if (labelPath.isEmpty()) 
    {
        return;
    }
    
    // 检查标签文件是否存在
    if (!QFile::exists(labelPath)) 
    {
        QMessageBox::warning(this, "警告", "标签文件不存在！");
        return;
    }
    
    // 加载标签文件
    if (dlProcessor_->LoadClassLabels(labelPath.toStdString()))
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

void DLExample::LoadModel()
{
    // 变量定义
    QString modelPath;      // 模型文件路径
    QString configPath;     // 配置文件路径
    bool success;           // 模型加载是否成功
    
    // 获取模型文件路径
    modelPath = modelPathEdit_->text().trimmed();
    if (modelPath.isEmpty()) 
    {
        QMessageBox::warning(this, "警告", "请先选择模型文件！");
        return;
    }
    
    statusLabel_->setText("正在加载模型...");
    statusLabel_->setStyleSheet("QLabel { color: orange; }");
    
    // 检查是否需要配置文件
    configPath = "";
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
    success = dlProcessor_->InitModel(modelPath.toStdString(), configPath.toStdString());
    
    if (success) 
    {
        isModelLoaded_ = true;
        statusLabel_->setText("模型加载成功");
        statusLabel_->setStyleSheet("QLabel { color: green; }");
        
        // 加载标签文件
        QString labelPath = labelPathEdit_->text().trimmed();
        if (!labelPath.isEmpty() && QFile::exists(labelPath)) 
        {
            dlProcessor_->LoadClassLabels(labelPath.toStdString());
        }
        
        resultLabel_->setText("模型已加载，可以开始分类");
        
        // 重置量化状态
        quantizationStatusLabel_->setText("未量化");
        quantizationStatusLabel_->setStyleSheet("QLabel { color: blue; font-style: italic; }");
        quantizationType_->setCurrentIndex(0);
        calibrationImages_.clear();
    } 
    else 
    {
        isModelLoaded_ = false;
        statusLabel_->setText("模型加载失败");
        statusLabel_->setStyleSheet("QLabel { color: red; }");
    }
}

void DLExample::SetParameters()
{
    // 变量定义
    bool ok;                // 转换是否成功
    float confidence;       // 置信度阈值
    int inputSize;          // 输入尺寸
    
    // 检查模型是否已加载
    if (!isModelLoaded_) 
    {
        QMessageBox::warning(this, "警告", "请先加载模型！");
        return;
    }
    
    // 获取并验证置信度阈值
    confidence = confidenceEdit_->text().toFloat(&ok);
    if (!ok || confidence < 0.0f || confidence > 1.0f) 
    {
        QMessageBox::warning(this, "警告", "置信度阈值必须在0.0-1.0之间！");
        return;
    }
    
    // 获取并验证输入尺寸
    inputSize = inputSizeEdit_->text().toInt(&ok);
    if (!ok || inputSize < 32 || inputSize > 1024) 
    {
        QMessageBox::warning(this, "警告", "输入尺寸必须在32-1024之间！");
        return;
    }
    
    // 设置模型参数
    dlProcessor_->SetModelParams(confidence, 0.4f);
    dlProcessor_->SetInputSize(Size(inputSize, inputSize));
    
    // 更新状态显示
    statusLabel_->setText(QString("参数已更新: 置信度=%1, 输入尺寸=%2x%2").arg(confidence).arg(inputSize));
    statusLabel_->setStyleSheet("QLabel { color: blue; }");
}

void DLExample::SelectImage()
{
    // 变量定义
    QString fileName;       // 选择的图像文件路径
    QPixmap pixmap;         // 图像像素映射
    QPixmap scaledPixmap;   // 缩放后的图像像素映射
    
    // 打开文件对话框选择图像文件
    fileName = QFileDialog::getOpenFileName(this, "选择要分类的图像", "", "图像文件 (*.jpg *.jpeg *.png *.bmp *.tiff);;所有文件 (*.*)");
    
    if (!fileName.isEmpty()) 
    {
        currentImagePath_ = fileName;
        
        // 加载并显示图像
        pixmap = QPixmap(fileName);
        if (!pixmap.isNull()) 
        {
            scaledPixmap = pixmap.scaled(imageLabel_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
            imageLabel_->setPixmap(scaledPixmap);
            
            statusLabel_->setText("图像已加载: " + QFileInfo(fileName).fileName());
            statusLabel_->setStyleSheet("QLabel { color: blue; }");
        }
    }
}

void DLExample::ClassifyImage()
{
    // 变量定义
    Mat image;                      // OpenCV图像矩阵
    ClassificationResult result;    // 分类结果
    
    // 检查模型是否已加载
    if (!isModelLoaded_) 
    {
        QMessageBox::warning(this, "警告", "请先加载模型！");
        return;
    }
    
    // 检查是否已选择图像
    if (currentImagePath_.isEmpty()) 
    {
        QMessageBox::warning(this, "警告", "请先选择图像！");
        return;
    }
    
    // 加载图像
    image = imread(currentImagePath_.toStdString());
    if (image.empty()) 
    {
        QMessageBox::warning(this, "错误", "无法加载图像文件！");
        return;
    }
    
    statusLabel_->setText("正在分类...");
    statusLabel_->setStyleSheet("QLabel { color: orange; }");
    
    // 执行分类
    if (dlProcessor_->ClassifyImage(image, result))
    {
        // 结果会通过信号槽处理
    }
     else 
    {
        statusLabel_->setText("分类失败");
        statusLabel_->setStyleSheet("QLabel { color: red; }");
    }
}

void DLExample::BatchClassify()
{
    // 变量定义
    QStringList fileNames;                // 选择的文件列表
    vector<Mat> images;                   // OpenCV图像矩阵列表
    QStringList validFiles;               // 有效的文件名列表
    vector<ClassificationResult> results; // 批量分类结果列表
    
    // 检查模型是否已加载
    if (!isModelLoaded_) 
    {
        QMessageBox::warning(this, "警告", "请先加载模型！");
        return;
    }
    
    // 打开文件对话框选择多个图像文件
    fileNames = QFileDialog::getOpenFileNames(this, "选择要批量分类的图像", "", "图像文件 (*.jpg *.jpeg *.png *.bmp *.tiff)");
    
    if (fileNames.isEmpty()) 
    {
        return;
    }
    
    // 加载所有图像
    for (const QString& fileName : fileNames) 
    {
        Mat image = imread(fileName.toStdString());
        if (!image.empty()) 
        {
            images.push_back(image);
            validFiles.append(fileName);
        }
    }
    
    // 检查是否有有效的图像
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
    dlProcessor_->ClassifyBatch(images, results);
}

void DLExample::OnClassificationComplete(const ClassificationResult& result)
{
    // 变量定义
    QString resultText; // 结果文本
    
    // 构建结果文本
    resultText = QString("分类结果:\n"
                         "类别: %1 (ID: %2)\n"
                         "置信度: %3\n"
                         "状态: %4")
                        .arg(QString::fromStdString(result.className))
                        .arg(result.classId)
                        .arg(result.confidence, 0, 'f', 4)
                        .arg(result.isValid ? "有效" : "无效");
    
    // 显示结果
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

void DLExample::OnBatchProcessingComplete(const vector<ClassificationResult>& results)
{
    // 变量定义
    QString resultText; // 结果文本
    QString fileName;   // 文件名
    
    // 隐藏进度条
    progressBar_->setVisible(false);
    
    // 构建结果文本
    resultText = QString("批量分类完成，共处理 %1 张图像:\n\n").arg(results.size());
    
    // 添加每个图像的结果
    for (size_t i = 0; i < results.size() && i < static_cast<size_t>(batchFileNames_.size()); ++i) 
    {
        fileName = QFileInfo(batchFileNames_[i]).fileName();
        const auto& result = results[i];
        
        resultText += QString("%1: %2 (%.3f)\n").arg(fileName).arg(QString::fromStdString(result.className)).arg(result.confidence);
    }
    
    // 显示结果
    resultLabel_->setText(resultText);
    
    // 更新状态
    statusLabel_->setText("批量分类完成");
    statusLabel_->setStyleSheet("QLabel { color: green; }");
}

void DLExample::OnDLError(const QString& error)
{
    // 显示错误消息框
    QMessageBox::critical(this, "深度学习错误", error);
    
    // 更新状态标签
    statusLabel_->setText("错误: " + error);
    statusLabel_->setStyleSheet("QLabel { color: red; }");
    
    // 隐藏进度条
    progressBar_->setVisible(false);
}

void DLExample::SelectCalibrationImages()
{
    // 变量定义
    QStringList fileNames;    // 选择的文件列表
    
    // 打开文件对话框选择多个图像文件
    fileNames = QFileDialog::getOpenFileNames(this, "选择校准图像", "", "图像文件 (*.jpg *.jpeg *.png *.bmp *.tiff)");
    
    if (!fileNames.isEmpty())
    {
        calibrationImages_ = fileNames;
        statusLabel_->setText(QString("已选择 %1 张校准图像").arg(calibrationImages_.size()));
        statusLabel_->setStyleSheet("QLabel { color: blue; }");
    }
}

void DLExample::QuantizeModel()
{
    // 变量定义
    QString quantType;       // 量化类型
    vector<Mat> calibImages; // 校准图像列表
    bool success;            // 量化是否成功
    
    // 检查模型是否已加载
    if (!isModelLoaded_)
    {
        QMessageBox::warning(this, "警告", "请先加载模型！");
        return;
    }
    
    // 获取量化类型
    quantType = quantizationType_->currentText();
    if (quantType == "None")
    {
        QMessageBox::warning(this, "警告", "请选择有效的量化类型！");
        return;
    }
    
    // 检查是否需要校准图像
    if ((quantType == "INT8" || quantType == "UINT8") && calibrationImages_.isEmpty())
    {
        QMessageBox::warning(this, "警告", "INT8/UINT8量化需要校准图像，请先选择校准图像！");
        return;
    }
    
    // 加载校准图像
    if (quantType == "INT8" || quantType == "UINT8")
    {
        statusLabel_->setText("正在准备校准图像...");
        statusLabel_->setStyleSheet("QLabel { color: orange; }");
        
        for (const QString& fileName : calibrationImages_)
        {
            Mat image = imread(fileName.toStdString());
            if (!image.empty())
            {
                calibImages.push_back(image);
            }
        }
        
        if (calibImages.empty())
        {
            QMessageBox::warning(this, "错误", "无法加载有效的校准图像！");
            return;
        }
    }
    
    // 开始量化
    statusLabel_->setText("正在量化模型...");
    statusLabel_->setStyleSheet("QLabel { color: orange; }");
    
    // 执行量化
    success = dlProcessor_->QuantizeModel(quantType.toStdString(), calibImages);
    
    if (success)
    {
        // 更新状态
        statusLabel_->setText("模型量化成功");
        statusLabel_->setStyleSheet("QLabel { color: green; }");
        
        quantizationStatusLabel_->setText(QString("已量化 (" + quantType + ")"));
        quantizationStatusLabel_->setStyleSheet("QLabel { color: green; font-style: italic; font-weight: bold; }");
        
        // 显示模型信息
        resultLabel_->setText(QString::fromStdString(dlProcessor_->GetModelInfo()));
    }
}

// 任务类型改变处理槽函数
void DLExample::OnTaskTypeChanged(int index)
{
    // 当前只有图像分类任务类型，无需其他处理
    classifyBtn_->setText("开始分类");
    batchBtn_->setText("批量分类");
}

// 通用图像处理方法
void DLExample::ProcessImage()
{
    // 变量定义
    Mat image;                      // OpenCV图像矩阵
    ClassificationResult result;    // 分类结果
    Mat processedImage;             // 处理后的图像

    // 检查模型是否已加载
    if (!isModelLoaded_) 
    {
        QMessageBox::warning(this, "警告", "请先加载模型！");
        return;
    }

    // 检查是否已选择图像
    if (currentImagePath_.isEmpty()) 
    {
        QMessageBox::warning(this, "警告", "请先选择图像！");
        return;
    }

    // 加载图像
    image = imread(currentImagePath_.toStdString());
    if (image.empty()) 
    {
        QMessageBox::warning(this, "错误", "无法加载图像文件！");
        return;
    }

    statusLabel_->setText("正在处理...");
    statusLabel_->setStyleSheet("QLabel { color: orange; }");

    // 获取当前任务类型
    int taskType = taskTypeComboBox_->currentIndex();
    
    // 根据任务类型执行不同操作
    if (taskType == 0) // 图像分类
    {
        if (dlProcessor_->ClassifyImage(image, result))
        {
            // 结果会通过信号槽处理
        }
        else 
        {
            statusLabel_->setText("处理失败");
            statusLabel_->setStyleSheet("QLabel { color: red; }");
        }
    }
    else if (taskType == 1 || taskType == 2) // 目标检测或实例分割
    {
        if (dlProcessor_->ProcessYoloFrame(image, processedImage))
        {
            // 结果会通过信号槽处理
        }
        else 
        {
            statusLabel_->setText("处理失败");
            statusLabel_->setStyleSheet("QLabel { color: red; }");
        }
    }
}

// 通用批量处理方法
void DLExample::BatchProcess()
{
    // 变量定义
    QStringList fileNames;                // 选择的文件列表
    vector<Mat> images;                   // OpenCV图像矩阵列表
    QStringList validFiles;               // 有效的文件名列表
    vector<ClassificationResult> results; // 批量分类结果列表

    // 检查模型是否已加载
    if (!isModelLoaded_) 
    {
        QMessageBox::warning(this, "警告", "请先加载模型！");
        return;
    }

    // 打开文件对话框选择多个图像文件
    fileNames = QFileDialog::getOpenFileNames(this, "选择要处理的图像", "", "图像文件 (*.jpg *.jpeg *.png *.bmp *.tiff)");

    if (fileNames.isEmpty()) 
    {
        return;
    }

    // 获取当前任务类型
    int taskType = taskTypeComboBox_->currentIndex();
    
    // 加载所有图像
    for (const QString& fileName : fileNames) 
    {
        Mat image = imread(fileName.toStdString());
        if (!image.empty()) 
        {
            images.push_back(image);
            validFiles.append(fileName);
        }
    }

    // 检查是否有有效的图像
    if (images.empty()) 
    {
        QMessageBox::warning(this, "错误", "没有有效的图像文件！");
        return;
    }

    // 显示进度条
    progressBar_->setVisible(true);
    progressBar_->setRange(0, images.size());
    progressBar_->setValue(0);

    statusLabel_->setText(QString("正在批量处理 %1 张图像...").arg(images.size()));
    statusLabel_->setStyleSheet("QLabel { color: orange; }");

    // 保存文件名用于结果显示
    batchFileNames_ = validFiles;

    // 根据任务类型执行不同的批量处理
    if (taskType == 0) // 图像分类
    {
        dlProcessor_->ClassifyBatch(images, results);
    }
    else // 目标检测或实例分割
    {
        // 这里可以扩展支持YOLO模型的批量处理
        QMessageBox::information(this, "信息", "YOLO批量处理功能即将推出");
        progressBar_->setVisible(false);
    }
}

// YOLO处理结果显示槽函数
void DLExample::OnProcessingComplete(const cv::Mat& resultImage)
{
    // 变量定义
    Mat rgbImage; // RGB格式图像
    QImage qImage; // Qt图像对象
    QPixmap pixmap; // Qt像素图对象
    
    try
    {
        // 将OpenCV图像转换为Qt格式
        cvtColor(resultImage, rgbImage, COLOR_BGR2RGB);
        qImage = QImage(rgbImage.data, rgbImage.cols, rgbImage.rows, rgbImage.step, QImage::Format_RGB888);
        
        // 调整图像大小以适应显示区域
        pixmap = QPixmap::fromImage(qImage).scaled(imageLabel_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
        
        // 显示处理后的图像
        imageLabel_->setPixmap(pixmap);
        
        // 更新状态
        statusLabel_->setText("处理完成");
        statusLabel_->setStyleSheet("QLabel { color: green; }");
        
        // 显示简单的结果统计
        resultLabel_->setText("YOLO处理完成 - 已在图像上绘制检测框和分割掩码");
    }
    catch (const exception& e)
    {
        statusLabel_->setText("结果显示失败");
        statusLabel_->setStyleSheet("QLabel { color: red; }");
        qDebug() << "Error displaying result image:" << QString::fromStdString(e.what());
    }
}
