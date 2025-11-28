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

/**
 * @brief DLExample类构造函数
 * @param parent 父窗口指针
 * 
 * 初始化深度学习示例应用，创建DLProcessor实例，设置UI和连接信号槽
 */
DLExample::DLExample(QWidget *parent)
    : QWidget(parent)
    , dlProcessor_(new DLProcessor(this))  // 创建DLProcessor实例，用于深度学习推理
    , isModelLoaded_(false)                // 模型加载状态，默认未加载
    , quantizationType_(nullptr)           // 量化类型选择框
    , calibrationImagesBtn_(nullptr)       // 校准图像选择按钮
    , quantizeBtn_(nullptr)                // 量化模型按钮
    , quantizationStatusLabel_(nullptr)    // 量化状态显示标签
    , calibrationImages_()                 // 校准图像路径列表
{
    // 设置用户界面
    SetupUI();
    // 连接信号和槽
    ConnectSignals();
}

/**
 * @brief DLExample类析构函数
 * 
 * 清理资源，释放内存
 */
DLExample::~DLExample()
{
    // 析构函数，自动释放资源
}

/**
 * @brief 设置用户界面
 * 
 * 创建并布局所有UI组件，包括模型加载、参数设置、图像显示和结果展示区域
 */
void DLExample::SetupUI()
{
    // 设置窗口标题和最小尺寸
    setWindowTitle("深度学习");
    setMinimumSize(800, 600);
    
    // 主布局 - 垂直布局
    QVBoxLayout* mainLayout = new QVBoxLayout(this);
    
    // 模型加载区域 - 水平布局
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
    
    // 标签文件区域 - 水平布局
    QHBoxLayout* labelLayout = new QHBoxLayout();
    labelLayout->addWidget(new QLabel("标签文件:"));
    labelPathEdit_ = new QLineEdit();
    labelPathEdit_->setText("Data/Labels/class_labels.txt");  // 默认标签文件路径
    labelLayout->addWidget(labelPathEdit_);
    
    QPushButton* browseLabelBtn = new QPushButton("浏览");
    connect(browseLabelBtn, &QPushButton::clicked, this, &DLExample::BrowseLabels);
    labelLayout->addWidget(browseLabelBtn);
    
    mainLayout->addLayout(labelLayout);
    
    // 参数设置区域 - 水平布局
    QHBoxLayout* paramLayout = new QHBoxLayout();
    
    // 任务类型选择
    paramLayout->addWidget(new QLabel("任务类型:"));
    taskTypeComboBox_ = new QComboBox();
    taskTypeComboBox_->addItems({"图像分类"});  // 当前仅支持图像分类
    connect(taskTypeComboBox_, static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, &DLExample::OnTaskTypeChanged);
    paramLayout->addWidget(taskTypeComboBox_);
    
    // 置信度阈值设置
    paramLayout->addWidget(new QLabel("置信度阈值:"));
    confidenceEdit_ = new QLineEdit("0.5");  // 默认置信度阈值0.5
    confidenceEdit_->setMaximumWidth(80);
    paramLayout->addWidget(confidenceEdit_);
    
    // 输入尺寸设置
    paramLayout->addWidget(new QLabel("输入尺寸:"));
    inputSizeEdit_ = new QLineEdit("224");  // 默认输入尺寸224x224
    inputSizeEdit_->setMaximumWidth(80);
    paramLayout->addWidget(inputSizeEdit_);
    
    // 参数设置按钮
    QPushButton* setParamsBtn = new QPushButton("设置参数");
    connect(setParamsBtn, &QPushButton::clicked, this, &DLExample::SetParameters);
    paramLayout->addWidget(setParamsBtn);
    
    paramLayout->addStretch();  // 右侧留白
    mainLayout->addLayout(paramLayout);
    
    // 模型量化设置区域 - 水平布局
    QHBoxLayout* quantLayout = new QHBoxLayout();
    quantLayout->addWidget(new QLabel("量化类型:"));
    
    quantizationType_ = new QComboBox();
    quantizationType_->addItems({"None", "FP16", "INT8", "UINT8"});  // 支持的量化类型
    quantLayout->addWidget(quantizationType_);
    
    // 校准图像选择按钮
    calibrationImagesBtn_ = new QPushButton("选择校准图像");
    connect(calibrationImagesBtn_, &QPushButton::clicked, this, &DLExample::SelectCalibrationImages);
    quantLayout->addWidget(calibrationImagesBtn_);
    
    // 量化模型按钮
    quantizeBtn_ = new QPushButton("量化模型");
    connect(quantizeBtn_, &QPushButton::clicked, this, &DLExample::QuantizeModel);
    quantLayout->addWidget(quantizeBtn_);
    
    quantLayout->addStretch();  // 右侧留白
    mainLayout->addLayout(quantLayout);
    
    // 量化状态显示
    quantizationStatusLabel_ = new QLabel("未量化");
    quantizationStatusLabel_->setStyleSheet("QLabel { color: blue; font-style: italic; }");
    mainLayout->addWidget(quantizationStatusLabel_);
    
    // 图像处理区域 - 水平布局
    QHBoxLayout* imageLayout = new QHBoxLayout();
    
    // 选择图像按钮
    QPushButton* selectImageBtn = new QPushButton("选择图像");
    connect(selectImageBtn, &QPushButton::clicked, this, &DLExample::SelectImage);
    imageLayout->addWidget(selectImageBtn);
    
    // 开始分类按钮
    classifyBtn_ = new QPushButton("开始分类");
    connect(classifyBtn_, &QPushButton::clicked, this, &DLExample::ProcessImage);
    imageLayout->addWidget(classifyBtn_);
    
    // 批量分类按钮
    batchBtn_ = new QPushButton("批量分类");
    connect(batchBtn_, &QPushButton::clicked, this, &DLExample::BatchProcess);
    imageLayout->addWidget(batchBtn_);
    
    imageLayout->addStretch();  // 右侧留白
    mainLayout->addLayout(imageLayout);
    
    // 进度条，用于显示批量处理进度
    progressBar_ = new QProgressBar();
    progressBar_->setVisible(false);  // 默认隐藏
    mainLayout->addWidget(progressBar_);
    
    // 结果显示区域
    resultLabel_ = new QLabel("等待分类结果...");
    resultLabel_->setStyleSheet("QLabel { background-color: #f0f0f0; padding: 10px; border: 1px solid #ccc; }");
    resultLabel_->setMinimumHeight(100);
    resultLabel_->setWordWrap(true);  // 自动换行
    mainLayout->addWidget(resultLabel_);
    
    // 图像显示区域
    imageLabel_ = new QLabel();
    imageLabel_->setMinimumSize(400, 300);
    imageLabel_->setStyleSheet("QLabel { border: 1px solid #ccc; background-color: white; }");
    imageLabel_->setAlignment(Qt::AlignCenter);  // 居中显示
    imageLabel_->setText("图像显示区域");
    mainLayout->addWidget(imageLabel_);
    
    // 状态标签，显示当前操作状态
    statusLabel_ = new QLabel("就绪");
    statusLabel_->setStyleSheet("QLabel { color: blue; }");
    mainLayout->addWidget(statusLabel_);
}

/**
 * @brief 连接信号和槽
 * 
 * 连接DLProcessor的信号到DLExample的槽函数，处理分类结果和错误信息
 */
void DLExample::ConnectSignals()
{
    // 连接DLProcessor信号到槽函数
    connect(dlProcessor_, &DLProcessor::classificationComplete, this, &DLExample::OnClassificationComplete);  // 单张图像分类完成
    connect(dlProcessor_, &DLProcessor::batchProcessingComplete, this, &DLExample::OnBatchProcessingComplete);  // 批量分类完成
    connect(dlProcessor_, &DLProcessor::errorOccurred, this, &DLExample::OnDLError);  // 错误处理
}

/**
 * @brief 浏览并选择模型文件
 * 
 * 打开文件对话框，让用户选择深度学习模型文件（支持ONNX、PB、Caffe等格式）
 */
void DLExample::BrowseModel()
{
    // 变量定义
    QString fileName; // 选择的模型文件路径
    
    // 打开文件对话框选择模型文件
    fileName = QFileDialog::getOpenFileName(this, "选择深度学习模型文件", "", "模型文件 (*.onnx *.pb *.caffemodel *.weights);;所有文件 (*.*)");
    
    if (!fileName.isEmpty()) 
    {
        // 更新模型路径输入框
        modelPathEdit_->setText(fileName);
    }
}

/**
 * @brief 浏览并选择标签文件
 * 
 * 打开文件对话框，让用户选择类别标签文件（文本文件，每行一个标签）
 */
void DLExample::BrowseLabels()
{
    // 变量定义
    QString fileName; // 选择的标签文件路径
    
    // 打开文件对话框选择标签文件
    fileName = QFileDialog::getOpenFileName(this, "选择类别标签文件", "", "文本文件 (*.txt);;所有文件 (*.*)");
    
    if (!fileName.isEmpty()) 
    {
        // 更新标签路径输入框
        labelPathEdit_->setText(fileName);
        // 立即加载标签文件，无论模型是否已加载
        LoadLabels();
    }
}

/**
 * @brief 加载标签文件
 * 
 * 从文件中读取类别标签，并更新DLProcessor的标签列表
 */
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
    
    // 加载标签文件到DLProcessor
    if (dlProcessor_->LoadClassLabels(labelPath.toStdString()))
    {
        // 标签加载成功
        statusLabel_->setText("标签文件加载成功");
        statusLabel_->setStyleSheet("QLabel { color: green; }");
        qDebug() << "标签加载成功:" << labelPath;
    } 
    else 
    {
        // 标签加载失败
        statusLabel_->setText("标签文件加载失败");
        statusLabel_->setStyleSheet("QLabel { color: red; }");
        qDebug() << "标签加载失败:" << labelPath;
    }
}

/**
 * @brief 加载深度学习模型
 * 
 * 从文件中加载深度学习模型，并初始化DLProcessor
 */
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
    
    // 更新状态显示
    statusLabel_->setText("正在加载模型...");
    statusLabel_->setStyleSheet("QLabel { color: orange; }");
    
    // 检查是否需要配置文件
    configPath = "";
    if (modelPath.endsWith(".caffemodel")) 
    {
        // Caffe模型需要.prototxt配置文件
        QString prototxt = modelPath;
        prototxt.replace(".caffemodel", ".prototxt");
        if (QFile::exists(prototxt)) 
        {
            configPath = prototxt;
        }
    } 
    else if (modelPath.endsWith(".weights")) 
    {
        // YOLO模型需要.cfg配置文件
        QString cfg = modelPath;
        cfg.replace(".weights", ".cfg");
        if (QFile::exists(cfg)) 
        {
            configPath = cfg;
        }
    }
    
    // 加载模型到DLProcessor
    success = dlProcessor_->InitModel(modelPath.toStdString(), configPath.toStdString());
    
    if (success) 
    {
        // 模型加载成功
        isModelLoaded_ = true;
        statusLabel_->setText("模型加载成功");
        statusLabel_->setStyleSheet("QLabel { color: green; }");
        
        // 加载标签文件
        QString labelPath = labelPathEdit_->text().trimmed();
        if (!labelPath.isEmpty() && QFile::exists(labelPath)) 
        {
            dlProcessor_->LoadClassLabels(labelPath.toStdString());
        }
        
        // 更新结果显示
        resultLabel_->setText("模型已加载，可以开始分类");
        
        // 重置量化状态
        quantizationStatusLabel_->setText("未量化");
        quantizationStatusLabel_->setStyleSheet("QLabel { color: blue; font-style: italic; }");
        quantizationType_->setCurrentIndex(0);
        calibrationImages_.clear();
    } 
    else 
    {
        // 模型加载失败
        isModelLoaded_ = false;
        statusLabel_->setText("模型加载失败");
        statusLabel_->setStyleSheet("QLabel { color: red; }");
    }
}

/**
 * @brief 设置模型参数
 * 
 * 从UI控件中获取参数，并设置到DLProcessor
 */
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
    
    // 获取并验证置信度阈值（0.0-1.0）
    confidence = confidenceEdit_->text().toFloat(&ok);
    if (!ok || confidence < 0.0f || confidence > 1.0f) 
    {
        QMessageBox::warning(this, "警告", "置信度阈值必须在0.0-1.0之间！");
        return;
    }
    
    // 获取并验证输入尺寸（32-1024）
    inputSize = inputSizeEdit_->text().toInt(&ok);
    if (!ok || inputSize < 32 || inputSize > 1024) 
    {
        QMessageBox::warning(this, "警告", "输入尺寸必须在32-1024之间！");
        return;
    }
    
    // 设置模型参数到DLProcessor
    dlProcessor_->SetModelParams(confidence, 0.4f);  // 设置置信度阈值和NMS阈值
    dlProcessor_->SetInputSize(Size(inputSize, inputSize));  // 设置输入尺寸
    
    // 更新状态显示
    statusLabel_->setText(QString("参数已更新: 置信度=%1, 输入尺寸=%2x%2").arg(confidence).arg(inputSize));
    statusLabel_->setStyleSheet("QLabel { color: blue; }");
}

/**
 * @brief 选择要分类的图像
 * 
 * 打开文件对话框，让用户选择单张图像文件
 */
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
        // 保存当前图像路径
        currentImagePath_ = fileName;
        
        // 加载并显示图像
        pixmap = QPixmap(fileName);
        if (!pixmap.isNull()) 
        {
            // 缩放图像以适应显示区域，保持宽高比
            scaledPixmap = pixmap.scaled(imageLabel_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
            imageLabel_->setPixmap(scaledPixmap);
            
            // 更新状态显示
            statusLabel_->setText("图像已加载: " + QFileInfo(fileName).fileName());
            statusLabel_->setStyleSheet("QLabel { color: blue; }");
        }
    }
}

/**
 * @brief 对单张图像进行分类
 * 
 * 加载当前选择的图像，调用DLProcessor进行分类，并显示结果
 */
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
    
    // 更新状态显示
    statusLabel_->setText("正在分类...");
    statusLabel_->setStyleSheet("QLabel { color: orange; }");
    
    // 执行分类，结果会通过信号槽处理
    if (dlProcessor_->ClassifyImage(image, result))
    {
        // 分类成功，结果会通过OnClassificationComplete槽函数处理
    }
     else 
    {
        // 分类失败
        statusLabel_->setText("分类失败");
        statusLabel_->setStyleSheet("QLabel { color: red; }");
    }
}

/**
 * @brief 批量分类多张图像
 * 
 * 选择多张图像，调用DLProcessor进行批量分类，并显示结果
 */
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
        return;  // 用户取消选择
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
    
    // 更新状态显示
    statusLabel_->setText(QString("正在批量分类 %1 张图像...").arg(images.size()));
    statusLabel_->setStyleSheet("QLabel { color: orange; }");
    
    // 保存文件名用于结果显示
    batchFileNames_ = validFiles;
    
    // 执行批量分类，结果会通过信号槽处理
    dlProcessor_->ClassifyBatch(images, results);
}

/**
 * @brief 处理单张图像分类完成信号
 * @param result 分类结果
 * 
 * 显示单张图像的分类结果，包括类别名称、ID、置信度和有效性
 */
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
                        .arg(result.confidence, 0, 'f', 4)  // 保留4位小数
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

/**
 * @brief 处理批量分类完成信号
 * @param results 批量分类结果列表
 * 
 * 显示批量分类的结果，包括每张图像的文件名、类别和置信度
 */
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
        fileName = QFileInfo(batchFileNames_[i]).fileName();  // 获取文件名（不含路径）
        const auto& result = results[i];
        
        // 添加当前图像的分类结果
        resultText += QString("%1: %2 (%.3f)\n").arg(fileName).arg(QString::fromStdString(result.className)).arg(result.confidence);
    }
    
    // 显示结果
    resultLabel_->setText(resultText);
    
    // 更新状态
    statusLabel_->setText("批量分类完成");
    statusLabel_->setStyleSheet("QLabel { color: green; }");
}

/**
 * @brief 处理深度学习错误信号
 * @param error 错误信息
 * 
 * 显示错误消息框，并更新状态标签
 */
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

/**
 * @brief 选择校准图像用于量化
 * 
 * 打开文件对话框，让用户选择用于模型量化的校准图像
 */
void DLExample::SelectCalibrationImages()
{
    // 变量定义
    QStringList fileNames;    // 选择的文件列表
    
    // 打开文件对话框选择多个图像文件
    fileNames = QFileDialog::getOpenFileNames(this, "选择校准图像", "", "图像文件 (*.jpg *.jpeg *.png *.bmp *.tiff)");
    
    if (!fileNames.isEmpty())
    {
        // 保存校准图像路径
        calibrationImages_ = fileNames;
        // 更新状态显示
        statusLabel_->setText(QString("已选择 %1 张校准图像").arg(calibrationImages_.size()));
        statusLabel_->setStyleSheet("QLabel { color: blue; }");
    }
}

/**
 * @brief 执行模型量化
 * 
 * 对已加载的模型进行量化，减小模型大小，加速推理
 */
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
    
    // 检查是否需要校准图像（INT8和UINT8量化需要）
    if ((quantType == "INT8" || quantType == "UINT8") && calibrationImages_.isEmpty())
    {
        QMessageBox::warning(this, "警告", "INT8/UINT8量化需要校准图像，请先选择校准图像！");
        return;
    }
    
    // 加载校准图像
    if (quantType == "INT8" || quantType == "UINT8")
    {
        // 更新状态显示
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
        
        // 更新量化状态显示
        quantizationStatusLabel_->setText(QString("已量化 (" + quantType + ")"));
        quantizationStatusLabel_->setStyleSheet("QLabel { color: green; font-style: italic; font-weight: bold; }");
        
        // 显示模型信息
        resultLabel_->setText(QString::fromStdString(dlProcessor_->GetModelInfo()));
    }
}

/**
 * @brief 任务类型改变处理槽函数
 * @param index 选择的任务类型索引
 * 
 * 当任务类型改变时，更新UI控件文本
 */
void DLExample::OnTaskTypeChanged(int index)
{
    // 当前只有图像分类任务类型，无需其他处理
    classifyBtn_->setText("开始分类");
    batchBtn_->setText("批量分类");
}

/**
 * @brief 图像分类处理方法
 * 
 * 加载当前选择的图像，调用DLProcessor进行分类，并显示结果
 */
void DLExample::ProcessImage()
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

    // 更新状态显示
    statusLabel_->setText("正在分类...");
    statusLabel_->setStyleSheet("QLabel { color: orange; }");

    // 执行图像分类，结果会通过信号槽处理
    if (dlProcessor_->ClassifyImage(image, result))
    {
        // 分类成功，结果会通过OnClassificationComplete槽函数处理
    }
    else 
    {
        // 分类失败
        statusLabel_->setText("分类失败");
        statusLabel_->setStyleSheet("QLabel { color: red; }");
    }
}

/**
 * @brief 批量分类处理方法
 * 
 * 选择多张图像，调用DLProcessor进行批量分类，并显示结果
 */
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
    fileNames = QFileDialog::getOpenFileNames(this, "选择要批量分类的图像", "", "图像文件 (*.jpg *.jpeg *.png *.bmp *.tiff)");

    if (fileNames.isEmpty()) 
    {
        return;  // 用户取消选择
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

    // 更新状态显示
    statusLabel_->setText(QString("正在批量分类 %1 张图像...").arg(images.size()));
    statusLabel_->setStyleSheet("QLabel { color: orange; }");

    // 保存文件名用于结果显示
    batchFileNames_ = validFiles;

    // 执行批量分类，结果会通过信号槽处理
    dlProcessor_->ClassifyBatch(images, results);
}


