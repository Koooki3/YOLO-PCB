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
#include <QDoubleValidator>
#include "configmanager.h"

using namespace cv;

/**
 * @brief YOLOExample构造函数
 * @param parent 父窗口指针
 * 
 * 初始化YOLO示例窗口，创建YOLO处理器实例，设置UI界面，连接信号和槽
 */
YOLOExample::YOLOExample(QWidget *parent)
    : QWidget(parent)
    , yoloProcessor_(new YOLOProcessorORT(this))  // 创建YOLO处理器实例
    , statusLabel_(nullptr)
    , imageLabel_(nullptr)
{
    SetupUI();      // 设置UI界面
    ConnectSignals(); // 连接信号和槽
    
    // 从配置管理器获取路径
    ConfigManager* config = ConfigManager::instance();
    QString modelPath = config->getModelPath() + "/arcuchi2.onnx";
    QString labelPath = config->getLabelPath() + "/class_labels.txt";
    
    // 加载模型
    bool modelOk = yoloProcessor_->InitModel(modelPath.toStdString(), false);
    if (modelOk)
    {
        statusLabel_->setText("状态: 模型已加载");
    }
    else
    {
        statusLabel_->setText("状态: 模型加载失败");
        return;
    }
    
    // 加载标签
    QFile f(labelPath);
    if (f.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        QStringList labels;
        while (!f.atEnd())
        {
            QByteArray line = f.readLine();
            QString s = QString::fromUtf8(line).trimmed();
            if (!s.isEmpty())
            {
                labels.append(s);
            }
        }
        f.close();
        
        // 保存标签并传递给处理器
        currentClassLabels_ = labels;
        vector<string> cls;
        cls.reserve(labels.size());
        for (const QString &qs : labels)
        {
            cls.push_back(qs.toStdString());
        }
        yoloProcessor_->SetClassLabels(cls);
        
        statusLabel_->setText(QString("状态: 模型和标签已加载，共 %1 类").arg(labels.size()));
    }
    else
    {
        statusLabel_->setText("状态: 模型已加载，但标签加载失败");
    }
}

/**
 * @brief YOLOExample析构函数
 */
YOLOExample::~YOLOExample()
{
    // 析构函数，自动释放内存
}

/**
 * @brief 设置UI界面
 * 
 * 创建并布局所有UI控件，包括模型选择、标签加载、图像选择、检测按钮等
 */
void YOLOExample::SetupUI()
{
    setWindowTitle("深度学习 - YOLO");  // 设置窗口标题
    setMinimumSize(800, 600);  // 设置窗口最小尺寸

    QVBoxLayout* mainLayout = new QVBoxLayout(this);  // 主布局

    // 图像选择和置信度设置布局
    QHBoxLayout* imgLayout = new QHBoxLayout();
    
    // 选择图像按钮
    QPushButton* selImg = new QPushButton("选择图像");
    connect(selImg, &QPushButton::clicked, this, &YOLOExample::SelectImage);
    imgLayout->addWidget(selImg);
    
    // 置信度阈值设置
    imgLayout->addWidget(new QLabel("置信度阈值 (%):"));
    confidenceEdit_ = new QLineEdit("50.0");
    confidenceEdit_->setPlaceholderText("0-100");
    confidenceEdit_->setValidator(new QDoubleValidator(0.0, 100.0, 2, this));
    // 置信度编辑框内容改变时自动更新并检测
    connect(confidenceEdit_, &QLineEdit::editingFinished, this, &YOLOExample::UpdateConfidenceThreshold);
    imgLayout->addWidget(confidenceEdit_);
    
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

/**
 * @brief 连接信号和槽
 * 
 * 连接YOLO处理器的信号到UI的槽函数
 */
void YOLOExample::ConnectSignals()
{
    // 连接处理完成信号到UI更新槽
    connect(yoloProcessor_, &YOLOProcessorORT::processingComplete, this, &YOLOExample::OnProcessingComplete);
    // 连接错误发生信号到UI错误显示槽
    connect(yoloProcessor_, &YOLOProcessorORT::errorOccurred, this, &YOLOExample::OnDLError);
}

/**
 * @brief 选择图像
 * 
 * 打开文件对话框选择图像文件，并在UI中显示
 */
void YOLOExample::SelectImage()
{
    // 打开文件对话框选择图像文件
    QString file = QFileDialog::getOpenFileName(this, "选择图像", ".", "Images (*.png *.jpg *.jpeg *.bmp)");
    if (!file.isEmpty()) 
    {
        currentImagePath_ = file;
        
        // 选择图片后自动进行检测
        RunDetect();
    }
}

/**
 * @brief 运行目标检测
 * 
 * 检查模型和图像是否已加载，调用YOLO处理器进行目标检测，绘制检测结果，并保存结果
 */
void YOLOExample::RunDetect()
{
    // 检查模型是否已加载
    if (!yoloProcessor_->IsModelLoaded()) 
    {
        statusLabel_->setText("状态: 请先加载模型");
        return;
    }
    
    // 检查图像是否已选择
    if (currentImagePath_.isEmpty()) 
    {
        statusLabel_->setText("状态: 请先选择图像");
        return;
    }
    
    // 读取图像
    Mat img = imread(currentImagePath_.toStdString());
    if (img.empty()) 
    {
        statusLabel_->setText("状态: 无法读取图像");
        return;
    }

    // 调用DetectObjects方法，使用引用参数获取结果
    std::vector<DetectionResult> results;
    bool success = yoloProcessor_->DetectObjects(img, results);
    if (!success) 
    {
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
    for (const auto &d : results) 
    {
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
    if (jsonFile.open(QIODevice::WriteOnly)) 
    {
        jsonFile.write(doc.toJson(QJsonDocument::Compact));
        jsonFile.close();
    }

    // 显示检测结果
    QPixmap pm = QPixmap::fromImage(QImage(out.data, out.cols, out.rows, out.step, QImage::Format_BGR888).rgbSwapped());
    if (!pm.isNull()) 
    {
        imageLabel_->setPixmap(pm.scaled(imageLabel_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
    
    // 更新状态
    statusLabel_->setText(QString("检测完成: %1 目标").arg(results.size()));
}

/**
 * @brief 处理完成信号槽
 * @param resultImage 处理后的图像
 * 
 * 显示处理完成的图像结果
 */
void YOLOExample::OnProcessingComplete(const cv::Mat& resultImage)
{
    Mat out = resultImage.clone();
    QPixmap pm = QPixmap::fromImage(QImage(out.data, out.cols, out.rows, out.step, QImage::Format_BGR888).rgbSwapped());
    if (!pm.isNull()) 
    {
        imageLabel_->setPixmap(pm.scaled(imageLabel_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
}

/**
 * @brief 更新置信度阈值
 * 
 * 从编辑框获取置信度阈值，更新YOLO处理器的阈值设置，并根据情况重新运行检测
 */
void YOLOExample::UpdateConfidenceThreshold()
{
    // 读取用户输入的置信度阈值
    QString confStr = confidenceEdit_->text();
    bool ok;
    double conf = confStr.toDouble(&ok);
    if (!ok || conf < 0.0 || conf > 100.0) 
    {
        statusLabel_->setText("状态: 置信度阈值无效，请输入0-100之间的数值");
        return;
    }

    float confThreshold = static_cast<float>(conf);
    
    // 更新处理器的置信度阈值，默认NMS阈值为0.45
    yoloProcessor_->SetThresholds(confThreshold, 0.45f);
    
    // 如果已经选择了图像，重新运行检测
    if (!currentImagePath_.isEmpty() && yoloProcessor_->IsModelLoaded()) 
    {
        RunDetect();
        statusLabel_->setText(QString("状态: 置信度阈值已更新为 %1%，重新检测完成").arg(confStr));
    } 
    else 
    {
        statusLabel_->setText(QString("状态: 置信度阈值已更新为 %1%").arg(confStr));
    }
}

/**
 * @brief 错误处理信号槽
 * @param error 错误信息
 * 
 * 显示YOLO处理器发送的错误信息
 */
void YOLOExample::OnDLError(const QString& error)
{
    statusLabel_->setText(QString("错误: %1").arg(error));
}
