/**
 * @file DLExample.h
 * @brief 深度学习二分类演示应用头文件
 * 
 * 该文件定义了DLExample类，提供了一个完整的GUI界面来演示DLProcessor的功能，
 * 包括模型加载、参数配置、单张图像分类、批量图像分类和实时结果显示等功能。
 * 
 * @author VisualRobot Team
 * @date 2025-12-30
 * @version 1.0
 */

#ifndef DLEXAMPLE_H
#define DLEXAMPLE_H

#include <QWidget>
#include <QLineEdit>
#include <QLabel>
#include <QProgressBar>
#include <QStringList>
#include "DLProcessor.h"

using namespace std;

QT_BEGIN_NAMESPACE
class QPushButton;
class QVBoxLayout;
class QHBoxLayout;
class QComboBox;
QT_END_NAMESPACE

/**
 * @brief 深度学习二分类演示应用
 * 
 * 这个类提供了一个完整的GUI界面来演示DLProcessor的功能，包括: 
 * - 模型加载和参数配置
 * - 单张图像分类
 * - 批量图像分类
 * - 实时结果显示
 * - 模型量化功能
 * 
 * @note 该类继承自QWidget，支持Qt的信号槽机制
 * @see DLProcessor
 */
class DLExample : public QWidget
{
    Q_OBJECT

public:
    /**
     * @brief 构造函数
     * 
     * 初始化深度学习示例应用，创建DLProcessor实例，设置UI和连接信号槽
     * 
     * @param parent 父窗口指针，默认为nullptr
     */
    explicit DLExample(QWidget *parent = nullptr);
    
    /**
     * @brief 析构函数
     * 
     * 清理资源，释放内存
     */
    ~DLExample();

private slots:
    /**
     * @brief 浏览并选择模型文件
     * 
     * 打开文件对话框，让用户选择深度学习模型文件（支持ONNX、PB、Caffe等格式）
     * 
     * @note 选择后会自动更新modelPathEdit_文本框
     * @see LoadModel()
     */
    void BrowseModel();
    
    /**
     * @brief 浏览并选择标签文件
     * 
     * 打开文件对话框，让用户选择类别标签文件（文本文件，每行一个标签）
     * 
     * @note 选择后会自动调用LoadLabels()加载标签
     * @see LoadLabels()
     */
    void BrowseLabels();
    
    /**
     * @brief 加载标签文件
     * 
     * 从文件中读取类别标签，并更新DLProcessor的标签列表
     * 
     * @note 会检查文件是否存在，加载成功后更新状态显示
     * @see DLProcessor::LoadClassLabels()
     */
    void LoadLabels();
    
    /**
     * @brief 加载深度学习模型
     * 
     * 从文件中加载深度学习模型，并初始化DLProcessor
     * 
     * @note 会根据模型格式自动查找配置文件（如Caffe的.prototxt）
     * @see DLProcessor::InitModel()
     */
    void LoadModel();
    
    /**
     * @brief 设置模型参数
     * 
     * 从UI控件中获取参数，并设置到DLProcessor
     * 
     * @note 参数包括置信度阈值和输入尺寸
     * @see DLProcessor::SetModelParams()
     * @see DLProcessor::SetInputSize()
     */
    void SetParameters();
    
    /**
     * @brief 选择要分类的图像
     * 
     * 打开文件对话框，让用户选择单张图像文件
     * 
     * @note 选择后会自动在图像显示区域预览
     * @see ClassifyImage()
     */
    void SelectImage();
    
    /**
     * @brief 对单张图像进行分类
     * 
     * 加载当前选择的图像，调用DLProcessor进行分类，并显示结果
     * 
     * @note 分类结果通过信号槽机制异步处理
     * @see OnClassificationComplete()
     */
    void ClassifyImage();
    
    /**
     * @brief 批量分类多张图像
     * 
     * 选择多张图像，调用DLProcessor进行批量分类，并显示结果
     * 
     * @note 会显示进度条，结果通过信号槽机制异步处理
     * @see OnBatchProcessingComplete()
     */
    void BatchClassify();
    
    /**
     * @brief 处理单张图像
     * 
     * 加载当前选择的图像，调用DLProcessor进行分类，并显示结果
     * 
     * @note 这是ClassifyImage()的别名，用于处理按钮点击事件
     * @see ClassifyImage()
     */
    void ProcessImage();
    
    /**
     * @brief 批量处理多张图像
     * 
     * 选择多张图像，调用DLProcessor进行批量分类，并显示结果
     * 
     * @note 这是BatchClassify()的别名，用于处理按钮点击事件
     * @see BatchClassify()
     */
    void BatchProcess();
    
    /**
     * @brief 处理单张图像分类完成信号
     * @param result 分类结果
     * 
     * 显示单张图像的分类结果，包括类别名称、ID、置信度和有效性
     * 
     * @note 该槽函数接收DLProcessor::classificationComplete信号
     * @see DLProcessor::classificationComplete
     */
    void OnClassificationComplete(const ClassificationResult& result);
    
    /**
     * @brief 处理批量分类完成信号
     * @param results 批量分类结果列表
     * 
     * 显示批量分类的结果，包括每张图像的文件名、类别和置信度
     * 
     * @note 该槽函数接收DLProcessor::batchProcessingComplete信号
     * @see DLProcessor::batchProcessingComplete
     */
    void OnBatchProcessingComplete(const vector<ClassificationResult>& results);
    
    /**
     * @brief 处理深度学习错误信号
     * @param error 错误信息
     * 
     * 显示错误消息框，并更新状态标签
     * 
     * @note 该槽函数接收DLProcessor::errorOccurred信号
     * @see DLProcessor::errorOccurred
     */
    void OnDLError(const QString& error);
    
    /**
     * @brief 选择校准图像用于量化
     * 
     * 打开文件对话框，让用户选择用于模型量化的校准图像
     * 
     * @note INT8和UINT8量化需要校准图像
     * @see QuantizeModel()
     */
    void SelectCalibrationImages();
    
    /**
     * @brief 执行模型量化
     * 
     * 对已加载的模型进行量化，减小模型大小，加速推理
     * 
     * @note 支持FP16、INT8、UINT8等量化类型
     * @see DLProcessor::QuantizeModel()
     */
    void QuantizeModel();
    
    /**
     * @brief 任务类型变更时的处理
     * @param index 选择的任务类型索引
     * 
     * 当任务类型改变时，更新UI控件文本
     * 
     * @note 当前仅支持图像分类任务
     */
    void OnTaskTypeChanged(int index);

private:
    /**
     * @brief 初始化用户界面
     * 
     * 创建并布局所有UI组件，包括模型加载、参数设置、图像显示和结果展示区域
     * 
     * @note 该函数在构造函数中调用
     */
    void SetupUI();
    
    /**
     * @brief 连接信号和槽
     * 
     * 连接DLProcessor的信号到DLExample的槽函数，处理分类结果和错误信息
     * 
     * @note 该函数在构造函数中调用
     */
    void ConnectSignals();

private:
    // 核心组件
    DLProcessor* dlProcessor_;          ///< 深度学习处理器，负责模型加载和推理
    bool isModelLoaded_;                ///< 模型是否已加载的标志
    
    // UI组件
    QLineEdit* modelPathEdit_;          ///< 模型路径输入框
    QLineEdit* labelPathEdit_;          ///< 标签文件路径输入框
    QLineEdit* confidenceEdit_;         ///< 置信度阈值输入框
    QLineEdit* inputSizeEdit_;          ///< 输入尺寸输入框
    QProgressBar* progressBar_;         ///< 进度条，用于显示批量处理进度
    QLabel* resultLabel_;               ///< 结果显示标签
    QLabel* imageLabel_;                ///< 图像显示标签
    QLabel* statusLabel_;               ///< 状态显示标签
    QComboBox* taskTypeComboBox_;       ///< 任务类型选择框
    QPushButton* classifyBtn_;          ///< 处理按钮
    QPushButton* batchBtn_;             ///< 批量处理按钮
    
    // 数据
    QString currentImagePath_;          ///< 当前选择的图像路径
    QStringList batchFileNames_;        ///< 批量处理的文件名列表
    
    // 量化相关组件
    QComboBox* quantizationType_;       ///< 量化类型选择框
    QPushButton* calibrationImagesBtn_; ///< 校准图像选择按钮
    QPushButton* quantizeBtn_;          ///< 量化按钮
    QLabel* quantizationStatusLabel_;   ///< 量化状态显示标签
    QStringList calibrationImages_;     ///< 校准图像路径列表
};

#endif // DLEXAMPLE_H
