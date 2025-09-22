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
QT_END_NAMESPACE

/**
 * @brief 深度学习二分类演示应用
 * 
 * 这个类提供了一个完整的GUI界面来演示DLProcessor的功能，包括：
 * - 模型加载和参数配置
 * - 单张图像分类
 * - 批量图像分类
 * - 实时结果显示
 */
class DLExample : public QWidget
{
    Q_OBJECT

public:
    /**
     * @brief 构造函数
     * @param parent 父窗口指针
     */
    explicit DLExample(QWidget *parent = nullptr);
    
    /**
     * @brief 析构函数
     */
    ~DLExample();

private slots:
    /**
     * @brief 浏览并选择模型文件
     */
    void BrowseModel();
    
    /**
     * @brief 浏览并选择标签文件
     */
    void BrowseLabels();
    
    /**
     * @brief 加载标签文件
     */
    void LoadLabels();
    
    /**
     * @brief 加载深度学习模型
     */
    void LoadModel();
    
    /**
     * @brief 设置模型参数
     */
    void SetParameters();
    
    /**
     * @brief 选择要分类的图像
     */
    void SelectImage();
    
    /**
     * @brief 对单张图像进行分类
     */
    void ClassifyImage();
    
    /**
     * @brief 批量分类多张图像
     */
    void BatchClassify();
    
    /**
     * @brief 处理单张图像分类完成信号
     * @param result 分类结果
     */
    void OnClassificationComplete(const ClassificationResult& result);
    
    /**
     * @brief 处理批量分类完成信号
     * @param results 批量分类结果
     */
    void OnBatchProcessingComplete(const vector<ClassificationResult>& results);
    
    /**
     * @brief 处理深度学习错误信号
     * @param error 错误信息
     */
    void OnDLError(const QString& error);

private:
    /**
     * @brief 初始化用户界面
     */
    void SetupUI();
    
    /**
     * @brief 连接信号和槽
     */
    void ConnectSignals();

private:
    // 核心组件
    DLProcessor* dlProcessor_;          ///< 深度学习处理器
    bool isModelLoaded_;                ///< 模型是否已加载
    
    // UI组件
    QLineEdit* modelPathEdit_;          ///< 模型路径输入框
    QLineEdit* labelPathEdit_;          ///< 标签文件路径输入框
    QLineEdit* confidenceEdit_;         ///< 置信度阈值输入框
    QLineEdit* inputSizeEdit_;          ///< 输入尺寸输入框
    QProgressBar* progressBar_;         ///< 进度条
    QLabel* resultLabel_;               ///< 结果显示标签
    QLabel* imageLabel_;                ///< 图像显示标签
    QLabel* statusLabel_;               ///< 状态显示标签
    
    // 数据
    QString currentImagePath_;          ///< 当前选择的图像路径
    QStringList batchFileNames_;        ///< 批量处理的文件名列表
};

#endif // DLEXAMPLE_H
