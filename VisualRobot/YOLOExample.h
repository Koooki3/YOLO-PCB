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

/**
 * @brief YOLO示例窗口类
 * 
 * 该类实现了YOLO目标检测的GUI界面，包括模型加载、图像选择、检测执行等功能
 * 用户可以通过该界面加载ONNX模型、选择图像、设置置信度阈值，并查看检测结果
 */
class YOLOExample : public QWidget
{
    Q_OBJECT

public:
    /**
     * @brief 构造函数
     * @param parent 父窗口指针
     */
    explicit YOLOExample(QWidget *parent = nullptr);
    
    /**
     * @brief 析构函数
     */
    ~YOLOExample();

private slots:
    /**
     * @brief 浏览模型文件
     */
    void BrowseModel();
    
    /**
     * @brief 加载模型
     */
    void LoadModel();
    
    /**
     * @brief 加载标签文件
     */
    void LoadLabels();
    
    /**
     * @brief 选择图像
     */
    void SelectImage();
    
    /**
     * @brief 运行目标检测
     */
    void RunDetect();
    
    /**
     * @brief 更新置信度阈值
     */
    void UpdateConfidenceThreshold();
    
    /**
     * @brief 处理完成信号槽
     * @param resultImage 处理后的图像
     */
    void OnProcessingComplete(const cv::Mat& resultImage);
    
    /**
     * @brief 错误处理信号槽
     * @param error 错误信息
     */
    void OnDLError(const QString& error);

private:
    /**
     * @brief 设置UI界面
     */
    void SetupUI();
    
    /**
     * @brief 连接信号和槽
     */
    void ConnectSignals();

    YOLOProcessorORT* yoloProcessor_;   // YOLO处理器指针
    QLineEdit* modelPathEdit_;          // 模型路径编辑框
    QLineEdit* labelPathEdit_;          // 标签路径编辑框
    QLineEdit* confidenceEdit_;         // 置信度阈值编辑框
    QLabel* statusLabel_;               // 状态标签
    QLabel* imageLabel_;                // 图像显示标签
    QPushButton* detectBtn_;            // 检测按钮
    QString currentImagePath_;          // 当前图像路径
    QStringList currentClassLabels_;     // 当前类别标签列表
};

#endif // YOLOEXAMPLE_H
