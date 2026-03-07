/**
 * @file YOLOExample.h
 * @brief YOLO目标检测示例模块头文件
 * 
 * 该文件定义了YOLOExample类，提供YOLO目标检测的GUI界面实现，
 * 包括模型加载、图像选择、检测执行、结果显示等功能。
 * 
 * @author VisualRobot Team
 * @date 2025-12-30
 * @version 1.0
 */

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
 * 
 * @note 该类使用YOLOProcessorORT进行底层的目标检测处理
 * @note 支持自动加载配置文件中的模型和标签路径
 * @note 检测结果会自动保存为图像文件和JSON格式的标注文件
 * @see YOLOProcessorORT, ConfigManager
 */
class YOLOExample : public QWidget
{
    Q_OBJECT

public:
    /**
     * @brief 构造函数
     * 
     * 初始化YOLO示例窗口，创建YOLO处理器实例，设置UI界面，连接信号和槽
     * 
     * @param parent 父窗口指针，默认为nullptr
     * @note 构造函数执行流程：
     *       1. 创建YOLO处理器实例
     *       2. 调用SetupUI()设置界面
     *       3. 调用ConnectSignals()连接信号槽
     *       4. 从配置管理器获取模型和标签路径
     *       5. 自动加载ONNX模型和类别标签
     * @see SetupUI(), ConnectSignals(), YOLOProcessorORT::InitModel()
     */
    explicit YOLOExample(QWidget *parent = nullptr);
    
    /**
     * @brief 析构函数
     * 
     * 清理资源，YOLO处理器会自动释放内存
     * 
     * @note 由于使用Qt的父子对象机制，大部分资源会自动释放
     */
    ~YOLOExample();

private slots:
    /**
     * @brief 选择图像槽函数
     * 
     * 打开文件对话框选择图像文件，更新当前图像路径，并自动运行目标检测
     * 
     * @note 支持的图像格式：PNG, JPG, JPEG, BMP
     * @note 选择图像后会自动调用RunDetect()进行检测
     * @see RunDetect()
     */
    void SelectImage();
    
    /**
     * @brief 运行目标检测槽函数
     * 
     * 执行完整的目标检测流程，包括图像读取、检测、结果绘制和保存
     * 
     * @note 检测流程：
     *       1. 检查模型和图像是否已加载
     *       2. 读取图像并调用YOLO处理器进行检测
     *       3. 绘制检测结果到图像
     *       4. 保存结果图像和JSON标注文件
     *       5. 在UI中显示结果
     * @note 结果文件命名规则：
     *       - 图像：原文件名_detections.jpg
     *       - JSON：原文件名_detections.json
     * @see YOLOProcessorORT::DetectObjects(), YOLOProcessorORT::DrawDetectionResults()
     */
    void RunDetect();
    
    /**
     * @brief 更新置信度阈值槽函数
     * 
     * 从置信度编辑框读取值，更新YOLO处理器的阈值设置，并根据情况重新运行检测
     * 
     * @note 阈值范围：0-100%
     * @note 如果已加载图像，会自动重新运行检测
     * @note 无效输入会显示错误信息
     * @see YOLOProcessorORT::SetThresholds()
     */
    void UpdateConfidenceThreshold();
    
    /**
     * @brief 处理完成信号槽
     * 
     * 接收YOLO处理器的处理完成信号，显示处理结果
     * 
     * @param resultImage 处理后的图像（包含检测框）
     * @note 将OpenCV Mat转换为QPixmap并在UI中显示
     * @see YOLOProcessorORT::processingComplete
     */
    void OnProcessingComplete(const cv::Mat& resultImage);
    
    /**
     * @brief 错误处理信号槽
     * 
     * 接收YOLO处理器的错误信号，显示错误信息
     * 
     * @param error 错误信息字符串
     * @note 在状态标签中显示错误信息
     * @see YOLOProcessorORT::errorOccurred
     */
    void OnDLError(const QString& error);

private:
    /**
     * @brief 设置UI界面
     * 
     * 创建并布局所有UI控件，包括图像选择按钮、置信度设置、图像显示区域、状态标签等
     * 
     * @note 创建的控件：
     *       - 选择图像按钮
     *       - 置信度阈值编辑框（带数值验证器）
     *       - 图像显示标签（带边框）
     *       - 状态标签
     * @note 窗口最小尺寸：800x600
     * @note 置信度编辑框使用QDoubleValidator限制输入范围0-100
     */
    void SetupUI();
    
    /**
     * @brief 连接信号和槽
     * 
     * 连接YOLO处理器的信号到UI的槽函数
     * 
     * @note 连接的信号：
     *       - processingComplete → OnProcessingComplete
     *       - errorOccurred → OnDLError
     * @see OnProcessingComplete(), OnDLError()
     */
    void ConnectSignals();

    YOLOProcessorORT* yoloProcessor_;   ///< YOLO处理器指针，负责底层检测逻辑
    QLineEdit* confidenceEdit_;         ///< 置信度阈值编辑框，用户输入百分比值
    QLabel* statusLabel_;               ///< 状态标签，显示当前状态和错误信息
    QLabel* imageLabel_;                ///< 图像显示标签，显示原始图像和检测结果
    QString currentImagePath_;          ///< 当前选择的图像路径
    QStringList currentClassLabels_;    ///< 当前类别标签列表
};

#endif // YOLOEXAMPLE_H
