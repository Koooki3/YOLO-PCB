#ifndef MAINWINDOW_H
#define MAINWINDOW_H

// Qt 核心库
#include <QMainWindow>
#include <QFileDialog>
#include <QPointF>
#include <QVector>
#include <QLabel>
#include <QMutex>
#include <QThread>

// 标准库
#include <vector>
#include <mutex>

// OpenCV 库
#include <opencv2/opencv.hpp>

// 项目自定义库
#include "MvCamera.h"
#include "MvCameraControl.h"
#include "SystemMonitor.h"
#include "DLExample.h"
#include "DefectDetection.h"

using namespace std;
using namespace cv;

namespace Ui {
class MainWindow;
}

/**
 * @brief 主窗口类，负责整个应用程序的界面和功能管理
 * 
 * 该类继承自QMainWindow，负责：
 * - 相机设备管理（枚举、打开、关闭、参数设置等）
 * - 图像采集和显示
 * - 系统监控（CPU、内存、温度）
 * - 图像处理功能（多边形绘制、矩形选择、缺陷检测等）
 * - 日志记录和显示
 */
class MainWindow : public QMainWindow
{
    Q_OBJECT

signals:
    /// @brief 清晰度值更新信号
    /// @param sharpness 计算得到的清晰度值
    void sharpnessValueUpdated(double sharpness);

public:
    /**
     * @brief 构造函数
     * @param parent 父窗口指针
     */
    explicit MainWindow(QWidget *parent = 0);
    
    /**
     * @brief 析构函数
     */
    ~MainWindow();

    // 图像回调相关函数
    void static __stdcall ImageCallBack(unsigned char * pData, MV_FRAME_OUT_INFO_EX* pFrameInfo, void* pUser);
    void ImageCallBackInner(unsigned char * pData, MV_FRAME_OUT_INFO_EX* pFrameInfo);
    
    /**
     * @brief 添加日志信息
     * @param message 日志消息
     * @param logType 日志类型（0:INFO, 1:WARNING, 2:ERROR）
     * @param value 可选数值参数
     */
    void AppendLog(const QString &message, int logType, double value = 0.0);
    
    /**
     * @brief 在显示窗口2上绘制测量结果叠加层
     * @param length 长度值
     * @param width 宽度值
     * @param angle 角度值
     */
    void DrawOverlayOnDisplay2(double length, double width, double angle);

private:
    /**
     * @brief 显示错误信息窗口
     * @param csMessage 错误消息
     * @param nErrorNum 错误代码
     */
    void ShowErrorMsg(QString csMessage, unsigned int nErrorNum);

private slots:
    // 相机设备管理槽函数
    void on_bnEnum_clicked();           ///< 枚举设备按钮点击
    void on_bnOpen_clicked();           ///< 打开设备按钮点击
    void on_bnClose_clicked();          ///< 关闭设备按钮点击
    void on_bnContinuesMode_clicked();  ///< 连续模式按钮点击
    void on_bnTriggerMode_clicked();    ///< 触发模式按钮点击
    void on_bnStart_clicked();          ///< 开始抓图按钮点击
    void on_bnStop_clicked();           ///< 停止抓图按钮点击
    void on_cbSoftTrigger_clicked();    ///< 软件触发复选框点击
    void on_bnTriggerExec_clicked();    ///< 触发执行按钮点击
    void on_bnGetParam_clicked();       ///< 获取参数按钮点击
    void on_bnSetParam_clicked();       ///< 设置参数按钮点击
    
    // 图像处理功能槽函数
    void on_pushButton_clicked();       ///< 保存图片按钮点击
    void on_GetLength_clicked();        ///< 获取长度按钮点击
    void on_genMatrix_clicked();        ///< 生成矩阵按钮点击
    void on_CallDLwindow_clicked();     ///< 调用深度学习窗口按钮点击

private:
    Ui::MainWindow *ui;                            ///< UI界面指针

    // 相机设备相关成员变量
    void *m_hWnd;                                  ///< 显示窗口句柄
    MV_CC_DEVICE_INFO_LIST  m_stDevList;           ///< 设备信息链表
    CMvCamera*              m_pcMyCamera;          ///< 相机类设备实例
    bool                    m_bGrabbing;           ///< 是否开始抓图标志

    // 图像帧缓存相关
    vector<unsigned char>   m_lastFrame;           ///< 缓存的原始帧数据
    MV_FRAME_OUT_INFO_EX    m_lastInfo{};          ///< 帧信息结构体
    mutex                   m_frameMtx;            ///< 帧数据互斥锁保护
    bool                    m_hasFrame = false;    ///< 是否已有可用帧标志

    // 坐标数据相关
    vector<double> Row;                            ///< 行坐标数据
    vector<double> Col;                            ///< 列坐标数据
    //HTuple Row, Col;                             ///< Halcon坐标数据（已注释）

    QVector<QPointF> WorldCoord;                   ///< 世界坐标数据
    QVector<QPointF> PixelCoord;                   ///< 像素坐标数据

    // 系统监控相关成员变量
    SystemMonitor* m_sysMonitor;                   ///< 系统监控对象指针
    QLabel* m_cpuLabel;                            ///< CPU使用率显示标签
    QLabel* m_memLabel;                            ///< 内存使用率显示标签
    QLabel* m_tempLabel;                           ///< 温度显示标签
    QLabel* m_threadLabel;                         ///< 线程信息显示标签
    QLabel* m_sharpnessLabel;                      ///< 清晰度显示标签

    // 缺陷检测相关
    DefectDetection* m_defectDetection;            ///< 缺陷检测库对象指针

private slots:
    /**
     * @brief 更新系统状态显示
     * @param cpuUsage CPU使用率
     * @param memUsage 内存使用率
     * @param temperature 温度
     * @param totalThreads 总线程数
     * @param activeThreads 活跃线程数
     */
    void updateSystemStats(float cpuUsage, float memUsage, float temperature, int totalThreads, int activeThreads);
    
    /**
     * @brief 打开调试手册按钮点击
     */
    void on_btnOpenManual_clicked();
    
    /**
     * @brief 更新清晰度显示
     * @param sharpness 清晰度值
     */
    void updateSharpnessDisplay(double sharpness);

    /**
     * @brief 设置模板按钮点击
     */
    void on_setTemplate_clicked();
    
    /**
     * @brief 缺陷检测按钮点击
     */
    void on_detect_clicked();

private:
    // 清晰度计算相关
    /**
     * @brief 计算Tenengrad清晰度
     * @param image 输入图像
     * @return 清晰度值
     */
    double CalculateTenengradSharpness(const Mat& image);

    // 多边形绘制功能相关成员变量
    QVector<QPoint> m_polygonPoints;                    ///< 存储多边形点坐标
    QPixmap m_originalPixmap;                           ///< 原始图片
    bool m_isImageLoaded;                               ///< 标记是否有图片加载
    bool m_polygonCompleted;                            ///< 标记多边形是否完成绘制
    QPixmap m_croppedPixmap;                            ///< 裁剪后的图片
    bool m_hasCroppedImage;                             ///< 标记是否有裁剪图片

    // 多边形绘制功能相关函数
    bool eventFilter(QObject* obj, QEvent* event);
    void SetupPolygonDrawing();                         ///< 初始化多边形绘制功能
    void HandleMouseClickOnDisplay2(const QPoint& pos); ///< 处理鼠标点击
    void HandleEnterKeyPress();                         ///< 处理Enter键按下
    void DrawPolygonOnImage();                          ///< 在图片上绘制多边形
    QPoint ConvertToImageCoordinates(const QPoint& widgetPoint); ///< 坐标转换
    void CropImageToPolygon();                          ///< 裁剪多边形区域图像
    void ClearPolygonDisplay();                         ///< 清除多边形显示
    QColor SampleBorderColor(const QImage& image, const QPolygon& polygon); ///< 取样多边形边缘颜色
    void HandleEscKeyPress();                           ///< 处理ESC键按下

    // 矩形拖动选取功能相关成员变量
    bool m_isDragging;                                  ///< 标记是否正在拖动
    QPoint m_dragStartPoint;                            ///< 拖动起始点
    QPoint m_dragEndPoint;                              ///< 拖动结束点
    QRect m_selectedRect;                               ///< 选中的矩形区域
    bool m_rectCompleted;                               ///< 标记矩形选择是否完成
    bool m_useRectangleMode;                            ///< 标记当前使用矩形模式(true)或多边形模式(false)

    // 矩形拖动选取功能相关函数
    void HandleMousePressOnDisplay2(const QPoint& pos); ///< 处理鼠标按下
    void HandleMouseMoveOnDisplay2(const QPoint& pos);  ///< 处理鼠标移动
    void HandleMouseReleaseOnDisplay2(const QPoint& pos); ///< 处理鼠标释放
    void DrawRectangleOnImage();                        ///< 绘制矩形区域
    void CropImageToRectangle();                        ///< 裁剪矩形区域图像
    void HandleSpaceKeyPress();                         ///< 处理空格键按下
    void SwitchSelectionMode();                         ///< 切换选择模式

    // 键盘事件处理
    void keyPressEvent(QKeyEvent *event) override;

    // 日志保存
    void closeEvent(QCloseEvent *event);

private:
    // 模板法缺陷检测相关参数（已移至DefectDetection库，保留参数用于兼容性）
    double  m_diffThresh = 25.0;     ///< 差异二值阈值（配准后的 absdiff 后再高斯平滑）
    double  m_minDefectArea = 1200;  ///< 过滤小区域（像素），按你的分辨率可调
    int     m_orbFeatures = 1500;    ///< ORB特征点数量（配准用）

    // 图像处理相关函数
    /**
     * @brief 将缓存的最新一帧转为BGR Mat
     * @param outBGR 输出的BGR图像
     * @return 成功返回true，失败返回false
     */
    bool GrabLastFrameBGR(Mat& outBGR);
    
    /**
     * @brief Mat 转 QPixmap（显示用）
     * @param bgr 输入的BGR图像
     * @return 转换后的QPixmap
     */
    static QPixmap MatToQPixmap(const Mat& bgr);
    
    /**
     * @brief 把当前帧设为模板
     * @return 成功返回true，失败返回false
     */
    bool SetTemplateFromCurrent();
    
    /**
     * @brief 从文件设置模板
     * @param path 模板文件路径
     * @return 成功返回true，失败返回false
     */
    bool SetTemplateFromFile(const QString& path);
    
    /**
     * @brief 配准：计算 H（模板 <- 当前）
     * @param curGray 当前灰度图像
     * @param H 输出的单应性矩阵
     * @param dbgMatches 调试匹配点（可选）
     * @return 成功返回true，失败返回false
     */
    bool ComputeHomography(const Mat& curGray, Mat& H, vector<DMatch>* dbgMatches=nullptr);
    
    /**
     * @brief 检测：根据配准后差异，得到在"当前图像坐标系"的缺陷外接框
     * @param curBGR 当前BGR图像
     * @param H 单应性矩阵
     * @param dbgMask 调试掩码（可选）
     * @return 缺陷外接框列表
     */
    vector<Rect> DetectDefects(const Mat& curBGR, const Mat& H, Mat* dbgMask=nullptr);

    // 实时检测相关成员变量
    bool m_realTimeDetectionRunning;                ///< 实时检测运行标志
    QMutex m_realTimeDetectionMutex;                ///< 实时检测互斥锁

    // 实时检测相关函数
    void RealTimeDetectionThread();                 ///< 实时检测线程函数
    void StartRealTimeDetection();                  ///< 开始实时检测
    void HandleQKeyPress();                         ///< 处理Q键，退出实时检测模式
};

#endif // MAINWINDOW_H
