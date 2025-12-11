#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "MvCamera.h"
#include <QFileDialog>
#include <vector>
#include <mutex>
#include <QPointF>
#include <QVector>
#include <vector>
#include "SystemMonitor.h"
#include <QLabel>
#include "DLExample.h"
#include "MvCameraControl.h"
#include <opencv2/opencv.hpp>
#include "DefectDetection.h"
#include "FeatureAlignment.h"
#include <QMutex>
#include <QThread>
#include <QTime>
#include "YOLOProcessorORT.h"
#include <QWaitCondition>

using namespace std;
using namespace cv;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

signals:
    void sharpnessValueUpdated(double sharpness);
    void newFrameAvailable();  // 新帧可用信号，用于主线程显示

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

    void static __stdcall ImageCallBack(unsigned char * pData, MV_FRAME_OUT_INFO_EX* pFrameInfo, void* pUser);
    void ImageCallBackInner(unsigned char * pData, MV_FRAME_OUT_INFO_EX* pFrameInfo);
    void AppendLog(const QString &message, int logType, double value = 0.0);
    void DrawOverlayOnDisplay2(double length, double width, double diagonal);

private:
    void ShowErrorMsg(QString csMessage, unsigned int nErrorNum); // ch:显示错误信息窗口 | en: Show the window of error message

private slots:

    void on_bnEnum_clicked();
    void on_bnOpen_clicked();
    void on_bnClose_clicked();
    void on_bnContinuesMode_clicked();
    void on_bnTriggerMode_clicked();
    void on_bnStart_clicked();
    void on_bnStop_clicked();
    void on_cbSoftTrigger_clicked();
    void on_bnTriggerExec_clicked();
    void on_bnGetParam_clicked();
    void on_bnSetParam_clicked();
    void on_pushButton_clicked();
    void on_GetLength_clicked();
    void on_genMatrix_clicked();
    void on_CallDLwindow_clicked();
    // 新增的槽函数
    void on_bnCapture_clicked();                // 拍照按钮点击事件

private:
    Ui::MainWindow *ui;

    void *m_hWnd;                                  // ch:显示窗口句柄 | en:The Handle of Display Window

    MV_CC_DEVICE_INFO_LIST  m_stDevList;           // ch:设备信息链表 | en:The list of device info
    CMvCamera*              m_pcMyCamera;          // ch:相机类设备实例 | en:The instance of CMvCamera
    bool                    m_bGrabbing;           // ch:是否开始抓图 | en:The flag of Grabbing

    vector<unsigned char>   m_lastFrame;           // 缓存的原始帧数据
    MV_FRAME_OUT_INFO_EX    m_lastInfo{};          // 帧信息
    mutex                   m_frameMtx;            // 互斥锁保护
    bool                    m_hasFrame = false;    // 是否已有可用帧

    vector<double> Row, Col;
    //HTuple Row, Col;

    QVector<QPointF> WorldCoord;
    QVector<QPointF> PixelCoord;

    // 系统监控相关
    SystemMonitor* m_sysMonitor;
    QLabel* m_cpuLabel;
    QLabel* m_memLabel;
    QLabel* m_tempLabel;
    QLabel* m_sharpnessLabel;  // 清晰度显示标签

    // 缺陷检测库
    DefectDetection* m_defectDetection;

private slots:
    void updateSystemStats(float cpuUsage, float memUsage, float temperature);
    void on_btnOpenManual_clicked();
    void updateSharpnessDisplay(double sharpness);
    void updateDisplay();  // 主线程图像显示槽

    void on_setTemplate_clicked();   // 可选：从当前帧设模板（或弹框选择文件）
    void on_detect_clicked();        // 你要的检测按钮

private:
    // 清晰度计算相关
    double CalculateTenengradSharpness(const Mat& image);

    // 多边形绘制功能相关
    QVector<QPoint> m_polygonPoints;                                        // 存储多边形点坐标
    QPixmap m_originalPixmap;                                               // 原始图片
    bool m_isImageLoaded;                                                   // 标记是否有图片加载
    bool m_polygonCompleted;                                                // 标记多边形是否完成绘制
    QPixmap m_croppedPixmap;                                                // 裁剪后的图片
    bool m_hasCroppedImage;                                                 // 标记是否有裁剪图片
    bool eventFilter(QObject* obj, QEvent* event);
    void SetupPolygonDrawing();                                             // 初始化多边形绘制功能
    void HandleMouseClickOnDisplay2(const QPoint& pos);                     // 处理鼠标点击
    void HandleEnterKeyPress();                                             // 处理Enter键按下
    void DrawPolygonOnImage();                                              // 在图片上绘制多边形
    QPoint ConvertToImageCoordinates(const QPoint& widgetPoint);            // 坐标转换
    void CropImageToPolygon();                                              // 裁剪多边形区域图像
    void ClearPolygonDisplay();                                             // 清除多边形显示
    QColor SampleBorderColor(const QImage& image, const QPolygon& polygon); // 取样多边形边缘颜色
    void HandleEscKeyPress();                                               // 处理ESC键按下

    // 矩形拖动选取功能相关
    bool m_isDragging;                                    // 标记是否正在拖动
    QPoint m_dragStartPoint;                              // 拖动起始点
    QPoint m_dragEndPoint;                                // 拖动结束点
    QRect m_selectedRect;                                 // 选中的矩形区域
    bool m_rectCompleted;                                 // 标记矩形选择是否完成
    bool m_useRectangleMode;                              // 标记当前使用矩形模式(true)或多边形模式(false)
    void HandleMousePressOnDisplay2(const QPoint& pos);   // 处理鼠标按下
    void HandleMouseMoveOnDisplay2(const QPoint& pos);    // 处理鼠标移动
    void HandleMouseReleaseOnDisplay2(const QPoint& pos); // 处理鼠标释放
    void DrawRectangleOnImage();                          // 绘制矩形区域
    void CropImageToRectangle();                          // 裁剪矩形区域图像
    void HandleSpaceKeyPress();                           // 处理空格键按下
    void SwitchSelectionMode();                           // 切换选择模式

    // 键盘事件处理
    void keyPressEvent(QKeyEvent *event) override;

    // 日志保存
    void closeEvent(QCloseEvent *event);

private:
    // 模板法缺陷检测相关参数（已移至DefectDetection库，保留参数用于兼容性）
    double  m_diffThresh = 25.0;     // 差异二值阈值（配准后的 absdiff 后再高斯平滑）
    double  m_minDefectArea = 1200;  // 过滤小区域（像素），按你的分辨率可调
    int     m_orbFeatures = 1500;    // ORB特征点数量（配准用）
    
    // 几何参数变换系数
    double  m_biasLength = 0.0;      // 长度变换系数
    double  m_biasWidth = 0.0;       // 宽度变换系数

    // 将缓存的最新一帧转为BGR Mat（经SDK内存编码为JPEG后imdecode，稳妥）
    bool GrabLastFrameBGR(Mat& outBGR);
    // Mat 转 QPixmap（显示用）
    static QPixmap MatToQPixmap(const Mat& bgr);
    // 把当前帧设为模板（从当前帧或文件）
    bool SetTemplateFromCurrent();
    bool SetTemplateFromFile(const QString& path);
    // 配准：计算 H（模板 <- 当前）
    bool ComputeHomography(const Mat& curGray, Mat& H, vector<DMatch>* dbgMatches=nullptr);
    // 检测：根据配准后差异，得到在"当前图像坐标系"的缺陷外接框
    vector<Rect> DetectDefects(const Mat& curBGR, const Mat& H, Mat* dbgMask=nullptr);

    // 实时检测相关
    bool m_realTimeDetectionRunning;
    QMutex m_realTimeDetectionMutex;
    QThread* m_realTimeDetectionThread;  // 实时检测线程指针
    void RealTimeDetectionThread();
    void StartRealTimeDetection();
    void StopRealTimeDetection();  // 新增停止函数
    void HandleQKeyPress(); // 处理Q键, 退出实时检测模式
    
    // 日志优化相关
    int m_lastDefectCount;        // 上次记录的缺陷数量
    QTime m_lastLogTime;          // 上次记录日志的时间
    QTime m_stableStateStartTime; // 当前稳定状态开始时间
    bool m_isStableState;         // 是否处于稳定状态
    
    // 设备热拔插自动枚举相关
    QTimer* m_deviceEnumTimer;    // 定时器用于自动枚举设备
    int m_lastDeviceCount;        // 上次枚举的设备数量
    void autoEnumDevices();       // 自动枚举设备的槽函数
    
    // YOLO实时检测相关
    YOLOProcessorORT* m_yoloProcessor;  // YOLO处理器实例
    bool m_yoloDetectionRunning;        // YOLO检测运行标志
    QMutex m_yoloDetectionMutex;        // YOLO检测互斥锁
    QThread* m_yoloDetectionThread;     // YOLO检测线程
    QTimer* m_yoloStatsTimer;           // YOLO统计信息定时器
    
    // YOLO显示线程相关
    bool m_yoloDisplayRunning;          // YOLO显示运行标志
    QMutex m_yoloDisplayMutex;          // YOLO显示互斥锁
    QThread* m_yoloDisplayThread;       // YOLO显示线程
    int m_displayUpdateInterval;        // 显示更新间隔（毫秒）
    
    // YOLO检测结果双缓冲
    struct YoloResultBuffer {
        Mat frame;                               // 检测的帧
        vector<DetectionResult> results;         // 检测结果
        double processingTime;                   // 处理时间
        bool newDataAvailable;                   // 是否有新数据
    };
    
    YoloResultBuffer m_yoloBuffers[2];           // 双缓冲
    int m_writeBufferIndex;                      // 当前写入缓冲区索引
    int m_readBufferIndex;                       // 当前读取缓冲区索引
    QMutex m_bufferMutex;                        // 缓冲区互斥锁
    QWaitCondition m_bufferReady;                // 缓冲区就绪条件变量
    
    // 显示定时器
    QTimer* m_displayTimer;                     // 显示更新定时器
    
    // YOLO统计信息
    int m_yoloFrameCount;                       // 帧计数
    QTime m_yoloStatsStartTime;                 // 统计开始时间
    double m_yoloTotalProcessingTime;           // 总处理时间
    
    void YoloRealTimeDetectionThread();         // YOLO实时检测线程
    void YoloDisplayThread();                   // YOLO显示线程
    void StartYoloRealTimeDetection();          // 开始YOLO实时检测
    void StopYoloRealTimeDetection();           // 停止YOLO实时检测
    void UpdateYoloStats();                     // 更新YOLO统计信息
    void DrawYoloResults(Mat& frame, const vector<DetectionResult>& results); // 绘制YOLO检测结果
    void UpdateYoloDisplay();                   // 更新YOLO显示
    
};

#endif // MAINWINDOW_H
