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

using namespace std;
using namespace cv;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    // 构造函数和析构函数
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

    // 公共方法
    void static __stdcall ImageCallBack(unsigned char * pData, MV_FRAME_OUT_INFO_EX* pFrameInfo, void* pUser);
    void ImageCallBackInner(unsigned char * pData, MV_FRAME_OUT_INFO_EX* pFrameInfo);
    void AppendLog(const QString &message, int logType, double value = 0.0);
    void DrawOverlayOnDisplay2(double length, double width, double diagonal);

private:
    // 基础UI和窗口成员
    Ui::MainWindow *ui;
    void *m_hWnd;                                  // 显示窗口句柄

    // 相机相关成员
    MV_CC_DEVICE_INFO_LIST  m_stDevList;           // 设备信息链表
    CMvCamera*              m_pcMyCamera;          // 相机类设备实例
    bool                    m_bGrabbing;           // 是否开始抓图

    // 图像帧缓存
    vector<unsigned char>   m_lastFrame;           // 缓存的原始帧数据
    MV_FRAME_OUT_INFO_EX    m_lastInfo{};          // 帧信息
    mutex                   m_frameMtx;            // 互斥锁保护
    bool                    m_hasFrame = false;    // 是否已有可用帧

    // 几何坐标转换相关
    vector<double> Row, Col;
    QVector<QPointF> WorldCoord;
    QVector<QPointF> PixelCoord;

    // 系统监控相关
    SystemMonitor* m_sysMonitor;
    QLabel* m_cpuLabel;
    QLabel* m_memLabel;
    QLabel* m_tempLabel;
    QLabel* m_sharpnessLabel;          // 清晰度显示标签

    // 缺陷检测相关
    DefectDetection* m_defectDetection;
    double  m_diffThresh = 25.0;       // 差异二值阈值
    double  m_minDefectArea = 1200;    // 过滤小区域（像素）
    int     m_orbFeatures = 1500;      // ORB特征点数量（配准用）
    
    // 几何参数变换系数
    double  m_biasLength = 0.0;        // 长度变换系数
    double  m_biasWidth = 0.0;         // 宽度变换系数

    // 多边形绘制功能相关
    QVector<QPoint> m_polygonPoints;   // 存储多边形点坐标
    QPixmap m_originalPixmap;          // 原始图片
    bool m_isImageLoaded;              // 标记是否有图片加载
    bool m_polygonCompleted;           // 标记多边形是否完成绘制
    QPixmap m_croppedPixmap;           // 裁剪后的图片
    bool m_hasCroppedImage;            // 标记是否有裁剪图片

    // 矩形拖动选取功能相关
    bool m_isDragging;                 // 标记是否正在拖动
    QPoint m_dragStartPoint;           // 拖动起始点
    QPoint m_dragEndPoint;             // 拖动结束点
    QRect m_selectedRect;              // 选中的矩形区域
    bool m_rectCompleted;              // 标记矩形选择是否完成
    bool m_useRectangleMode;           // 标记当前使用矩形模式(true)或多边形模式(false)

    // 实时检测相关
    bool m_realTimeDetectionRunning;
    QMutex m_realTimeDetectionMutex;

    // 缩放和平移功能相关
    double m_scaleFactor = 1.0;        // 当前缩放因子
    double m_minScaleFactor = 0.1;     // 最小缩放因子
    double m_maxScaleFactor = 5.0;     // 最大缩放因子
    QPoint m_lastPanPos;               // 上次平移位置
    bool m_isPanning = false;          // 是否正在平移

    // 辅助方法
    void ShowErrorMsg(QString csMessage, unsigned int nErrorNum);
    double CalculateTenengradSharpness(const Mat& image);
    bool GrabLastFrameBGR(Mat& outBGR);
    static QPixmap MatToQPixmap(const Mat& bgr);
    bool SetTemplateFromCurrent();
    bool SetTemplateFromFile(const QString& path);
    bool ComputeHomography(const Mat& curGray, Mat& H, vector<DMatch>* dbgMatches=nullptr);
    vector<Rect> DetectDefects(const Mat& curBGR, const Mat& H, Mat* dbgMask=nullptr);

    // 初始化方法
    void SetupPolygonDrawing();                                             // 初始化多边形绘制功能

    // 事件过滤器
    bool eventFilter(QObject* obj, QEvent* event);

    // 鼠标事件处理
    void HandleMouseClickOnDisplay2(const QPoint& pos);                     // 处理鼠标点击
    void HandleMousePressOnDisplay2(const QPoint& pos);                     // 处理鼠标按下
    void HandleMouseMoveOnDisplay2(const QPoint& pos);                      // 处理鼠标移动
    void HandleMouseReleaseOnDisplay2(const QPoint& pos);                   // 处理鼠标释放

    // 平移相关事件处理
    void HandleMousePressForPan(QMouseEvent* event);                        // 处理鼠标按下用于平移
    void HandleMouseMoveForPan(QMouseEvent* event);                         // 处理鼠标移动用于平移
    void HandleMouseReleaseForPan(QMouseEvent* event);                      // 处理鼠标释放结束平移

    // 滚轮事件处理
    void HandleWheelEvent(QWheelEvent* event);                              // 处理滚轮事件

    // 键盘事件处理
    void keyPressEvent(QKeyEvent *event) override;
    void HandleEnterKeyPress();                                             // 处理Enter键按下
    void HandleEscKeyPress();                                               // 处理ESC键按下
    void HandleSpaceKeyPress();                                             // 处理空格键按下
    void HandleQKeyPress();                                                 // 处理Q键，退出实时检测模式

    // 模式切换
    void SwitchSelectionMode();                                             // 切换选择模式

    // 图像绘制和处理
    void DrawPolygonOnImage();                                              // 在图片上绘制多边形
    void DrawRectangleOnImage();                                            // 绘制矩形区域
    void CropImageToPolygon();                                              // 裁剪多边形区域图像
    void CropImageToRectangle();                                            // 裁剪矩形区域图像
    void ScaleImage(double factor);                                         // 缩放图像
    void ResetZoom();                                                       // 重置缩放

    // 坐标转换
    QPoint ConvertToImageCoordinates(const QPoint& widgetPoint);            // 坐标转换

    // 显示相关
    void ClearPolygonDisplay();                                             // 清除多边形显示
    QColor SampleBorderColor(const QImage& image, const QPolygon& polygon); // 取样多边形边缘颜色

    // 实时检测相关方法
    void RealTimeDetectionThread();
    void StartRealTimeDetection();

    // 窗口事件
    void closeEvent(QCloseEvent *event);

signals:
    void sharpnessValueUpdated(double sharpness);

private slots:
    // 相机控制相关槽函数
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

    // 检测和处理相关槽函数
    void on_GetLength_clicked();
    void on_genMatrix_clicked();
    void on_CallDLwindow_clicked();
    void on_setTemplate_clicked();
    void on_detect_clicked();
    void on_btnOpenManual_clicked();

    // 系统更新相关槽函数
    void updateSystemStats(float cpuUsage, float memUsage, float temperature);
    void updateSharpnessDisplay(double sharpness);
};

};

#endif // MAINWINDOW_H
