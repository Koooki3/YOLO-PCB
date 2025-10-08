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

using namespace std;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

signals:
    void sharpnessValueUpdated(double sharpness);

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

    void static __stdcall ImageCallBack(unsigned char * pData, MV_FRAME_OUT_INFO_EX* pFrameInfo, void* pUser);
    void ImageCallBackInner(unsigned char * pData, MV_FRAME_OUT_INFO_EX* pFrameInf);
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

private slots:
    void updateSystemStats(float cpuUsage, float memUsage, float temperature);
    void on_btnOpenManual_clicked();
    void updateSharpnessDisplay(double sharpness);

private:
    // 清晰度计算相关
    double CalculateTenengradSharpness(const cv::Mat& image);

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

    // 日志保存
    void closeEvent(QCloseEvent *event);
};

#endif // MAINWINDOW_H
