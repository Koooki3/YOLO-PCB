#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "MvCamera.h"
#include <QFileDialog>
#include <vector>
#include <mutex>
#include <halconcpp/HalconCpp.h>
#include <Halcon.h>
#include <halconcpp/HDevThread.h>
#include <QPointF>
#include <QVector>
#include <vector>
#include "SystemMonitor.h"
#include <QLabel>

using namespace HalconCpp;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

    void static __stdcall ImageCallBack(unsigned char * pData, MV_FRAME_OUT_INFO_EX* pFrameInfo, void* pUser);
    void ImageCallBackInner(unsigned char * pData, MV_FRAME_OUT_INFO_EX* pFrameInf);
    void appendLog(const QString &message, int logType, double value = 0.0);
    void drawOverlayOnDisplay2(double length, double width, double diagonal);

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

private:
    Ui::MainWindow *ui;

    void *m_hWnd;                          // ch:显示窗口句柄 | en:The Handle of Display Window

    MV_CC_DEVICE_INFO_LIST  m_stDevList;   // ch:设备信息链表 | en:The list of device info
    CMvCamera*              m_pcMyCamera;  // ch:相机类设备实例 | en:The instance of CMvCamera
    bool                    m_bGrabbing;   // ch:是否开始抓图 | en:The flag of Grabbing

    std::vector<unsigned char> m_lastFrame;      // 缓存的原始帧数据
    MV_FRAME_OUT_INFO_EX       m_lastInfo{};     // 帧信息
    std::mutex                 m_frameMtx;       // 互斥锁保护
    bool                       m_hasFrame = false; // 是否已有可用帧

    std::vector<double> Row, Col;
    //HTuple Row, Col;

    QVector<QPointF> WorldCoord;
    QVector<QPointF> PixelCoord;

    // 系统监控相关
    SystemMonitor* m_sysMonitor;
    QLabel* m_cpuLabel;
    QLabel* m_memLabel;
    QLabel* m_tempLabel;

private slots:
    void updateSystemStats(float cpuUsage, float memUsage, float temperature);

};

#endif // MAINWINDOW_H
