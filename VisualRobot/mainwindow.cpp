#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QMessageBox>
#include <QStyleFactory>
#include <QTextCodec>
#include <stdio.h>
#include <ctime>
#include <QDir>
#include <QDateTime>
#include <QImage>
#include <cstdlib>
#include <QStandardPaths>
#include <QFile>
#include <QScrollBar>
#include <memory>
#include "DIP.h"
#include <QPointF>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include "eigen3/Eigen/Dense"
#include <QPainter>
#include <QFont>
#include <QFontMetrics>
#include <QDesktopServices>
#include <QUrl>
#include <QCoreApplication>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <string.h>
#include <QPainterPath>

#define ERROR 2
#define WARNNING 1
#define INFO 0

using namespace Eigen;
using namespace std;
using namespace cv;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    // 初始化UI
    ui->setupUi(this);

    // 初始化相机相关变量
    memset(&m_stDevList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
    m_pcMyCamera = NULL;
    m_bGrabbing = false;
    m_hWnd = (void*)ui->widgetDisplay->winId();

    // 初始化系统监控
    m_sysMonitor = new SystemMonitor(this);
    
    // 创建系统信息标签
    m_cpuLabel = new QLabel(this);
    m_memLabel = new QLabel(this);
    m_tempLabel = new QLabel(this);
    m_sharpnessLabel = new QLabel(this);

    // 设置标签样式
    QString labelStyle = "QLabel { color: white; background-color: rgba(0, 0, 0, 150); padding: 5px; border-radius: 5px; }";
    m_cpuLabel->setStyleSheet(labelStyle);
    m_memLabel->setStyleSheet(labelStyle);
    m_tempLabel->setStyleSheet(labelStyle);
    m_sharpnessLabel->setStyleSheet(labelStyle);

    // 添加标签到状态栏
    statusBar()->addWidget(m_cpuLabel);
    statusBar()->addWidget(m_memLabel);
    statusBar()->addWidget(m_tempLabel);
    statusBar()->addWidget(m_sharpnessLabel);

    // 连接监控信号
    connect(m_sysMonitor, &SystemMonitor::systemStatsUpdated, this, &MainWindow::updateSystemStats);

    // 启动监控
    m_sysMonitor->startMonitoring(1000); // 每秒更新一次

    ui->genMatrix->setEnabled(false);

    // 连接清晰度信号
    connect(this, &MainWindow::sharpnessValueUpdated, this, &MainWindow::updateSharpnessDisplay);
    
    // 初始化多边形绘制功能
    SetupPolygonDrawing();
}

MainWindow::~MainWindow()
{
    if (m_sysMonitor) 
    {
        m_sysMonitor->stopMonitoring();
    }
    delete ui;
}

// 显示错误信息
void MainWindow::ShowErrorMsg(QString csMessage, unsigned int nErrorNum)
{
    // 变量定义
    QString errorMsg = csMessage;  // 错误消息字符串
    QString TempMsg;               // 临时消息字符串

    if (nErrorNum != 0)
    {
        TempMsg.asprintf(": Error = %x: ", nErrorNum);
        errorMsg += TempMsg;
    }

    switch(nErrorNum)
    {
    case MV_E_HANDLE:           errorMsg += "Error or invalid handle ";                                         break;
    case MV_E_SUPPORT:          errorMsg += "Not supported function ";                                          break;
    case MV_E_BUFOVER:          errorMsg += "Cache is full ";                                                   break;
    case MV_E_CALLORDER:        errorMsg += "Function calling order error ";                                    break;
    case MV_E_PARAMETER:        errorMsg += "Incorrect parameter ";                                             break;
    case MV_E_RESOURCE:         errorMsg += "Applying resource failed ";                                        break;
    case MV_E_NODATA:           errorMsg += "No data ";                                                         break;
    case MV_E_PRECONDITION:     errorMsg += "Precondition error, or running environment changed ";              break;
    case MV_E_VERSION:          errorMsg += "Version mismatches ";                                              break;
    case MV_E_NOENOUGH_BUF:     errorMsg += "Insufficient memory ";                                             break;
    case MV_E_ABNORMAL_IMAGE:   errorMsg += "Abnormal image, maybe incomplete image because of lost packet ";   break;
    case MV_E_UNKNOW:           errorMsg += "Unknown error ";                                                   break;
    case MV_E_GC_GENERIC:       errorMsg += "General error ";                                                   break;
    case MV_E_GC_ACCESS:        errorMsg += "Node accessing condition error ";                                  break;
    case MV_E_ACCESS_DENIED:	errorMsg += "No permission ";                                                   break;
    case MV_E_BUSY:             errorMsg += "Device is busy, or network disconnected ";                         break;
    case MV_E_NETER:            errorMsg += "Network error ";                                                   break;
    }

    QMessageBox::information(NULL, "PROMPT", errorMsg);
}

void __stdcall MainWindow::ImageCallBack(unsigned char * pData, MV_FRAME_OUT_INFO_EX* pFrameInfo, void* pUser)
{
    if (pUser)
    {
        MainWindow* pMainWindow = static_cast<MainWindow*>(pUser);  // 将 pUser 强制转换为 MainWindow 指针
        pMainWindow->ImageCallBackInner(pData, pFrameInfo);  // 调用内部处理函数
    }

    // 1) 可选：仍然显示
    if (pUser)
    {
        MainWindow* pMainWindow = static_cast<MainWindow*>(pUser);  // 确保pUser非空后再访问
        MV_DISPLAY_FRAME_INFO disp{};
        disp.hWnd        = pMainWindow->m_hWnd;
        disp.pData       = pData;
        disp.nDataLen    = pFrameInfo->nFrameLen;
        disp.nWidth      = pFrameInfo->nWidth;
        disp.nHeight     = pFrameInfo->nHeight;
        disp.enPixelType = pFrameInfo->enPixelType;
        if (pMainWindow->m_pcMyCamera)
        { 
            pMainWindow->m_pcMyCamera->DisplayOneFrame(&disp);
        }
    }

    // 2) 缓存：深拷贝最新一帧到成员变量
    vector<unsigned char> tempFrame;  // 临时变量用于清晰度计算
    if (pUser)
    {
        MainWindow* pMainWindow = static_cast<MainWindow*>(pUser);  // 再次获取 MainWindow 指针
        lock_guard<mutex> lk(pMainWindow->m_frameMtx);
        pMainWindow->m_lastFrame.resize(pFrameInfo->nFrameLen);
        memcpy(pMainWindow->m_lastFrame.data(), pData, pFrameInfo->nFrameLen);
        pMainWindow->m_lastInfo = *pFrameInfo;   // 结构体按值拷贝
        pMainWindow->m_hasFrame = true;

        // 拷贝一份到临时变量，用于清晰度计算
        tempFrame = pMainWindow->m_lastFrame;
    }

    // 3) 计算清晰度并发射信号
    if (pUser && !tempFrame.empty())
    {
        MainWindow* pMainWindow = static_cast<MainWindow*>(pUser);
        cv::Mat grayImage;
        // 根据像素类型转换到灰度图
        switch(pFrameInfo->enPixelType) 
        {
            case PixelType_Gvsp_Mono8:
                grayImage = cv::Mat(pFrameInfo->nHeight, pFrameInfo->nWidth, CV_8UC1, tempFrame.data());
                break;
            case PixelType_Gvsp_RGB8_Packed:
                {
                    cv::Mat colorImage = cv::Mat(pFrameInfo->nHeight, pFrameInfo->nWidth, CV_8UC3, tempFrame.data());
                    cv::cvtColor(colorImage, grayImage, cv::COLOR_RGB2GRAY);
                }
                break;
            case PixelType_Gvsp_BGR8_Packed:
                {
                    cv::Mat colorImage = cv::Mat(pFrameInfo->nHeight, pFrameInfo->nWidth, CV_8UC3, tempFrame.data());
                    cv::cvtColor(colorImage, grayImage, cv::COLOR_BGR2GRAY);
                }
                break;
            default:
                // 不支持的格式，跳过清晰度计算
                return;
        }
        double sharpness = pMainWindow->CalculateTenengradSharpness(grayImage);
        // 发射信号
        emit pMainWindow->sharpnessValueUpdated(sharpness);
    }
}

void MainWindow::ImageCallBackInner(unsigned char * pData, MV_FRAME_OUT_INFO_EX* pFrameInfo)
{
    // 变量定义
    MV_DISPLAY_FRAME_INFO stDisplayInfo;  // 显示帧信息结构体

    memset(&stDisplayInfo, 0, sizeof(MV_DISPLAY_FRAME_INFO));

    stDisplayInfo.hWnd = m_hWnd;
    stDisplayInfo.pData = pData;
    stDisplayInfo.nDataLen = pFrameInfo->nFrameLen;
    stDisplayInfo.nWidth = pFrameInfo->nWidth;
    stDisplayInfo.nHeight = pFrameInfo->nHeight;
    stDisplayInfo.enPixelType = pFrameInfo->enPixelType;
    m_pcMyCamera->DisplayOneFrame(&stDisplayInfo);
}

void MainWindow::on_bnEnum_clicked()
{
    // 变量定义
    int nRet;                            // 返回值变量
    unsigned int i;                      // 循环索引
    MV_CC_DEVICE_INFO* pDeviceInfo;      // 设备信息指针
    char strUserName[256] = {0};         // 设备名称字符串

    ui->ComboDevices->clear();
    QTextCodec::setCodecForLocale(QTextCodec::codecForName("GBK"));
    ui->ComboDevices->setStyle(QStyleFactory::create("Windows"));
    
    // ch:枚举子网内所有设备 | en:Enumerate all devices within subnet
    memset(&m_stDevList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
    nRet = CMvCamera::EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE | MV_GENTL_CAMERALINK_DEVICE | MV_GENTL_CXP_DEVICE | MV_GENTL_XOF_DEVICE, &m_stDevList);
    
    if (MV_OK != nRet)
    {
        AppendLog("枚举设备错误", ERROR);
        return;
    }
    
    AppendLog(QString("枚举到 %1 个设备").arg(m_stDevList.nDeviceNum), INFO);
    
    // ch:将值加入到信息列表框中并显示出来 | en:Add value to the information list box and display
    for (i = 0; i < m_stDevList.nDeviceNum; i++)
    {
        pDeviceInfo = m_stDevList.pDeviceInfo[i];
        if (NULL == pDeviceInfo)
        {
            continue;
        }
        
        if (pDeviceInfo->nTLayerType == MV_GIGE_DEVICE)
        {
            int nIp1 = ((m_stDevList.pDeviceInfo[i]->SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24);
            int nIp2 = ((m_stDevList.pDeviceInfo[i]->SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16);
            int nIp3 = ((m_stDevList.pDeviceInfo[i]->SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8);
            int nIp4 = (m_stDevList.pDeviceInfo[i]->SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff);

            if (strcmp("", (char*)pDeviceInfo->SpecialInfo.stGigEInfo.chUserDefinedName) != 0)
            {
                snprintf(strUserName, 256, "[%d]GigE:   %s (%s) (%d.%d.%d.%d)", i, pDeviceInfo->SpecialInfo.stGigEInfo.chUserDefinedName,
                         pDeviceInfo->SpecialInfo.stGigEInfo.chSerialNumber, nIp1, nIp2, nIp3, nIp4);
            }
            else
            {
                snprintf(strUserName, 256, "[%d]GigE:   %s (%s) (%d.%d.%d.%d)", i, pDeviceInfo->SpecialInfo.stGigEInfo.chModelName,
                         pDeviceInfo->SpecialInfo.stGigEInfo.chSerialNumber, nIp1, nIp2, nIp3, nIp4);
            }
        }
        else if (pDeviceInfo->nTLayerType == MV_USB_DEVICE)
        {
            if (strcmp("", (char*)pDeviceInfo->SpecialInfo.stUsb3VInfo.chUserDefinedName) != 0)
            {
                snprintf(strUserName, 256, "[%d]UsbV3:  %s (%s)", i, pDeviceInfo->SpecialInfo.stUsb3VInfo.chUserDefinedName,
                         pDeviceInfo->SpecialInfo.stUsb3VInfo.chSerialNumber);
            }
            else
            {
                snprintf(strUserName, 256, "[%d]UsbV3:  %s (%s)", i, pDeviceInfo->SpecialInfo.stUsb3VInfo.chModelName,
                         pDeviceInfo->SpecialInfo.stUsb3VInfo.chSerialNumber);
            }
        }
        else if (pDeviceInfo->nTLayerType == MV_GENTL_CAMERALINK_DEVICE)
        {
            if (strcmp("", (char*)pDeviceInfo->SpecialInfo.stCMLInfo.chUserDefinedName) != 0)
            {
                snprintf(strUserName, 256, "[%d]CML:  %s (%s)", i, pDeviceInfo->SpecialInfo.stCMLInfo.chUserDefinedName,
                         pDeviceInfo->SpecialInfo.stCMLInfo.chSerialNumber);
            }
            else
            {
                snprintf(strUserName, 256, "[%d]CML:  %s (%s)", i, pDeviceInfo->SpecialInfo.stCMLInfo.chModelName,
                         pDeviceInfo->SpecialInfo.stCMLInfo.chSerialNumber);
            }
        }
        else if (pDeviceInfo->nTLayerType == MV_GENTL_CXP_DEVICE)
        {
            if (strcmp("", (char*)pDeviceInfo->SpecialInfo.stCXPInfo.chUserDefinedName) != 0)
            {
                snprintf(strUserName, 256, "[%d]CXP:  %s (%s)", i, pDeviceInfo->SpecialInfo.stCXPInfo.chUserDefinedName,
                         pDeviceInfo->SpecialInfo.stCXPInfo.chSerialNumber);
            }
            else
            {
                snprintf(strUserName, 256, "[%d]CXP:  %s (%s)", i, pDeviceInfo->SpecialInfo.stCXPInfo.chModelName,
                         pDeviceInfo->SpecialInfo.stCXPInfo.chSerialNumber);
            }
        }
        else if (pDeviceInfo->nTLayerType == MV_GENTL_XOF_DEVICE)
        {
            if (strcmp("", (char*)pDeviceInfo->SpecialInfo.stXoFInfo.chUserDefinedName) != 0)
            {
                snprintf(strUserName, 256, "[%d]XOF:  %s (%s)", i, pDeviceInfo->SpecialInfo.stXoFInfo.chUserDefinedName,
                         pDeviceInfo->SpecialInfo.stXoFInfo.chSerialNumber);
            }
            else
            {
                snprintf(strUserName, 256, "[%d]XOF:  %s (%s)", i, pDeviceInfo->SpecialInfo.stXoFInfo.chModelName,
                         pDeviceInfo->SpecialInfo.stXoFInfo.chSerialNumber);
            }
        }
        else
        {
            ShowErrorMsg("Unknown device enumerated", 0);
            AppendLog("未知设备被枚举", WARNNING);
        }
        ui->ComboDevices->addItem(QString::fromLocal8Bit(strUserName));
    }

    if (0 == m_stDevList.nDeviceNum)
    {
        ShowErrorMsg("No device", 0);
        AppendLog("没有设备", WARNNING);
        return;
    }
    ui->ComboDevices->setCurrentIndex(0);
}

void MainWindow::on_bnOpen_clicked()
{
    // 变量定义
    int nIndex;        // 设备索引
    int nRet;          // 返回值
    unsigned int nPacketSize;  // 数据包大小

    nIndex = ui->ComboDevices->currentIndex();
    if ((nIndex < 0) | (nIndex >= MV_MAX_DEVICE_NUM))
    {
        ShowErrorMsg("Please select device", 0);
        AppendLog("请选择设备", WARNNING);
        return;
    }

    // ch:由设备信息创建设备实例 | en:Device instance created by device information
    if (NULL == m_stDevList.pDeviceInfo[nIndex])
    {
        ShowErrorMsg("Device does not exist", 0);
        AppendLog("设备不存在", ERROR);
        return;
    }

    if (m_pcMyCamera == NULL)
    {
        m_pcMyCamera = new (nothrow) CMvCamera;
        if (NULL == m_pcMyCamera)
        {
            ShowErrorMsg("new CMvCamera Instance failed", MV_E_RESOURCE);
            AppendLog("新建相机实例失败", ERROR);
            return;
        }
    }

    nRet = m_pcMyCamera->Open(m_stDevList.pDeviceInfo[nIndex]);
    if (MV_OK != nRet)
    {
        delete m_pcMyCamera;
        m_pcMyCamera = NULL;
        ShowErrorMsg("Open Fail", nRet);
        AppendLog("相机打开失败", ERROR);
        return;
    }

    AppendLog("设备打开成功", INFO);

    // ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
    if (m_stDevList.pDeviceInfo[nIndex]->nTLayerType == MV_GIGE_DEVICE)
    {
        nRet = m_pcMyCamera->GetOptimalPacketSize(&nPacketSize);
        if (nRet == MV_OK)
        {
            nRet = m_pcMyCamera->SetIntValue("GevSCPSPacketSize",nPacketSize);
            if(nRet != MV_OK)
            {
                ShowErrorMsg("Warning: Set Packet Size fail!", nRet);
            }
        }
        else
        {
            ShowErrorMsg("Warning: Get Packet Size fail!", nRet);
        }
    }

    m_pcMyCamera->SetEnumValue("AcquisitionMode", MV_ACQ_MODE_CONTINUOUS);
    m_pcMyCamera->SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF);

    on_bnGetParam_clicked(); // ch:获取参数 | en:Get Parameter

    ui->bnOpen->setEnabled(false);
    ui->bnClose->setEnabled(true);
    ui->bnStart->setEnabled(true);
    ui->bnStop->setEnabled(false);
    ui->bnContinuesMode->setEnabled(true);
    ui->bnContinuesMode->setChecked(true);
    ui->bnTriggerMode->setEnabled(true);
    ui->cbSoftTrigger->setEnabled(false);
    ui->bnTriggerExec->setEnabled(false);

    ui->tbExposure->setEnabled(true);
    ui->tbGain->setEnabled(true);
    ui->tbFrameRate->setEnabled(true);
    ui->bnSetParam->setEnabled(true);
    ui->bnGetParam->setEnabled(true);
}

void MainWindow::on_bnClose_clicked()
{
    if (m_pcMyCamera)
    {
        m_pcMyCamera->Close();
        delete m_pcMyCamera;
        m_pcMyCamera = NULL;
    }
    m_bGrabbing = false;

    ui->bnOpen->setEnabled(true);
    ui->bnClose->setEnabled(false);
    ui->bnStart->setEnabled(false);
    ui->bnStop->setEnabled(false);
    ui->bnContinuesMode->setEnabled(false);
    ui->bnTriggerMode->setEnabled(false);
    ui->cbSoftTrigger->setEnabled(false);
    ui->bnTriggerExec->setEnabled(false);

    ui->tbExposure->setEnabled(false);
    ui->tbGain->setEnabled(false);
    ui->tbFrameRate->setEnabled(false);
    ui->bnSetParam->setEnabled(false);
    ui->bnGetParam->setEnabled(false);

    AppendLog("设备关闭成功", INFO);
}

void MainWindow::on_bnContinuesMode_clicked()
{
    if (true == ui->bnContinuesMode->isChecked())
    {
        m_pcMyCamera->SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF);
        ui->cbSoftTrigger->setEnabled(false);
        ui->bnTriggerExec->setEnabled(false);
    }
    AppendLog("连续模式已开启", INFO);
}

void MainWindow::on_bnTriggerMode_clicked()
{
    if (true == ui->bnTriggerMode->isChecked())
    {
        m_pcMyCamera->SetEnumValue("TriggerMode", MV_TRIGGER_MODE_ON);
        if (true == ui->cbSoftTrigger->isChecked())
        {
            m_pcMyCamera->SetEnumValue("TriggerSource", MV_TRIGGER_SOURCE_SOFTWARE);
            if (m_bGrabbing)
            {
                ui->bnTriggerExec->setEnabled(true);
            }
        }
        else
        {
            m_pcMyCamera->SetEnumValue("TriggerSource", MV_TRIGGER_SOURCE_LINE0);
        }
        ui->cbSoftTrigger->setEnabled(true);
    }
    AppendLog("触发模式已开启", INFO);
}

void MainWindow::on_bnStart_clicked()
{
    // 变量定义
    int nRet;  // 返回值

    m_pcMyCamera->RegisterImageCallBack(ImageCallBack, this);

    nRet = m_pcMyCamera->StartGrabbing();
    if (MV_OK != nRet)
    {
        ShowErrorMsg("Start grabbing fail", nRet);
        AppendLog("开始抓图失败", ERROR);
        return;
    }
    AppendLog("开始抓图成功", INFO);
    m_bGrabbing = true;

    ui->bnStart->setEnabled(false);
    ui->bnStop->setEnabled(true);
    ui->pushButton->setEnabled(true);
    ui->GetLength->setEnabled(false);
    if (true == ui->bnTriggerMode->isChecked() && ui->cbSoftTrigger->isChecked())
    {
        ui->bnTriggerExec->setEnabled(true);
    }
}

void MainWindow::on_bnStop_clicked()
{
    // 变量定义
    int nRet;  // 返回值

    nRet = m_pcMyCamera->StopGrabbing();
    if (MV_OK != nRet)
    {
        ShowErrorMsg("Stop grabbing fail", nRet);
        AppendLog("停止抓图失败", ERROR);
        return;
    }
    AppendLog("停止抓图成功", INFO);
    m_bGrabbing = false;

    ui->bnStart->setEnabled(true);
    ui->bnStop->setEnabled(false);
    ui->bnTriggerExec->setEnabled(false);
    ui->pushButton->setEnabled(false);
    ui->GetLength->setEnabled(false);
}

void MainWindow::on_cbSoftTrigger_clicked()
{
    if (true == ui->cbSoftTrigger->isChecked())
    {
        m_pcMyCamera->SetEnumValue("TriggerSource", MV_TRIGGER_SOURCE_SOFTWARE);
        if (m_bGrabbing)
        {
            ui->bnTriggerExec->setEnabled(true);
            AppendLog("软件触发已开启", INFO);
        }
    }
    else
    {
        m_pcMyCamera->SetEnumValue("TriggerSource", MV_TRIGGER_SOURCE_LINE0);
        AppendLog("软件触发已关闭", INFO);
        ui->bnTriggerExec->setEnabled(false);
    }
}

void MainWindow::on_bnTriggerExec_clicked()
{
    // 变量定义
    int nRet;  // 返回值

    nRet = m_pcMyCamera->CommandExecute("TriggerSoftware");
    if(MV_OK != nRet)
    {
        ShowErrorMsg("Trigger Software fail", nRet);
        AppendLog("软件触发失败", ERROR);
    }
    else
    {
        AppendLog("软件触发成功", INFO);
    }
}

void MainWindow::on_bnGetParam_clicked()
{
    // 变量定义
    MVCC_FLOATVALUE stFloatValue;  // 浮点数值结构体
    int nRet;                      // 返回值

    memset(&stFloatValue, 0, sizeof(MVCC_FLOATVALUE));

    nRet = m_pcMyCamera->GetFloatValue("ExposureTime", &stFloatValue);
    if (MV_OK != nRet)
    {
        ShowErrorMsg("Get Exposure Time Fail", nRet);
        AppendLog("获取曝光时间失败", ERROR);
    }
    else
    {
        ui->tbExposure->setText(QString("%1").arg(stFloatValue.fCurValue));
        AppendLog(QString("获取曝光时间成功: %1").arg(stFloatValue.fCurValue), INFO);
    }

    nRet = m_pcMyCamera->GetFloatValue("Gain", &stFloatValue);
    if (MV_OK != nRet)
    {
        ShowErrorMsg("Get Gain Fail", nRet);
        AppendLog("获取增益失败", ERROR);
    }
    else
    {
        ui->tbGain->setText(QString("%1").arg(stFloatValue.fCurValue));
        AppendLog(QString("获取增益成功: %1").arg(stFloatValue.fCurValue), INFO);
    }

    nRet = m_pcMyCamera->GetFloatValue("ResultingFrameRate", &stFloatValue);
    if (MV_OK != nRet)
    {
        ShowErrorMsg("Get Frame Rate Fail", nRet);
        AppendLog("获取帧率失败", ERROR);
    }
    else
    {
        ui->tbFrameRate->setText(QString("%1").arg(stFloatValue.fCurValue));
        AppendLog(QString("获取帧率成功: %1").arg(stFloatValue.fCurValue), INFO);
    }
}

void MainWindow::on_bnSetParam_clicked()
{
    // 变量定义
    int nRet;  // 返回值

    m_pcMyCamera->SetEnumValue("ExposureAuto", 0);
    nRet = m_pcMyCamera->SetFloatValue("ExposureTime", ui->tbExposure->text().toFloat());
    if (MV_OK != nRet)
    {
        ShowErrorMsg("Set Exposure Time Fail", nRet);
        AppendLog("设置曝光时间失败", ERROR);
    }
    else
    {
        AppendLog(QString("设置曝光时间成功: %1").arg(ui->tbExposure->text()), INFO);
    }

    m_pcMyCamera->SetEnumValue("GainAuto", 0);
    nRet = m_pcMyCamera->SetFloatValue("Gain", ui->tbGain->text().toFloat());
    if (MV_OK != nRet)
    {
        ShowErrorMsg("Set Gain Fail", nRet);
        AppendLog("设置增益失败", ERROR);
    }
    else
    {
        AppendLog(QString("设置增益成功: %1").arg(ui->tbGain->text()), INFO);
    }

    nRet = m_pcMyCamera->SetFloatValue("AcquisitionFrameRate", ui->tbFrameRate->text().toFloat());
    if (MV_OK != nRet)
    {
        ShowErrorMsg("Set Frame Rate Fail", nRet);
        AppendLog("设置帧率失败", ERROR);
    }
    else
    {
        AppendLog(QString("设置帧率成功: %1").arg(ui->tbFrameRate->text()), INFO);
    }
}

void MainWindow::on_pushButton_clicked()
{
    // 变量定义
    QMessageBox msgBox;                          // 消息对话框
    QPushButton *cornerButton;                   // 角点检测按钮
    QPushButton *circleButton;                   // 最小外接矩形检测按钮
    QPushButton *cancelButton;                   // 取消按钮
    QAbstractButton *clicked;                    // 被点击的按钮
    bool needDetection;                          // 是否需要检测
    vector<unsigned char> frame;                 // 图像帧数据
    MV_FRAME_OUT_INFO_EX info{};                 // 帧信息
    unsigned int dstMax;                         // 目标缓冲区大小
    unique_ptr<unsigned char[]> pDst;            // 目标缓冲区指针
    MV_SAVE_IMAGE_PARAM_EX3 save{};              // 保存图像参数
    QString dir;                                 // 目录路径
    QString fpath;                               // 文件路径
    QFile f;                                     // 文件对象
    int nRet;                                    // 返回值
    int ProcessedOK;                             // 处理结果
    QString imagePath;                           // 图像路径
    QPixmap pixmap;                              // 图像对象
    QPixmap scaledPixmap;                        // 缩放后的图像

    if (!m_pcMyCamera)
    {
        QMessageBox::warning(this, "保存图片", "相机对象无效！");
        AppendLog("相机对象无效", ERROR);
        return;
    }

    // 创建选择对话框
    msgBox.setWindowTitle("选择检测模式");
    msgBox.setText("请选择检测模式：");
    cornerButton = msgBox.addButton("角点检测模式", QMessageBox::ActionRole);
    circleButton = msgBox.addButton("最小外接矩形检测模式", QMessageBox::ActionRole);
    cancelButton = msgBox.addButton(QMessageBox::Cancel);
    
    msgBox.exec();

    clicked = msgBox.clickedButton();
    if (clicked == cancelButton) 
    {
        return;
    }

    needDetection = ((clicked == cornerButton) || (clicked != circleButton));
    

    // 取出一份缓存快照
    {
        lock_guard<mutex> lk(m_frameMtx);
        if (!m_hasFrame || m_lastFrame.empty())
        {
            QMessageBox::warning(this, "保存图片", "暂无可用图像，请先开始采集。");
            AppendLog("暂无可用图像，请先开始采集", WARNNING);
            return;
        }
        frame = m_lastFrame;   // 拷贝到本地变量，避免持锁编码
        info  = m_lastInfo;
    }

    // 预分配编码缓冲（给足空间）
    dstMax = info.nWidth * info.nHeight * 3 + 4096;
    pDst = make_unique<unsigned char[]>(dstMax);
    if (!pDst)
    {
        QMessageBox::warning(this, "保存图片", "内存不足（编码缓冲）！");
        AppendLog("内存不足（编码缓冲）", ERROR);
        return;
    }

    // 让 SDK 负责像素转换 + JPEG/PNG 编码（与头文件一致）
    save.enImageType   = MV_Image_Jpeg;              // 也可 MV_Image_Png
    save.enPixelType   = info.enPixelType;           // 源像素格式（SDK内部转）
    save.nWidth        = info.nWidth;
    save.nHeight       = info.nHeight;
    save.nDataLen      = info.nFrameLen;
    save.pData         = frame.data();
    save.pImageBuffer  = pDst.get();
    save.nImageLen     = dstMax;                     // 入参为目标缓冲大小
    save.nJpgQuality   = 90;

    nRet = m_pcMyCamera->SaveImage(&save);
    if (MV_OK != nRet || save.nImageLen == 0)
    {
        ShowErrorMsg("保存失败（编码阶段）", nRet);
        AppendLog("保存失败（编码阶段）", ERROR);
        return;
    }

    // 生成保存路径
    dir = "/home/orangepi/Desktop/VisualRobot_Local/Img/";
    fpath = QDir(dir).filePath(QString("capture.jpg"));

    f.setFileName(fpath);
    if (!f.open(QIODevice::WriteOnly))
    {
        QMessageBox::warning(this, "保存图片", "无法打开文件进行写入！");
        AppendLog("保存图片，无法打开文件进行写入", ERROR);
        return;
    }
    f.write(reinterpret_cast<const char*>(pDst.get()), save.nImageLen);
    f.close();

    QMessageBox::information(this, "保存图片", QString("保存成功：%1").arg(fpath));
    AppendLog(QString("保存成功，地址为：%1").arg(fpath), INFO);
    ui->GetLength->setEnabled(true);
    m_hasCroppedImage = false;
    m_polygonCompleted = false;

    // 如果选择了角点检测，才执行检测算法
    if (needDetection) 
    {
        //OpenCV版本
        ProcessedOK = DetectRectangleOpenCV(fpath.toStdString(), Row, Col);
        cout << "原始数据" << endl;
        cout << "Row = [";
        for (size_t i = 0; i < Row.size(); ++i)
        {
            cout << Row[i] << (i < Row.size()-1 ? "," : "");
        }
        cout << "]" << endl;
        for (size_t i = 0; i < Col.size(); ++i)
        {
            cout << Col[i] << (i < Col.size()-1 ? "," : "");
        }
        cout << "]" << endl;
        if (ProcessedOK)
        {
            AppendLog("角点检测算法执行失败，请调整曝光或者重选待测物体", ERROR);
            return;
        }
        else
        {
            AppendLog("待检测图像读取成功，角点检测算法执行完毕", INFO);
        }

        if (Row.size() != Col.size())
        {
            AppendLog("获取到的x,y参数数量不匹配", ERROR);
            return;
        }
        else if (Row.empty() || Col.empty())
        {
            AppendLog("未能正确检测到角点", ERROR);
            return;
        }
        else
        {
            for (size_t i = 0; i < Row.size(); ++i)
            {
                AppendLog(QString("第%1个角点(x,y)像素坐标为:(%2, %3)").arg(i+1).arg(Col[i]).arg(Row[i]), INFO);
            }
        }

        imagePath = "../detectedImg.jpg";

        // 加载图片
        pixmap = QPixmap(imagePath);
        if (pixmap.isNull()) 
        {
            QMessageBox::warning(this, "加载图片失败", "无法加载保存的图片！");
            return;
        }

        // 缩放图片以适应QLabel大小，保持宽高比
        scaledPixmap = pixmap.scaled(ui->widgetDisplay_2->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
        ui->widgetDisplay_2->setPixmap(scaledPixmap);
        ui->widgetDisplay_2->setAlignment(Qt::AlignCenter);

        AppendLog("检测后图像显示成功", INFO);
    }
    else
    {
        imagePath = "../Img/capture.jpg";

        // 加载图片
        pixmap = QPixmap(imagePath);
        if (pixmap.isNull())
        {
            QMessageBox::warning(this, "加载图片失败", "无法加载保存的图片！");
            return;
        }

        // 缩放图片以适应QLabel大小，保持宽高比
        scaledPixmap = pixmap.scaled(ui->widgetDisplay_2->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
        ui->widgetDisplay_2->setPixmap(scaledPixmap);
        ui->widgetDisplay_2->setAlignment(Qt::AlignCenter);

        AppendLog("原始图像显示成功", INFO);
    }
}

// 调试信息打印函数（输入：调试信息（QString）+宏定义调试信息等级）
void MainWindow::AppendLog(const QString &message, int logType, double value)
{
    // 变量定义
    QString timeStamp;      // 时间戳字符串
    QString fullMessage;    // 完整消息字符串
    QTextCursor cursor;     // 文本光标
    QTextCharFormat format; // 文本格式
    QScrollBar *scrollbar;  // 滚动条指针

    // 添加时间戳
    timeStamp = QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss : ");
    fullMessage = timeStamp + message;

    // 检查value是否为有效值（非零或非默认值），您可以根据需要调整条件
    // 这里使用了一个简单的检查：如果value不是0.0，则追加它
    // 注意：这可能不适用于所有情况，您可能需要更精确的检查（例如与NaN比较）
    if (value != 0.0) 
    {
        fullMessage += " " + QString::number(value);
    }

    // 获取文本光标
    cursor = QTextCursor(ui->displayLogMsg->document());
    cursor.movePosition(QTextCursor::End);

    // 根据调试信息等级配置文本颜色 (ERROR: 2, WARNNING: 1, INFO: 0)
    switch(logType)
    {
    case 0:
        format.setForeground(Qt::black);              // 黑色为信息
        break;
    case 1:
        format.setForeground(QColor(255, 140, 0));    // 橙色为警告
        break;
    case 2:
        format.setForeground(Qt::red);                // 红色为错误
        break;
    default:
        break;
    }

    // 插入带颜色格式的调试信息
    cursor.insertText(fullMessage + "\n", format);

    // 滚动条设置
    scrollbar = ui->displayLogMsg->verticalScrollBar();
    if (scrollbar) 
    {
        scrollbar->setValue(scrollbar->maximum());
    }
}

void MainWindow::on_GetLength_clicked()
{
    // 变量定义
    string inputPath;        // 输入图像路径
    string outputPath;       // 输出图像路径
    QString output;          // 输出图像路径(QString)
    Mat inputImage;          // 输入图像
    Params params;           // 处理参数
    double bias;             // 偏差值
    Result result;           // 处理结果
    bool success;            // 保存成功标志
    QPixmap pixmap;          // 图像对象
    QPixmap scaledPixmap;    // 缩放后的图像

    ui->GetLength->setEnabled(false);

    // 情况一：如果多边形未完成或没有裁剪图像，清除多边形显示并处理原始图像
    if (!m_polygonCompleted || !m_hasCroppedImage) 
    {
        QElapsedTimer timer;     // 计时器
        timer.start();           // 开始计时

        ClearPolygonDisplay();
        
        inputPath = "/home/orangepi/Desktop/VisualRobot_Local/Img/capture.jpg";
        outputPath = "../detectedImg.jpg";
        output = "../detectedImg.jpg";

        // 读取输入图像
        inputImage = imread(inputPath);
        if (inputImage.empty()) 
        {
            cerr << "无法读取输入图像: " << inputPath << endl;
            AppendLog("无法读取输入图像", ERROR);
            return;
        }

        // 设置参数（可根据需要修改）
        params.thresh = 127;
        params.maxval = 255;
        params.blurK = 5;
        params.areaMin = 100.0;

        // 处理图像
        bias = 1.0;
        result = CalculateLength(inputImage, params, bias);

        // 保存输出图像
        if (!result.image.empty()) 
        {
            success = imwrite(outputPath, result.image);
            if (success) 
            {
                cout << "输出图像已保存到: " << outputPath << endl;
            } 
            else 
            {
                cerr << "保存输出图像失败: " << outputPath << endl;
                return;
            }
        } 
        else 
        {
            cerr << "处理后的图像为空，无法保存" << endl;
            return;
        }

        qint64 elapsed = timer.elapsed(); // 获取经过的时间（毫秒）
        AppendLog(QString("图像处理时间（毫秒）：%1").arg(elapsed), INFO);

        // 加载图片
        pixmap = QPixmap(output);
        if (pixmap.isNull()) 
        {
            QMessageBox::warning(this, "加载图片失败", "无法加载保存的图片！");
            return;
        }

        // 缩放图片以适应QLabel大小，保持宽高比
        scaledPixmap = pixmap.scaled(ui->widgetDisplay_2->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
        ui->widgetDisplay_2->setPixmap(scaledPixmap);
        ui->widgetDisplay_2->setAlignment(Qt::AlignCenter);

        AppendLog("检测后图像显示成功", INFO);
        AppendLog("检长算法执行完成", INFO);
        AppendLog(QString("物件长度（mm）：%1").arg((double)result.heights[0]), INFO);
        AppendLog(QString("物件宽度（mm）：%1").arg((double)result.widths[0]), INFO);
        AppendLog(QString("物件倾角（°）：%1").arg((double)result.angles[0]), INFO);

        DrawOverlayOnDisplay2((double)result.heights[0], (double)result.widths[0], (double)result.angles[0]);
    }
    // 情况二：如果多边形已完成且有裁剪图像，处理裁剪后的图像
    else 
    {
        QElapsedTimer timer;     // 计时器
        timer.start();           // 开始计时

        AppendLog("检测到多边形区域，将处理裁剪后的图像", INFO);
        
        // 保存裁剪后的图像到临时文件
        QString tempCroppedPath = "../Img/cropped_polygon.jpg";
        m_croppedPixmap.save(tempCroppedPath);
        
        inputPath = tempCroppedPath.toStdString();
        outputPath = "../detectedImg.jpg";
        output = "../detectedImg.jpg";

        // 读取裁剪后的图像
        inputImage = imread(inputPath);
        if (inputImage.empty()) 
        {
            cerr << "无法读取裁剪后的图像: " << inputPath << endl;
            AppendLog("无法读取裁剪后的图像", ERROR);
            return;
        }

        // 设置参数（可根据需要修改）
        params.thresh = 127;
        params.maxval = 255;
        params.blurK = 5;
        params.areaMin = 100.0;

        // 处理图像
        bias = 1.0;
        result = CalculateLength(inputImage, params, bias);

        // 保存输出图像
        if (!result.image.empty()) 
        {
            success = imwrite(outputPath, result.image);
            if (success) 
            {
                cout << "输出图像已保存到: " << outputPath << endl;
            } 
            else 
            {
                cerr << "保存输出图像失败: " << outputPath << endl;
                return;
            }
        } 
        else 
        {
            cerr << "处理后的图像为空，无法保存" << endl;
            return;
        }

        qint64 elapsed = timer.elapsed(); // 获取经过的时间（毫秒）
        AppendLog(QString("裁剪区域图像处理时间（毫秒）：%1").arg(elapsed), INFO);

        // 加载图片
        pixmap = QPixmap(output);
        if (pixmap.isNull()) 
        {
            QMessageBox::warning(this, "加载图片失败", "无法加载保存的图片！");
            return;
        }

        // 缩放图片以适应QLabel大小，保持宽高比
        scaledPixmap = pixmap.scaled(ui->widgetDisplay_2->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
        ui->widgetDisplay_2->setPixmap(scaledPixmap);
        ui->widgetDisplay_2->setAlignment(Qt::AlignCenter);

        AppendLog("裁剪区域检测后图像显示成功", INFO);
        AppendLog("裁剪区域检长算法执行完成", INFO);
        AppendLog(QString("裁剪区域物件长度（mm）：%1").arg((double)result.heights[0]), INFO);
        AppendLog(QString("裁剪区域物件宽度（mm）：%1").arg((double)result.widths[0]), INFO);
        AppendLog(QString("裁剪区域物件倾角（°）：%1").arg((double)result.angles[0]), INFO);

        DrawOverlayOnDisplay2((double)result.heights[0], (double)result.widths[0], (double)result.angles[0]);
    }
}

void MainWindow::on_genMatrix_clicked()
{
    // 变量定义
    int getCoordsOk;                // 坐标获取结果
    Matrix3d transformationMatrix;  // 变换矩阵
    int result;                     // 计算结果
    QString matrixStr;              // 矩阵字符串
    int i;                          // 循环索引
    Vector3d pixelHomogeneous;      // 像素齐次坐标
    Vector3d worldTransformed;      // 变换后的世界坐标
    double x_transformed;           // 变换后的X坐标
    double y_transformed;           // 变换后的Y坐标
    double error_x;                 // X方向误差
    double error_y;                 // Y方向误差
    double total_error;             // 总误差
    QString message;                // 消息字符串

    WorldCoord.clear();
    PixelCoord.clear();
    getCoordsOk = GetCoordsOpenCV(WorldCoord, PixelCoord, 100.0);
    if (getCoordsOk != 0)
    {
        AppendLog("坐标获取错误", ERROR);
    }

    // 在命令行显示坐标结果
    cout << "=== 坐标检测结果 ===" << endl;
    cout << "检测到的坐标对数量: " << WorldCoord.size() << endl << endl;

    cout << "世界坐标 (单位:mm):" << endl;
    cout << "索引\tX坐标\t\tY坐标" << endl;
    cout << "----\t------\t\t------" << endl;
    for (i = 0; i < WorldCoord.size(); i++)
    {
        cout << i << "\t"
             << fixed << setprecision(3) << WorldCoord[i].x() << "\t\t"
             << fixed << setprecision(3) << WorldCoord[i].y() << endl;
    }

    cout << endl << "像素坐标 (单位:像素):" << endl;
    cout << "索引\tX坐标\t\tY坐标" << endl;
    cout << "----\t------\t\t------" << endl;
    for (i = 0; i < PixelCoord.size(); i++)
    {
        cout << i << "\t"
             << fixed << setprecision(1) << PixelCoord[i].x() << "\t\t"
             << fixed << setprecision(1) << PixelCoord[i].y() << endl;
    }
    cout << "==========================" << endl << endl;

    // 调用函数计算变换矩阵并保存到文件
    result = CalculateTransformationMatrix(WorldCoord, PixelCoord, transformationMatrix, "../matrix.bin");

    if (result == 0)
    {
        cout << "变换矩阵计算并保存成功!" << endl;
        AppendLog("变换矩阵计算并保存成功", INFO);

        // 使用变换矩阵将像素坐标转换回世界坐标
        cout << endl << "=== 使用变换矩阵转换像素坐标 ===" << endl;
        cout << "索引\t原始世界坐标\t\t转换后坐标\t\t误差" << endl;
        cout << "----\t------------\t\t------------\t\t------" << endl;

        for (i = 0; i < PixelCoord.size(); i++)
        {
            // 将像素坐标转换为齐次坐标 (x, y, 1)
            pixelHomogeneous = Vector3d(PixelCoord[i].x(), PixelCoord[i].y(), 1.0);

            // 应用变换矩阵
            worldTransformed = transformationMatrix * pixelHomogeneous;

            // 转换为非齐次坐标 (除以w分量)
            x_transformed = worldTransformed[0] / worldTransformed[2];
            y_transformed = worldTransformed[1] / worldTransformed[2];

            // 计算误差
            error_x = fabs(WorldCoord[i].x() - x_transformed);
            error_y = fabs(WorldCoord[i].y() - y_transformed);
            total_error = sqrt(error_x * error_x + error_y * error_y);

            // 输出结果
            cout << i << "\t"
                 << fixed << setprecision(3) << "(" << WorldCoord[i].x() << "," << WorldCoord[i].y() << ")"
                 << "\t\t(" << x_transformed << "," << y_transformed << ")"
                 << "\t\t" << total_error << " mm" << endl;
        }
        cout << "=============================================" << endl;

        // 首先显示变换矩阵
        matrixStr = "变换矩阵:\n";
        for (int i = 0; i < 3; i++) 
        {
            matrixStr += "| ";
            for (int j = 0; j < 3; j++) 
            {
                matrixStr += QString::number(transformationMatrix(i, j), 'g', 15) + " ";
            }
            matrixStr += "|\n";
        }
        AppendLog(matrixStr, INFO);

        // 然后显示误差结果
        for (i = 0; i < PixelCoord.size(); i++) 
        {
            // 将像素坐标转换为齐次坐标 (x, y, 1)
            pixelHomogeneous = Vector3d(PixelCoord[i].x(), PixelCoord[i].y(), 1.0);

            // 应用变换矩阵
            worldTransformed = transformationMatrix * pixelHomogeneous;

            // 转换为非齐次坐标 (除以w分量)
            x_transformed = worldTransformed[0] / worldTransformed[2];
            y_transformed = worldTransformed[1] / worldTransformed[2];

            // 计算误差
            error_x = fabs(WorldCoord[i].x() - x_transformed);
            error_y = fabs(WorldCoord[i].y() - y_transformed);
            total_error = sqrt(error_x * error_x + error_y * error_y);

            // 创建格式化的输出消息
            message = QString("点 %1: 理论世界坐标(%2, %3) -> 变换后世界坐标(%4, %5)").arg(i).arg(WorldCoord[i].x(), 0, 'f', 3).arg(WorldCoord[i].y(), 0, 'f', 3).arg(x_transformed, 0, 'f', 3).arg(y_transformed, 0, 'f', 3);

            // 调用日志函数显示结果
            AppendLog(message, INFO, total_error); // 使用信息级别，并将误差作为value传递
        }
    }
    else
    {
        cout << "变换矩阵计算失败，错误码: " << result << endl;
        AppendLog(QString("变换矩阵计算失败，错误码:%1").arg(result), ERROR);
    }
}

void MainWindow::on_btnOpenManual_clicked()
{
    // 变量定义
    QString pdfPath;  // PDF文件路径

    // 构建PDF文件的路径
    pdfPath = QDir(QCoreApplication::applicationDirPath()).filePath("../Doc/调试信息手册.pdf");
    
    // 尝试使用系统默认程序打开PDF
    QDesktopServices::openUrl(QUrl::fromLocalFile(pdfPath));
    
    AppendLog("已尝试打开调试信息手册", INFO);
}

void MainWindow::DrawOverlayOnDisplay2(double length, double width, double angle)
{
    // 变量定义
    QPixmap src;                    // 源图像
    QPixmap annotated;              // 标注后的图像
    QPainter p;                     // 绘图对象
    QString text;                   // 显示文本
    QFont font;                     // 字体
    QFontMetrics fm;                // 字体度量
    const int margin = 10;          // 边距
    const int pad = 8;              // 内边距
    QRect textRect;                 // 文本矩形区域
    QRect boxRect;                  // 框矩形区域

    // 使用value方式获取pixmap
    src = ui->widgetDisplay_2->pixmap(Qt::ReturnByValue);
    if (src.isNull()) 
    {
        AppendLog("没有可叠加的图像（widgetDisplay_2 为空）", WARNNING);
        return;
    }

    annotated = src.copy();
    p.begin(&annotated);
    p.setRenderHint(QPainter::Antialiasing, true);
    p.setRenderHint(QPainter::TextAntialiasing, true);

    text = QString("长: %1 mm\n宽: %2 mm\n角度: %3 °").arg(length,   0, 'f', 3).arg(width,    0, 'f', 3).arg(angle, 0, 'f', 3);

    font.setPointSize(12); 
    font.setBold(true);
    p.setFont(font);

    fm = QFontMetrics(font);
    textRect = fm.boundingRect(QRect(0, 0, annotated.width()/2, annotated.height()), Qt::AlignRight | Qt::AlignTop | Qt::TextWordWrap, text);

    boxRect = QRect(annotated.width() - textRect.width() - 2*pad - margin, margin, textRect.width() + 2*pad, textRect.height() + 2*pad);

    // 半透明底框
    p.setPen(Qt::NoPen);
    p.setBrush(QColor(0, 0, 0, 140));
    p.drawRoundedRect(boxRect, 8, 8);

    // 红色文字
    p.setPen(QColor(255, 0, 0));
    p.drawText(boxRect.adjusted(pad, pad, -pad, -pad), Qt::AlignRight | Qt::AlignTop, text);

    p.end();
    ui->widgetDisplay_2->setPixmap(annotated);
}

void MainWindow::on_CallDLwindow_clicked()
{
    // 变量定义
    DLExample* dlExample;  // 深度学习示例窗口指针

    dlExample = new DLExample(nullptr);
    dlExample->setAttribute(Qt::WA_DeleteOnClose);
    dlExample->show();
    AppendLog("深度学习二分类示例窗口已打开", INFO);
}

// Tenengrad清晰度计算函数
double MainWindow::CalculateTenengradSharpness(const cv::Mat& image)
{
    // 变量定义
    cv::Mat imageGrey;      // 灰度图像
    cv::Mat imageSobel;     // Sobel梯度图像
    double meanValue;       // 平均值

    // 转换为灰度图
    if (image.channels() == 3) 
    {
        cv::cvtColor(image, imageGrey, cv::COLOR_BGR2GRAY);
    } 
    else 
    {
        imageGrey = image.clone();
    }
    
    // 计算Sobel梯度
    cv::Sobel(imageGrey, imageSobel, CV_16U, 1, 1);
    
    // 计算梯度的平方和
    meanValue = cv::mean(imageSobel)[0];
    
    return meanValue;
}

// 更新清晰度显示
void MainWindow::updateSharpnessDisplay(double sharpness)
{
    // 变量定义
    QString sharpnessText;  // 清晰度文本

    // 更新状态栏的清晰度标签
    if (m_sharpnessLabel) 
    {
        sharpnessText = QString("清晰度: %1").arg(sharpness, 0, 'f', 2);
        m_sharpnessLabel->setText(sharpnessText);
    }
}

// 初始化多边形绘制功能
void MainWindow::SetupPolygonDrawing()
{
    // 初始化成员变量
    m_polygonPoints.clear();
    m_isImageLoaded = false;
    m_polygonCompleted = false;
    m_hasCroppedImage = false;
    
    // 初始化矩形拖动选取变量
    m_isDragging = false;
    m_rectCompleted = false;
    
    // 设置widgetDisplay_2接受鼠标和键盘事件
    ui->widgetDisplay_2->setMouseTracking(true);
    ui->widgetDisplay_2->setFocusPolicy(Qt::StrongFocus); // 允许获得焦点
    ui->widgetDisplay_2->installEventFilter(this);
    
    AppendLog("多边形绘制和矩形拖动功能已初始化", INFO);
}

// 事件过滤器，用于捕获widgetDisplay_2的鼠标点击和键盘事件
bool MainWindow::eventFilter(QObject* obj, QEvent* event)
{
    // 变量定义
    QMouseEvent* mouseEvent;  // 鼠标事件
    QKeyEvent* keyEvent;      // 键盘事件

    if (obj == ui->widgetDisplay_2) 
    {
        if (event->type() == QEvent::MouseButtonPress) 
        {
            mouseEvent = static_cast<QMouseEvent*>(event);
            if (mouseEvent->button() == Qt::LeftButton) 
            {
                // 根据当前状态选择处理方式
                if (m_polygonPoints.isEmpty() && !m_isDragging) 
                {
                    // 开始矩形拖动
                    HandleMousePressOnDisplay2(mouseEvent->pos());
                    return true;
                } 
                else 
                {
                    // 继续多边形绘制
                    HandleMouseClickOnDisplay2(mouseEvent->pos());
                    return true;
                }
            }
        }
        else if (event->type() == QEvent::MouseMove) 
        {
            mouseEvent = static_cast<QMouseEvent*>(event);
            if (m_isDragging) 
            {
                HandleMouseMoveOnDisplay2(mouseEvent->pos());
                return true;
            }
        }
        else if (event->type() == QEvent::MouseButtonRelease) 
        {
            mouseEvent = static_cast<QMouseEvent*>(event);
            if (mouseEvent->button() == Qt::LeftButton && m_isDragging) 
            {
                HandleMouseReleaseOnDisplay2(mouseEvent->pos());
                return true;
            }
        }
        else if (event->type() == QEvent::KeyPress) 
        {
            keyEvent = static_cast<QKeyEvent*>(event);
            if (keyEvent->key() == Qt::Key_Return || keyEvent->key() == Qt::Key_Enter) 
            {
                HandleEnterKeyPress();
                return true;
            }
        }
    }
    
    // 如果widgetDisplay_2有焦点，也捕获主窗口的Enter键事件
    if (ui->widgetDisplay_2->hasFocus() && event->type() == QEvent::KeyPress) 
    {
        keyEvent = static_cast<QKeyEvent*>(event);
        if (keyEvent->key() == Qt::Key_Return || keyEvent->key() == Qt::Key_Enter) 
        {
            HandleEnterKeyPress();
            return true;
        }
    }
    
    return QMainWindow::eventFilter(obj, event);
}

// 处理鼠标点击事件
void MainWindow::HandleMouseClickOnDisplay2(const QPoint& pos)
{
    // 变量定义
    QPoint imagePoint;        // 图像坐标点
    QPixmap currentPixmap;    // 当前图像
    QPainter painter;         // 绘图对象
    int i;                    // 循环索引

    // 检查是否有图片加载
    if (!ui->widgetDisplay_2->pixmap() || ui->widgetDisplay_2->pixmap()->isNull()) 
    {
        AppendLog("请在widgetDisplay_2上显示图片后再进行点击", WARNNING);
        return;
    }
    
    // 让widgetDisplay_2获得焦点，以便接收键盘事件
    ui->widgetDisplay_2->setFocus();
    
    // 保存原始图片（如果尚未保存）
    if (!m_isImageLoaded) 
    {
        m_originalPixmap = ui->widgetDisplay_2->pixmap(Qt::ReturnByValue);
        m_isImageLoaded = true;
    }
    
    // 将控件坐标转换为图像坐标
    imagePoint = ConvertToImageCoordinates(pos);
    
    // 添加点到多边形点列表
    m_polygonPoints.append(imagePoint);
    
    // 在图片上显示点击点
    currentPixmap = m_originalPixmap.copy();
    painter.begin(&currentPixmap);
    painter.setRenderHint(QPainter::Antialiasing, true);
    
    // 绘制点击点
    painter.setPen(QPen(Qt::red, 3));
    painter.setBrush(QBrush(Qt::red));
    painter.drawEllipse(imagePoint, 5, 5);
    
    // 绘制点序号
    painter.setPen(QPen(Qt::white, 2));
    painter.setFont(QFont("Arial", 10, QFont::Bold));
    painter.drawText(imagePoint + QPoint(8, -8), QString::number(m_polygonPoints.size()));
    
    // 如果已有多个点，绘制连线
    if (m_polygonPoints.size() > 1) 
    {
        painter.setPen(QPen(Qt::green, 2, Qt::DashLine));
        for (i = 1; i < m_polygonPoints.size(); ++i) 
        {
            painter.drawLine(m_polygonPoints[i-1], m_polygonPoints[i]);
        }
        // 如果是最后一个点，连接到第一个点形成闭合多边形预览
        if (m_polygonPoints.size() >= 3) 
        {
            painter.drawLine(m_polygonPoints.last(), m_polygonPoints.first());
        }
    }
    
    painter.end();
    
    // 更新显示
    ui->widgetDisplay_2->setPixmap(currentPixmap);
    
    AppendLog(QString("已添加第%1个点，坐标(%2, %3)").arg(m_polygonPoints.size()).arg(imagePoint.x()).arg(imagePoint.y()), INFO);
}

// 处理Enter键按下事件
void MainWindow::HandleEnterKeyPress()
{
    if (m_polygonPoints.size() < 3) 
    {
        AppendLog(QString("需要至少3个点才能绘制多边形，当前只有%1个点").arg(m_polygonPoints.size()), WARNNING);
        return;
    }
    
    AppendLog(QString("开始绘制多边形，共%1个点").arg(m_polygonPoints.size()), INFO);
    DrawPolygonOnImage();
}

// 坐标转换：将控件坐标转换为图像坐标
QPoint MainWindow::ConvertToImageCoordinates(const QPoint& widgetPoint)
{
    // 变量定义
    QSize originalSize;    // 原始图像尺寸
    QSize widgetSize;      // 控件尺寸
    double widgetAspect;   // 控件宽高比
    double imageAspect;    // 图像宽高比
    QRect displayRect;     // 显示区域
    double relX;           // 相对X位置
    double relY;           // 相对Y位置
    int imageX;            // 图像X坐标
    int imageY;            // 图像Y坐标

    if (!ui->widgetDisplay_2->pixmap() || ui->widgetDisplay_2->pixmap()->isNull()) 
    {
        return widgetPoint;
    }
    
    // 获取原始图像尺寸
    originalSize = m_originalPixmap.size();
    if (originalSize.isEmpty()) 
    {
        originalSize = ui->widgetDisplay_2->pixmap(Qt::ReturnByValue).size();
    }
    
    // 获取控件尺寸
    widgetSize = ui->widgetDisplay_2->size();
    
    // 计算图像在控件中的显示区域（保持宽高比居中显示）
    widgetAspect = static_cast<double>(widgetSize.width()) / widgetSize.height();
    imageAspect = static_cast<double>(originalSize.width()) / originalSize.height();
    
    if (widgetAspect > imageAspect) 
    {
        // 控件更宽，图像在垂直方向填充
        int displayHeight = widgetSize.height();
        int displayWidth = static_cast<int>(displayHeight * imageAspect);
        int offsetX = (widgetSize.width() - displayWidth) / 2;
        displayRect = QRect(offsetX, 0, displayWidth, displayHeight);
    } 
    else 
    {
        // 控件更高，图像在水平方向填充
        int displayWidth = widgetSize.width();
        int displayHeight = static_cast<int>(displayWidth / imageAspect);
        int offsetY = (widgetSize.height() - displayHeight) / 2;
        displayRect = QRect(0, offsetY, displayWidth, displayHeight);
    }
    
    // 检查点击是否在图像显示区域内
    if (!displayRect.contains(widgetPoint)) 
    {
        // 如果点击在图像区域外，返回无效点
        return QPoint(-1, -1);
    }
    
    // 计算相对位置比例
    relX = static_cast<double>(widgetPoint.x() - displayRect.x()) / displayRect.width();
    relY = static_cast<double>(widgetPoint.y() - displayRect.y()) / displayRect.height();
    
    // 映射到原始图像坐标
    imageX = static_cast<int>(relX * originalSize.width());
    imageY = static_cast<int>(relY * originalSize.height());
    
    // 确保坐标在图像范围内
    imageX = qMax(0, qMin(imageX, originalSize.width() - 1));
    imageY = qMax(0, qMin(imageY, originalSize.height() - 1));
    
    return QPoint(imageX, imageY);
}

// 绘制多边形
void MainWindow::DrawPolygonOnImage()
{
    // 变量定义
    QPixmap currentPixmap;    // 当前图像
    QPainter painter;         // 绘图对象
    QPolygon polygon;         // 多边形
    QString infoText;         // 信息文本
    int i;                    // 循环索引

    if (m_polygonPoints.size() < 3) 
    {
        AppendLog("需要至少3个点才能绘制多边形", WARNNING);
        return;
    }
    
    // 恢复原始图片
    currentPixmap = m_originalPixmap.copy();
    painter.begin(&currentPixmap);
    painter.setRenderHint(QPainter::Antialiasing, true);
    
    // 绘制多边形
    painter.setPen(QPen(Qt::blue, 3));
    painter.setBrush(QBrush(QColor(0, 0, 255, 50))); // 半透明蓝色填充
    
    for (const QPoint& point : m_polygonPoints) 
    {
        polygon << point;
    }
    painter.drawPolygon(polygon);
    
    // 绘制顶点和序号
    painter.setPen(QPen(Qt::red, 3));
    painter.setBrush(QBrush(Qt::red));
    for (i = 0; i < m_polygonPoints.size(); ++i) 
    {
        painter.drawEllipse(m_polygonPoints[i], 5, 5);
        painter.setPen(QPen(Qt::white, 2));
        painter.setFont(QFont("Arial", 10, QFont::Bold));
        painter.drawText(m_polygonPoints[i] + QPoint(8, -8), QString::number(i+1));
        painter.setPen(QPen(Qt::red, 3));
    }
    
    // 在图像上显示多边形信息
    infoText = QString("多边形: %1个顶点").arg(m_polygonPoints.size());
    painter.setPen(QPen(Qt::white, 2));
    painter.setFont(QFont("Arial", 12, QFont::Bold));
    painter.drawText(10, 30, infoText);
    
    painter.end();
    
    // 更新显示
    ui->widgetDisplay_2->setPixmap(currentPixmap);
    
    // 在日志中显示所有点的坐标
    AppendLog("多边形绘制完成，顶点坐标：", INFO);
    for (i = 0; i < m_polygonPoints.size(); ++i) 
    {
        AppendLog(QString("顶点%1: (%2, %3)").arg(i+1).arg(m_polygonPoints[i].x()).arg(m_polygonPoints[i].y()), INFO);
    }
    
    // 标记多边形完成并裁剪图像
    m_polygonCompleted = true;
    CropImageToPolygon();
    
    // 清空点列表，准备下一次绘制
    m_polygonPoints.clear();
    m_isImageLoaded = false;
}

// 裁剪多边形区域图像并补全背景为矩形
void MainWindow::CropImageToPolygon()
{
    // 变量定义
    QImage originalImage;     // 原始图像
    QPolygon polygon;         // 多边形
    QRect boundingRect;       // 边界矩形
    int maxSize;              // 最大尺寸
    QRect squareRect;         // 正方形区域
    QImage croppedImage;      // 裁剪后的图像
    QColor backgroundColor;   // 背景颜色
    QPainter painter;         // 绘图对象
    QPolygon relativePolygon; // 相对多边形
    QPainterPath clipPath;    // 裁剪路径

    if (m_polygonPoints.size() < 3) 
    {
        AppendLog("需要至少3个点才能裁剪图像", WARNNING);
        return;
    }

    // 将QPixmap转换为QImage
    originalImage = m_originalPixmap.toImage();

    // 计算多边形的边界框
    for (const QPoint& point : m_polygonPoints) 
    {
        polygon << point;
    }

    boundingRect = polygon.boundingRect();

    // 确保边界框在图像范围内
    boundingRect = boundingRect.intersected(QRect(0, 0, originalImage.width(), originalImage.height()));

    if (boundingRect.isEmpty()) 
    {
        AppendLog("多边形区域超出图像范围", WARNNING);
        return;
    }

    // 计算边界框的最大边长，确保为正方形
    maxSize = qMax(boundingRect.width(), boundingRect.height());
    squareRect = QRect(boundingRect.x(), boundingRect.y(), maxSize, maxSize);

    // 调整正方形区域确保在图像范围内
    if (squareRect.right() >= originalImage.width()) 
    {
        squareRect.moveLeft(originalImage.width() - maxSize);
    }
    if (squareRect.bottom() >= originalImage.height()) 
    {
        squareRect.moveTop(originalImage.height() - maxSize);
    }
    if (squareRect.left() < 0) 
    {
        squareRect.moveLeft(0);
    }
    if (squareRect.top() < 0) 
    {
        squareRect.moveTop(0);
    }

    // 创建新的正方形图像
    croppedImage = QImage(maxSize, maxSize, QImage::Format_ARGB32);

    // 取样多边形边缘颜色作为背景色
    backgroundColor = SampleBorderColor(originalImage, polygon);

    // 用背景色填充整个图像
    croppedImage.fill(backgroundColor);

    // 创建QPainter来绘制多边形区域
    painter.begin(&croppedImage);
    painter.setRenderHint(QPainter::Antialiasing, true);

    // 设置剪裁路径为多边形（相对于正方形区域的坐标）
    for (const QPoint& point : m_polygonPoints) 
    {
        relativePolygon << QPoint(point.x() - squareRect.x(), point.y() - squareRect.y());
    }

    // 修改这里：使用addPolygon而不是fromPolygon
    clipPath.addPolygon(relativePolygon);
    painter.setClipPath(clipPath);

    // 将原始图像的多边形区域绘制到新图像上
    painter.drawImage(0, 0, originalImage, squareRect.x(), squareRect.y(), squareRect.width(), squareRect.height());

    painter.end();

    // 转换为QPixmap
    m_croppedPixmap = QPixmap::fromImage(croppedImage);
    m_hasCroppedImage = true;

    // 将裁剪后的图像显示到widgetDisplay_2上
    if (!m_croppedPixmap.isNull()) 
    {
        // 缩放图片以适应widgetDisplay_2大小，保持宽高比
        QPixmap scaledPixmap = m_croppedPixmap.scaled(ui->widgetDisplay->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
        ui->widgetDisplay_2->setPixmap(scaledPixmap);
        ui->widgetDisplay_2->setAlignment(Qt::AlignCenter);
        AppendLog("裁剪后的图像已显示", INFO);
    }

    AppendLog(QString("多边形区域图像裁剪完成，尺寸: %1x%2 像素").arg(maxSize).arg(maxSize), INFO);
    AppendLog(QString("背景颜色: RGB(%1, %2, %3)").arg(backgroundColor.red()).arg(backgroundColor.green()).arg(backgroundColor.blue()), INFO);
    ui->GetLength->setEnabled(true);
}

// 取样多边形边缘颜色作为背景色
QColor MainWindow::SampleBorderColor(const QImage& image, const QPolygon& polygon)
{
    // 变量定义
    QVector<QRgb> borderPixels;  // 边缘像素数组
    int sampleCount;             // 采样计数
    int i;                       // 循环索引
    int j;                       // 内层循环索引
    QPoint p1;                   // 多边形点1
    QPoint p2;                   // 多边形点2
    int steps;                   // 步数
    float t;                     // 插值参数
    QPoint samplePoint;          // 采样点
    long long r;                 // 红色分量总和
    long long g;                 // 绿色分量总和
    long long b;                 // 蓝色分量总和

    borderPixels = QVector<QRgb>();
    sampleCount = 0;
    
    // 在多边形边缘取样像素颜色
    for (i = 0; i < polygon.size(); ++i) 
    {
        p1 = polygon[i];
        p2 = polygon[(i + 1) % polygon.size()];
        
        // 沿着边缘取样
        steps = qMax(qAbs(p2.x() - p1.x()), qAbs(p2.y() - p1.y()));
        if (steps > 0) 
        {
            for (j = 0; j <= steps; ++j) 
            {
                t = (float)j / steps;
                samplePoint = QPoint(
                    qRound(p1.x() * (1 - t) + p2.x() * t),
                    qRound(p1.y() * (1 - t) + p2.y() * t)
                );
                
                // 确保采样点在图像范围内
                if (samplePoint.x() >= 0 && samplePoint.x() < image.width() && samplePoint.y() >= 0 && samplePoint.y() < image.height()) 
                { 
                    borderPixels.append(image.pixel(samplePoint));
                    sampleCount++;
                    
                    // 限制采样数量以提高性能
                    if (sampleCount >= 100) 
                    {
                        break;
                    }
                }
            }
        }
        if (sampleCount >= 100) 
        {
            break;
        }
    }
    
    if (borderPixels.isEmpty()) 
    {
        return Qt::white; // 默认背景色
    }
    
    // 计算平均颜色
    r = 0;
    g = 0;
    b = 0;
    for (QRgb pixel : borderPixels) 
    {
        r += qRed(pixel);
        g += qGreen(pixel);
        b += qBlue(pixel);
    }
    
    return QColor(r / borderPixels.size(), g / borderPixels.size(), b / borderPixels.size());
}

// 清除多边形显示
void MainWindow::ClearPolygonDisplay()
{
    // 恢复原始图片显示
    if (!m_originalPixmap.isNull()) 
    {
        ui->widgetDisplay_2->setPixmap(m_originalPixmap);
    }
    
    // 重置状态
    m_polygonPoints.clear();
    m_polygonCompleted = false;
    m_hasCroppedImage = false;
    m_isImageLoaded = false;
    
    AppendLog("已清除多边形显示", INFO);
}

// 矩形拖动选取功能实现

// 处理鼠标按下事件（开始矩形拖动）
void MainWindow::HandleMousePressOnDisplay2(const QPoint& pos)
{
    // 检查是否有图片加载
    if (!ui->widgetDisplay_2->pixmap() || ui->widgetDisplay_2->pixmap()->isNull()) 
    {
        AppendLog("请在widgetDisplay_2上显示图片后再进行拖动", WARNNING);
        return;
    }
    
    // 让widgetDisplay_2获得焦点，以便接收键盘事件
    ui->widgetDisplay_2->setFocus();
    
    // 保存原始图片（如果尚未保存）
    if (!m_isImageLoaded) 
    {
        m_originalPixmap = ui->widgetDisplay_2->pixmap(Qt::ReturnByValue);
        m_isImageLoaded = true;
    }
    
    // 开始拖动
    m_isDragging = true;
    m_dragStartPoint = pos;
    m_dragEndPoint = pos;
    
    AppendLog("开始矩形拖动选取", INFO);
}

// 处理鼠标移动事件（矩形拖动预览）
void MainWindow::HandleMouseMoveOnDisplay2(const QPoint& pos)
{
    // 变量定义
    QPixmap currentPixmap;  // 当前图像
    QPainter painter;       // 绘图对象
    QRect dragRect;         // 拖动矩形
    QPoint startImagePoint; // 起始图像坐标
    QPoint endImagePoint;   // 结束图像坐标
    QRect imageRect;        // 图像矩形
    QString sizeText;       // 尺寸文本

    if (!m_isDragging) 
    {
        return;
    }
    
    m_dragEndPoint = pos;
    
    // 显示矩形预览
    currentPixmap = m_originalPixmap.copy();
    painter.begin(&currentPixmap);
    painter.setRenderHint(QPainter::Antialiasing, true);
    
    // 计算矩形区域
    dragRect = QRect(m_dragStartPoint, m_dragEndPoint).normalized();
    
    // 绘制半透明矩形
    painter.setPen(QPen(Qt::red, 2));
    painter.setBrush(QBrush(QColor(255, 0, 0, 50))); // 半透明红色填充
    painter.drawRect(dragRect);
    
    // 显示矩形尺寸信息
    painter.setPen(QPen(Qt::white, 2));
    painter.setFont(QFont("Arial", 12, QFont::Bold));
    
    // 将控件坐标转换为图像坐标
    startImagePoint = ConvertToImageCoordinates(m_dragStartPoint);
    endImagePoint = ConvertToImageCoordinates(m_dragEndPoint);
    imageRect = QRect(startImagePoint, endImagePoint).normalized();
    
    sizeText = QString("%1 x %2 像素").arg(imageRect.width()).arg(imageRect.height());
    painter.drawText(dragRect.bottomRight() + QPoint(5, 15), sizeText);
    
    painter.end();
    
    // 更新显示
    ui->widgetDisplay_2->setPixmap(currentPixmap);
}

// 处理鼠标释放事件（完成矩形选择）
void MainWindow::HandleMouseReleaseOnDisplay2(const QPoint& pos)
{
    // 变量定义
    QPoint startImagePoint;  // 起始图像坐标
    QPoint endImagePoint;    // 结束图像坐标

    if (!m_isDragging) 
    {
        return;
    }
    
    m_dragEndPoint = pos;
    m_isDragging = false;
    
    // 将控件坐标转换为图像坐标
    startImagePoint = ConvertToImageCoordinates(m_dragStartPoint);
    endImagePoint = ConvertToImageCoordinates(m_dragEndPoint);
    
    // 计算选中的矩形区域
    m_selectedRect = QRect(startImagePoint, endImagePoint).normalized();
    
    // 确保矩形区域有效
    if (m_selectedRect.width() < 10 || m_selectedRect.height() < 10) 
    {
        AppendLog("选择的矩形区域太小，请重新选择", WARNNING);
        // 恢复原始图片
        ui->widgetDisplay_2->setPixmap(m_originalPixmap);
        return;
    }
    
    AppendLog(QString("矩形选择完成，区域: %1x%2 像素").arg(m_selectedRect.width()).arg(m_selectedRect.height()), INFO);
    
    // 绘制最终矩形并裁剪图像
    DrawRectangleOnImage();
}

// 绘制矩形区域
void MainWindow::DrawRectangleOnImage()
{
    // 变量定义
    QPixmap currentPixmap;  // 当前图像
    QPainter painter;       // 绘图对象
    QString infoText;       // 信息文本

    if (m_selectedRect.isEmpty()) 
    {
        AppendLog("无效的矩形区域", WARNNING);
        return;
    }
    
    // 恢复原始图片
    currentPixmap = m_originalPixmap.copy();
    painter.begin(&currentPixmap);
    painter.setRenderHint(QPainter::Antialiasing, true);
    
    // 绘制矩形边框
    painter.setPen(QPen(Qt::blue, 3));
    painter.setBrush(QBrush(QColor(0, 0, 255, 30))); // 半透明蓝色填充
    painter.drawRect(m_selectedRect);
    
    // 绘制矩形信息
    painter.setPen(QPen(Qt::white, 2));
    painter.setFont(QFont("Arial", 12, QFont::Bold));
    
    infoText = QString("矩形区域: %1x%2").arg(m_selectedRect.width()).arg(m_selectedRect.height());
    painter.drawText(m_selectedRect.topLeft() + QPoint(5, -5), infoText);
    
    painter.end();
    
    // 更新显示
    ui->widgetDisplay_2->setPixmap(currentPixmap);
    
    // 标记矩形完成并裁剪图像
    m_rectCompleted = true;
    CropImageToRectangle();
}

// 裁剪矩形区域图像
void MainWindow::CropImageToRectangle()
{
    // 变量定义
    QImage originalImage;  // 原始图像
    QRect validRect;       // 有效矩形区域
    QImage croppedImage;   // 裁剪后的图像
    QPixmap scaledPixmap;  // 缩放后的图像

    if (m_selectedRect.isEmpty()) 
    {
        AppendLog("无效的矩形区域", WARNNING);
        return;
    }
    
    // 将QPixmap转换为QImage
    originalImage = m_originalPixmap.toImage();
    
    // 确保矩形区域在图像范围内
    validRect = m_selectedRect.intersected(QRect(0, 0, originalImage.width(), originalImage.height()));
    
    if (validRect.isEmpty()) 
    {
        AppendLog("矩形区域超出图像范围", WARNNING);
        return;
    }
    
    // 裁剪图像
    croppedImage = originalImage.copy(validRect);
    
    // 转换为QPixmap
    m_croppedPixmap = QPixmap::fromImage(croppedImage);
    m_hasCroppedImage = true;
    
    // 将裁剪后的图像显示到widgetDisplay_2上
    if (!m_croppedPixmap.isNull()) 
    {
        // 缩放图片以适应widgetDisplay_2大小，保持宽高比
        scaledPixmap = m_croppedPixmap.scaled(ui->widgetDisplay->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
        ui->widgetDisplay_2->setPixmap(scaledPixmap);
        ui->widgetDisplay_2->setAlignment(Qt::AlignCenter);
        AppendLog("裁剪后的矩形区域图像已显示", INFO);
    }
    
    AppendLog(QString("矩形区域图像裁剪完成，尺寸: %1x%2 像素").arg(validRect.width()).arg(validRect.height()), INFO);
    ui->GetLength->setEnabled(true);
}
