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
}

MainWindow::~MainWindow()
{
    if (m_sysMonitor) 
    {
        m_sysMonitor->stopMonitoring();
    }
    delete ui;
}

// ch:显示错误信息 | en:Show error message
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
        switch(pFrameInfo->enPixelType) {
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
        double sharpness = pMainWindow->calculateTenengradSharpness(grayImage);
        // 发射信号
        emit pMainWindow->sharpnessValueUpdated(sharpness);
    }
}

void MainWindow::ImageCallBackInner(unsigned char * pData, MV_FRAME_OUT_INFO_EX* pFrameInfo)
{
    MV_DISPLAY_FRAME_INFO stDisplayInfo;
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
    ui->ComboDevices->clear();
    QTextCodec::setCodecForLocale(QTextCodec::codecForName("GBK"));
    ui->ComboDevices->setStyle(QStyleFactory::create("Windows"));
    // ch:枚举子网内所有设备 | en:Enumerate all devices within subnet
    memset(&m_stDevList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
    int nRet = CMvCamera::EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE | MV_GENTL_CAMERALINK_DEVICE | MV_GENTL_CXP_DEVICE | MV_GENTL_XOF_DEVICE, &m_stDevList);
    if (MV_OK != nRet)
    {
        AppendLog("枚举设备错误", ERROR);
        return;
    }
    AppendLog(QString("枚举到 %1 个设备").arg(m_stDevList.nDeviceNum), INFO);
    // ch:将值加入到信息列表框中并显示出来 | en:Add value to the information list box and display
    for (unsigned int i = 0; i < m_stDevList.nDeviceNum; i++)
    {
        MV_CC_DEVICE_INFO* pDeviceInfo = m_stDevList.pDeviceInfo[i];
        if (NULL == pDeviceInfo)
        {
            continue;
        }
        char strUserName[256] = {0};
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
    int nIndex = ui->ComboDevices->currentIndex();
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

    int nRet = m_pcMyCamera->Open(m_stDevList.pDeviceInfo[nIndex]);
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
        unsigned int nPacketSize = 0;
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
    m_pcMyCamera->RegisterImageCallBack(ImageCallBack, this);

    int nRet = m_pcMyCamera->StartGrabbing();
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
    if (true == ui->bnTriggerMode->isChecked() && ui->cbSoftTrigger->isChecked())
    {
        ui->bnTriggerExec->setEnabled(true);
    }
}

void MainWindow::on_bnStop_clicked()
{
    int nRet = m_pcMyCamera->StopGrabbing();
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
    int nRet = m_pcMyCamera->CommandExecute("TriggerSoftware");
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
    MVCC_FLOATVALUE stFloatValue;
    memset(&stFloatValue, 0, sizeof(MVCC_FLOATVALUE));

    int nRet = m_pcMyCamera->GetFloatValue("ExposureTime", &stFloatValue);
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
    m_pcMyCamera->SetEnumValue("ExposureAuto", 0);
    int nRet = m_pcMyCamera->SetFloatValue("ExposureTime", ui->tbExposure->text().toFloat());
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
    if (!m_pcMyCamera)
    {
        QMessageBox::warning(this, "保存图片", "相机对象无效！");
        AppendLog("相机对象无效", ERROR);
        return;
    }

    // 创建选择对话框
    QMessageBox msgBox;
    msgBox.setWindowTitle("选择检测模式");
    msgBox.setText("请选择检测模式：");
    QPushButton *cornerButton = msgBox.addButton("角点检测模式", QMessageBox::ActionRole);
    QPushButton *circleButton = msgBox.addButton("最小外接矩形检测模式", QMessageBox::ActionRole);
    QPushButton *cancelButton = msgBox.addButton(QMessageBox::Cancel);
    
    msgBox.exec();

    QAbstractButton *clicked =msgBox.clickedButton();
    if (clicked == cancelButton) 
    {
        return;
    }

    bool needDetection = ((clicked == cornerButton) || (clicked != circleButton));
    

    // 取出一份缓存快照
    vector<unsigned char> frame;
    MV_FRAME_OUT_INFO_EX info{};
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
    unsigned int dstMax = info.nWidth * info.nHeight * 3 + 4096;
    unique_ptr<unsigned char[]> pDst(new (nothrow) unsigned char[dstMax]);
    if (!pDst)
    {
        QMessageBox::warning(this, "保存图片", "内存不足（编码缓冲）！");
        AppendLog("内存不足（编码缓冲）", ERROR);
        return;
    }

    // 让 SDK 负责像素转换 + JPEG/PNG 编码（与头文件一致）
    MV_SAVE_IMAGE_PARAM_EX3 save{};
    save.enImageType   = MV_Image_Jpeg;              // 也可 MV_Image_Png
    save.enPixelType   = info.enPixelType;           // 源像素格式（SDK内部转）
    save.nWidth        = info.nWidth;
    save.nHeight       = info.nHeight;
    save.nDataLen      = info.nFrameLen;
    save.pData         = frame.data();
    save.pImageBuffer  = pDst.get();
    save.nImageLen     = dstMax;                     // 入参为目标缓冲大小
    save.nJpgQuality   = 90;

    int nRet = m_pcMyCamera->SaveImage(&save);
    if (MV_OK != nRet || save.nImageLen == 0)
    {
        ShowErrorMsg("保存失败（编码阶段）", nRet);
        AppendLog("保存失败（编码阶段）", ERROR);
        return;
    }

    // 生成保存路径
    QString dir = "/home/orangepi/Desktop/VisualRobot_Local/Img/";
    QString fpath = QDir(dir).filePath(QString("capture.jpg"));

    QFile f(fpath);
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

    // 如果选择了角点检测，才执行检测算法
    if (needDetection) 
    {
        //OpenCV版本
        int ProcessedOK = DetectRectangleOpenCV(fpath.toStdString(), Row, Col);
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
                AppendLog(QString("第%1个角点(x,y)像素坐标为:(%2, %3)")
                          .arg(i+1)
                          .arg(Col[i])
                          .arg(Row[i]),
                          INFO);
            }
        }

        QString imagePath = "../detectedImg.jpg";

        // 加载图片
        QPixmap pixmap(imagePath);
        if (pixmap.isNull()) 
        {
            QMessageBox::warning(this, "加载图片失败", "无法加载保存的图片！");
            return;
        }

        // 缩放图片以适应QLabel大小，保持宽高比
        QPixmap scaledPixmap = pixmap.scaled(ui->widgetDisplay_2->size(),
                                             Qt::KeepAspectRatio,
                                             Qt::SmoothTransformation);
        ui->widgetDisplay_2->setPixmap(scaledPixmap);
        ui->widgetDisplay_2->setAlignment(Qt::AlignCenter);

        AppendLog("图片显示成功", INFO);
    }
}

// 调试信息打印函数（输入：调试信息（QString）+宏定义调试信息等级）
void MainWindow::AppendLog(const QString &message, int logType, double value)
{
    // 添加时间戳
    QString timeStamp = QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss : ");
    QString fullMessage = timeStamp + message;

    // 检查value是否为有效值（非零或非默认值），您可以根据需要调整条件
    // 这里使用了一个简单的检查：如果value不是0.0，则追加它
    // 注意：这可能不适用于所有情况，您可能需要更精确的检查（例如与NaN比较）
    if (value != 0.0) 
    {
        fullMessage += " " + QString::number(value);
    }

    // 获取文本光标
    QTextCursor cursor(ui->displayLogMsg->document());
    cursor.movePosition(QTextCursor::End);

    // 根据调试信息等级配置文本颜色 (ERROR: 2, WARNNING: 1, INFO: 0)
    QTextCharFormat format;
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
    QScrollBar *scrollbar = ui->displayLogMsg->verticalScrollBar();
    if (scrollbar) 
    {
        scrollbar->setValue(scrollbar->maximum());
    }
}

void MainWindow::on_GetLength_clicked()
{

//    vector<QPointF> points;
//    points.reserve(Row.size()); // 预分配空间
////    Matrix3d transMatrix = readTransformationMatrix("../matrix.bin");
////    AppendLog("成功读取变换矩阵，详见终端显示", INFO);
////    // 输出变换矩阵
////    cout << "变换矩阵:" << endl;
////    cout << fixed << setprecision(10); // 设置输出精度
////    cout << transMatrix << endl;

//    if (Row.size() <=0 || Col.size() <= 0 || (Row.size() != Col.size()))
//    {
//        AppendLog("未能正确读取角点", ERROR);
//        return;
//    }

//    for (size_t i = 0; i < Row.size(); ++i)
//    {
//        double x = Col[i];  // 获取列坐标（转换为 double）
//        double y = Row[i];  // 获取行坐标（转换为 double）

////        // 创建齐次坐标向量 [x, y, 1]
////        Vector3d s(x, y, 1.0);
////        // 应用变换矩阵
////        Vector3d d = transMatrix * s;
////        // 创建变换后的 QPointF
////        QPointF transformedPoint(d(0), d(1));
//        QPointF p(x, y);

//        // 添加到结果向量
//        points.push_back(p);
//        // 输出转换信息
//        cout << "点 " << i << ": 像素坐标 (" << x << ", " << y << ") -> ";
////        cout << "世界坐标 (" << transformedPoint.x() << ", " << transformedPoint.y() << ")" << endl;
////        AppendLog(QString("第%1个角点(x,y)世界坐标为：(%2, %3)")
////                  .arg(i+1)
////                  .arg(static_cast<double>(transformedPoint.x()))
////                  .arg(static_cast<double>(transformedPoint.y())),
////                  INFO);
//    }
////    AppendLog("坐标矩阵变换成功", INFO);

//    // 计算点之间的欧几里得距离
//    auto euclideanDistance = [](const QPointF& a, const QPointF& b)
//    {
//        // 采用非圆心标定版的检长策略
//        double dx = (a.x() - b.x()) * 0.03294; // 100.0 / 3032.0
//        double dy = (a.y() - b.y()) * 0.03306;
//        return sqrt(dx * dx + dy * dy);
//    };

//    vector<double> distances;
//    distances.reserve(6); // 预分配空间

//    distances.push_back(euclideanDistance(points[0], points[1]));
//    distances.push_back(euclideanDistance(points[0], points[2]));
//    distances.push_back(euclideanDistance(points[0], points[3]));
//    distances.push_back(euclideanDistance(points[1], points[2]));
//    distances.push_back(euclideanDistance(points[1], points[3]));
//    distances.push_back(euclideanDistance(points[2], points[3]));

//    // 从大到小排序
//    sort(distances.rbegin(), distances.rend());

//    double diagonal = ((double)distances[0] + (double)distances[1]) / 2;
//    AppendLog(QString("物件对角线（mm）：%1").arg(diagonal), INFO);
//    double length = ((double)distances[2] + (double)distances[3]) / 2;
//    AppendLog(QString("物件长度（mm）：%1").arg(length), INFO);
//    double width = ((double)distances[4] + (double)distances[5]) / 2;
//    AppendLog(QString("物件宽度（mm）：%1").arg(width), INFO);

//    // 输出结果
//    cout << "各点欧式距离（降序）:" << endl;
//    for (const auto& dist : distances)
//    {
//        cout << dist << endl;
//    }

    string inputPath = "/home/orangepi/Desktop/VisualRobot_Local/Img/capture.jpg";
    string outputPath = "../detectedImg.jpg";
    QString output = "../detectedImg.jpg";

    // 读取输入图像
    Mat inputImage = imread(inputPath);
    if (inputImage.empty()) 
    {
        cerr << "无法读取输入图像: " << inputPath << endl;
        AppendLog("无法读取输入图像", ERROR);
        return;
    }

    // 设置参数（可根据需要修改）
    Params params;
    params.thresh = 127;
    params.maxval = 255;
    params.blurK = 5;
    params.areaMin = 100.0;

    // 处理图像
    double bias = 1.0;
    Result result = CalculateLength(inputImage, params, bias);

    // 保存输出图像
    if (!result.image.empty()) 
    {
        bool success = imwrite(outputPath, result.image);
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

    // 加载图片
    QPixmap pixmap(output);
    if (pixmap.isNull()) 
    {
        QMessageBox::warning(this, "加载图片失败", "无法加载保存的图片！");
        return;
    }

    // 缩放图片以适应QLabel大小，保持宽高比
    QPixmap scaledPixmap = pixmap.scaled(ui->widgetDisplay_2->size(),
                                         Qt::KeepAspectRatio,
                                         Qt::SmoothTransformation);
    ui->widgetDisplay_2->setPixmap(scaledPixmap);
    ui->widgetDisplay_2->setAlignment(Qt::AlignCenter);

    AppendLog("图片显示成功", INFO);
    AppendLog("检长算法执行完成", INFO);
    AppendLog(QString("物件长度（mm）：%1").arg((double)result.heights[0]), INFO);
    AppendLog(QString("物件宽度（mm）：%1").arg((double)result.widths[0]), INFO);
    AppendLog(QString("物件倾角（°）：%1").arg((double)result.angles[0]), INFO);

    DrawOverlayOnDisplay2((double)result.heights[0], (double)result.widths[0], (double)result.angles[0]);
}

void MainWindow::on_genMatrix_clicked()
{
    WorldCoord.clear();
    PixelCoord.clear();
    int getCoordsOk = GetCoordsOpenCV(WorldCoord, PixelCoord, 100.0);
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
    for (int i = 0; i < WorldCoord.size(); i++)
    {
        cout << i << "\t"
             << fixed << setprecision(3) << WorldCoord[i].x() << "\t\t"
             << fixed << setprecision(3) << WorldCoord[i].y() << endl;
    }

    cout << endl << "像素坐标 (单位:像素):" << endl;
    cout << "索引\tX坐标\t\tY坐标" << endl;
    cout << "----\t------\t\t------" << endl;
    for (int i = 0; i < PixelCoord.size(); i++)
    {
        cout << i << "\t"
             << fixed << setprecision(1) << PixelCoord[i].x() << "\t\t"
             << fixed << setprecision(1) << PixelCoord[i].y() << endl;
    }
    cout << "==========================" << endl << endl;

    Matrix3d transformationMatrix;

    // 调用函数计算变换矩阵并保存到文件
    int result = CalculateTransformationMatrix(WorldCoord, PixelCoord, transformationMatrix, "../matrix.bin");

    if (result == 0)
    {
        cout << "变换矩阵计算并保存成功!" << endl;
        AppendLog("变换矩阵计算并保存成功", ERROR);

        // 使用变换矩阵将像素坐标转换回世界坐标
        cout << endl << "=== 使用变换矩阵转换像素坐标 ===" << endl;
        cout << "索引\t原始世界坐标\t\t转换后坐标\t\t误差" << endl;
        cout << "----\t------------\t\t------------\t\t------" << endl;

        for (int i = 0; i < PixelCoord.size(); i++)
        {
            // 将像素坐标转换为齐次坐标 (x, y, 1)
            Vector3d pixelHomogeneous(PixelCoord[i].x(), PixelCoord[i].y(), 1.0);

            // 应用变换矩阵
            Vector3d worldTransformed = transformationMatrix * pixelHomogeneous;

            // 转换为非齐次坐标 (除以w分量)
            double x_transformed = worldTransformed[0] / worldTransformed[2];
            double y_transformed = worldTransformed[1] / worldTransformed[2];

            // 计算误差
            double error_x = fabs(WorldCoord[i].x() - x_transformed);
            double error_y = fabs(WorldCoord[i].y() - y_transformed);
            double total_error = sqrt(error_x * error_x + error_y * error_y);

            // 输出结果
            cout << i << "\t"
                 << fixed << setprecision(3) << "(" << WorldCoord[i].x() << "," << WorldCoord[i].y() << ")"
                 << "\t\t(" << x_transformed << "," << y_transformed << ")"
                 << "\t\t" << total_error << " mm" << endl;
        }
        cout << "=============================================" << endl;

        // 首先显示变换矩阵
        QString matrixStr = "变换矩阵:\n";
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
        for (int i = 0; i < PixelCoord.size(); i++) 
        {
            // 将像素坐标转换为齐次坐标 (x, y, 1)
            Vector3d pixelHomogeneous(PixelCoord[i].x(), PixelCoord[i].y(), 1.0);

            // 应用变换矩阵
            Vector3d worldTransformed = transformationMatrix * pixelHomogeneous;

            // 转换为非齐次坐标 (除以w分量)
            double x_transformed = worldTransformed[0] / worldTransformed[2];
            double y_transformed = worldTransformed[1] / worldTransformed[2];

            // 计算误差
            double error_x = fabs(WorldCoord[i].x() - x_transformed);
            double error_y = fabs(WorldCoord[i].y() - y_transformed);
            double total_error = sqrt(error_x * error_x + error_y * error_y);

            // 创建格式化的输出消息
            QString message = QString("点 %1: 理论世界坐标(%2, %3) -> 变换后世界坐标(%4, %5)")
                                 .arg(i)
                                 .arg(WorldCoord[i].x(), 0, 'f', 3)
                                 .arg(WorldCoord[i].y(), 0, 'f', 3)
                                 .arg(x_transformed, 0, 'f', 3)
                                 .arg(y_transformed, 0, 'f', 3);

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
    // 构建PDF文件的路径
    QString pdfPath = QDir(QCoreApplication::applicationDirPath()).filePath("../Doc/调试信息手册.pdf");
    
    // 尝试使用系统默认程序打开PDF
    QDesktopServices::openUrl(QUrl::fromLocalFile(pdfPath));
    
    AppendLog("已尝试打开调试信息手册", INFO);
}

void MainWindow::DrawOverlayOnDisplay2(double length, double width, double angle)
{
    // 使用value方式获取pixmap
    QPixmap src = ui->widgetDisplay_2->pixmap(Qt::ReturnByValue);
    if (src.isNull()) 
    {
        AppendLog("没有可叠加的图像（widgetDisplay_2 为空）", WARNNING);
        return;
    }

    QPixmap annotated = src.copy();
    QPainter p(&annotated);
    p.setRenderHint(QPainter::Antialiasing, true);
    p.setRenderHint(QPainter::TextAntialiasing, true);

    QString text = QString("长: %1 mm\n宽: %2 mm\n角度: %3 °")
                       .arg(length,   0, 'f', 3)
                       .arg(width,    0, 'f', 3)
                       .arg(angle, 0, 'f', 3);

    QFont font; font.setPointSize(12); font.setBold(true);
    p.setFont(font);

    QFontMetrics fm(font);
    const int margin = 10, pad = 8;
    QRect textRect = fm.boundingRect(QRect(0, 0, annotated.width()/2, annotated.height()),
                                     Qt::AlignRight | Qt::AlignTop | Qt::TextWordWrap, text);

    QRect boxRect(annotated.width() - textRect.width() - 2*pad - margin,
                  margin,
                  textRect.width() + 2*pad,
                  textRect.height() + 2*pad);

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
    DLExample* dlExample = new DLExample(nullptr);
    dlExample->setAttribute(Qt::WA_DeleteOnClose);
    dlExample->show();
    AppendLog("深度学习二分类示例窗口已打开", INFO);
}

// Tenengrad清晰度计算函数
double MainWindow::calculateTenengradSharpness(const cv::Mat& image)
{
    cv::Mat imageGrey;
    
    // 转换为灰度图
    if (image.channels() == 3) {
        cv::cvtColor(image, imageGrey, cv::COLOR_BGR2GRAY);
    } else {
        imageGrey = image.clone();
    }
    
    cv::Mat imageSobel;
    // 计算Sobel梯度
    cv::Sobel(imageGrey, imageSobel, CV_16U, 1, 1);
    
    // 计算梯度的平方和
    double meanValue = cv::mean(imageSobel)[0];
    
    return meanValue;
}

// 更新清晰度显示
void MainWindow::updateSharpnessDisplay(double sharpness)
{
    // 更新状态栏的清晰度标签
    if (m_sharpnessLabel) {
        QString sharpnessText = QString("清晰度: %1").arg(sharpness, 0, 'f', 2);
        m_sharpnessLabel->setText(sharpnessText);
    }
    
    // 同时在日志中记录清晰度值（可选）
    AppendLog(QString("当前图像清晰度: %1").arg(sharpness, 0, 'f', 2), INFO);
}

// 在图像上绘制清晰度叠加信息（保留函数，但当前主要用于状态栏显示）
void MainWindow::drawSharpnessOverlay(double sharpness)
{
    // 此函数目前主要用于状态栏显示
    // 如果需要直接在图像上叠加显示，可以在此处添加OpenCV绘制逻辑
    // 但考虑到性能，建议使用状态栏显示
    
    // 示例：如果需要直接在图像上绘制，可以使用以下代码
    /*
    if (m_hasFrame && !m_lastFrame.empty()) {
        // 将缓存数据转换为OpenCV Mat
        cv::Mat image(m_lastInfo.nHeight, m_lastInfo.nWidth, CV_8UC1, m_lastFrame.data());
        
        // 在图像上绘制清晰度文本
        std::string sharpnessText = "Sharpness: " + std::to_string(sharpness);
        cv::putText(image, sharpnessText, cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    }
    */
}
