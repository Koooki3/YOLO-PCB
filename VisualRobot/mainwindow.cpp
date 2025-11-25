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
#include <QElapsedTimer>
#include <QTextStream>
#include <QInputDialog>
#include <QTimer>
#include "Undistort.h"
#include "DefectDetection.h"
#include "YOLOExample.h"

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

    // 自动创建log文件夹
    QDir logDir("../log");
    if (!logDir.exists()) 
    {
        if (logDir.mkpath(".")) 
        {
            AppendLog("log文件夹创建成功", INFO);
        } 
        else 
        {
            AppendLog("log文件夹创建失败", ERROR);
        }
    }

    // 初始化相机相关变量
    memset(&m_stDevList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
    m_pcMyCamera = NULL;
    m_bGrabbing = false;
    m_hWnd = (void*)ui->widgetDisplay->winId();
    
    // 初始化设备热拔插自动枚举相关变量
    m_deviceEnumTimer = new QTimer(this);
    m_lastDeviceCount = 0;
    
    // 连接定时器信号到槽函数
    connect(m_deviceEnumTimer, &QTimer::timeout, this, &MainWindow::autoEnumDevices);
    
    // 启动定时器，每2秒自动枚举一次设备
    m_deviceEnumTimer->start(2000);

    // 初始化系统监控
    m_sysMonitor = new SystemMonitor(this);
    
    // 创建系统信息标签
    m_cpuLabel = new QLabel(this);
    m_memLabel = new QLabel(this);
    m_tempLabel = new QLabel(this);
    m_sharpnessLabel = new QLabel(this);

    // 初始化缺陷检测库
    m_defectDetection = new DefectDetection();
    
    // 加载缺陷分类模板库
    if (m_defectDetection->LoadTemplateLibrary("../Img/Templates")) 
    {
        AppendLog("缺陷分类模板库加载成功", INFO);
        auto templateNames = m_defectDetection->GetTemplateNames();
        AppendLog(QString("加载的模板类型: %1").arg(templateNames.size()), INFO);
        for (const auto& name : templateNames) 
        {
            AppendLog(QString("模板: %1").arg(QString::fromStdString(name)), INFO);
        }
    } 
    else 
    {
        AppendLog("缺陷分类模板库加载失败", WARNNING);
    }

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

    ui->genMatrix->setEnabled(true);

    // 连接清晰度信号
    connect(this, &MainWindow::sharpnessValueUpdated, this, &MainWindow::updateSharpnessDisplay);
    
    // 初始化多边形绘制功能
    SetupPolygonDrawing();
    
    // 日志优化相关初始化
    m_lastDefectCount = -1;  // -1 表示首次运行
    m_lastLogTime = QTime::currentTime();
    m_stableStateStartTime = QTime::currentTime();
    m_isStableState = false;
    
    // 初始时执行一次设备枚举
    on_bnEnum_clicked();
}

MainWindow::~MainWindow()
{
    if (m_sysMonitor) 
    {
        m_sysMonitor->stopMonitoring();
    }
    delete ui;
}

// 程序关闭事件处理
void MainWindow::closeEvent(QCloseEvent *event)
{
    // 变量定义
    QString logContent;           // 日志内容
    QString logFileName;          // 日志文件名
    QString logFilePath;          // 日志文件路径
    QFile logFile;                // 日志文件对象
    QTextStream out;              // 文本流
    QDateTime currentTime;        // 当前时间

    // 获取displayLogMsg中的所有文本内容
    logContent = ui->displayLogMsg->toPlainText();
    
    if (logContent.isEmpty()) 
    {
        // 如果没有日志内容, 直接关闭
        event->accept();
        return;
    }

    // 生成时间戳文件名
    currentTime = QDateTime::currentDateTime();
    logFileName = QString("log_%1.txt").arg(currentTime.toString("yyyyMMdd_hhmmss"));
    logFilePath = QString("../log/%1").arg(logFileName);

    // 打开文件进行写入
    logFile.setFileName(logFilePath);
    if (!logFile.open(QIODevice::WriteOnly | QIODevice::Text)) 
    {
        AppendLog("无法创建日志文件", ERROR);
        event->accept();
        return;
    }

    // 写入日志内容
    out.setDevice(&logFile);
    out << "== VisualRobot Runtime Log ==\n";
    out << "Execution Time: " << currentTime.toString("yyyy-MM-dd hh:mm:ss") << "\n";
    out << "=============================\n";
    out << logContent;

    logFile.close();
    
    // 接受关闭事件
    event->accept();
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
        pMainWindow->ImageCallBackInner(pData, pFrameInfo);         // 调用内部处理函数
    }

    // 1) 可选: 仍然显示
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

    // 2) 缓存: 深拷贝最新一帧到成员变量
    vector<unsigned char> tempFrame;  // 临时变量用于清晰度计算
    if (pUser)
    {
        MainWindow* pMainWindow = static_cast<MainWindow*>(pUser);  // 再次获取 MainWindow 指针
        lock_guard<mutex> lk(pMainWindow->m_frameMtx);
        pMainWindow->m_lastFrame.resize(pFrameInfo->nFrameLen);
        memcpy(pMainWindow->m_lastFrame.data(), pData, pFrameInfo->nFrameLen);
        pMainWindow->m_lastInfo = *pFrameInfo;   // 结构体按值拷贝
        pMainWindow->m_hasFrame = true;

        // 拷贝一份到临时变量, 用于清晰度计算
        tempFrame = pMainWindow->m_lastFrame;
    }

    // 3) 计算清晰度并发射信号
    if (pUser && !tempFrame.empty())
    {
        MainWindow* pMainWindow = static_cast<MainWindow*>(pUser);
        Mat grayImage;
        // 根据像素类型转换到灰度图
        switch(pFrameInfo->enPixelType) 
        {
            case PixelType_Gvsp_Mono8:
            {
                grayImage = Mat(pFrameInfo->nHeight, pFrameInfo->nWidth, CV_8UC1, tempFrame.data());
                break;
            }
            case PixelType_Gvsp_RGB8_Packed:
            {
                Mat colorImage = Mat(pFrameInfo->nHeight, pFrameInfo->nWidth, CV_8UC3, tempFrame.data());
                cvtColor(colorImage, grayImage, COLOR_RGB2GRAY);
                break;
            }
            case PixelType_Gvsp_BGR8_Packed:
            {
                Mat colorImage = Mat(pFrameInfo->nHeight, pFrameInfo->nWidth, CV_8UC3, tempFrame.data());
                cvtColor(colorImage, grayImage, COLOR_BGR2GRAY);
                break;
            }
            default: return;
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
    msgBox.setText("请选择检测模式: ");
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
    lock_guard<mutex> lk(m_frameMtx);
    if (!m_hasFrame || m_lastFrame.empty())
    {
        QMessageBox::warning(this, "保存图片", "暂无可用图像, 请先开始采集。");
        AppendLog("暂无可用图像, 请先开始采集", WARNNING);
        return;
    }
    frame = m_lastFrame;   // 拷贝到本地变量, 避免持锁编码
    info  = m_lastInfo;

    // 预分配编码缓冲 (给足空间) 
    dstMax = info.nWidth * info.nHeight * 3 + 4096;
    pDst = make_unique<unsigned char[]>(dstMax);
    if (!pDst)
    {
        QMessageBox::warning(this, "保存图片", "内存不足 (编码缓冲) ！");
        AppendLog("内存不足 (编码缓冲) ", ERROR);
        return;
    }

    // 让 SDK 负责像素转换 + JPEG/PNG 编码 (与头文件一致) 
    save.enImageType   = MV_Image_Jpeg;              // 也可 MV_Image_Png
    save.enPixelType   = info.enPixelType;           // 源像素格式 (SDK内部转) 
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
        ShowErrorMsg("保存失败 (编码阶段) ", nRet);
        AppendLog("保存失败 (编码阶段) ", ERROR);
        return;
    }

    // 生成保存路径
    dir = "/home/orangepi/Desktop/VisualRobot_Local/Img/";
    fpath = QDir(dir).filePath(QString("capture.jpg"));

    f.setFileName(fpath);
    if (!f.open(QIODevice::WriteOnly))
    {
        QMessageBox::warning(this, "保存图片", "无法打开文件进行写入！");
        AppendLog("保存图片, 无法打开文件进行写入", ERROR);
        return;
    }
    f.write(reinterpret_cast<const char*>(pDst.get()), save.nImageLen);
    f.close();

    // 应用去畸变
    Mat capture = imread("../Img/capture.jpg");
    if (!capture.empty())
    {
        // 创建校准器
        // 设置棋盘格参数 (内角点数量)
        Size boardSize(6, 6);     // 宽度方向9个内角点，高度方向6个内角点
        float squareSize = 10.0f; // 棋盘格方格实际大小，单位：毫米
        CameraCalibrator calibrator(boardSize, squareSize);
        string calibratorFile = "../calibration_parameters.yml";
        bool loadSuccess = calibrator.loadCalibration(calibratorFile);
        if (loadSuccess)
        {
            AppendLog("去畸变参数载入成功", INFO);
            Mat undistort = calibrator.undistortImage(capture, true, 100, BORDER_REPLICATE);
            if (!undistort.empty())
            {
                imwrite("../Img/capture.jpg", undistort);
                AppendLog("去畸变后图像保存完毕", INFO);
            }
            else
            {
                AppendLog("去畸变后图像为空, 请检查去畸变是否出错", ERROR);
            }
        }
        else
        {
            AppendLog("去畸变失败, 请检查去畸变参数文件是否存在", ERROR);
        }
    }
    else
    {
        AppendLog("待去畸变图像不存在", ERROR);
    }

    QMessageBox::information(this, "保存图片", QString("保存成功: %1").arg(fpath));
    AppendLog(QString("保存成功, 地址为: %1").arg(fpath), INFO);
    ui->GetLength->setEnabled(true);
    m_hasCroppedImage = false;
    m_polygonCompleted = false;

    // 如果选择了角点检测, 才执行检测算法
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
            AppendLog("角点检测算法执行失败, 请调整曝光或者重选待测物体", ERROR);
            return;
        }
        else
        {
            AppendLog("待检测图像读取成功, 角点检测算法执行完毕", INFO);
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

        // 缩放图片以适应QLabel大小, 保持宽高比
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

        // 缩放图片以适应QLabel大小, 保持宽高比
        scaledPixmap = pixmap.scaled(ui->widgetDisplay_2->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
        ui->widgetDisplay_2->setPixmap(scaledPixmap);
        ui->widgetDisplay_2->setAlignment(Qt::AlignCenter);

        AppendLog("原始图像显示成功", INFO);
    }
}

// 调试信息打印函数 (输入: 调试信息 (QString) +宏定义调试信息等级) 
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

    // 检查value是否为有效值 (非零或非默认值) , 您可以根据需要调整条件
    // 这里使用了一个简单的检查: 如果value不是0.0, 则追加它
    // 注意: 这可能不适用于所有情况, 您可能需要更精确的检查 (例如与NaN比较) 
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
        case 0: format.setForeground(Qt::black); break;
        case 1: format.setForeground(QColor(255, 140, 0)); break;
        case 2: format.setForeground(Qt::red); break;
        default: break;
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

    // 情况一: 如果没有裁剪图像, 清除多边形显示并处理原始图像
    // 仅当 m_hasCroppedImage 为 false 时使用原始全图进行识别
    if (!m_hasCroppedImage)
    {
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

        // 设置参数 (可根据需要修改) 
        params.thresh = 127;
        params.maxval = 255;
        params.blurK = 5;
        params.areaMin = 100.0;

        // 询问用户是否导入新几何参数变换系数
        QMessageBox::StandardButton importCoeff;
        importCoeff = QMessageBox::question(this, "导入几何参数", "是否导入新几何参数变换系数？", QMessageBox::Yes | QMessageBox::No, QMessageBox::No);

        // 处理图像 - 使用多目标检长算法
        if (importCoeff == QMessageBox::Yes) 
        {
            // 首先用bias=1.0计算得到像素数据
            bias = 1.0;

            QElapsedTimer timer;     // 计时器
            timer.start();           // 开始计时
            result = CalculateLength(inputImage, params, bias);
            qint64 elapsed = timer.elapsed(); // 获取经过的时间 (毫秒)
            AppendLog(QString("图像处理时间 (毫秒) : %1").arg(elapsed), INFO);

            // 如果有检测到目标，计算统计平均值
            if (!result.heights.empty() && !result.widths.empty()) 
            {
                double avgHeight = 0.0, avgWidth = 0.0;
                for (size_t i = 0; i < result.heights.size(); i++) 
                {
                    avgHeight += result.heights[i];
                    avgWidth += result.widths[i];
                }
                avgHeight /= result.heights.size();
                avgWidth /= result.widths.size();

                // 弹出对话框让用户输入理想目标长宽参数
                bool okLength, okWidth;
                double idealLength = QInputDialog::getDouble(this, "输入理想长度", "请输入理想目标长度 (μm):", 100.0, 0.0, 10000.0, 2, &okLength);
                double idealWidth = 0.0;
                if (okLength) 
                {
                    idealWidth = QInputDialog::getDouble(this, "输入理想宽度", "请输入理想目标宽度 (μm):", 100.0, 0.0, 10000.0, 2, &okWidth);
                }

                // 计算并更新变换系数
                if (okLength && okWidth && avgHeight > 0 && avgWidth > 0) 
                {
                    m_biasLength = idealLength / avgHeight;
                    m_biasWidth = idealWidth / avgWidth;
                    AppendLog(QString("已更新几何参数变换系数 - 长度系数: %1, 宽度系数: %2").arg(m_biasLength).arg(m_biasWidth), INFO);
                }
            }
        } 
        else 
        {
            // 如果不导入新系数，先判断是否已有系数
            if (m_biasLength > 0 && m_biasWidth > 0) 
            {
                // 已有系数，直接使用bias=1.0计算像素数据，后续手动转换
                bias = 1.0;

                QElapsedTimer timer;     // 计时器
                timer.start();           // 开始计时
                result = CalculateLength(inputImage, params, bias);
                qint64 elapsed = timer.elapsed(); // 获取经过的时间 (毫秒)
                AppendLog(QString("图像处理时间 (毫秒) : %1").arg(elapsed), INFO);
            } 
            else 
            {
                // 没有系数，使用bias=1.0计算并输出像素数据
                bias = 1.0;

                QElapsedTimer timer;     // 计时器
                timer.start();           // 开始计时
                result = CalculateLength(inputImage, params, bias);
                qint64 elapsed = timer.elapsed(); // 获取经过的时间 (毫秒)
                AppendLog(QString("图像处理时间 (毫秒) : %1").arg(elapsed), INFO);
            }
        }

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
            cerr << "处理后的图像为空, 无法保存" << endl;
            return;
        }

        // 加载图片
        pixmap = QPixmap(output);
        if (pixmap.isNull()) 
        {
            QMessageBox::warning(this, "加载图片失败", "无法加载保存的图片！");
            return;
        }

        // 缩放图片以适应QLabel大小, 保持宽高比
        scaledPixmap = pixmap.scaled(ui->widgetDisplay_2->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
        ui->widgetDisplay_2->setPixmap(scaledPixmap);
        ui->widgetDisplay_2->setAlignment(Qt::AlignCenter);

        AppendLog("检测后图像显示成功", INFO);
        AppendLog("基于连通域的多目标检长算法执行完成", INFO);
        
        // 输出所有目标的检长结果到日志
        for (size_t i = 0; i < result.heights.size(); i++) 
        {
            // 判断是否有有效的变换系数
            if (m_biasLength > 0 && m_biasWidth > 0) 
            {
                // 使用变换系数计算真实尺寸
                double realLength = result.heights[i] * m_biasLength;
                double realWidth = result.widths[i] * m_biasWidth;
                AppendLog(QString("目标%1 - 长度 (μm) : %2").arg(i+1).arg(realLength), INFO);
                AppendLog(QString("目标%1 - 宽度 (μm) : %2").arg(i+1).arg(realWidth), INFO);
                AppendLog(QString("目标%1 - 倾角 (°) : %2").arg(i+1).arg((double)result.angles[i]), INFO);
            } 
            else 
            {
                // 没有变换系数，输出像素数据
                AppendLog(QString("目标%1 - 长度 (pixs) : %2").arg(i+1).arg((double)result.heights[i]), INFO);
                AppendLog(QString("目标%1 - 宽度 (pixs) : %2").arg(i+1).arg((double)result.widths[i]), INFO);
                AppendLog(QString("目标%1 - 倾角 (°) : %2").arg(i+1).arg((double)result.angles[i]), INFO);
            }
        }
    }
    // 情况二: 如果已有裁剪图像, 处理裁剪后的图像
    else
    {
        QElapsedTimer timer;     // 计时器
        timer.start();           // 开始计时

        AppendLog("检测到多边形区域, 将处理裁剪后的图像", INFO);
        
        // 如果之前 CropImageToRectangle/CropImageToPolygon 已经保存了裁剪图像到 ../Img/cropped_polygon.jpg
        QString tempCroppedPath = "../Img/cropped_polygon.jpg";
        // 尝试保存当前 m_croppedPixmap（如果未保存）以确保文件存在
        if (!m_croppedPixmap.isNull()) 
        {
            m_croppedPixmap.save(tempCroppedPath);
        }

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

        // 设置参数 (可根据需要修改) 
        params.thresh = 127;
        params.maxval = 255;
        params.blurK = 5;
        params.areaMin = 100.0;

        // 对于裁剪区域的处理，同样支持几何参数变换
        if (m_biasLength > 0 && m_biasWidth > 0) 
        {
            // 已有系数，直接使用bias=1.0计算像素数据，后续手动转换
            bias = 1.0;
            result = CalculateLength(inputImage, params, bias);
        } 
        else 
        {
            // 没有系数，使用bias=1.0计算
            bias = 1.0;
            result = CalculateLength(inputImage, params, bias);
        }

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
            cerr << "处理后的图像为空, 无法保存" << endl;
            return;
        }

        qint64 elapsed = timer.elapsed(); // 获取经过的时间 (毫秒) 
        AppendLog(QString("裁剪区域图像处理时间 (毫秒) : %1").arg(elapsed), INFO);

        // 加载图片
        pixmap = QPixmap(output);
        if (pixmap.isNull()) 
        {
            QMessageBox::warning(this, "加载图片失败", "无法加载保存的图片！");
            return;
        }

        // 缩放图片以适应QLabel大小, 保持宽高比
        scaledPixmap = pixmap.scaled(ui->widgetDisplay_2->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
        ui->widgetDisplay_2->setPixmap(scaledPixmap);
        ui->widgetDisplay_2->setAlignment(Qt::AlignCenter);

        AppendLog("裁剪区域检测后图像显示成功", INFO);
        AppendLog("裁剪区域检长算法执行完成", INFO);
        
        // 判断是否有有效的变换系数
        if (m_biasLength > 0 && m_biasWidth > 0) 
        {
            // 使用变换系数计算真实尺寸
            double realLength = result.heights[0] * m_biasLength;
            double realWidth = result.widths[0] * m_biasWidth;
            AppendLog(QString("裁剪区域物件长度 (μm) : %1").arg(realLength), INFO);
            AppendLog(QString("裁剪区域物件宽度 (μm) : %1").arg(realWidth), INFO);
            AppendLog(QString("裁剪区域物件倾角 (°) : %1").arg((double)result.angles[0]), INFO);
        } 
        else 
        {
            // 没有变换系数，输出像素数据
            AppendLog(QString("裁剪区域物件长度 (pixs) : %1").arg((double)result.heights[0]), INFO);
            AppendLog(QString("裁剪区域物件宽度 (pixs) : %1").arg((double)result.widths[0]), INFO);
            AppendLog(QString("裁剪区域物件倾角 (°) : %1").arg((double)result.angles[0]), INFO);
        }

        DrawOverlayOnDisplay2((double)result.heights[0], (double)result.widths[0], (double)result.angles[0]);
    }
}

void MainWindow::on_genMatrix_clicked()
{
//     // 变量定义
//     int getCoordsOk;                // 坐标获取结果
//     Matrix3d transformationMatrix;  // 变换矩阵
//     int result;                     // 计算结果
//     QString matrixStr;              // 矩阵字符串
//     int i;                          // 循环索引
//     Vector3d pixelHomogeneous;      // 像素齐次坐标
//     Vector3d worldTransformed;      // 变换后的世界坐标
//     double x_transformed;           // 变换后的X坐标
//     double y_transformed;           // 变换后的Y坐标
//     double error_x;                 // X方向误差
//     double error_y;                 // Y方向误差
//     double total_error;             // 总误差
//     QString message;                // 消息字符串

//     WorldCoord.clear();
//     PixelCoord.clear();
//     getCoordsOk = GetCoordsOpenCV(WorldCoord, PixelCoord, 100.0);
//     if (getCoordsOk != 0)
//     {
//         AppendLog("坐标获取错误", ERROR);
//     }

//     // 在命令行显示坐标结果
//     cout << "=== 坐标检测结果 ===" << endl;
//     cout << "检测到的坐标对数量: " << WorldCoord.size() << endl << endl;

//     cout << "世界坐标 (单位:mm):" << endl;
//     cout << "索引\tX坐标\t\tY坐标" << endl;
//     cout << "----\t------\t\t------" << endl;
//     for (i = 0; i < WorldCoord.size(); i++)
//     {
//         cout << i << "\t"
//              << fixed << setprecision(3) << WorldCoord[i].x() << "\t\t"
//              << fixed << setprecision(3) << WorldCoord[i].y() << endl;
//     }

//     cout << endl << "像素坐标 (单位:像素):" << endl;
//     cout << "索引\tX坐标\t\tY坐标" << endl;
//     cout << "----\t------\t\t------" << endl;
//     for (i = 0; i < PixelCoord.size(); i++)
//     {
//         cout << i << "\t"
//              << fixed << setprecision(1) << PixelCoord[i].x() << "\t\t"
//              << fixed << setprecision(1) << PixelCoord[i].y() << endl;
//     }
//     cout << "==========================" << endl << endl;

//     // 调用函数计算变换矩阵并保存到文件
//     result = CalculateTransformationMatrix(WorldCoord, PixelCoord, transformationMatrix, "../matrix.bin");

//     if (result == 0)
//     {
//         cout << "变换矩阵计算并保存成功!" << endl;
//         AppendLog("变换矩阵计算并保存成功", INFO);

//         // 使用变换矩阵将像素坐标转换回世界坐标
//         cout << endl << "=== 使用变换矩阵转换像素坐标 ===" << endl;
//         cout << "索引\t原始世界坐标\t\t转换后坐标\t\t误差" << endl;
//         cout << "----\t------------\t\t------------\t\t------" << endl;

//         for (i = 0; i < PixelCoord.size(); i++)
//         {
//             // 将像素坐标转换为齐次坐标 (x, y, 1)
//             pixelHomogeneous = Vector3d(PixelCoord[i].x(), PixelCoord[i].y(), 1.0);

//             // 应用变换矩阵
//             worldTransformed = transformationMatrix * pixelHomogeneous;

//             // 转换为非齐次坐标 (除以w分量)
//             x_transformed = worldTransformed[0] / worldTransformed[2];
//             y_transformed = worldTransformed[1] / worldTransformed[2];

//             // 计算误差
//             error_x = fabs(WorldCoord[i].x() - x_transformed);
//             error_y = fabs(WorldCoord[i].y() - y_transformed);
//             total_error = sqrt(error_x * error_x + error_y * error_y);

//             // 输出结果
//             cout << i << "\t"
//                  << fixed << setprecision(3) << "(" << WorldCoord[i].x() << "," << WorldCoord[i].y() << ")"
//                  << "\t\t(" << x_transformed << "," << y_transformed << ")"
//                  << "\t\t" << total_error << " mm" << endl;
//         }
//         cout << "=============================================" << endl;

//         // 首先显示变换矩阵
//         matrixStr = "变换矩阵:\n";
//         for (int i = 0; i < 3; i++) 
//         {
//             matrixStr += "| ";
//             for (int j = 0; j < 3; j++) 
//             {
//                 matrixStr += QString::number(transformationMatrix(i, j), 'g', 15) + " ";
//             }
//             matrixStr += "|\n";
//         }
//         AppendLog(matrixStr, INFO);

//         // 然后显示误差结果
//         for (i = 0; i < PixelCoord.size(); i++) 
//         {
//             // 将像素坐标转换为齐次坐标 (x, y, 1)
//             pixelHomogeneous = Vector3d(PixelCoord[i].x(), PixelCoord[i].y(), 1.0);

//             // 应用变换矩阵
//             worldTransformed = transformationMatrix * pixelHomogeneous;

//             // 转换为非齐次坐标 (除以w分量)
//             x_transformed = worldTransformed[0] / worldTransformed[2];
//             y_transformed = worldTransformed[1] / worldTransformed[2];

//             // 计算误差
//             error_x = fabs(WorldCoord[i].x() - x_transformed);
//             error_y = fabs(WorldCoord[i].y() - y_transformed);
//             total_error = sqrt(error_x * error_x + error_y * error_y);

//             // 创建格式化的输出消息
//             message = QString("点 %1: 理论世界坐标(%2, %3) -> 变换后世界坐标(%4, %5)").arg(i).arg(WorldCoord[i].x(), 0, 'f', 3).arg(WorldCoord[i].y(), 0, 'f', 3).arg(x_transformed, 0, 'f', 3).arg(y_transformed, 0, 'f', 3);

//             // 调用日志函数显示结果
//             AppendLog(message, INFO, total_error); // 使用信息级别, 并将误差作为value传递
//         }
//     }
//     else
//     {
//         cout << "变换矩阵计算失败, 错误码: " << result << endl;
//         AppendLog(QString("变换矩阵计算失败, 错误码:%1").arg(result), ERROR);
//     }

    // 新代码: 整合Undistort去畸变模块功能
    // 设置棋盘格参数 (内角点数量)
    Size boardSize(6, 6);                                      // 宽度方向6个内角点，高度方向6个内角点
    float squareSize = 10.0f;                                  // 棋盘格方格实际大小，单位：毫米
    CameraCalibrator calibrator(boardSize, squareSize);        // 相机标定器
    string imageFolder = "../ImgData";                         // 图像文件夹路径
    string calibrationFile  = "../calibration_parameters.yml"; // 标定参数保存路径
    int processedCount = 0;                                    // 处理图像数量
    double reprojectionError  = 0.0;                           // 重投影误差
    bool saveSuccess = false;                                  // 保存成功标志

    // 检查ImgData文件夹是否存在
    QDir imgDataDir(QString::fromStdString(imageFolder));
    if (!imgDataDir.exists())
    {
        AppendLog("ImgData文件夹不存在, 请创建该文件夹并放入标定图像", ERROR);
        return;
    }

    // 处理文件夹中的所有图像
    processedCount = calibrator.processImagesFromFolder(imageFolder, false);
    if (processedCount < 3)
    {
        AppendLog(QString("需要至少3张有效图像进行校准, 当前只有 %1 张").arg(processedCount), WARNNING);
        return;
    }

    AppendLog(QString("成功处理 %1 张标定图像").arg(processedCount), INFO);

    // 进行相机标定
    reprojectionError = calibrator.calibrate();
    if (reprojectionError >= 0)
    {
        AppendLog("相机去畸变标定成功", INFO);

        // 保存标定参数到文件
        saveSuccess = calibrator.saveCalibration(calibrationFile);

        if (saveSuccess)
        {
            AppendLog(QString("去畸变标定参数已保存到: %1").arg(QString::fromStdString(calibrationFile)), INFO);

            // 获取相机参数并且显示
            Mat cameraMatrix = calibrator.getCameraMatrix();
            Mat distCoeffs = calibrator.getDistCoeffs();

            // 显示标定结果信息
            QString cameraMatrixStr = "相机内参矩阵:\n";
            for (int i = 0; i < cameraMatrix.rows; i++)
            {
                cameraMatrixStr += "| ";
                for (int j = 0; j < cameraMatrix.cols; j++)
                {
                    QString numStr = QString::number(cameraMatrix.at<double>(i, j), 'f', 10);
                    // 调整到固定宽度
                    numStr = numStr.leftJustified(10, ' ');
                    cameraMatrixStr += numStr;
                    if (j < cameraMatrix.cols - 1)
                    {
                        cameraMatrixStr += ", ";
                    }
                }
                cameraMatrixStr += " |\n";
            }
            AppendLog(cameraMatrixStr, INFO);

            QString distCoeffsStr = "畸变系数: [";
            for (int i = 0; i < distCoeffs.cols; i++)
            {
                distCoeffsStr += QString::number(distCoeffs.at<double>(0,i), 'f', 10);
                if (i < distCoeffs.cols - 1)
                {
                    distCoeffsStr += ", ";
                }
            }
            distCoeffsStr += "]";

            AppendLog(distCoeffsStr, INFO);
            AppendLog(QString("重投影误差: %1").arg(reprojectionError), INFO);
        }
        else
        {
            AppendLog("保存去畸变标定参数失败", ERROR);
        }
    }
    else
    {
        AppendLog("相机去畸变标定失败, 请检查标定图像", ERROR);
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
    QFontMetrics fm(font);          // 字体度量
    const int margin = 10;          // 边距
    const int pad = 8;              // 内边距
    QRect textRect;                 // 文本矩形区域
    QRect boxRect;                  // 框矩形区域

    // 使用value方式获取pixmap
    src = ui->widgetDisplay_2->pixmap(Qt::ReturnByValue);
    if (src.isNull()) 
    {
        AppendLog("没有可叠加的图像 (widgetDisplay_2 为空) ", WARNNING);
        return;
    }

    annotated = src.copy();
    p.begin(&annotated);
    p.setRenderHint(QPainter::Antialiasing, true);
    p.setRenderHint(QPainter::TextAntialiasing, true);

    // 根据是否有有效的变换系数选择显示单位
    if (m_biasLength > 0 && m_biasWidth > 0) 
    {
        // 使用变换系数计算真实尺寸并显示μm单位
        double realLength = length * m_biasLength;
        double realWidth = width * m_biasWidth;
        text = QString("长: %1 μm\n宽: %2 μm\n角度: %3 °").arg(realLength, 0, 'f', 3).arg(realWidth, 0, 'f', 3).arg(angle, 0, 'f', 3);
    } 
    else 
    {
        // 没有变换系数，显示像素数据
        text = QString("长: %1 pixs\n宽: %2 pixs\n角度: %3 °").arg(length, 0, 'f', 3).arg(width, 0, 'f', 3).arg(angle, 0, 'f', 3);
    }

    font.setPointSize(12); 
    font.setBold(true);
    p.setFont(font);

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
    // 让用户选择深度学习模式: 二分类（原DLExample）或 YOLO (独立窗口)
    QMessageBox msg(this);
    msg.setWindowTitle("选择深度学习模式");
    msg.setText("请选择深度学习模式:");
    auto *btnDL = msg.addButton("二分类", QMessageBox::AcceptRole);
    auto *btnYOLO = msg.addButton("YOLO", QMessageBox::AcceptRole);
    msg.addButton(QMessageBox::Cancel);

    msg.exec();
    QAbstractButton* clicked = msg.clickedButton();
    if (clicked == btnYOLO) {
        YOLOExample* ywin = new YOLOExample(nullptr);
        ywin->setAttribute(Qt::WA_DeleteOnClose);
        ywin->show();
        AppendLog("YOLO 窗口已打开", INFO);
    } else if (clicked == btnDL) {
        DLExample* dlExample = new DLExample(nullptr);
        dlExample->setAttribute(Qt::WA_DeleteOnClose);
        dlExample->show();
        AppendLog("深度学习二分类示例窗口已打开", INFO);
    } else {
        AppendLog("已取消打开深度学习窗口", INFO);
    }
}

// Tenengrad清晰度计算函数
double MainWindow::CalculateTenengradSharpness(const Mat& image)
{
    // 变量定义
    Mat imageGrey;      // 灰度图像
    Mat imageSobel;     // Sobel梯度图像
    double meanValue;       // 平均值

    // 转换为灰度图
    if (image.channels() == 3) 
    {
        cvtColor(image, imageGrey, COLOR_BGR2GRAY);
    } 
    else 
    {
        imageGrey = image.clone();
    }
    
    // 计算Sobel梯度
    Sobel(imageGrey, imageSobel, CV_16U, 1, 1);
    
    // 计算梯度的平方和
    meanValue = mean(imageSobel)[0];
    
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
    m_useRectangleMode = false; // 默认使用多边形模式
    
    // 设置widgetDisplay_2接受鼠标和键盘事件
    ui->widgetDisplay_2->setMouseTracking(true);
    ui->widgetDisplay_2->setFocusPolicy(Qt::StrongFocus); // 允许获得焦点
    ui->widgetDisplay_2->installEventFilter(this);
    
    AppendLog("多边形绘制和矩形拖动功能已初始化, 当前模式: 多边形点击模式", INFO);
}

// 事件过滤器, 用于捕获widgetDisplay_2的鼠标点击和键盘事件
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
                // 根据当前模式选择处理方式
                if (m_useRectangleMode) 
                {
                    HandleMousePressOnDisplay2(mouseEvent->pos());
                }
                else 
                {
                    HandleMouseClickOnDisplay2(mouseEvent->pos());
                }
                return true;
            }
        }
        else if (event->type() == QEvent::MouseMove) 
        {
            mouseEvent = static_cast<QMouseEvent*>(event);
            if (m_useRectangleMode && m_isDragging) 
            {
                HandleMouseMoveOnDisplay2(mouseEvent->pos());
                return true;
            }
        }
        else if (event->type() == QEvent::MouseButtonRelease) 
        {
            mouseEvent = static_cast<QMouseEvent*>(event);
            if (m_useRectangleMode && mouseEvent->button() == Qt::LeftButton && m_isDragging) 
            {
                HandleMouseReleaseOnDisplay2(mouseEvent->pos());
                return true;
            }
        }
        else if (event->type() == QEvent::KeyPress) 
        {
            QKeyEvent* keyEvent = static_cast<QKeyEvent*>(event);
            if (keyEvent->key() == Qt::Key_Return || keyEvent->key() == Qt::Key_Enter) 
            {
                HandleEnterKeyPress();
                return true;
            }
            else if (keyEvent->key() == Qt::Key_Escape) 
            {
                HandleEscKeyPress();
                return true;
            }
            else if (keyEvent->key() == Qt::Key_Space) 
            {
                HandleSpaceKeyPress();
                return true;
            }
        }
    }
    
    // 如果widgetDisplay_2有焦点, 也捕获主窗口的Enter键事件
    if (ui->widgetDisplay_2->hasFocus() && event->type() == QEvent::KeyPress) 
    {
        keyEvent = static_cast<QKeyEvent*>(event);
        if (keyEvent->key() == Qt::Key_Return || keyEvent->key() == Qt::Key_Enter) 
        {
            HandleEnterKeyPress();
            return true;
        }
        else if (keyEvent->key() == Qt::Key_Space) 
        {
            HandleSpaceKeyPress();
            return true;
        }
    }
    
    return QMainWindow::eventFilter(obj, event);
}

// 主窗口键盘事件处理
void MainWindow::keyPressEvent(QKeyEvent *event)
{
    if (event->key() == Qt::Key_Space) 
    {
        // 处理空格键, 切换选择模式
        HandleSpaceKeyPress();
        event->accept(); // 标记事件已处理, 阻止传播
    }
    else if (event->key() == Qt::Key_Q)
    {
        // 处理Q键, 退出实时检测模式
        HandleQKeyPress();
        event->accept(); // 标记事件已处理, 阻止传播
    }
    else 
    {
        // 其他按键交给父类处理
        QMainWindow::keyPressEvent(event);
    }
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
    
    // 让widgetDisplay_2获得焦点, 以便接收键盘事件
    ui->widgetDisplay_2->setFocus();
    
    // 保存原始图片 (如果尚未保存) 
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
    
    // 如果已有多个点, 绘制连线
    if (m_polygonPoints.size() > 1) 
    {
        painter.setPen(QPen(Qt::green, 2, Qt::DashLine));
        for (i = 1; i < m_polygonPoints.size(); ++i) 
        {
            painter.drawLine(m_polygonPoints[i-1], m_polygonPoints[i]);
        }
        // 如果是最后一个点, 连接到第一个点形成闭合多边形预览
        if (m_polygonPoints.size() >= 3) 
        {
            painter.drawLine(m_polygonPoints.last(), m_polygonPoints.first());
        }
    }
    
    painter.end();
    
    // 更新显示
    ui->widgetDisplay_2->setPixmap(currentPixmap);
    
    AppendLog(QString("已添加第%1个点, 坐标(%2, %3)").arg(m_polygonPoints.size()).arg(imagePoint.x()).arg(imagePoint.y()), INFO);
}

// 坐标转换: 将控件坐标转换为图像坐标
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
    
    // 计算图像在控件中的显示区域 (保持宽高比居中显示) 
    widgetAspect = static_cast<double>(widgetSize.width()) / widgetSize.height();
    imageAspect = static_cast<double>(originalSize.width()) / originalSize.height();
    
    if (widgetAspect > imageAspect) 
    {
        // 控件更宽, 图像在垂直方向填充
        int displayHeight = widgetSize.height();
        int displayWidth = static_cast<int>(displayHeight * imageAspect);
        int offsetX = (widgetSize.width() - displayWidth) / 2;
        displayRect = QRect(offsetX, 0, displayWidth, displayHeight);
    } 
    else 
    {
        // 控件更高, 图像在水平方向填充
        int displayWidth = widgetSize.width();
        int displayHeight = static_cast<int>(displayWidth / imageAspect);
        int offsetY = (widgetSize.height() - displayHeight) / 2;
        displayRect = QRect(0, offsetY, displayWidth, displayHeight);
    }
    
    // 检查点击是否在图像显示区域内
    if (!displayRect.contains(widgetPoint)) 
    {
        // 如果点击在图像区域外, 返回无效点
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
    AppendLog("多边形绘制完成, 顶点坐标: ", INFO);
    for (i = 0; i < m_polygonPoints.size(); ++i) 
    {
        AppendLog(QString("顶点%1: (%2, %3)").arg(i+1).arg(m_polygonPoints[i].x()).arg(m_polygonPoints[i].y()), INFO);
    }
    
    // 标记多边形完成并裁剪图像
    m_polygonCompleted = true;
    CropImageToPolygon();
    
    // 清空点列表, 准备下一次绘制
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

    // 计算边界框的最大边长, 确保为正方形
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

    // 设置剪裁路径为多边形 (相对于正方形区域的坐标) 
    for (const QPoint& point : m_polygonPoints) 
    {
        relativePolygon << QPoint(point.x() - squareRect.x(), point.y() - squareRect.y());
    }

    // 修改这里: 使用addPolygon而不是fromPolygon
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
        // 缩放图片以适应widgetDisplay_2大小, 保持宽高比
        QPixmap scaledPixmap = m_croppedPixmap.scaled(ui->widgetDisplay->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
        ui->widgetDisplay_2->setPixmap(scaledPixmap);
        ui->widgetDisplay_2->setAlignment(Qt::AlignCenter);
        AppendLog("裁剪后的图像已显示", INFO);
    }

    AppendLog(QString("多边形区域图像裁剪完成, 尺寸: %1x%2 像素").arg(maxSize).arg(maxSize), INFO);
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
                samplePoint = QPoint(qRound(p1.x() * (1 - t) + p2.x() * t), qRound(p1.y() * (1 - t) + p2.y() * t));
                
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

// 处理ESC键按下事件
void MainWindow::HandleEscKeyPress()
{
    // 检查是否有鼠标点击产生的点或矩形框
    if (!m_polygonPoints.isEmpty() || m_rectCompleted) 
    {
        // 清空所有点并还原状态
        m_polygonPoints.clear();
        m_isImageLoaded = false;
        m_polygonCompleted = false;
        m_hasCroppedImage = false;
        m_rectCompleted = false;
        m_isDragging = false;
        
        // 恢复原始图片显示
        if (!m_originalPixmap.isNull()) 
        {
            ui->widgetDisplay_2->setPixmap(m_originalPixmap);
        }
        
        AppendLog("已清空所有已选点和矩形框", INFO);
    }
    else 
    {
        // 如果没有选点, 显示警告信息
        AppendLog("无ROI选点", WARNNING);
    }
}

// 处理空格键按下事件
void MainWindow::HandleSpaceKeyPress()
{
    SwitchSelectionMode();
}

// 处理Q键按下事件 - 退出实时检测模式
void MainWindow::HandleQKeyPress()
{
    {
        QMutexLocker locker(&m_realTimeDetectionMutex);
        if (m_realTimeDetectionRunning)
        {
            // 停止实时检测
            m_realTimeDetectionRunning = false;
            AppendLog("已按Q键退出实时检测模式", INFO);
            
            // 清空widgetDisplay_2内容
            QMetaObject::invokeMethod(this, [this]() {
                ui->widgetDisplay_2->clear();
                AppendLog("已清空实时检测显示", INFO);
            }, Qt::QueuedConnection);
        }
        else
        {
            AppendLog("当前未在实时检测模式中", WARNNING);
        }
    }
}

// 切换选择模式
void MainWindow::SwitchSelectionMode()
{
    // 切换模式
    m_useRectangleMode = !m_useRectangleMode;
    
    // 清空当前状态
    m_polygonPoints.clear();
    m_rectCompleted = false;
    m_isDragging = false;
    
    // 恢复原始图片显示
    if (!m_originalPixmap.isNull()) 
    {
        ui->widgetDisplay_2->setPixmap(m_originalPixmap);
    }
    
    // 记录模式切换
    if (m_useRectangleMode) 
    {
        AppendLog("已切换到矩形拖动模式", INFO);
    }
    else 
    {
        AppendLog("已切换到多边形点击模式", INFO);
    }
}

// 处理鼠标按下事件（开始矩形拖动）
void MainWindow::HandleMousePressOnDisplay2(const QPoint& pos)
{
    // 检查是否有图片加载
    if (!ui->widgetDisplay_2->pixmap() || ui->widgetDisplay_2->pixmap()->isNull()) 
    {
        AppendLog("请在widgetDisplay_2上显示图片后再进行拖动", WARNNING);
        return;
    }
    
    // 让widgetDisplay_2获得焦点, 以便接收键盘事件
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
    if (!m_isDragging) 
    {
        return;
    }
    
    m_dragEndPoint = pos;
    
    // 显示矩形预览
    QPixmap currentPixmap = m_originalPixmap.copy();
    QPainter painter(&currentPixmap);
    painter.setRenderHint(QPainter::Antialiasing, true);
    
    // 计算矩形区域
    QRect dragRect = QRect(m_dragStartPoint, m_dragEndPoint).normalized();
    
    // 绘制半透明矩形
    painter.setPen(QPen(Qt::red, 2));
    painter.setBrush(QBrush(QColor(255, 0, 0, 50))); // 半透明红色填充
    painter.drawRect(dragRect);
    
    // 显示矩形尺寸信息
    painter.setPen(QPen(Qt::white, 2));
    painter.setFont(QFont("Arial", 12, QFont::Bold));
    
    // 将控件坐标转换为图像坐标
    QPoint startImagePoint = ConvertToImageCoordinates(m_dragStartPoint);
    QPoint endImagePoint = ConvertToImageCoordinates(m_dragEndPoint);
    QRect imageRect = QRect(startImagePoint, endImagePoint).normalized();
    
    QString sizeText = QString("%1 x %2 像素").arg(imageRect.width()).arg(imageRect.height());
    painter.drawText(dragRect.bottomRight() + QPoint(5, 15), sizeText);
    
    painter.end();
    
    // 更新显示
    ui->widgetDisplay_2->setPixmap(currentPixmap);
}

// 处理鼠标释放事件（完成矩形选择）
void MainWindow::HandleMouseReleaseOnDisplay2(const QPoint& pos)
{
    if (!m_isDragging) 
    {
        return;
    }
    
    m_dragEndPoint = pos;
    m_isDragging = false;
    
    // 将控件坐标转换为图像坐标
    QPoint startImagePoint = ConvertToImageCoordinates(m_dragStartPoint);
    QPoint endImagePoint = ConvertToImageCoordinates(m_dragEndPoint);
    
    // 计算选中的矩形区域
    m_selectedRect = QRect(startImagePoint, endImagePoint).normalized();
    
    // 确保矩形区域有效
    if (m_selectedRect.width() < 10 || m_selectedRect.height() < 10) 
    {
        AppendLog("选择的矩形区域太小, 请重新选择", WARNNING);
        // 恢复原始图片
        ui->widgetDisplay_2->setPixmap(m_originalPixmap);
        return;
    }
    
    AppendLog(QString("矩形选择完成, 区域: %1x%2 像素").arg(m_selectedRect.width()).arg(m_selectedRect.height()), INFO);
    
    // 绘制最终矩形并裁剪图像
    DrawRectangleOnImage();
}

// 绘制矩形区域
void MainWindow::DrawRectangleOnImage()
{
    if (m_selectedRect.isEmpty()) 
    {
        AppendLog("无效的矩形区域", WARNNING);
        return;
    }
    
    // 恢复原始图片
    QPixmap currentPixmap = m_originalPixmap.copy();
    QPainter painter(&currentPixmap);
    painter.setRenderHint(QPainter::Antialiasing, true);
    
    // 绘制矩形边框
    painter.setPen(QPen(Qt::blue, 3));
    painter.setBrush(QBrush(QColor(0, 0, 255, 30))); // 半透明蓝色填充
    painter.drawRect(m_selectedRect);
    
    // 绘制矩形信息
    painter.setPen(QPen(Qt::white, 2));
    painter.setFont(QFont("Arial", 12, QFont::Bold));
    
    QString infoText = QString("矩形区域: %1x%2").arg(m_selectedRect.width()).arg(m_selectedRect.height());
    painter.drawText(m_selectedRect.topLeft() + QPoint(5, -5), infoText);
    
    painter.end();
    
    // 更新显示
    ui->widgetDisplay_2->setPixmap(currentPixmap);
    
    // 标记矩形完成
    m_rectCompleted = true;
    AppendLog("矩形区域已绘制完成, 按Enter键确认裁剪", INFO);
}

// 裁剪矩形区域图像
void MainWindow::CropImageToRectangle()
{
    // 变量定义
    QString savePath;        // 保存路径
    bool saveSuccess;        // 保存成功标志

    if (m_selectedRect.isEmpty()) 
    {
        AppendLog("无效的矩形区域", WARNNING);
        return;
    }
    
    // 将QPixmap转换为QImage
    QImage originalImage = m_originalPixmap.toImage();
    
    // 确保矩形区域在图像范围内
    QRect validRect = m_selectedRect.intersected(QRect(0, 0, originalImage.width(), originalImage.height()));
    
    if (validRect.isEmpty()) 
    {
        AppendLog("矩形区域超出图像范围", WARNNING);
        return;
    }
    
    // 裁剪图像
    QImage croppedImage = originalImage.copy(validRect);
    
    // 转换为QPixmap
    m_croppedPixmap = QPixmap::fromImage(croppedImage);
    m_hasCroppedImage = true;
    
    // 保存裁剪后的图像到Img文件夹
    savePath = "../Img/cropped_polygon.jpg";
    saveSuccess = m_croppedPixmap.save(savePath);
    
    if (saveSuccess) 
    {
        AppendLog(QString("裁剪后的矩形区域图像已保存到: %1").arg(savePath), INFO);
    } 
    else 
    {
        AppendLog("保存裁剪后的矩形区域图像失败", WARNNING);
    }
    
    // 将裁剪后的图像显示到widgetDisplay_2上
    if (!m_croppedPixmap.isNull()) 
    {
        // 缩放图片以适应widgetDisplay_2大小, 保持宽高比
        QPixmap scaledPixmap = m_croppedPixmap.scaled(ui->widgetDisplay_2->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
        ui->widgetDisplay_2->setPixmap(scaledPixmap);
        ui->widgetDisplay_2->setAlignment(Qt::AlignCenter);
        AppendLog("裁剪后的矩形区域图像已显示", INFO);
    }
    
    AppendLog(QString("矩形区域图像裁剪完成, 尺寸: %1x%2 像素").arg(validRect.width()).arg(validRect.height()), INFO);
    ui->GetLength->setEnabled(true);
}

// 修改Enter键处理函数以支持矩形模式
void MainWindow::HandleEnterKeyPress()
{
    if (m_useRectangleMode) 
    {
        // 矩形模式处理
        if (!m_rectCompleted) 
        {
            AppendLog("请先完成矩形选择", WARNNING);
            return;
        }
        
        AppendLog("开始裁剪矩形区域", INFO);
        CropImageToRectangle();
    }
    else 
    {
        // 多边形模式处理
        if (m_polygonPoints.size() < 3) 
        {
            AppendLog(QString("需要至少3个点才能绘制多边形, 当前只有%1个点").arg(m_polygonPoints.size()), WARNNING);
            return;
        }
        
        AppendLog(QString("开始绘制多边形, 共%1个点").arg(m_polygonPoints.size()), INFO);
        DrawPolygonOnImage();
    }
}

// Mat 转 QPixmap
QPixmap MainWindow::MatToQPixmap(const Mat& bgr)
{
    if (bgr.empty()) 
    {
        return QPixmap();
    }
    Mat rgb;
    cvtColor(bgr, rgb, COLOR_BGR2RGB);
    QImage img(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
    return QPixmap::fromImage(img.copy());
}

// 从缓存的最后一帧取图：BGR
bool MainWindow::GrabLastFrameBGR(Mat& outBGR)
{
    vector<unsigned char> frameCopy;
    MV_FRAME_OUT_INFO_EX info{};
    {
        lock_guard<mutex> lk(m_frameMtx);
        if (!m_hasFrame || m_lastFrame.empty()) 
        {
            AppendLog("暂无可用图像，请先开始采集。", WARNNING);
            return false;
        }
        frameCopy = m_lastFrame;
        info      = m_lastInfo;
    }

    // 用 SDK 把原始帧编码成 JPEG，然后用 OpenCV 解码得到BGR
    unsigned int dstMax = info.nWidth * info.nHeight * 3 + 4096;
    unique_ptr<unsigned char[]> pDst(new (nothrow) unsigned char[dstMax]);
    if (!pDst) 
    {
        AppendLog("抓帧转换：内存不足（编码缓冲）", ERROR);
        return false;
    }

    MV_SAVE_IMAGE_PARAM_EX3 save{};
    save.enImageType   = MV_Image_Jpeg;
    save.enPixelType   = info.enPixelType;
    save.nWidth        = info.nWidth;
    save.nHeight       = info.nHeight;
    save.nDataLen      = info.nFrameLen;
    save.pData         = frameCopy.data();
    save.pImageBuffer  = pDst.get();
    save.nImageLen     = dstMax;
    save.nJpgQuality   = 90;

    int nRet = m_pcMyCamera ? m_pcMyCamera->SaveImage(&save) : MV_E_HANDLE;
    if (MV_OK != nRet || save.nImageLen == 0) 
    {
        AppendLog("抓帧转换失败（编码阶段）", ERROR);
        return false;
    }

    // OpenCV 解码
    Mat encoded(1, static_cast<int>(save.nImageLen), CV_8U, pDst.get());
    Mat bgr = imdecode(encoded, IMREAD_COLOR);
    if (bgr.empty()) 
    {
        AppendLog("imdecode 失败", ERROR);
        return false;
    }
    outBGR = bgr.clone();
    return true;
}

// 将当前帧设为模板
bool MainWindow::SetTemplateFromCurrent()
{
    Mat bgr;
    if (!GrabLastFrameBGR(bgr))
    {
        return false;
    }

    if (m_defectDetection->SetTemplateFromCurrent(bgr))
    {
        AppendLog(QString("已将当前帧设为模板，尺寸：%1x%2").arg(bgr.cols).arg(bgr.rows), INFO);
        return true;
    }
    return false;
}

bool MainWindow::SetTemplateFromFile(const QString& path)
{
    if (m_defectDetection->SetTemplateFromFile(path.toStdString()))
    {
        AppendLog(QString("已从文件加载模板：%1").arg(path), INFO);
        return true;
    }
    AppendLog("读取模板文件失败：" + path, ERROR);
    return false;
}

// 计算单应性（模板 <- 当前）
bool MainWindow::ComputeHomography(const Mat& curGray, Mat& H, vector<DMatch>* dbgMatches)
{
    return m_defectDetection->ComputeHomography(curGray, H, dbgMatches);
}

// 检测核心：返回在"当前图像坐标系"的外接框
vector<Rect> MainWindow::DetectDefects(const Mat& curBGR, const Mat& H, Mat* dbgMask)
{
    return m_defectDetection->DetectDefects(curBGR, H, dbgMask);
}

// 设置模板按钮
void MainWindow::on_setTemplate_clicked()
{
    // 弹窗选择：当前帧 / 从文件
    QMessageBox msg(this);
    msg.setWindowTitle("设置模板");
    msg.setText("选择模板来源：");
    auto *btnCur = msg.addButton("使用当前帧", QMessageBox::AcceptRole);
    auto *btnFile= msg.addButton("从文件选择...", QMessageBox::ActionRole);
    msg.addButton(QMessageBox::Cancel);

    msg.exec();
    QAbstractButton* clicked = msg.clickedButton();
    if (clicked == btnCur) 
    {
        if (SetTemplateFromCurrent())
        {
            AppendLog("模板已更新（来自当前帧）", INFO);
        }
    }
    else if (clicked == btnFile) 
    {
        QString path = QFileDialog::getOpenFileName(this, "选择模板图像", ".", "Images (*.png *.jpg *.jpeg *.bmp)");
        if (!path.isEmpty() && SetTemplateFromFile(path))
        {
            AppendLog("模板已更新（来自文件）", INFO);
        }
    }
}

// 实时检测线程控制变量
bool m_realTimeDetectionRunning = false;
QMutex m_realTimeDetectionMutex;

// 实时检测线程函数
void MainWindow::RealTimeDetectionThread()
{
    while (true)
    {
        {
            QMutexLocker locker(&m_realTimeDetectionMutex);
            if (!m_realTimeDetectionRunning)
            {
                break;
            }
        }

        // 取当前帧
        Mat curBGR;
        if (!GrabLastFrameBGR(curBGR))
        {
            QThread::msleep(50); // 等待50ms再试
            continue;
        }

        // 计算配准 H（模板 <- 当前）
        Mat curGray;
        cvtColor(curBGR, curGray, COLOR_BGR2GRAY);
        GaussianBlur(curGray, curGray, cv::Size(3,3), 0);

        Mat H;
        if (!ComputeHomography(curGray, H))
        {
            QThread::msleep(50);
            continue;
        }

        // 检测
        QElapsedTimer t; t.start();
        Mat dbgMask;
        auto boxes = DetectDefects(curBGR, H, &dbgMask);
        
        // 绘制结果
        Mat draw = curBGR.clone();
        for (size_t i = 0; i < boxes.size(); ++i) 
        {
            auto& b = boxes[i];
            rectangle(draw, b, Scalar(0,0,255), 2); // 红框
            
            // 对缺陷区域进行分类
            Mat defectROI = curBGR(b);
            std::string defectType = m_defectDetection->ClassifyDefect(defectROI);
            
            // 在缺陷框左上角显示缺陷类型
            QString typeText = QString::fromStdString(defectType);
            int fontFace = FONT_HERSHEY_SIMPLEX;
            double fontScale = 0.6;
            int thickness = 2;
            int baseline = 0;
            Size textSize = getTextSize(typeText.toStdString(), fontFace, fontScale, thickness, &baseline);
            
            // 计算文本位置（框内左上角）
            Point textOrg(b.x + 5, b.y + textSize.height + 5);
            
            // 绘制半透明背景
            rectangle(draw,  Point(b.x, b.y),  Point(b.x + textSize.width + 10, b.y + textSize.height + 10), Scalar(0, 0, 0), -1);
            
            // 绘制文本
            putText(draw, typeText.toStdString(), textOrg, fontFace, fontScale, Scalar(0, 255, 0), thickness);
            
            // 记录分类结果
            // 只在非稳定状态下记录具体缺陷类型，避免日志过多
            if (!m_isStableState || QTime::currentTime().msecsTo(m_lastLogTime) > 60000) 
            { // 每分钟记录一次详细信息
                QMetaObject::invokeMethod(this, [this, i, typeText]() {
                    AppendLog(QString("缺陷框 %1: 类型=%2").arg(i+1).arg(typeText), INFO);
                }, Qt::QueuedConnection);
            }
        }

        // 在图像上显示检测信息
        QString infoText = QString("REALTIME DETECTION - DEFECTS: %1").arg(boxes.size());
        putText(draw, infoText.toStdString(), Point(100, 150), FONT_HERSHEY_SIMPLEX, 4, Scalar(0, 255, 0), 10);
        
        // 优化日志记录：仅在缺陷数发生变化并保持2秒以上时记录
        int currentDefectCount = boxes.size();
        QTime currentTime = QTime::currentTime();
        
        // 在主线程中处理日志逻辑
        QMetaObject::invokeMethod(this, [this, currentDefectCount, currentTime]() {
            // 如果是首次运行或者缺陷数量发生变化
            if (m_lastDefectCount == -1 || m_lastDefectCount != currentDefectCount) 
            {
                // 更新状态为非稳定，并记录开始时间
                m_isStableState = false;
                m_stableStateStartTime = currentTime;
                m_lastDefectCount = currentDefectCount;
            } 
            else 
            {
                // 检查当前状态是否已经稳定2秒以上
                if (!m_isStableState && m_stableStateStartTime.secsTo(currentTime) >= 2) 
                {
                    // 状态稳定，记录日志
                    m_isStableState = true;
                    AppendLog(QString("实时检测 - 缺陷数量稳定为: %1 个，已持续 2 秒以上").arg(currentDefectCount), INFO);
                    m_lastLogTime = currentTime;
                }
            }
        }, Qt::QueuedConnection);

        // 展示到 widgetDisplay_2
        QPixmap pm = MatToQPixmap(draw);
        if (!pm.isNull()) 
        {
            // 在主线程中更新UI
            QMetaObject::invokeMethod(this, [this, pm]() {
                QPixmap scaled = pm.scaled(ui->widgetDisplay_2->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
                ui->widgetDisplay_2->setPixmap(scaled);
                ui->widgetDisplay_2->setAlignment(Qt::AlignCenter);
            }, Qt::QueuedConnection);
        }

        QThread::msleep(33); // 约30fps
    }
}

// 开始实时检测
void MainWindow::StartRealTimeDetection()
{
    {
        QMutexLocker locker(&m_realTimeDetectionMutex);
        if (m_realTimeDetectionRunning)
        {
            AppendLog("实时检测已在运行", WARNNING);
            return;
        }
        m_realTimeDetectionRunning = true;
    }

    // 启动实时检测线程
    QThread* thread = QThread::create([this]() { RealTimeDetectionThread(); });
    connect(thread, &QThread::finished, thread, &QThread::deleteLater);
    thread->start();
    
    // 重置日志状态变量
    m_lastDefectCount = -1;
    m_lastLogTime = QTime::currentTime();
    m_stableStateStartTime = QTime::currentTime();
    m_isStableState = false;

    AppendLog("实时缺陷检测已启动", INFO);
}

// 缺陷检测按钮
void MainWindow::on_detect_clicked()
{
    // 创建选择对话框
    QMessageBox msgBox(this);
    msgBox.setWindowTitle("选择检测模式");
    msgBox.setText("请选择检测模式:");
    
    QPushButton *singleButton = msgBox.addButton("单次", QMessageBox::ActionRole);
    QPushButton *realTimeButton = msgBox.addButton("实时", QMessageBox::ActionRole);
    QPushButton *cancelButton = msgBox.addButton("取消", QMessageBox::RejectRole);
    
    msgBox.exec();
    
    QAbstractButton* clicked = msgBox.clickedButton();
    if (clicked == cancelButton) 
    {
        AppendLog("检测已取消", INFO);
        return;
    }
    else if (clicked == realTimeButton)
    {
        // 实时检测模式
        if (!m_defectDetection->HasTemplate()) 
        {
            AppendLog("尚未设置模板，请先设置模板。", WARNNING);
            // 尝试直接用当前帧设模板
            if (!SetTemplateFromCurrent())
            {
                return;
            }
        }
        StartRealTimeDetection();
        return;
    }
    else if (clicked == singleButton)
    {
        // 单次检测模式（集成特征对齐）
        if (!m_defectDetection->HasTemplate()) 
        {
            AppendLog("尚未设置模板，请先设置模板。", WARNNING);
            // 尝试直接用当前帧设模板，继续流程（也可直接 return）
            if (!SetTemplateFromCurrent())
            {
                return;
            }
        }

        // 取当前帧
        Mat curBGR;
        if (!GrabLastFrameBGR(curBGR))
        {
            return;
        }

        // 使用特征对齐进行图像配准
        QElapsedTimer alignTimer;
        alignTimer.start();
        
        Mat H;
        if (!m_defectDetection->ComputeHomographyWithFeatureAlignment(curBGR, H))
        {
            AppendLog("特征对齐失败，使用传统方法", WARNNING);
            // 回退到传统方法
            Mat curGray;
            cvtColor(curBGR, curGray, COLOR_BGR2GRAY);
            GaussianBlur(curGray, curGray, cv::Size(3,3), 0);
            
            if (!ComputeHomography(curGray, H))
            {
                return;
            }
        }
        else
        {
            AppendLog(QString("特征对齐成功，耗时: %1 ms").arg(alignTimer.elapsed()), INFO);
        }

        // 重构待检测图像（使用特征对齐后的变换矩阵）
        Mat alignedBGR = m_defectDetection->AlignAndWarpImage(curBGR);
        if (alignedBGR.empty())
        {
            AppendLog("图像重构失败，使用原始图像", WARNNING);
            alignedBGR = curBGR.clone();
        }
        else
        {
            imwrite("../Img/alignedBGR.jpg", alignedBGR);
            AppendLog("图像重构成功，已对齐到模板坐标系", INFO);
        }

        AppendLog("重构图像已保存: ../Img/alignedBGR.jpg", INFO);

        // 步骤6: 从文件重新读取模板图和重构图像进行缺陷检测
        AppendLog("步骤6: 从文件重新读取图像进行缺陷检测...", INFO);
        
        // 读取模板图像
        Mat templateBGR = imread("../Img/templateBGR.jpg");
        if (templateBGR.empty()) 
        {
            AppendLog("错误: 无法加载模板图像: ../Img/templateBGR.jpg", ERROR);
            return;
        }
        AppendLog("模板图像加载成功", INFO);
        
        // 读取重构后的待检测图像
        Mat testImage = imread("../Img/alignedBGR.jpg");
        if (testImage.empty()) 
        {
            AppendLog("错误: 无法加载待检测图像: ../Img/alignedBGR.jpg", ERROR);
            return;
        }
        AppendLog("待检测图像加载成功", INFO);

        // 创建新的缺陷检测对象进行独立检测
        DefectDetection fileDetector;
        
        // 设置模板
        if (!fileDetector.SetTemplateFromFile("../Img/templateBGR.jpg")) 
        {
            AppendLog("错误: 无法设置模板图像", ERROR);
            return;
        }
        AppendLog("模板设置成功", INFO);

        // 使用ORB方法计算单应性矩阵
        QElapsedTimer orbTimer;
        orbTimer.start();
        
        Mat testGray;
        cvtColor(testImage, testGray, COLOR_BGR2GRAY);
        GaussianBlur(testGray, testGray, cv::Size(3,3), 0);
        
        Mat fileHomography;
        vector<DMatch> fileDebugMatches;
        if (!fileDetector.ComputeHomography(testGray, fileHomography, &fileDebugMatches)) 
        {
            AppendLog("ORB方法配准失败，无法进行缺陷检测", ERROR);
            return;
        }
        AppendLog(QString("ORB方法配准成功，耗时: %1 ms，匹配点数量: %2").arg(orbTimer.elapsed()).arg(fileDebugMatches.size()), INFO);

        // 缺陷检测
        QElapsedTimer detectTimer;
        detectTimer.start();
        Mat fileDebugMask;
        auto boxes = fileDetector.DetectDefects(testImage, fileHomography, &fileDebugMask);
        AppendLog(QString("基于模板的缺陷检测耗时: %1 ms，检测到缺陷框数: %2").arg(detectTimer.elapsed()).arg((int)boxes.size()), INFO);

        // 绘制检测结果
        Mat resultImage = testImage.clone();
        for (size_t i = 0; i < boxes.size(); i++) 
        {
            Rect rect = boxes[i];
        }

        // 保存结果图像
        imwrite("../Img/detection_result.jpg", resultImage);
        AppendLog("检测结果已保存: ../Img/detection_result.jpg", INFO);

        // 显示结果
        QPixmap pm = MatToQPixmap(resultImage);
        if (!pm.isNull()) 
        {
            // 自适应显示
            QPixmap scaled = pm.scaled(ui->widgetDisplay_2->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
            ui->widgetDisplay_2->setPixmap(scaled);
            ui->widgetDisplay_2->setAlignment(Qt::AlignCenter);
            AppendLog("缺陷结果已显示（基于文件读取的模板和重构图像）。", INFO);
        } 
        else 
        {
            AppendLog("显示失败: QPixmap 为空。", ERROR);
        }

        // 输出检测统计信息
        AppendLog("=== 检测统计 ===", INFO);
        AppendLog("模板图像: ../Img/templateBGR.jpg", INFO);
        AppendLog("测试图像: ../Img/alignedBGR.jpg", INFO);
        AppendLog(QString("检测到的缺陷数量: %1").arg(boxes.size()), INFO);
        AppendLog(QString("最小缺陷面积: %1").arg(fileDetector.GetMinDefectArea()), INFO);
        AppendLog(QString("差异阈值: %1").arg(fileDetector.GetTemplateDiffThreshold()), INFO);

        // 日志每个框
        for (size_t i=0; i<boxes.size(); ++i)
        {
            AppendLog(QString("缺陷框 %1: (x=%2, y=%3, w=%4, h=%5) 面积: %6").arg(i+1).arg(boxes[i].x).arg(boxes[i].y).arg(boxes[i].width).arg(boxes[i].height).arg(boxes[i].area()), INFO);
        }
    }
    // 如需同时叠加尺寸/角度的浮窗，可复用已有的 drawOverlayOnDisplay2()
}

// 设备热拔插自动枚举槽函数
void MainWindow::autoEnumDevices()
{
    // 只有在没有打开相机的情况下才进行自动枚举
    if (m_pcMyCamera && m_pcMyCamera->IsDeviceConnected())
    {
        return;
    }

    // 临时存储设备信息
    MV_CC_DEVICE_INFO_LIST stDevList;
    memset(&stDevList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
    
    // 枚举设备
    int nRet = CMvCamera::EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE | MV_GENTL_CAMERALINK_DEVICE | MV_GENTL_CXP_DEVICE | MV_GENTL_XOF_DEVICE, &stDevList);
    
    if (MV_OK != nRet)
    {
        // 枚举失败，不做处理，避免频繁日志
        return;
    }
    
    // 检查设备数量是否变化
    if (stDevList.nDeviceNum != m_lastDeviceCount)
    {
        // 设备数量变化，更新设备列表
        AppendLog(QString("设备数量变化: %1 -> %2，自动更新设备列表").arg(m_lastDeviceCount).arg(stDevList.nDeviceNum), INFO);
        
        // 调用现有的枚举设备按钮的槽函数
        on_bnEnum_clicked();
        
        // 更新上次设备数量
        m_lastDeviceCount = stDevList.nDeviceNum;
    }
}
