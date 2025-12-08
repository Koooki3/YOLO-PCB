#include "ImageAcquisition.h"
#include "DVPCamera.h"

//RGB转BGR和BGR转RGB是一样的方法
static bool BGR2RGB(unsigned char* pRgbData, unsigned int nWidth, unsigned int nHeight)
{
    if (NULL == pRgbData)
    {
        return false;
    }

    for (unsigned int j = 0; j < nHeight; j++)
    {
        for (unsigned int i = 0; i < nWidth; i++)
        {
            unsigned char red = pRgbData[j * (nWidth * 3) + i * 3];
            pRgbData[j * (nWidth * 3) + i * 3] = pRgbData[j * (nWidth * 3) + i * 3 + 2];
            pRgbData[j * (nWidth * 3) + i * 3 + 2] = red;
        }
    }
    return true;
}

QImageAcquisition::QImageAcquisition(dvpHandle &handle, QObject *parent):
    QObject(parent)
{
    m_pThread = new QThread();
    this->moveToThread(m_pThread);
    // 启动采集线程
    m_pThread->start();

    // 初始化成员变量
    m_handle = handle;
//    m_bAcquireImg = false;
    pBuffer = NULL;

    m_timer = new QTimer(this);

    //将定时器超时信号与槽(功能函数)联系起来
    connect( m_timer, SIGNAL(timeout()), this, SLOT(slotGrabFrames()));

    //定时器每30毫秒采集一次，也就是显示帧率大概维持在30帧每秒
    m_timer->start(30);
    qDebug()<< __FUNCTION__ << " id = "<< QThread::currentThreadId();
}

QImageAcquisition::~QImageAcquisition()
{
    //等待采集线程结束
    m_pThread->wait();

    //删除线程对象
    if (m_pThread != NULL)
    {
        delete m_pThread;
        m_pThread = NULL;
    }
}

bool QImageAcquisition::IsValidHandle(dvpHandle handle)
{
    bool bValidHandle = false;
    dvpIsValid(handle, &bValidHandle);

    return bValidHandle;
}

void QImageAcquisition::slotGrabFrames()
{
    dvpStatus status;
    status = dvpGetFrame(m_handle, &m_pFrame, &pBuffer, m_uGrabTimeout);

    if (status == DVP_STATUS_OK)
    {
        //这里将采集图像、图像转换放置在工作线程中实现，解决主界面在高帧率显示时卡顿问题
        if(m_pFrame.format==FORMAT_BGR24)
        {
            m_threadMutex.lock();

            // 保存原始图像数据，用于后续处理
            // 注意：不要修改原始数据，因为它会被相机驱动回收
            // 我们只在signalDisplay信号处理中读取和复制数据

#if (QT_VERSION >= QT_VERSION_CHECK(5,14,0))
            // Qt5.14版本新增QImage::Format_BGR888类型
            m_ShowImage = QPixmap::fromImage(QImage((uchar*)pBuffer,m_pFrame.iWidth, m_pFrame.iHeight,m_pFrame.iWidth*3, QImage::Format_BGR888,0,0)); // 5.13
#else
            //其他版本先把BGR数据转成RGB数据，再用RGB数据转QImage
            // 注意：这里我们创建一个副本，避免修改原始数据
            uchar* tempBuffer = new uchar[m_pFrame.iWidth * m_pFrame.iHeight * 3];
            memcpy(tempBuffer, pBuffer, m_pFrame.iWidth * m_pFrame.iHeight * 3);
            BGR2RGB(tempBuffer, m_pFrame.iWidth, m_pFrame.iHeight);
            m_ShowImage = QPixmap::fromImage(QImage(tempBuffer, m_pFrame.iWidth, m_pFrame.iHeight, m_pFrame.iWidth*3, QImage::Format_RGB888, [](void* ptr){ delete[] static_cast<uchar*>(ptr); }, tempBuffer));
#endif
            m_threadMutex.unlock();
            emit signalDisplay();
        }
    }
    else
    {
        // 使用msleep代替usleep，减少CPU占用
        QThread::msleep(5);
    }
}

void QImageAcquisition::StopThread()
{
    // 停止定时器
    if (m_timer && m_timer->isActive())
    {
        m_timer->stop();
        delete m_timer;
        m_timer = nullptr;
    }
    
    // 等待线程结束
    if(m_pThread->isRunning())
    {
        m_pThread->exit();
        m_pThread->wait(1000); // 等待1秒
    }
}


























