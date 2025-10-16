/**
 * @file SystemMonitor.cpp
 * @brief 系统监控类实现文件
 * @details 实现系统CPU、内存和温度的监控功能
 */

#include "SystemMonitor.h"
#include <QDebug>
#include <QProcess>
#include <QFile>
#include <QDir>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdio>

#ifdef __arm__
#include <unistd.h>
#include <sys/types.h>
#include <sys/sysinfo.h>
#endif

using namespace std;

SystemMonitor::SystemMonitor(QObject *parent)
    : QObject(parent)
    , m_updateTimer(new QTimer(this))
    , m_lastTotalUser(0)
    , m_lastTotalUserLow(0)
    , m_lastTotalSys(0)
    , m_lastTotalIdle(0)
{
    connect(m_updateTimer, &QTimer::timeout, this, &SystemMonitor::updateSystemStats);
}

SystemMonitor::~SystemMonitor()
{
    stopMonitoring();
}

void SystemMonitor::startMonitoring(int interval)
{
    m_updateTimer->start(interval);
}

void SystemMonitor::stopMonitoring()
{
    m_updateTimer->stop();
}

float SystemMonitor::GetCpuUsage()
{
    // 变量定义
    ifstream statFile;               ///< /proc/stat 文件流
    string line;                     ///< 读取的文件行内容
    istringstream iss;               ///< 字符串流用于解析数据
    string cpu;                      ///< CPU标识符
    unsigned long long totalUser;    ///< 用户态CPU时间
    unsigned long long totalUserLow; ///< 低优先级用户态CPU时间
    unsigned long long totalSys;     ///< 内核态CPU时间
    unsigned long long totalIdle;    ///< 空闲CPU时间
    unsigned long long iowait;       ///< I/O等待时间
    unsigned long long irq;          ///< 中断时间
    unsigned long long softirq;      ///< 软中断时间
    unsigned long long steal;        ///< 虚拟化环境中的steal时间

    // 直接尝试读取/proc/stat，不管是什么平台
    statFile.open("/proc/stat");
    
    if (!statFile.is_open()) 
    {
        qDebug() << "Can't open /proc/stat file";
        return 0.0f;
    }

    // 读取CPU统计信息
    if (getline(statFile, line)) 
    {
        iss.str(line);
        
        // 尝试读取所有CPU时间值
        if (!(iss >> cpu >> totalUser >> totalUserLow >> totalSys >> totalIdle >> iowait >> irq >> softirq >> steal))
        {
            qDebug() << "Failed to read CPU data: " << QString::fromStdString(line);
            return 0.0f;
        }
        
        // 首次运行，初始化数值
        if (m_lastTotalUser == 0) 
        {
            m_lastTotalUser = totalUser;
            m_lastTotalUserLow = totalUserLow;
            m_lastTotalSys = totalSys;
            m_lastTotalIdle = totalIdle;
            return 0.0f;
        }
        
        // 计算时间差
        unsigned long long userTimeDelta = totalUser - m_lastTotalUser;
        unsigned long long userLowDelta = totalUserLow - m_lastTotalUserLow;
        unsigned long long sysTimeDelta = totalSys - m_lastTotalSys;
        unsigned long long idleTimeDelta = totalIdle - m_lastTotalIdle;
        
        // 计算总的时间差
        unsigned long long totalDelta = userTimeDelta + userLowDelta + sysTimeDelta + idleTimeDelta;
        
        // 计算CPU使用率
        float cpuUsage = 0.0f;
        if (totalDelta > 0) 
        {
            cpuUsage = 100.0f * (totalDelta - idleTimeDelta) / totalDelta;
        }
        
        // 更新上一次的值
        m_lastTotalUser = totalUser;
        m_lastTotalUserLow = totalUserLow;
        m_lastTotalSys = totalSys;
        m_lastTotalIdle = totalIdle;
        
        return min(100.0f, max(0.0f, cpuUsage));
    }
    
    qDebug() << "Failed to read CPU data";
    return 0.0f;
}

float SystemMonitor::GetMemoryUsage()
{
    // 变量定义
    unsigned long totalMem = 0;    ///< 总内存大小 (KB)
    unsigned long freeMem = 0;     ///< 空闲内存大小 (KB)
    unsigned long buffers = 0;     ///< 缓冲区大小 (KB)
    unsigned long cached = 0;      ///< 缓存大小 (KB)
    string line;                   ///< 读取的每行数据

    // 尝试从/proc/meminfo读取内存信息
    ifstream memFile("/proc/meminfo");
    if (!memFile.is_open()) 
    {
        qDebug() << "Can't open /proc/meminfo file";
        return 0.0f;
    }
    
    while (getline(memFile, line)) 
    {
        if (line.find("MemTotal:") != string::npos) 
        {
            sscanf(line.c_str(), "MemTotal: %lu", &totalMem);
        } 
        else if (line.find("MemFree:") != string::npos) 
        {
            sscanf(line.c_str(), "MemFree: %lu", &freeMem);
        } 
        else if (line.find("Buffers:") != string::npos) 
        {
            sscanf(line.c_str(), "Buffers: %lu", &buffers);
        } 
        else if (line.find("Cached:") != string::npos) 
        {
            sscanf(line.c_str(), "Cached: %lu", &cached);
            break;  // 已经找到所需的所有信息
        }
    }

    if (totalMem == 0) 
    {
        qDebug() << "Unable to read the total memory capacity";
        return 0.0f;
    }

    // 计算实际使用的内存 (考虑缓存和缓冲区)
    unsigned long usedMem = totalMem - freeMem - buffers - cached;
    float memUsage = 100.0f * static_cast<float>(usedMem) / totalMem;
    
    return min(100.0f, max(0.0f, memUsage));
}

float SystemMonitor::GetTemperature()
{
    // 变量定义
    QFile tempFile;                 ///< 温度文件对象
    QString tempStr;                ///< 读取的温度字符串
    bool ok;                        ///< 转换结果标志
    float tempValue;                ///< 温度值

    // 尝试多个可能的温度传感器路径
    QStringList tempPaths = {
        "/sys/class/thermal/thermal_zone0/temp",  ///< 标准路径
        "/sys/devices/virtual/thermal/thermal_zone0/temp", ///< 虚拟设备路径
        "/sys/class/hwmon/hwmon0/temp1_input"    ///< hwmon路径
    };

    // 遍历所有可能的温度传感器路径
    for (const QString& path : tempPaths) 
    {
        tempFile.setFileName(path);
        
        if (tempFile.exists() && tempFile.open(QIODevice::ReadOnly)) 
        {
            tempStr = tempFile.readAll().trimmed();
            tempFile.close();
            
            tempValue = tempStr.toFloat(&ok);
            
            if (ok) 
            {
                // 根据读取值的范围来决定是否需要转换单位
                if (tempValue > 1000) 
                {
                    tempValue /= 1000.0f; // 转换为摄氏度
                }
                return tempValue;
            } 
            else 
            {
                qDebug() << "Failed to analyze temperature data:" << tempStr << "(Read from" << path << ")";
            }
        } 
        else 
        {
            qDebug() << "Can't open temperature file:" << path;
        }
    }

    // 如果以上方法都失败，尝试通过系统命令获取温度
    QProcess process;
    process.start("sh", QStringList() << "-c" << "cat /sys/class/thermal/thermal_zone*/temp");
    process.waitForFinished(1000);
    
    if (process.exitCode() == 0) 
    {
        QString output = process.readAllStandardOutput().trimmed();
        tempValue = output.toFloat(&ok) / 1000.0f;
        
        if (ok) 
        {
            return tempValue;
        }
    }

    qDebug() << "Can't get system temperature";
    return 0.0f;
}

bool SystemMonitor::GetThreadInfo(int &totalThreads, int &activeThreads)
{
    // 初始化返回值
    totalThreads = 0;
    activeThreads = 0;

    // 方法1: 通过/proc/stat获取系统总线程数
    ifstream statFile("/proc/stat");
    if (statFile.is_open()) 
    {
        string line;
        while (getline(statFile, line)) 
        {
            if (line.find("procs_running") != string::npos) 
            {
                istringstream iss(line);
                string key;
                iss >> key >> activeThreads;
            }
            else if (line.find("processes") != string::npos) 
            {
                istringstream iss(line);
                string key;
                iss >> key >> totalThreads;
            }
        }
        statFile.close();
        
        if (totalThreads > 0 && activeThreads > 0) 
        {
            return true;
        }
    }

    // 方法2: 通过系统命令获取线程信息
    QProcess process;
    process.start("sh", QStringList() << "-c" << "ps -eLf | wc -l");
    process.waitForFinished(1000);
    
    if (process.exitCode() == 0) 
    {
        QString output = process.readAllStandardOutput().trimmed();
        bool ok;
        int threadCount = output.toInt(&ok);
        
        if (ok && threadCount > 0) 
        {
            totalThreads = threadCount - 1; // 减去标题行
            activeThreads = totalThreads;   // 简化处理，假设所有线程都活跃
            return true;
        }
    }

    // 方法3: 通过/proc文件系统统计线程
    QDir procDir("/proc");
    QStringList processDirs = procDir.entryList(QDir::Dirs | QDir::NoDotAndDotDot);
    
    int threadCount = 0;
    for (const QString& dir : processDirs) 
    {
        bool isPid;
        dir.toInt(&isPid);
        if (isPid) 
        {
            QDir taskDir(QString("/proc/%1/task").arg(dir));
            if (taskDir.exists()) 
            {
                threadCount += taskDir.entryList(QDir::Dirs | QDir::NoDotAndDotDot).count();
            }
        }
    }
    
    if (threadCount > 0) 
    {
        totalThreads = threadCount;
        activeThreads = threadCount; // 简化处理
        return true;
    }

    qDebug() << "Can't get thread information";
    return false;
}

void SystemMonitor::updateSystemStats()
{
    // 获取系统状态数据
    float cpuUsage = GetCpuUsage();
    float memUsage = GetMemoryUsage();
    float temperature = GetTemperature();
    
    // 获取线程信息
    int totalThreads = 0;
    int activeThreads = 0;
    bool threadInfoAvailable = GetThreadInfo(totalThreads, activeThreads);
    
    // 如果无法获取线程信息，使用默认值
    if (!threadInfoAvailable) 
    {
        totalThreads = 0;
        activeThreads = 0;
    }
    
    // 发送系统状态更新信号
    emit systemStatsUpdated(cpuUsage, memUsage, temperature, totalThreads, activeThreads);
}
