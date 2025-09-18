#include "SystemMonitor.h"
#include <QDebug>
#include <QProcess>
#include <QFile>
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

float SystemMonitor::getCpuUsage()
{
    // 直接尝试读取/proc/stat，不管是什么平台
    ifstream statFile("/proc/stat");
    if (!statFile.is_open()) 
    {
        qDebug() << "Can't open /proc/stat file";
        return 0.0f;
    }

    string line;
    if (getline(statFile, line)) 
    {
        istringstream iss(line);
        string cpu;
        unsigned long long totalUser, totalUserLow, totalSys, totalIdle, iowait, irq, softirq, steal;
        
        // 尝试读取所有CPU时间值
        if (!(iss >> cpu >> totalUser >> totalUserLow >> totalSys >> totalIdle 
              >> iowait >> irq >> softirq >> steal)) 
        {
            qDebug() << "Failed to read CPU data: " << QString::fromStdString(line);
            return 0.0f;
        }
        
        if (m_lastTotalUser == 0) 
        {
            // 首次运行，初始化数值
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
        unsigned long long totalDelta = userTimeDelta + userLowDelta + 
                                      sysTimeDelta + idleTimeDelta;
        
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

float SystemMonitor::getMemoryUsage()
{
    // 尝试从/proc/meminfo读取内存信息
    ifstream memFile("/proc/meminfo");
    if (!memFile.is_open()) 
    {
        qDebug() << "Can't open /proc/meminfo file";
        return 0.0f;
    }

    unsigned long totalMem = 0, freeMem = 0, buffers = 0, cached = 0;
    string line;
    
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

    // 计算实际使用的内存（考虑缓存和缓冲区）
    unsigned long usedMem = totalMem - freeMem - buffers - cached;
    float memUsage = 100.0f * static_cast<float>(usedMem) / totalMem;
    
    return min(100.0f, max(0.0f, memUsage));
}

float SystemMonitor::getTemperature()
{
    // 尝试多个可能的温度传感器路径
    QStringList tempPaths = {
        "/sys/class/thermal/thermal_zone0/temp",  // 标准路径
        "/sys/devices/virtual/thermal/thermal_zone0/temp", // 虚拟设备路径
        "/sys/class/hwmon/hwmon0/temp1_input"    // hwmon路径
    };

    for (const QString& path : tempPaths) 
    {
        QFile tempFile(path);
        if (tempFile.exists() && tempFile.open(QIODevice::ReadOnly)) 
        {
            QString tempStr = tempFile.readAll().trimmed();
            tempFile.close();
            
            bool ok;
            float temp = tempStr.toFloat(&ok);
            if (ok) 
            {
                // 根据读取值的范围来决定是否需要转换单位
                if (temp > 1000) 
                {
                    temp /= 1000.0f; // 转换为摄氏度
                }
                return temp;
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
        bool ok;
        float temp = output.toFloat(&ok) / 1000.0f;
        if (ok) 
        {
            return temp;
        }
    }

    qDebug() << "Can't get system temperature";
    return 0.0f;
}

void SystemMonitor::updateSystemStats()
{
    float cpuUsage = getCpuUsage();
    float memUsage = getMemoryUsage();
    float temperature = getTemperature();
    
    emit systemStatsUpdated(cpuUsage, memUsage, temperature);
}
