#include "SystemMonitor.h"
#include <QDebug>
#include <QProcess>
#include <cmath>

#ifdef __arm__
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <sys/types.h>
#include <sys/sysinfo.h>
#endif

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
#ifdef __arm__
    std::ifstream statFile("/proc/stat");
    std::string line;
    if (std::getline(statFile, line)) {
        std::istringstream iss(line);
        std::string cpu;
        unsigned long long totalUser, totalUserLow, totalSys, totalIdle;
        
        iss >> cpu >> totalUser >> totalUserLow >> totalSys >> totalIdle;
        
        if (m_lastTotalUser == 0) {
            // 首次运行，初始化数值
            m_lastTotalUser = totalUser;
            m_lastTotalUserLow = totalUserLow;
            m_lastTotalSys = totalSys;
            m_lastTotalIdle = totalIdle;
            return 0.0f;
        }
        
        unsigned long long totalDelta = (totalUser - m_lastTotalUser) +
                                      (totalUserLow - m_lastTotalUserLow) +
                                      (totalSys - m_lastTotalSys);
        unsigned long long idleDelta = totalIdle - m_lastTotalIdle;
        
        float cpuUsage = static_cast<float>(totalDelta - idleDelta) / totalDelta * 100.0f;
        
        // 更新上一次的值
        m_lastTotalUser = totalUser;
        m_lastTotalUserLow = totalUserLow;
        m_lastTotalSys = totalSys;
        m_lastTotalIdle = totalIdle;
        
        return std::min(100.0f, std::max(0.0f, cpuUsage));
    }
#endif
    return 0.0f;
}

float SystemMonitor::getMemoryUsage()
{
#ifdef __arm__
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        unsigned long totalRam = si.totalram;
        unsigned long freeRam = si.freeram;
        float memUsage = (1.0f - static_cast<float>(freeRam) / totalRam) * 100.0f;
        return std::min(100.0f, std::max(0.0f, memUsage));
    }
#endif
    return 0.0f;
}

float SystemMonitor::getGpuUsage()
{
#ifdef __arm__
    // RK3588 GPU使用率通过mali节点获取
    QFile gpuFile("/sys/class/mali/utilization");
    if (gpuFile.open(QIODevice::ReadOnly)) {
        QString gpuStr = gpuFile.readAll().trimmed();
        gpuFile.close();
        bool ok;
        float gpuUsage = gpuStr.toFloat(&ok);
        if (ok) {
            return std::min(100.0f, std::max(0.0f, gpuUsage));
        }
    }
#endif
    return 0.0f;
}

float SystemMonitor::getTemperature()
{
#ifdef __arm__
    // RK3588温度通过thermal_zone节点获取
    QFile tempFile("/sys/class/thermal/thermal_zone0/temp");
    if (tempFile.open(QIODevice::ReadOnly)) {
        QString tempStr = tempFile.readAll().trimmed();
        tempFile.close();
        bool ok;
        float temp = tempStr.toFloat(&ok) / 1000.0f; // 转换为摄氏度
        if (ok) {
            return temp;
        }
    }
#endif
    return 0.0f;
}

void SystemMonitor::updateSystemStats()
{
    float cpuUsage = getCpuUsage();
    float memUsage = getMemoryUsage();
    float gpuUsage = getGpuUsage();
    float temperature = getTemperature();
    
    emit systemStatsUpdated(cpuUsage, memUsage, gpuUsage, temperature);
}
