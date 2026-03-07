/**
 * @file SystemMonitor.cpp
 * @brief 系统监控模块实现文件
 * 
 * 该文件实现了SystemMonitor类的所有方法，提供系统资源监控功能，
 * 包括CPU使用率、内存使用率和系统温度的实时监控。
 * 
 * @author VisualRobot Team
 * @date 2025-12-30
 * @version 1.0
 */

#include "SystemMonitor.h"
#include <QDebug>
#include <QProcess>
#include <QFile>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdio>
#include <QTime>

#ifdef __arm__
#include <unistd.h>
#include <sys/types.h>
#include <sys/sysinfo.h>
#endif

using namespace std;

/**
 * @brief SystemMonitor构造函数
 * 
 * 初始化系统监控对象，创建定时器并连接信号槽
 * 
 * @param parent 父对象指针
 * @note 初始化步骤：
 *       - 调用父类构造函数
 *       - 创建定时器对象
 *       - 初始化CPU统计变量为0
 *       - 连接定时器超时信号到更新系统状态槽函数
 * @see updateSystemStats()
 */
SystemMonitor::SystemMonitor(QObject *parent)
    : QObject(parent)
    , m_updateTimer(new QTimer(this))  // 创建定时器
    , m_lastTotalUser(0)               // 初始化CPU统计变量
    , m_lastTotalUserLow(0)
    , m_lastTotalSys(0)
    , m_lastTotalIdle(0)
{
    // 连接定时器超时信号到更新系统状态槽函数
    connect(m_updateTimer, &QTimer::timeout, this, &SystemMonitor::updateSystemStats);
}

/**
 * @brief SystemMonitor析构函数
 * 
 * 停止监控并释放资源
 * 
 * @note 调用stopMonitoring()停止定时器
 * @see stopMonitoring()
 */
SystemMonitor::~SystemMonitor()
{
    stopMonitoring();  // 停止监控
}

/**
 * @brief 开始监控系统状态
 * 
 * 启动定时器，开始定期更新系统状态
 * 
 * @param interval 监控更新间隔（毫秒）
 * @note 定时器启动后会定期触发updateSystemStats()槽函数
 * @see stopMonitoring(), updateSystemStats()
 */
void SystemMonitor::startMonitoring(int interval)
{
    m_updateTimer->start(interval);  // 启动定时器
}

/**
 * @brief 停止监控系统状态
 * 
 * 停止定时器，不再更新系统状态
 * 
 * @see startMonitoring()
 */
void SystemMonitor::stopMonitoring()
{
    m_updateTimer->stop();  // 停止定时器
}

/**
 * @brief 获取CPU使用率
 * 
 * 通过读取/proc/stat文件获取CPU统计信息，计算CPU使用率
 * 
 * @return CPU使用率（%），范围0-100
 * @note 计算原理：
 *       - 读取/proc/stat文件获取CPU时间统计
 *       - 计算时间差：用户态、内核态、空闲时间
 *       - CPU使用率 = (总时间 - 空闲时间) / 总时间 * 100%
 * @note 首次调用返回0.0，用于初始化基准值
 * @note 如果无法打开/proc/stat文件，返回0.0
 * @see m_lastTotalUser, m_lastTotalUserLow, m_lastTotalSys, m_lastTotalIdle
 */
float SystemMonitor::GetCpuUsage()
{
    // 变量定义
    ifstream statFile;               // /proc/stat 文件流
    string line;                     // 读取的文件行内容
    istringstream iss;               // 字符串流用于解析数据
    string cpu;                      // CPU标识符
    unsigned long long totalUser;    // 用户态CPU时间
    unsigned long long totalUserLow; // 低优先级用户态CPU时间
    unsigned long long totalSys;     // 内核态CPU时间
    unsigned long long totalIdle;    // 空闲CPU时间
    unsigned long long iowait;       // I/O等待时间
    unsigned long long irq;          // 中断时间
    unsigned long long softirq;      // 软中断时间
    unsigned long long steal;        // 虚拟化环境中的steal时间

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
            // CPU使用率 = (总时间 - 空闲时间) / 总时间 * 100%
            cpuUsage = 100.0f * (totalDelta - idleTimeDelta) / totalDelta;
        }
        
        // 更新上一次的值
        m_lastTotalUser = totalUser;
        m_lastTotalUserLow = totalUserLow;
        m_lastTotalSys = totalSys;
        m_lastTotalIdle = totalIdle;
        
        // 确保CPU使用率在0-100%之间
        return min(100.0f, max(0.0f, cpuUsage));
    }
    
    qDebug() << "Failed to read CPU data";
    return 0.0f;
}

/**
 * @brief 获取内存使用率
 * 
 * 通过读取/proc/meminfo文件获取内存信息，计算内存使用率
 * 
 * @return 内存使用率（%），范围0-100
 * @note 计算原理：
 *       - 读取MemTotal、MemFree、Buffers、Cached
 *       - 实际使用内存 = 总内存 - 空闲内存 - 缓冲区 - 缓存
 *       - 内存使用率 = 实际使用内存 / 总内存 * 100%
 * @note 考虑了Linux的内存缓存机制，计算更准确
 * @note 如果无法打开/proc/meminfo文件或总内存为0，返回0.0
 */
float SystemMonitor::GetMemoryUsage()
{
    // 变量定义
    unsigned long totalMem = 0;    // 总内存大小 (KB) 
    unsigned long freeMem = 0;     // 空闲内存大小 (KB) 
    unsigned long buffers = 0;     // 缓冲区大小 (KB) 
    unsigned long cached = 0;      // 缓存大小 (KB) 
    string line;                   // 读取的每行数据

    // 尝试从/proc/meminfo读取内存信息
    ifstream memFile("/proc/meminfo");
    if (!memFile.is_open()) 
    {
        qDebug() << "Can't open /proc/meminfo file";
        return 0.0f;
    }
    
    // 读取内存信息
    while (getline(memFile, line)) 
    {
        if (line.find("MemTotal:") != string::npos) 
        {
            sscanf(line.c_str(), "MemTotal: %lu", &totalMem);  // 读取总内存
        } 
        else if (line.find("MemFree:") != string::npos) 
        {
            sscanf(line.c_str(), "MemFree: %lu", &freeMem);  // 读取空闲内存
        } 
        else if (line.find("Buffers:") != string::npos) 
        {
            sscanf(line.c_str(), "Buffers: %lu", &buffers);  // 读取缓冲区大小
        } 
        else if (line.find("Cached:") != string::npos) 
        {
            sscanf(line.c_str(), "Cached: %lu", &cached);  // 读取缓存大小
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
    
    // 确保内存使用率在0-100%之间
    return min(100.0f, max(0.0f, memUsage));
}

/**
 * @brief 获取系统温度
 * 
 * 尝试从多个可能的温度传感器路径读取温度，失败则尝试通过系统命令获取
 * 
 * @return 系统温度（摄氏度）
 * @note 尝试的路径：
 *       - /sys/class/thermal/thermal_zone0/temp
 *       - /sys/devices/virtual/thermal/thermal_zone0/temp
 *       - /sys/class/hwmon/hwmon0/temp1_input
 * @note 如果文件读取失败，尝试通过系统命令获取
 * @note 如果温度值大于1000，会自动转换为摄氏度（除以1000）
 * @note 如果所有方法都失败，返回0.0
 */
float SystemMonitor::GetTemperature()
{
    // 变量定义
    QFile tempFile;                 // 温度文件对象
    QString tempStr;                // 读取的温度字符串
    bool ok;                        // 转换结果标志
    float tempValue;                // 温度值

    // 尝试多个可能的温度传感器路径
    QStringList tempPaths = {
        "/sys/class/thermal/thermal_zone0/temp",  // 标准路径
        "/sys/devices/virtual/thermal/thermal_zone0/temp", // 虚拟设备路径
        "/sys/class/hwmon/hwmon0/temp1_input"    // hwmon路径
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

/**
 * @brief 更新系统状态槽函数
 * 
 * 定期调用该函数获取系统状态并发送信号
 * 
 * @note 处理流程：
 *       1. 调用GetCpuUsage()获取CPU使用率
 *       2. 调用GetMemoryUsage()获取内存使用率
 *       3. 调用GetTemperature()获取系统温度
 *       4. 发出systemStatsUpdated()信号
 * @note 每60秒输出一次详细的内存监控日志，包括实际使用量和总容量
 * @see GetCpuUsage(), GetMemoryUsage(), GetTemperature(), systemStatsUpdated()
 */
void SystemMonitor::updateSystemStats()
{
    // 获取系统状态数据
    float cpuUsage = GetCpuUsage();        // 获取CPU使用率
    float memUsage = GetMemoryUsage();     // 获取内存使用率
    float temperature = GetTemperature();  // 获取系统温度
    
    // 发送系统状态更新信号
    emit systemStatsUpdated(cpuUsage, memUsage, temperature);
    
    // 添加内存监控日志，每分钟记录一次内存使用趋势（增强版：包括实际使用量）
    static QTime lastLogTime = QTime::currentTime();
    if (lastLogTime.msecsTo(QTime::currentTime()) > 60000) 
    {  // 每60秒
        // 计算实际内存使用量 (MB)
        unsigned long totalMemKB = 0, usedMemKB = 0;
        ifstream memFile("/proc/meminfo");
        if (memFile.is_open()) 
        {
            string line;
            while (getline(memFile, line)) 
            {
                if (line.find("MemTotal:") != string::npos) 
                {
                    sscanf(line.c_str(), "MemTotal: %lu", &totalMemKB);
                } 
                else if (line.find("MemFree:") != string::npos) 
                {
                    unsigned long freeMemKB = 0, buffersKB = 0, cachedKB = 0;
                    sscanf(line.c_str(), "MemFree: %lu", &freeMemKB);
                    // 简单估算：读取后续Buffers和Cached
                    if (getline(memFile, line) && line.find("Buffers:") != string::npos) 
                    {
                        sscanf(line.c_str(), "Buffers: %lu", &buffersKB);
                    }
                    if (getline(memFile, line) && line.find("Cached:") != string::npos) 
                    {
                        sscanf(line.c_str(), "Cached: %lu", &cachedKB);
                    }
                    usedMemKB = totalMemKB - freeMemKB - buffersKB - cachedKB;
                    break;
                }
            }
            memFile.close();
        }
        double usedMemMB = usedMemKB / 1024.0;
        double totalMemMB = totalMemKB / 1024.0;
        
        QString logEntry = QString("内存监控 - 使用率: %1%, 已用: %2 MB / 总: %3 MB, 时间: %4")
                              .arg(memUsage, 0, 'f', 2)
                              .arg(usedMemMB, 0, 'f', 1)
                              .arg(totalMemMB, 0, 'f', 1)
                              .arg(QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss"));
        qDebug() << logEntry;  // 输出到控制台，可扩展到文件日志
        // TODO: 可集成到主程序的AppendLog系统
        lastLogTime = QTime::currentTime();
    }
}
