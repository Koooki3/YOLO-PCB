/**
 * @file SystemMonitor.h
 * @brief 系统监控模块头文件
 * 
 * 该文件定义了SystemMonitor类，提供系统资源监控功能，包括CPU使用率、内存使用率
 * 和系统温度的实时监控，通过定时器定期发送系统状态更新信号。
 * 
 * @author VisualRobot Team
 * @date 2025-12-30
 * @version 1.0
 */

#ifndef SYSTEMMONITOR_H
#define SYSTEMMONITOR_H

#include <QObject>
#include <QTimer>
#include <QString>
#include <QFile>

/**
 * @brief 系统监控类
 * 
 * 该类用于监控系统的CPU使用率、内存使用率和系统温度，并通过信号定期发送系统状态更新
 * 
 * @note 该类基于Linux系统的/proc文件系统获取系统信息
 * @note 支持ARM架构的嵌入式系统
 * @see startMonitoring(), stopMonitoring(), systemStatsUpdated()
 */
class SystemMonitor : public QObject
{
    Q_OBJECT

public:
    /**
     * @brief 构造函数
     * 
     * 初始化系统监控对象，创建定时器并连接信号槽
     * 
     * @param parent 父对象指针，默认为nullptr
     * @note 初始化步骤：
     *       - 创建定时器对象
     *       - 初始化CPU统计变量为0
     *       - 连接定时器超时信号到更新系统状态槽函数
     * @see updateSystemStats()
     */
    explicit SystemMonitor(QObject *parent = nullptr);
    
    /**
     * @brief 析构函数
     * 
     * 停止监控并释放资源
     * 
     * @note 调用stopMonitoring()停止定时器
     * @see stopMonitoring()
     */
    ~SystemMonitor();

    /**
     * @brief 开始监控系统状态
     * 
     * 启动定时器，开始定期更新系统状态
     * 
     * @param interval 监控更新间隔（毫秒），默认1000毫秒
     * @note 定时器启动后会定期触发updateSystemStats()槽函数
     * @see stopMonitoring(), updateSystemStats()
     */
    void startMonitoring(int interval = 1000);
    
    /**
     * @brief 停止监控系统状态
     * 
     * 停止定时器，不再更新系统状态
     * 
     * @see startMonitoring()
     */
    void stopMonitoring();

signals:
    /**
     * @brief 系统状态更新信号
     * 
     * 当监控到系统状态变化时发出此信号
     * 
     * @param cpuUsage CPU使用率（%），范围0-100
     * @param memUsage 内存使用率（%），范围0-100
     * @param temperature 系统温度（摄氏度）
     * @note 该信号由updateSystemStats()槽函数定期发出
     * @see updateSystemStats()
     */
    void systemStatsUpdated(float cpuUsage, float memUsage, float temperature);

private slots:
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
     * @note 每60秒输出一次详细的内存监控日志
     * @see GetCpuUsage(), GetMemoryUsage(), GetTemperature()
     */
    void updateSystemStats();

private:
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
     * @see m_lastTotalUser, m_lastTotalUserLow, m_lastTotalSys, m_lastTotalIdle
     */
    float GetCpuUsage();
    
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
     */
    float GetMemoryUsage();
    
    /**
     * @brief 获取系统温度
     * 
     * 尝试从多个可能的温度传感器路径读取温度
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
    float GetTemperature();
    
    QTimer *m_updateTimer;  ///< 定时更新系统状态的定时器
    
    // 用于CPU使用率计算的辅助变量
    unsigned long long m_lastTotalUser;     ///< 上一次的用户态CPU时间
    unsigned long long m_lastTotalUserLow;  ///< 上一次的低优先级用户态CPU时间
    unsigned long long m_lastTotalSys;      ///< 上一次的内核态CPU时间
    unsigned long long m_lastTotalIdle;     ///< 上一次的空闲CPU时间
};

#endif // SYSTEMMONITOR_H
