/**
 * @file SystemMonitor.h
 * @brief 系统监控类头文件
 * @details 提供CPU使用率、内存使用率和系统温度的监控功能
 */

#ifndef SYSTEMMONITOR_H
#define SYSTEMMONITOR_H

#include <QObject>
#include <QTimer>
#include <QString>
#include <QFile>

/**
 * @class SystemMonitor
 * @brief 系统监控类
 * 
 * 该类用于监控系统的CPU使用率、内存使用率和系统温度，
 * 并通过信号机制实时更新系统状态信息。
 */
class SystemMonitor : public QObject
{
    Q_OBJECT

public:
    /**
     * @brief 构造函数
     * @param parent 父对象指针
     */
    explicit SystemMonitor(QObject *parent = nullptr);
    
    /**
     * @brief 析构函数
     */
    ~SystemMonitor();

    /**
     * @brief 开始监控
     * @param interval 监控间隔时间，单位毫秒，默认1000ms
     */
    void startMonitoring(int interval = 1000);
    
    /**
     * @brief 停止监控
     */
    void stopMonitoring();

signals:
    /**
     * @brief 系统状态更新信号
     * @param cpuUsage CPU使用率（百分比）
     * @param memUsage 内存使用率（百分比）
     * @param temperature 系统温度（摄氏度）
     * @param totalThreads 总线程数
     * @param activeThreads 活跃线程数
     */
    void systemStatsUpdated(float cpuUsage, float memUsage, float temperature, int totalThreads, int activeThreads);

private slots:
    /**
     * @brief 更新系统状态
     */
    void updateSystemStats();

private:
    QTimer *m_updateTimer;  ///< 定时器，用于定期更新系统状态
    
    /**
     * @brief 获取CPU使用率
     * @return CPU使用率（百分比）
     */
    float GetCpuUsage();
    
    /**
     * @brief 获取内存使用率
     * @return 内存使用率（百分比）
     */
    float GetMemoryUsage();
    
    /**
     * @brief 获取系统温度
     * @return 系统温度（摄氏度）
     */
    float GetTemperature();
    
    /**
     * @brief 获取线程信息
     * @param totalThreads 总线程数
     * @param activeThreads 活跃线程数
     * @return 成功返回true，失败返回false
     */
    bool GetThreadInfo(int &totalThreads, int &activeThreads);
    
    // 用于CPU使用率计算的辅助变量
    unsigned long long m_lastTotalUser;     ///< 上一次用户态CPU时间
    unsigned long long m_lastTotalUserLow;  ///< 上一次低优先级用户态CPU时间
    unsigned long long m_lastTotalSys;      ///< 上一次内核态CPU时间
    unsigned long long m_lastTotalIdle;     ///< 上一次空闲CPU时间
};

#endif // SYSTEMMONITOR_H
