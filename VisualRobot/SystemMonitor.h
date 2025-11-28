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
     * @brief 开始监控系统状态
     * @param interval 监控更新间隔（毫秒），默认1000毫秒
     */
    void startMonitoring(int interval = 1000);  // 默认每秒更新一次
    
    /**
     * @brief 停止监控系统状态
     */
    void stopMonitoring();

signals:
    /**
     * @brief 系统状态更新信号
     * @param cpuUsage CPU使用率（%）
     * @param memUsage 内存使用率（%）
     * @param temperature 系统温度（摄氏度）
     */
    void systemStatsUpdated(float cpuUsage, float memUsage, float temperature);

private slots:
    /**
     * @brief 更新系统状态槽函数
     * 
     * 定期调用该函数获取系统状态并发送信号
     */
    void updateSystemStats();

private:
    /**
     * @brief 获取CPU使用率
     * @return CPU使用率（%）
     */
    float GetCpuUsage();
    
    /**
     * @brief 获取内存使用率
     * @return 内存使用率（%）
     */
    float GetMemoryUsage();
    
    /**
     * @brief 获取系统温度
     * @return 系统温度（摄氏度）
     */
    float GetTemperature();
    
    QTimer *m_updateTimer;  // 定时更新系统状态的定时器
    
    // 用于CPU使用率计算的辅助变量
    unsigned long long m_lastTotalUser;     // 上一次的用户态CPU时间
    unsigned long long m_lastTotalUserLow;  // 上一次的低优先级用户态CPU时间
    unsigned long long m_lastTotalSys;      // 上一次的内核态CPU时间
    unsigned long long m_lastTotalIdle;     // 上一次的空闲CPU时间
};

#endif // SYSTEMMONITOR_H
