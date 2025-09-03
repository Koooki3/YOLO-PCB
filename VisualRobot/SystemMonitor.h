#ifndef SYSTEMMONITOR_H
#define SYSTEMMONITOR_H

#include <QObject>
#include <QTimer>
#include <QString>
#include <QFile>

class SystemMonitor : public QObject
{
    Q_OBJECT

public:
    explicit SystemMonitor(QObject *parent = nullptr);
    ~SystemMonitor();

    void startMonitoring(int interval = 1000);  // 默认每秒更新一次
    void stopMonitoring();

signals:
    void systemStatsUpdated(float cpuUsage, float memUsage, float gpuUsage, float temperature);

private slots:
    void updateSystemStats();

private:
    QTimer *m_updateTimer;
    
    float getCpuUsage();
    float getMemoryUsage();
    float getGpuUsage();
    float getTemperature();
    
    // 用于CPU使用率计算的辅助变量
    unsigned long long m_lastTotalUser;
    unsigned long long m_lastTotalUserLow;
    unsigned long long m_lastTotalSys;
    unsigned long long m_lastTotalIdle;
};

#endif // SYSTEMMONITOR_H
