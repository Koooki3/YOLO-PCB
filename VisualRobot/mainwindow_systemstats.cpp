// 此文件包含 MainWindow 类的系统监控相关实现

#include "mainwindow.h"
#include "ui_mainwindow.h"

void MainWindow::updateSystemStats(float cpuUsage, float memUsage, float gpuUsage, float temperature)
{
    m_cpuLabel->setText(QString("CPU: %1%").arg(cpuUsage, 0, 'f', 1));
    m_memLabel->setText(QString("内存: %1%").arg(memUsage, 0, 'f', 1));
    m_gpuLabel->setText(QString("GPU: %1%").arg(gpuUsage, 0, 'f', 1));
    m_tempLabel->setText(QString("温度: %1°C").arg(temperature, 0, 'f', 1));

    // 根据数值设置颜色
    auto setLabelColor = [](QLabel* label, float value, float threshold1, float threshold2) {
        QString style = "QLabel { color: %1; background-color: rgba(0, 0, 0, 150); padding: 5px; border-radius: 5px; }";
        if (value >= threshold2) {
            label->setStyleSheet(style.arg("red"));
        } else if (value >= threshold1) {
            label->setStyleSheet(style.arg("yellow"));
        } else {
            label->setStyleSheet(style.arg("white"));
        }
    };

    setLabelColor(m_cpuLabel, cpuUsage, 70, 90);    // CPU 使用率阈值
    setLabelColor(m_memLabel, memUsage, 80, 90);    // 内存使用率阈值
    setLabelColor(m_gpuLabel, gpuUsage, 70, 90);    // GPU 使用率阈值
    setLabelColor(m_tempLabel, temperature, 70, 85); // 温度阈值

    // 记录到日志（只在数值显著变化时）
    static float lastCpuUsage = 0;
    static float lastMemUsage = 0;
    static float lastGpuUsage = 0;
    static float lastTemp = 0;

    const float threshold = 5.0f; // 变化阈值

    if (std::abs(cpuUsage - lastCpuUsage) > threshold) {
        appendLog(QString("CPU使用率变化: %1%").arg(cpuUsage), INFO);
        lastCpuUsage = cpuUsage;
    }
    if (std::abs(memUsage - lastMemUsage) > threshold) {
        appendLog(QString("内存使用率变化: %1%").arg(memUsage), INFO);
        lastMemUsage = memUsage;
    }
    if (std::abs(gpuUsage - lastGpuUsage) > threshold) {
        appendLog(QString("GPU使用率变化: %1%").arg(gpuUsage), INFO);
        lastGpuUsage = gpuUsage;
    }
    if (std::abs(temperature - lastTemp) > 2.0f) {  // 温度使用更小的阈值
        appendLog(QString("系统温度变化: %1°C").arg(temperature), INFO);
        lastTemp = temperature;
    }
}
