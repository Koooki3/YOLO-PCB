// 此文件包含 MainWindow 类的系统监控相关实现

#include "mainwindow.h"
#include "ui_mainwindow.h"

void MainWindow::updateSystemStats(float cpuUsage, float memUsage, float temperature)
{
    m_cpuLabel->setText(QString("CPU: %1%").arg(cpuUsage, 0, 'f', 1));
    m_memLabel->setText(QString("内存: %1%").arg(memUsage, 0, 'f', 1));
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
    setLabelColor(m_tempLabel, temperature, 70, 85); // 温度阈值
}
