/**
 * @file mainwindow_systemstats.cpp
 * @brief MainWindow 类的系统监控相关实现
 * 
 * 此文件包含 MainWindow 类的系统监控相关功能实现，
 * 包括CPU、内存、温度等系统状态的更新和显示。
 */

#include "mainwindow.h"
#include "ui_mainwindow.h"

/**
 * @brief 更新系统状态显示
 * 
 * 该函数接收系统监控数据并更新状态栏中的显示标签，
 * 同时根据数值大小设置不同的颜色提示。
 * 
 * @param cpuUsage CPU使用率（百分比）
 * @param memUsage 内存使用率（百分比）
 * @param temperature 系统温度（摄氏度）
 * @param totalThreads 总线程数
 * @param activeThreads 活跃线程数
 */
void MainWindow::updateSystemStats(float cpuUsage, float memUsage, float temperature, int totalThreads, int activeThreads)
{
    // 更新标签文本显示
    m_cpuLabel->setText(QString("CPU: %1%").arg(cpuUsage, 0, 'f', 1));
    m_memLabel->setText(QString("内存: %1%").arg(memUsage, 0, 'f', 1));
    m_tempLabel->setText(QString("温度: %1°C").arg(temperature, 0, 'f', 1));
    m_threadLabel->setText(QString("线程: %1/%2").arg(activeThreads).arg(totalThreads));

    /**
     * @brief 设置标签颜色
     * 
     * 根据数值大小设置标签颜色：
     * - 正常：白色
     * - 警告：黄色（超过阈值1）
     * - 危险：红色（超过阈值2）
     * 
     * @param label 标签指针
     * @param value 数值
     * @param threshold1 警告阈值
     * @param threshold2 危险阈值
     */
    auto setLabelColor = [](QLabel* label, float value, float threshold1, float threshold2) 
    {
        QString style = "QLabel { color: %1; background-color: rgba(0, 0, 0, 150); padding: 5px; border-radius: 5px; }";
        if (value >= threshold2) 
        {
            label->setStyleSheet(style.arg("red"));        // 危险状态：红色
        } 
        else if (value >= threshold1) 
        {
            label->setStyleSheet(style.arg("yellow"));     // 警告状态：黄色
        } 
        else 
        {
            label->setStyleSheet(style.arg("white"));      // 正常状态：白色
        }
    };

    /**
     * @brief 设置线程标签颜色
     * 
     * 根据线程负载设置颜色：
     * - 正常：白色（线程数正常）
     * - 警告：黄色（线程数较多）
     * - 危险：红色（线程数过多）
     * 
     * @param label 标签指针
     * @param activeThreads 活跃线程数
     * @param threshold1 警告阈值
     * @param threshold2 危险阈值
     */
    auto setThreadLabelColor = [](QLabel* label, int activeThreads, int threshold1, int threshold2) 
    {
        QString style = "QLabel { color: %1; background-color: rgba(0, 0, 0, 150); padding: 5px; border-radius: 5px; }";
        if (activeThreads >= threshold2) 
        {
            label->setStyleSheet(style.arg("red"));        // 危险状态：红色
        } 
        else if (activeThreads >= threshold1) 
        {
            label->setStyleSheet(style.arg("yellow"));     // 警告状态：黄色
        } 
        else 
        {
            label->setStyleSheet(style.arg("white"));      // 正常状态：白色
        }
    };

    // 设置各标签的颜色阈值
    setLabelColor(m_cpuLabel, cpuUsage, 70, 90);    // CPU 使用率阈值：70%警告，90%危险
    setLabelColor(m_memLabel, memUsage, 80, 90);    // 内存使用率阈值：80%警告，90%危险
    setLabelColor(m_tempLabel, temperature, 70, 85); // 温度阈值：70°C警告，85°C危险
    
    // 设置线程标签的颜色阈值（根据系统负载调整）
    int threadThreshold1 = 500;  // 警告阈值：500个活跃线程
    int threadThreshold2 = 1000; // 危险阈值：1000个活跃线程
    setThreadLabelColor(m_threadLabel, activeThreads, threadThreshold1, threadThreshold2);
}
