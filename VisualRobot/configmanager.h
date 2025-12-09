#ifndef CONFIGMANAGER_H
#define CONFIGMANAGER_H

#include <QObject>
#include <QJsonObject>
#include <QJsonDocument>
#include <QFile>
#include <QDir>
#include <QDebug>
#include <QJsonArray>
#include <QJsonValue>

class ConfigManager : public QObject
{
    Q_OBJECT
public:
    explicit ConfigManager(QObject *parent = nullptr);
    ~ConfigManager();
    
    // 初始化配置管理器
    bool init(const QString& configFilePath = "");
    
    // 加载指定硬件配置
    bool loadHardwareConfig(const QString& configName);
    
    // 获取当前硬件配置名称
    QString getCurrentConfigName() const;
    
    // 获取所有可用的硬件配置名称
    QStringList getAvailableConfigs() const;
    
    // 获取依赖库信息
    QString getDependencyIncludePath(const QString& depName) const;
    QString getDependencyLibPath(const QString& depName) const;
    QStringList getDependencyLibs(const QString& depName) const;
    
    // 获取加速选项
    bool isAcceleratorEnabled(const QString& accName) const;
    
    // 获取路径配置
    QString getModelPath() const;
    QString getLabelPath() const;
    QString getImagePath() const;
    
    // 获取优化参数
    int getIntraOpNumThreads() const;
    int getInterOpNumThreads() const;
    QString getGraphOptimizationLevel() const;
    QString getExecutionMode() const;
    bool getEnableCpuMemArena() const;
    bool getEnableMemPattern() const;
    
    // 获取硬件信息
    QString getHardwareName() const;
    QString getHardwareModel() const;
    QString getHardwareSystem() const;
    QString getHardwareArchitecture() const;
    
    // 静态方法：获取单例实例
    static ConfigManager* instance();
    
private:
    // 解析配置文件
    bool parseConfigFile(const QString& filePath);
    
    // 当前配置
    QString currentConfigName_;
    QJsonObject currentConfig_;
    
    // 所有配置
    QJsonObject allConfigs_;
    
    // 单例实例
    static ConfigManager* instance_;
};

#endif // CONFIGMANAGER_H
