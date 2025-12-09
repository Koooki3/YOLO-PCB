#include "configmanager.h"
#include <QJsonArray>
#include <QJsonObject>
#include <QJsonValue>
#include <QJsonDocument>

// 初始化单例实例
ConfigManager* ConfigManager::instance_ = nullptr;

ConfigManager::ConfigManager(QObject *parent) 
    : QObject(parent)
    , currentConfigName_("")
{
}

ConfigManager::~ConfigManager()
{
}

ConfigManager* ConfigManager::instance()
{
    if (!instance_)
    {
        instance_ = new ConfigManager();
    }
    return instance_;
}

bool ConfigManager::init(const QString& configFilePath)
{
    QString filePath = configFilePath;
    
    // 如果没有提供文件路径，使用默认路径
    if (filePath.isEmpty())
    {
        // 尝试从应用程序目录获取配置文件
        filePath = QDir::currentPath() + "/hardware_config.json";
        
        // 如果不存在，尝试从资源文件获取
        if (!QFile::exists(filePath))
        {
            filePath = ":/hardware_config.json";
        }
    }
    
    qDebug() << "Loading config file:" << filePath;
    
    // 解析配置文件
    if (!parseConfigFile(filePath))
    {
        qWarning() << "Failed to parse config file:" << filePath;
        return false;
    }
    
    // 加载默认配置
    QString defaultConfig = allConfigs_.value("default_config").toString();
    if (defaultConfig.isEmpty())
    {
        qWarning() << "No default config specified in config file";
        return false;
    }
    
    return loadHardwareConfig(defaultConfig);
}

bool ConfigManager::parseConfigFile(const QString& filePath)
{
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        qWarning() << "Failed to open config file:" << filePath;
        return false;
    }
    
    QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
    if (doc.isNull() || !doc.isObject())
    {
        qWarning() << "Invalid JSON format in config file:" << filePath;
        return false;
    }
    
    allConfigs_ = doc.object();
    return true;
}

bool ConfigManager::loadHardwareConfig(const QString& configName)
{
    QJsonObject hardwareConfigs = allConfigs_.value("hardware_configs").toObject();
    if (hardwareConfigs.isEmpty())
    {
        qWarning() << "No hardware configs found in config file";
        return false;
    }
    
    QJsonObject config = hardwareConfigs.value(configName).toObject();
    if (config.isEmpty())
    {
        qWarning() << "Hardware config not found:" << configName;
        return false;
    }
    
    currentConfigName_ = configName;
    currentConfig_ = config;
    
    qDebug() << "Loaded hardware config:" << configName;
    return true;
}

QString ConfigManager::getCurrentConfigName() const
{
    return currentConfigName_;
}

QStringList ConfigManager::getAvailableConfigs() const
{
    QJsonObject hardwareConfigs = allConfigs_.value("hardware_configs").toObject();
    return hardwareConfigs.keys();
}

QString ConfigManager::getDependencyIncludePath(const QString& depName) const
{
    QJsonObject deps = currentConfig_.value("dependencies").toObject();
    QJsonObject dep = deps.value(depName).toObject();
    return dep.value("include_path").toString();
}

QString ConfigManager::getDependencyLibPath(const QString& depName) const
{
    QJsonObject deps = currentConfig_.value("dependencies").toObject();
    QJsonObject dep = deps.value(depName).toObject();
    return dep.value("lib_path").toString();
}

QStringList ConfigManager::getDependencyLibs(const QString& depName) const
{
    QStringList libs;
    QJsonObject deps = currentConfig_.value("dependencies").toObject();
    QJsonObject dep = deps.value(depName).toObject();
    QJsonArray libArray = dep.value("libs").toArray();
    
    for (const QJsonValue& lib : libArray)
    {
        libs << lib.toString();
    }
    
    return libs;
}

bool ConfigManager::isAcceleratorEnabled(const QString& accName) const
{
    QJsonObject accs = currentConfig_.value("accelerators").toObject();
    return accs.value(accName).toBool(false);
}

QString ConfigManager::getModelPath() const
{
    QJsonObject paths = currentConfig_.value("paths").toObject();
    return paths.value("models").toString("../models");
}

QString ConfigManager::getLabelPath() const
{
    QJsonObject paths = currentConfig_.value("paths").toObject();
    return paths.value("labels").toString("../Data/Labels");
}

QString ConfigManager::getImagePath() const
{
    QJsonObject paths = currentConfig_.value("paths").toObject();
    return paths.value("images").toString("../Img");
}

int ConfigManager::getIntraOpNumThreads() const
{
    QJsonObject opt = currentConfig_.value("optimization").toObject();
    return opt.value("intra_op_num_threads").toInt(4);
}

int ConfigManager::getInterOpNumThreads() const
{
    QJsonObject opt = currentConfig_.value("optimization").toObject();
    return opt.value("inter_op_num_threads").toInt(1);
}

QString ConfigManager::getGraphOptimizationLevel() const
{
    QJsonObject opt = currentConfig_.value("optimization").toObject();
    return opt.value("graph_optimization_level").toString("ORT_ENABLE_EXTENDED");
}

QString ConfigManager::getExecutionMode() const
{
    QJsonObject opt = currentConfig_.value("optimization").toObject();
    return opt.value("execution_mode").toString("ORT_SEQUENTIAL");
}

bool ConfigManager::getEnableCpuMemArena() const
{
    QJsonObject opt = currentConfig_.value("optimization").toObject();
    return opt.value("enable_cpu_mem_arena").toBool(true);
}

bool ConfigManager::getEnableMemPattern() const
{
    QJsonObject opt = currentConfig_.value("optimization").toObject();
    return opt.value("enable_mem_pattern").toBool(true);
}

QString ConfigManager::getHardwareName() const
{
    return currentConfig_.value("name").toString();
}

QString ConfigManager::getHardwareModel() const
{
    return currentConfig_.value("model").toString();
}

QString ConfigManager::getHardwareSystem() const
{
    return currentConfig_.value("system").toString();
}

QString ConfigManager::getHardwareArchitecture() const
{
    return currentConfig_.value("architecture").toString();
}
