/**
 * @file configmanager.cpp
 * @brief 配置管理器实现文件
 * 
 * 该文件实现了ConfigManager类的所有方法，提供配置文件的加载、解析和管理功能。
 * 
 * @author VisualRobot Team
 * @date 2025-12-29
 * @version 1.0
 */

#include "configmanager.h"
#include <QJsonArray>
#include <QJsonObject>
#include <QJsonValue>
#include <QJsonDocument>

// 初始化单例实例
ConfigManager* ConfigManager::instance_ = nullptr;

/**
 * @brief 构造函数
 * 
 * 初始化配置管理器，设置父对象和默认配置名称
 * 
 * @param parent 父对象指针，默认为nullptr
 */
ConfigManager::ConfigManager(QObject *parent) 
    : QObject(parent)
    , currentConfigName_("")
{
}

/**
 * @brief 析构函数
 * 
 * 清理配置管理器占用的资源
 */
ConfigManager::~ConfigManager()
{
}

/**
 * @brief 获取配置管理器单例实例
 * 
 * 获取ConfigManager的单例实例，如果实例不存在则创建新实例
 * 
 * @return ConfigManager* 配置管理器实例指针
 * @note 使用单例模式，确保全局唯一性
 * @see init()
 */
ConfigManager* ConfigManager::instance()
{
    if (!instance_)
    {
        instance_ = new ConfigManager();
    }
    return instance_;
}

/**
 * @brief 初始化配置管理器
 * 
 * 加载配置文件并初始化默认硬件配置
 * 
 * @param configFilePath 配置文件路径，默认为空字符串时使用默认路径
 * @return bool 初始化成功返回true，失败返回false
 * @retval true 初始化成功
 * @retval false 初始化失败（文件不存在、格式错误等）
 * @note 默认路径优先级：
 *       -# 当前目录下的hardware_config.json
 *       -# 资源文件中的:/hardware_config.json
 * @see parseConfigFile()
 * @see loadHardwareConfig()
 */
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

/**
 * @brief 解析配置文件
 * 
 * 从指定路径的JSON配置文件中读取并解析配置数据
 * 
 * @param filePath 配置文件路径
 * @return bool 解析成功返回true，失败返回false
 * @retval true 文件存在且JSON格式正确
 * @retval false 文件不存在或JSON格式错误
 * @see init()
 */
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

/**
 * @brief 加载指定硬件配置
 * 
 * 从已加载的配置中查找并激活指定的硬件配置
 * 
 * @param configName 硬件配置名称
 * @return bool 加载成功返回true，失败返回false
 * @retval true 配置加载成功
 * @retval false 配置不存在或格式错误
 * @see getAvailableConfigs()
 * @see getCurrentConfigName()
 */
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

/**
 * @brief 获取当前硬件配置名称
 * 
 * 返回当前激活的硬件配置的名称
 * 
 * @return QString 当前配置的名称字符串
 * @note 如果未初始化或未加载配置，返回空字符串
 * @see loadHardwareConfig()
 */
QString ConfigManager::getCurrentConfigName() const
{
    return currentConfigName_;
}

/**
 * @brief 获取所有可用的硬件配置名称
 * 
 * 从配置文件中提取所有可用的硬件配置名称列表
 * 
 * @return QStringList 可用配置名称列表
 * @note 返回的列表可用于loadHardwareConfig()的参数
 * @see loadHardwareConfig()
 */
QStringList ConfigManager::getAvailableConfigs() const
{
    QJsonObject hardwareConfigs = allConfigs_.value("hardware_configs").toObject();
    return hardwareConfigs.keys();
}

/**
 * @brief 获取依赖库包含路径
 * 
 * 获取指定依赖库的头文件包含路径
 * 
 * @param depName 依赖库名称
 * @return QString 依赖库包含路径字符串
 * @note 路径格式为标准的包含路径格式
 * @see getDependencyLibPath()
 * @see getDependencyLibs()
 */
QString ConfigManager::getDependencyIncludePath(const QString& depName) const
{
    QJsonObject deps = currentConfig_.value("dependencies").toObject();
    QJsonObject dep = deps.value(depName).toObject();
    return dep.value("include_path").toString();
}

/**
 * @brief 获取依赖库文件路径
 * 
 * 获取指定依赖库的库文件所在目录路径
 * 
 * @param depName 依赖库名称
 * @return QString 依赖库文件路径字符串
 * @note 路径格式为标准的库文件路径格式
 * @see getDependencyIncludePath()
 * @see getDependencyLibs()
 */
QString ConfigManager::getDependencyLibPath(const QString& depName) const
{
    QJsonObject deps = currentConfig_.value("dependencies").toObject();
    QJsonObject dep = deps.value(depName).toObject();
    return dep.value("lib_path").toString();
}

/**
 * @brief 获取依赖库文件列表
 * 
 * 获取指定依赖库的所有库文件名称列表
 * 
 * @param depName 依赖库名称
 * @return QStringList 依赖库文件名称列表
 * @note 返回的文件名不包含路径，仅文件名
 * @see getDependencyIncludePath()
 * @see getDependencyLibPath()
 */
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

/**
 * @brief 检查加速器是否启用
 * 
 * 查询指定加速器（如GPU、NPU等）的启用状态
 * 
 * @param accName 加速器名称
 * @return bool 加速器启用返回true，未启用返回false
 * @retval true 加速器已启用
 * @retval false 加速器未启用或不存在
 * @see isAcceleratorEnabled()
 */
bool ConfigManager::isAcceleratorEnabled(const QString& accName) const
{
    QJsonObject accs = currentConfig_.value("accelerators").toObject();
    return accs.value(accName).toBool(false);
}

/**
 * @brief 获取模型文件路径
 * 
 * 获取ONNX模型文件的存储路径
 * 
 * @return QString 模型文件路径字符串
 * @note 默认路径为"../models"
 * @see getLabelPath()
 * @see getImagePath()
 */
QString ConfigManager::getModelPath() const
{
    QJsonObject paths = currentConfig_.value("paths").toObject();
    return paths.value("models").toString("../models");
}

/**
 * @brief 获取标签文件路径
 * 
 * 获取分类标签文件的存储路径
 * 
 * @return QString 标签文件路径字符串
 * @note 默认路径为"../labels"
 * @see getModelPath()
 * @see getImagePath()
 */
QString ConfigManager::getLabelPath() const
{
    QJsonObject paths = currentConfig_.value("paths").toObject();
    return paths.value("labels").toString("../labels");
}

/**
 * @brief 获取图像文件路径
 * 
 * 获取测试图像文件的存储路径
 * 
 * @return QString 图像文件路径字符串
 * @note 默认路径为"../Img"
 * @see getModelPath()
 * @see getLabelPath()
 */
QString ConfigManager::getImagePath() const
{
    QJsonObject paths = currentConfig_.value("paths").toObject();
    return paths.value("images").toString("../Img");
}

/**
 * @brief 获取IntraOp线程数
 * 
 * 获取ONNX Runtime单操作并行线程数
 * 
 * @return int 线程数整数值
 * @note 默认值为4
 * @see getInterOpNumThreads()
 */
int ConfigManager::getIntraOpNumThreads() const
{
    QJsonObject opt = currentConfig_.value("optimization").toObject();
    return opt.value("intra_op_num_threads").toInt(4);
}

/**
 * @brief 获取InterOp线程数
 * 
 * 获取ONNX Runtime跨操作并行线程数
 * 
 * @return int 线程数整数值
 * @note 默认值为1
 * @see getIntraOpNumThreads()
 */
int ConfigManager::getInterOpNumThreads() const
{
    QJsonObject opt = currentConfig_.value("optimization").toObject();
    return opt.value("inter_op_num_threads").toInt(1);
}

/**
 * @brief 获取图优化级别
 * 
 * 获取ONNX Runtime图优化级别配置
 * 
 * @return QString 优化级别字符串
 * @note 常见值：ORT_ENABLE_BASIC, ORT_ENABLE_EXTENDED, ORT_ENABLE_ALL
 * @see getExecutionMode()
 */
QString ConfigManager::getGraphOptimizationLevel() const
{
    QJsonObject opt = currentConfig_.value("optimization").toObject();
    return opt.value("graph_optimization_level").toString("ORT_ENABLE_EXTENDED");
}

/**
 * @brief 获取执行模式
 * 
 * 获取ONNX Runtime执行模式配置
 * 
 * @return QString 执行模式字符串
 * @note 常见值：ORT_SEQUENTIAL, ORT_PARALLEL
 * @see getGraphOptimizationLevel()
 */
QString ConfigManager::getExecutionMode() const
{
    QJsonObject opt = currentConfig_.value("optimization").toObject();
    return opt.value("execution_mode").toString("ORT_SEQUENTIAL");
}

/**
 * @brief 检查是否启用CPU内存区域
 * 
 * 查询CPU内存区域优化是否启用
 * 
 * @return bool 启用返回true，未启用返回false
 * @retval true CPU内存区域已启用
 * @retval false CPU内存区域未启用
 * @see getEnableMemPattern()
 */
bool ConfigManager::getEnableCpuMemArena() const
{
    QJsonObject opt = currentConfig_.value("optimization").toObject();
    return opt.value("enable_cpu_mem_arena").toBool(true);
}

/**
 * @brief 检查是否启用内存模式
 * 
 * 查询内存模式优化是否启用
 * 
 * @return bool 启用返回true，未启用返回false
 * @retval true 内存模式已启用
 * @retval false 内存模式未启用
 * @see getEnableCpuMemArena()
 */
bool ConfigManager::getEnableMemPattern() const
{
    QJsonObject opt = currentConfig_.value("optimization").toObject();
    return opt.value("enable_mem_pattern").toBool(true);
}

/**
 * @brief 获取硬件名称
 * 
 * 获取当前配置的硬件设备名称
 * 
 * @return QString 硬件名称字符串
 * @see getHardwareModel()
 * @see getHardwareSystem()
 * @see getHardwareArchitecture()
 */
QString ConfigManager::getHardwareName() const
{
    return currentConfig_.value("name").toString();
}

/**
 * @brief 获取硬件型号
 * 
 * 获取当前配置的硬件设备型号
 * 
 * @return QString 硬件型号字符串
 * @see getHardwareName()
 * @see getHardwareSystem()
 * @see getHardwareArchitecture()
 */
QString ConfigManager::getHardwareModel() const
{
    return currentConfig_.value("model").toString();
}

/**
 * @brief 获取硬件系统
 * 
 * 获取当前配置的硬件操作系统信息
 * 
 * @return QString 硬件系统字符串
 * @see getHardwareName()
 * @see getHardwareModel()
 * @see getHardwareArchitecture()
 */
QString ConfigManager::getHardwareSystem() const
{
    return currentConfig_.value("system").toString();
}

/**
 * @brief 获取硬件架构
 * 
 * 获取当前配置的硬件架构信息
 * 
 * @return QString 硬件架构字符串
 * @see getHardwareName()
 * @see getHardwareModel()
 * @see getHardwareSystem()
 */
QString ConfigManager::getHardwareArchitecture() const
{
    return currentConfig_.value("architecture").toString();
}
