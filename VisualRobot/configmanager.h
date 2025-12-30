/**
 * @file configmanager.h
 * @brief 配置管理器模块头文件
 * 
 * 该模块负责管理应用程序的配置信息，包括硬件配置、依赖库信息、
 * 路径配置、优化参数等。支持从JSON配置文件加载和管理多个硬件配置。
 * 
 * @author VisualRobot Team
 * @date 2025-12-29
 * @version 1.0
 * @see ConfigManager
 */

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

/**
 * @class ConfigManager
 * @brief 配置管理器类
 * 
 * 该类负责管理应用程序的所有配置信息，采用单例模式确保全局唯一性。
 * 支持从JSON配置文件加载和管理多个硬件配置，提供便捷的配置获取接口。
 * 
 * 配置管理器提供以下主要功能：
 * - 硬件配置管理（CPU、GPU、ARM等）
 * - 依赖库路径和库文件管理
 * - 路径配置（模型路径、标签路径、图像路径）
 * - 性能优化参数配置
 * - 硬件信息获取
 * 
 * @note 使用前需要调用init()方法初始化配置管理器
 * @see QObject
 */
class ConfigManager : public QObject
{
    Q_OBJECT
public:
    /**
     * @brief 构造函数
     * @param parent 父对象指针，默认为nullptr
     * @warning 由于使用单例模式，建议通过instance()方法获取实例
     */
    explicit ConfigManager(QObject *parent = nullptr);
    
    /**
     * @brief 析构函数
     * 
     * 清理配置管理器占用的资源
     */
    ~ConfigManager();
    
    /**
     * @brief 初始化配置管理器
     * 
     * 加载配置文件并初始化默认硬件配置
     * 
     * @param configFilePath 配置文件路径，默认为空字符串时使用默认路径
     * @return bool 初始化成功返回true，失败返回false
     * @retval true 初始化成功
     * @retval false 初始化失败（文件不存在、格式错误等）
     * @note 默认路径优先级：当前目录hardware_config.json -> 资源文件:/hardware_config.json
     * @see parseConfigFile()
     * @see loadHardwareConfig()
     */
    bool init(const QString& configFilePath = "");
    
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
    bool loadHardwareConfig(const QString& configName);
    
    /**
     * @brief 获取当前硬件配置名称
     * 
     * 返回当前激活的硬件配置的名称
     * 
     * @return QString 当前配置的名称字符串
     * @note 如果未初始化或未加载配置，返回空字符串
     * @see loadHardwareConfig()
     */
    QString getCurrentConfigName() const;
    
    /**
     * @brief 获取所有可用的硬件配置名称
     * 
     * 从配置文件中提取所有可用的硬件配置名称列表
     * 
     * @return QStringList 可用配置名称列表
     * @note 返回的列表可用于loadHardwareConfig()的参数
     * @see loadHardwareConfig()
     */
    QStringList getAvailableConfigs() const;
    
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
    QString getDependencyIncludePath(const QString& depName) const;
    
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
    QString getDependencyLibPath(const QString& depName) const;
    
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
    QStringList getDependencyLibs(const QString& depName) const;
    
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
    bool isAcceleratorEnabled(const QString& accName) const;
    
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
    QString getModelPath() const;
    
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
    QString getLabelPath() const;
    
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
    QString getImagePath() const;
    
    /**
     * @brief 获取IntraOp线程数
     * 
     * 获取ONNX Runtime单操作并行线程数
     * 
     * @return int 线程数整数值
     * @note 默认值为4
     * @see getInterOpNumThreads()
     */
    int getIntraOpNumThreads() const;
    
    /**
     * @brief 获取InterOp线程数
     * 
     * 获取ONNX Runtime跨操作并行线程数
     * 
     * @return int 线程数整数值
     * @note 默认值为1
     * @see getIntraOpNumThreads()
     */
    int getInterOpNumThreads() const;
    
    /**
     * @brief 获取图优化级别
     * 
     * 获取ONNX Runtime图优化级别配置
     * 
     * @return QString 优化级别字符串
     * @note 常见值：ORT_ENABLE_BASIC, ORT_ENABLE_EXTENDED, ORT_ENABLE_ALL
     * @see getExecutionMode()
     */
    QString getGraphOptimizationLevel() const;
    
    /**
     * @brief 获取执行模式
     * 
     * 获取ONNX Runtime执行模式配置
     * 
     * @return QString 执行模式字符串
     * @note 常见值：ORT_SEQUENTIAL, ORT_PARALLEL
     * @see getGraphOptimizationLevel()
     */
    QString getExecutionMode() const;
    
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
    bool getEnableCpuMemArena() const;
    
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
    bool getEnableMemPattern() const;
    
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
    QString getHardwareName() const;
    
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
    QString getHardwareModel() const;
    
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
    QString getHardwareSystem() const;
    
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
    QString getHardwareArchitecture() const;
    
    /**
     * @brief 获取配置管理器单例实例
     * 
     * 获取ConfigManager的单例实例，如果实例不存在则创建
     * 
     * @return ConfigManager* 配置管理器实例指针
     * @note 使用单例模式，确保全局唯一性
     * @see init()
     */
    static ConfigManager* instance();

private:
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
    bool parseConfigFile(const QString& filePath);
    
    /// @brief 当前配置的名称
    QString currentConfigName_;
    
    /// @brief 当前激活的配置数据
    QJsonObject currentConfig_;
    
    /// @brief 所有可用的配置数据
    QJsonObject allConfigs_;
    
    /// @brief 配置管理器单例实例
    static ConfigManager* instance_;
};

#endif // CONFIGMANAGER_H
