# VisualRobot

基于 Qt 和 C++ 开发的工业视觉机器人控制系统，提供完整的相机控制和图像处理解决方案。

## ✨ 主要特性

- 🎥 支持 GigE 和 USB3.0 接口的工业相机控制
- 🖼️ 强大的数字图像处理功能
- 🌐 支持中英文双语界面
- 📊 实时系统状态监控
- 🛠️ 可配置的相机参数设置

## 📁 项目结构

```plaintext
VisualRobot/
├── Doc/                          # 项目文档
│   └── 调试信息手册.pdf            # 系统调试和维护手册
├── Img/                          # 图像资源目录
│   ├── capture.jpg               # 相机采集的原始图像
│   └── circle_detected.jpg       # 圆形检测的结果图像
├── VisualRobot/                  # 核心源代码目录
│   ├── Core                      # 核心功能模块
│   │   ├── MvCamera.*            # 工业相机控制实现
│   │   ├── DIP.*                 # 数字图像处理算法
│   │   ├── Format.*              # 数据格式转换工具
│   │   └── SystemMonitor.*       # 系统状态监控模块
│   ├── UI                        # 用户界面相关
│   │   ├── mainwindow.*          # 主窗口实现
│   │   └── *.ui                  # Qt Designer UI 文件
│   ├── Resources                 # 资源文件
│   │   ├── VisualRobot_zh_EN.ts  # 中英文翻译源文件
│   │   └── VisualRobot_zh_EN.qm  # 编译后的翻译文件
│   ├── main.cpp                  # 程序入口点
│   └── VisualRobot.pro           # Qt 项目配置文件
├── Data                          # 数据文件
    ├── matrix.bin                # 系统配置矩阵
    └── detectedImg.jpg           # 最新检测结果
```

### 目录说明

#### 📂 核心源代码 (VisualRobot/)

- **Core** - 核心功能实现

  - `MvCamera.*` - 工业相机控制，支持 GigE/USB3.0
  - `DIP.*` - 数字图像处理，包含图像增强和特征检测
  - `Format.*` - 数据格式转换，处理不同格式间的转换
  - `SystemMonitor.*` - 系统监控，追踪性能和资源使用
- **UI** - 用户界面组件

  - `mainwindow.*` - 主窗口实现，包含界面逻辑
  - `*.ui` - Qt Designer 界面描述文件
- **Resources** - 项目资源

  - 多语言支持文件
  - 图标和样式表

### 关键文件说明

- `VisualRobot.pro` - Qt 项目配置，包含:

  - 构建设置
  - 依赖配置
  - 资源管理
  - 编译选项
- `main.cpp` - 程序入口点，负责:

  - 初始化应用程序
  - 配置日志系统
  - 启动主界面

## 💡 核心模块

### 相机控制模块 (MvCamera)

- 工业相机初始化与配置
- 实时图像采集
- 相机参数动态调节

### 图像处理模块 (DIP)

- 图像预处理与增强
- 特征检测与提取
- 目标识别与定位

### 系统监控模块 (SystemMonitor)

- 实时性能监控
- 系统资源管理
- 运行状态追踪

### 界面模块 (MainWindow)

- 直观的用户交互界面
- 相机实时预览
- 多语言支持

## 🔧 环境要求

### 开发环境

- Qt 5.x 或更高版本
- C++ 编译器 (支持 C++11 标准)
- CMake 3.x 或更高版本

### 依赖库

1. MVS (Machine Vision Software) SDK

```cpp
INCLUDEPATH += /opt/MVS/include
LIBS += -L/opt/MVS/bin/ -L/opt/MVS/lib/64/
```

2. HALCON 图像处理库

```cpp
INCLUDEPATH += /home/orangepi/MVTec/HALCON-25.05-Progress/include
LIBS += -L/home/orangepi/MVTec/HALCON-25.05-Progress/lib/aarch64-linux/
```

3. Eigen3 数学库

```cpp
INCLUDEPATH += /usr/include/eigen3/Eigen
```

## 📦 安装与配置

### 1. 安装依赖

```bash
# 安装 Qt 开发环境
# Windows: 使用 Qt 在线安装器
# Linux: 
sudo apt-get install qt5-default
sudo apt-get install qtcreator

# 安装必要的开发工具
sudo apt-get install build-essential
sudo apt-get install cmake
sudo apt-get install libomp-dev
```

### 配置项目

1. 克隆项目：

```bash
git clone [项目地址]
cd VisualRobot
```

## 🚀 编译与运行

1. 使用 Qt Creator 打开 `VisualRobot.pro`
2. 配置编译选项
3. 编译项目
4. 运行程序

## ⚠️ 注意事项

1. 首次运行前确认：

   - 所有依赖库已正确安装
   - 相机驱动已安装
   - 相机连接正常
2. 运行时注意：

   - 检查相机连接状态
   - 确保权限设置正确
   - 监控系统资源占用
3. 调试参考：

   - 查看 `Doc/调试信息手册.docx`
   - 使用系统监控工具跟踪性能

## 📝 开发规范

1. 代码风格

   - 使用统一的代码格式化工具
   - 遵循 Qt 编码规范
   - 保持注释的完整性
2. 版本控制

   - 使用语义化版本号
   - 提交信息要清晰明确
   - 保持分支管理的规范性

## 📄 许可证

待添加许可证信息

## 👥 维护者

待添加维护者信息

## 🤝 贡献

欢迎提交问题和改进建议！

---

最后更新日期：2025年9月3日
