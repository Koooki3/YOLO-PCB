# VisualRobot 项目说明

这是一个基于 Qt 和 C++ 开发的视觉机器人控制系统，集成了相机控制和图像处理功能。

## 项目结构

```plaintext
├── Doc/                    # 文档目录
│   └── 调试信息手册.docx   # 调试相关文档
├── Img/                    # 图像资源目录
│   ├── capture.jpg        # 捕获的图像
│   └── circle_detected.jpg # 圆形检测结果
├── VisualRobot/           # 主程序源代码目录
│   ├── DIP.*             # 数字图像处理相关代码
│   ├── Format.*          # 格式转换相关代码
│   ├── MvCamera.*        # 相机控制模块
│   ├── main.cpp          # 程序入口
│   ├── mainwindow.*      # 主窗口界面
│   └── VisualRobot.pro   # Qt 项目文件
├── detectedImg.jpg        # 检测结果图像
└── matrix.bin            # 矩阵数据文件
```

## 主要模块说明

### 1. 相机控制模块 (MvCamera.*)

- 封装了相机SDK的操作接口
- 提供相机的初始化、图像采集、参数设置等功能
- 支持 GigE 和 USB3.0 接口的工业相机

### 2. 图像处理模块 (DIP.*)

- 实现数字图像处理算法
- 包含图像预处理、特征提取等功能

### 3. 格式转换模块 (Format.*)

- 处理各种数据格式的转换
- 支持图像格式转换和数据格式化

### 4. 主窗口界面 (mainwindow.*)

- 实现用户交互界面
- 集成相机预览、参数配置等功能
- 支持中英文界面切换

## 依赖项配置

项目依赖以下库和SDK，使用前需要正确配置路径：

1. MVS (Machine Vision Software) SDK

```cpp
INCLUDEPATH += /opt/MVS/include
LIBS += -L/opt/MVS/bin/ -L/opt/MVS/lib/64/ -lMvCameraControl
```

2. HALCON

```cpp
INCLUDEPATH += /home/kooki/MVTec/HALCON-25.05-Progress/include
LIBS += -L/home/kooki/MVTec/HALCON-25.05-Progress/lib/x64-linux/
```

3. Eigen3

```cpp
INCLUDEPATH += /usr/include/eigen3/Eigen
```

## 配置说明

### 需要修改的路径

1. 在 `VisualRobot.pro` 中修改以下路径为您的实际安装路径：

   - MVS SDK 路径：`/opt/MVS/`
   - HALCON 路径：`/home/kooki/MVTec/HALCON-25.05-Progress/`
   - Eigen3 路径：`/usr/include/eigen3/`

2. 确保翻译文件 `VisualRobot_zh_EN.qm` 和 `VisualRobot_zh_EN.ts` 在可执行文件的同一目录下

## 编译和运行

1. 确保已安装 Qt 开发环境（建议 Qt 5.x 或更高版本）
2. 使用 Qt Creator 打开 `VisualRobot.pro` 文件
3. 配置编译环境
4. 点击编译运行即可

## 注意事项

1. 请确保所有依赖库都已正确安装并配置
2. 相机驱动必须正确安装
3. 首次运行时请检查相机连接状态
4. 建议参考 `Doc/调试信息手册.docx` 进行系统调试

## 项目中的路径使用

### 1. 配置相关路径（VisualRobot.pro）

#### 系统库路径

MVS库路径：

- `/opt/MVS/include`（INCLUDEPATH）
- `/opt/MVS/bin/`（LIBS）
- `/opt/MVS/lib/64/`（LIBS）

HALCON库路径：

- `/home/kooki/MVTec/HALCON-25.05-Progress/include`（INCLUDEPATH）
- `/home/kooki/MVTec/HALCON-25.05-Progress/lib/x64-linux/`（LIBS）

Eigen3库路径：

- `/usr/include/eigen3/Eigen`（INCLUDEPATH）

部署路径：

- QNX系统：`/tmp/$${TARGET}/bin`
- Unix系统：`/opt/$${TARGET}/bin`

#### 项目源文件

源代码文件：

- DIP 模块：`DIP.cpp`, `DIP.h`
- 格式转换模块：`Format.cpp`, `Format.h`
- 相机控制模块：`MvCamera.cpp`, `MvCamera.h`
- 主程序：`main.cpp`
- 主窗口：`mainwindow.cpp`, `mainwindow.h`, `mainwindow.ui`

### 2. 代码中的路径引用

#### DIP.cpp

图像处理相关路径：

- 相对路径：
  - 输出检测图像：`"../detectedImg.jpg"`（函数 `Algorithm` 中，DumpWindow 函数调用）

- 绝对路径：
  - 输入图像路径：`"/home/kooki/VisualRobot/Img/capture.jpg"`（函数 `getCoords` 中，ReadImage 函数调用）

#### main.cpp

资源文件：

- 翻译文件：`"VisualRobot_zh_EN.qm"`（主函数中）
- Eigen库：`"eigen3/Eigen/Dense"`（头文件引用）

#### mainwindow.cpp

头文件引用：

- 相对路径：

- 绝对路径：

### 3. 开发环境配置路径

#### 构建目录设置

调试和发布目录：

- `/home/kooki/VisualRobot/QT5_15_13-Debug`
- `/home/kooki2/VisualRobot/build-VisualRobot-Qt_5_12_8_in_PATH_qt5-Debug`
- `/home/kooki2/VisualRobot/build-VisualRobot-Qt_5_12_8_in_PATH_qt5-Release`
- `/home/kooki2/VisualRobot/build-VisualRobot-Qt_5_12_8_in_PATH_qt5-Profile`

#### 工程文件位置

项目主文件：

- `/home/kooki2/VisualRobot/VisualRobot/VisualRobot.pro`

注意：这些路径在不同的开发环境中可能需要相应调整。确保在部署时正确设置所有路径。
