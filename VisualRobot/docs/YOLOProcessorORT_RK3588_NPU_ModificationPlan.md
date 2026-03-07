# YOLOProcessorORT 模块分析与 RK3588 NPU 可选配置修改计划

## 一、YOLOProcessorORT 模块设计分析

### 1.1 架构概览

- **职责**：基于 ONNX Runtime 的 YOLO 目标检测，包括模型加载、预处理、推理、后处理、结果绘制与延时统计。
- **依赖**：`onnxruntime_cxx_api.h`、OpenCV、ConfigManager、`DLProcessor`（DetectionResult）。
- **加速**：通过 `ConfigManager::isAcceleratorEnabled(accName)` 和 `SessionOptions::AddConfigEntry` / `AppendExecutionProvider` 选择执行端。

### 1.2 关键流程

| 阶段 | 位置 | 说明 |
|------|------|------|
| 构造 | 构造函数 | 从 ConfigManager 读取线程、内存、图优化、执行模式等，配置 `sessionOptions_` |
| 初始化 | `InitModel(modelPath, useCUDA)` | 根据 `accelerators.opencl` 写 OpenCL 相关 `AddConfigEntry`，创建 `Ort::Session` |
| 推理 | `DetectObjects` | Letterbox 预处理 → 创建 `Ort::Value` 输入 → `session_->Run` → `OrtOutputToMats` → `PostProcess` |
| 后处理 | `PostProcess` | 解析 `[1,10,8400]` 等输出、阈值过滤、letterbox 逆变换、NMS，输出 `DetectionResult` |

### 1.3 现有加速与配置

- **OpenCL**：在 `InitModel` 中若 `isAcceleratorEnabled("opencl")`，则 `AddConfigEntry("session.enable_opencl","1")` 等；未使用 `AppendExecutionProvider_OpenCL`。
- **CUDA**：`InitModel(..., useCUDA)` 的 `useCUDA` 当前未使用；CUDA 可由 `accelerators.cuda` 与后续 EP 扩展。
- **配置来源**：`hardware_config.json` → `hardware_configs.<name>.accelerators`（如 `opencl`、`cuda`、`tensorrt`），由 `ConfigManager::isAcceleratorEnabled(accName)` 读取。

### 1.4 与 RK3588 的关联

- 已有 `orangepi5_rk3588s` 配置，`accelerators` 含 `opencl: true`，无 `npu`。
- RK3588 带 NPU（RKNPU2），常见用法：
  - **方式 A**：ONNX Runtime + RKNPU 社区 EP，继续用 ONNX 模型，通过 `OrtSessionOptionsAppendExecutionProvider_RKNPU` 选用 NPU。
  - **方式 B**：RKNN2 原生（librknn_api + .rknn 模型），需单独推理路径与模型格式，改动较大。

本计划**仅实现方式 A**，保持 ONNX 模型与现有 `DetectObjects`/`PostProcess` 不变，通过“可选配置 + 条件编译”接入 RKNPU EP。

---

## 二、RK3588 NPU 可选配置修改计划

### 2.1 设计原则

- **可选**：`accelerators.npu` 为配置项，默认 `false`；启用 NPU 时需 ONNX Runtime 带 RKNPU EP 且系统有对应驱动。
- **可裁剪**：通过 `USE_RKNPU_EP` 宏控制 NPU 相关代码与头文件，未定义时零依赖、零行为变化。
- **兼容**：不改变 `InitModel(modelPath, useCUDA)` 的对外签名；不改变 `DetectObjects`/`PostProcess` 的输入输出与调用方。

### 2.2 修改项清单

| # | 文件 | 修改内容 |
|---|------|----------|
| 1 | `hardware_config.json` | 在 `orangepi5_rk3588s`、`raspberrypi4`、`x86_64_pc` 的 `accelerators` 中增加 `"npu": false`；RK3588 可改为 `true` 以启用。 |
| 2 | `YOLOProcessorORT.cpp` | 在 `InitModel` 中，在创建 `Ort::Session` 之前：若 `USE_RKNPU_EP` 且 `isAcceleratorEnabled("npu")`，则 `Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_RKNPU(sessionOptions_))`；对应头文件在 `#ifdef USE_RKNPU_EP` 中条件包含。 |
| 3 | `YOLOProcessorORT.h` | 在类/文件头注释中补充：支持通过 `accelerators.npu` 与 ONNX Runtime RKNPU EP 在 RK3588 上做可选 NPU 推理。 |
| 4 | `VisualRobot.pro` | 增加 `CONFIG+=rknpu` 可选：`contains(CONFIG, rknpu) { DEFINES += USE_RKNPU_EP }`；在 `orangepi5_rk3588s` 块中可加注释说明 `qmake CONFIG+=rknpu` 启用 NPU。 |

### 2.3 代码级说明

#### 2.3.1 InitModel 中 NPU 逻辑顺序

建议顺序（在 `session_ = make_unique<Ort::Session>(...)` 之前）：

1. 已有：线程、内存、图优化、执行模式、MemPattern、OpenCL 的 `AddConfigEntry`。
2. **新增**：若 `#ifdef USE_RKNPU_EP` 且 `config->isAcceleratorEnabled("npu")`，则调用 `OrtSessionOptionsAppendExecutionProvider_RKNPU`，以便 NPU 优先于其他 EP。
3. 创建 `Ort::Session`。

#### 2.3.2 RKNPU EP 的 C API 与头文件

- 使用方式（与官方 RKNPU EP 文档一致）：
  - `Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_RKNPU(sessionOptions_));`
- 头文件：需使用带 RKNPU EP 的 ONNX Runtime 构建所提供头文件，例如 `onnxruntime_rknpu_provider_factory.h` 或构建产物的 `rknpu_provider_factory.h`，在 `#ifdef USE_RKNPU_EP` 内 `#include`，避免未启用 NPU 的 build 找不到该头文件。

#### 2.3.3 SessionOptions 的传入方式

- 若 `OrtSessionOptionsAppendExecutionProvider_RKNPU` 的 C 接口形如 `(OrtSessionOptions* options)`，而 `Ort::SessionOptions` 无直接隐式转换，则需使用其取原生句柄的接口（如 `GetHandle()`、`GetSessionOptions()` 等，视 ONNX Runtime 版本而定）传入；若官方示例中可直接传 `Ort::SessionOptions` 变量，则按示例传入 `sessionOptions_` 即可。

### 2.4 构建与运行

- **启用 NPU 构建**：  
  `qmake CONFIG+=rknpu`（或按 .pro 中实际 CONFIG 名），并确保 ONNX Runtime 使用带 RKNPU EP 的构建，且 INCLUDEPATH/LIBS 能解析 RKNPU 相关头文件与符号。
- **启用 NPU 运行**：  
  在 `hardware_config.json` 的 `orangepi5_rk3588s.accelerators` 中设 `"npu": true`；若 EP 或驱动不可用，ORT 会按实现回退或报错，可再设回 `false` 使用 OpenCL/CPU。

### 2.5 限制与后续扩展

- **RKNPU EP 支持范围**：当前社区 RKNPU EP 文档写明支持 RK1808，未明确 RK3588；RK3588 使用 RKNPU2，需自行验证 EP 或 ORT 定制构建是否支持。若不支持，保持 `npu: false`，用 OpenCL/CPU。
- **头文件与句柄**：`OrtSessionOptionsAppendExecutionProvider_RKNPU` 的头文件名依 ORT/RKNPU EP 构建而定（如 `onnxruntime_rknpu_provider_factory.h` 或 `rknpu_provider_factory.h`）；若传参类型报错，可改用 `Ort::SessionOptions` 的 native 句柄 getter（视 ONNX Runtime 版本）传入。
- **RKNN2 原生（.rknn）**：若需用 .rknn 模型与 librknn_api，可作为后续扩展：例如新增 `YOLOProcessorRKNN` 或在现有类中按 `modelPath` 后缀/配置切换推理后端，并复用预处理/后处理逻辑。

---

## 三、修改后自检要点

- [ ] `accelerators.npu` 仅在 `true` 且 `USE_RKNPU_EP` 定义时才会调用 `AppendExecutionProvider_RKNPU`。
- [ ] 未定义 `USE_RKNPU_EP` 时，不包含 RKNPU 头文件、不引用 RKNPU 符号，可在无 RKNPU 的 ORT 上正常编译。
- [ ] `InitModel` 的签名与 `mainwindow.cpp`、`YOLOExample.cpp` 等调用处无需变更。
- [ ] `npu: false` 或未配置时行为与修改前一致；`npu: true` 且 EP 可用时，ORT 将优先尝试在 NPU 上执行。

---

## 四、执行顺序

1. `hardware_config.json`：为三套硬件配置的 `accelerators` 添加 `"npu": false`，`orangepi5_rk3588s` 可保留为 `false` 或按需改为 `true`。
2. `YOLOProcessorORT.cpp`：在 `InitModel` 中按 2.3.1 插入 RKNPU 分支；在文件顶部 `#ifdef USE_RKNPU_EP` 中 `#include` RKNPU  provider 头文件。
3. `YOLOProcessorORT.h`：更新文件/类注释，说明 NPU 可选支持。
4. `VisualRobot.pro`：增加 `contains(CONFIG, rknpu) { DEFINES += USE_RKNPU_EP }` 及 `orangepi5_rk3588s` 下的说明注释。
