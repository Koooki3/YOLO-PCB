NEU Pipeline (Qt5 + C++ + OpenCV)

说明：
本示例提供一个基于 `DefectDetection` 的训练与评估流水线，适配 NEU Surface Defect Dataset。

文件：
- neu_pipeline.cpp：Qt5 控制台程序，支持两种模式：train / eval。
  - 现在支持多类训练与评估：训练时会把训练目录下每个子目录视为一个类别，并按子目录顺序生成 `models/defect_detector_classes.txt` 保存类别映射；评估时会读取该映射以保证测试标签与训练对齐。
- DefectDetection.h / DefectDetection.cpp：缺陷检测模块（特征提取、PCA、SVM、保存/加载）。

构建（qmake / QtCreator）示例：
1. 在项目根创建或修改 .pro 文件，添加 neu_pipeline.cpp 并链接 OpenCV：

# 在 VisualRobot.pro 中加入示例 target（伪代码）
# CONFIG += console
# SOURCES += neu_pipeline.cpp DefectDetection.cpp
# INCLUDEPATH += C:/path/to/opencv/build/include
# LIBS += -LC:/path/to/opencv/build/x64/vc15/lib -lopencv_core420 -lopencv_imgproc420 -lopencv_imgcodecs420 -lopencv_ml420

2. 在 QtCreator 中打开 .pro，运行 qmake 并构建。

命令行运行示例（PowerShell）：
# 训练
.\neu_pipeline.exe -m train -r ..\Data\NEU -p 32
# 评估
.\neu_pipeline.exe -m eval -r ..\Data\NEU -o neu_results.csv

目录结构（预期）：
NEU/
  train/
    normal/
    defect/
  test/
    normal/
    defect/

输出：
- models/defect_detector_svm.yml
- models/defect_detector_pca.yml
- neu_results.csv (评估时生成)

注意：
- 请根据本机 OpenCV 版本修改库名（示例中为 opencv 4.2.0 的命名格式），并在 .pro 文件中正确设置 INCLUDEPATH 与 LIBS。
- 如果样本较大，建议增加 PCA 维度或使用交叉验证调整 SVM 超参。

