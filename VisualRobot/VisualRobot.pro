QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets
TRANSLATIONS = VisualRobot_zh_EN.ts
CONFIG += c++17

CONFIG += precompile_header
PRECOMPILE_HEADER = qglobal.h

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    DIP.cpp \
    DLProcessor.cpp \
    DataProcessor.cpp \
    MvCamera.cpp \
    SystemMonitor.cpp \
    main.cpp \
    mainwindow.cpp \
    mainwindow_systemstats.cpp \
    DLExample.cpp \
    YOLOExample.cpp \
    YOLOProcessorORT.cpp \
    Undistort.cpp \
    DefectDetection.cpp \
    FeatureAlignment.cpp \

HEADERS += \
    DIP.h \
    DLProcessor.h \
    DataProcessor.h \
    YOLOExample.h \
    YOLOProcessorORT.h \
    MvCamera.h \
    SystemMonitor.h \
    mainwindow.h \
    DLExample.h \
    Undistort.h \
    DefectDetection.h \
    FeatureAlignment.h \

FORMS += \
    mainwindow.ui

INCLUDEPATH += /opt/MVS/include

INCLUDEPATH += /usr/include/eigen3/Eigen

INCLUDEPATH += /usr/local/include \
               /usr/local/include/opencv4 \
               /usr/local/include/opencv4/opencv2

INCLUDEPATH += /home/orangepi/Desktop/VisualRobot_Local/onnxruntime-linux-aarch64-1.23.2/include

LIBS += -L/home/orangepi/Desktop/VisualRobot_Local/onnxruntime-linux-aarch64-1.23.2/lib -lonnxruntime

LIBS += -L/usr/local/lib/ -lopencv_calib3d -lopencv_core -lopencv_dnn -lopencv_features2d \
        -lopencv_flann -lopencv_gapi -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc \
        -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_video \
        -lopencv_videoio

LIBS += -L/opt/MVS/lib/aarch64/ -lMvCameraControl -lMvCameraControlWrapper -lMVGigEVisionSDK -lMvUsb3vTL

LIBS += -L/usr/lib/aarch64-linux-gnu -lmali -lOpenCL

# 编译优化选项
unix {
    # 启用所有警告
    QMAKE_CXXFLAGS += -Wall -Wextra

    # 优化选项
    QMAKE_CXXFLAGS_RELEASE -= -O2
    QMAKE_CXXFLAGS_RELEASE += -O3

    # 架构特定优化
    QMAKE_CXXFLAGS += -march=armv8.2-a -mtune=cortex-a76.cortex-a55

    # SIMD和向量化优化
    QMAKE_CXXFLAGS += -ftree-vectorize -ftree-slp-vectorize

    # 链接时优化
    QMAKE_CXXFLAGS += -flto
    QMAKE_LFLAGS += -flto

    # 缓存和内存优化
    QMAKE_CXXFLAGS += -fprefetch-loop-arrays
    QMAKE_CXXFLAGS += -falign-functions=64
}

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

# Bundle the UI stylesheet into the application resources so it is available at runtime
RESOURCES += styles.qrc
