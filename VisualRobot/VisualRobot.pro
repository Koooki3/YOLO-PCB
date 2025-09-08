QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets
TRANSLATIONS = VisualRobot_zh_EN.ts
CONFIG += c++11

# 启用OpenMP支持
QMAKE_CXXFLAGS += -fopenmp
QMAKE_LFLAGS += -fopenmp
LIBS += -lgomp

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
    Format.cpp \
    MvCamera.cpp \
    SystemMonitor.cpp \
    main.cpp \
    mainwindow.cpp \
    mainwindow_systemstats.cpp

HEADERS += \
    DIP.h \
    Format.h \
    MvCamera.h \
    SystemMonitor.h \
    mainwindow.h \
    ThreadPool.h \
    ImageMemoryPool.h \
    ImageCache.h

FORMS += \
    mainwindow.ui

INCLUDEPATH += /opt/MVS/include

INCLUDEPATH += /home/orangepi/MVTec/HALCON-25.05-Progress/include

INCLUDEPATH += /usr/include/eigen3/Eigen

INCLUDEPATH += /usr/local/include \
               /usr/local/include/opencv4 \
               /usr/local/include/opencv4/opencv2

LIBS += -L/usr/local/lib/ -lopencv_calib3d -lopencv_core -lopencv_dnn -lopencv_features2d \
        -lopencv_flann -lopencv_gapi -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc \
        -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_video \
        -lopencv_videoio

LIBS += -L/opt/MVS/lib/aarch64/ -lMvCameraControl -lMvCameraControlWrapper -lMVGigEVisionSDK -lMvUsb3vTL

LIBS += -L/home/orangepi/MVTec/HALCON-25.05-Progress/lib/aarch64-linux/ \
        -lhalcon -lhalconc -lhalconcpp -lhalcondl -lhdevenginecpp

# 编译优化选项
unix {
    # 基本警告
    QMAKE_CXXFLAGS += -Wall
    
    # 基本优化选项
    QMAKE_CXXFLAGS_RELEASE += -O2
    
    # ARM64基本优化
    contains(QMAKE_HOST.arch, aarch64) {
        QMAKE_CXXFLAGS += -march=armv8-a
    }
}

# 资源限制
DEFINES += MAX_POOL_SIZE=536870912
DEFINES += MAX_CACHE_SIZE=1073741824

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

# 添加版本信息
VERSION = 1.2.0
DEFINES += APP_VERSION=\\\"$$VERSION\\\"
