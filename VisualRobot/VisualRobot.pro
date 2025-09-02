QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets
TRANSLATIONS = VisualRobot_zh_EN.ts
CONFIG += c++11

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
    main.cpp \
    mainwindow.cpp

HEADERS += \
    DIP.h \
    Format.h \
    MvCamera.h \
    mainwindow.h

FORMS += \
    mainwindow.ui

INCLUDEPATH += /opt/MVS/include

INCLUDEPATH += /home/orangepi/MVTec/HALCON-25.05-Progress/include

INCLUDEPATH += /usr/include/eigen3/Eigen

LIBS += -L/opt/MVS/lib/aarch64/ -lMvCameraControl -lMvCameraControlWrapper -lMVGigEVisionSDK -lMvUsb3vTL

LIBS += -L/home/orangepi/MVTec/HALCON-25.05-Progress/lib/aarch64-linux/ \
        -lhalcon -lhalconc -lhalconcpp -lhalcondl -lhdevenginecpp

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
