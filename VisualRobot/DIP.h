/************************************************************************/
/* 提供基于Halcon和Eigen的坐标变换矩阵创建、计算、保存、读取支持和现有待测物体边缘检测和拟合算法支持 */
/************************************************************************/

#ifndef DIP_H
#define DIP_H

#include <QVector>
#include <QPointF>
#include <QDir>
#include <QCoreApplication>
#include "eigen3/Eigen/Dense"
#include <halconcpp/HalconCpp.h>
#include <Halcon.h>
#include <halconcpp/HDevThread.h>

using namespace HalconCpp;
using namespace Eigen;
using namespace std;

bool createDirectory(const string& path);

int TransMatrix(const QVector<QPointF>& WorldCoord, const QVector<QPointF>& PixelCoord, Matrix3d& matrix, const string& filename);

int Algorithm(const string& imgPath, HTuple& Row, HTuple& Col);

Matrix3d readTransformationMatrix(const string& filePath);

int getCoords(QVector<QPointF>& WorldCoord, QVector<QPointF>& PixelCoord, double size);

#endif // DIP_H
