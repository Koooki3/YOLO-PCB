#include <iostream>
#include <opencv2/opencv.hpp>
#include "../VisualRobot/FeatureAlignment.h"

using namespace cv;
using namespace std;

int main() {
    cout << "特征对齐功能测试" << endl;
    cout << "==================" << endl;
    
    // 创建特征对齐对象
    FeatureAlignment alignment;
    
    // 测试图像路径
    string templatePath = "../Img/Templates/template01.jpg";
    string testPath = "../Img/test1.jpg";
    
    // 读取测试图像
    Mat templateImg = imread(templatePath);
    Mat testImg = imread(testPath);
    
    if (templateImg.empty() || testImg.empty()) 
    {
        cout << "无法读取测试图像，请检查文件路径" << endl;
        return -1;
    }
    
    cout << "模板图像尺寸: " << templateImg.cols << "x" << templateImg.rows << endl;
    cout << "测试图像尺寸: " << testImg.cols << "x" << testImg.rows << endl;
    
    // 设置对齐参数
    AlignmentParams params;
    params.minInliers = 10;  // 匹配到10个内点即可停止
    params.enableParallel = true;
    params.numThreads = 4;
    
    cout << "开始特征对齐测试..." << endl;
    
    // 测试快速对齐
    AlignmentResult result = alignment.FastAlignImages(testImg, templateImg, params);
    
    if (result.success) 
    {
        cout << "特征对齐成功!" << endl;
        cout << "内点数量: " << result.inlierCount << endl;
        cout << "重投影误差: " << result.reprojectionError << endl;
        cout << "变换矩阵: " << endl << result.transformMatrix << endl;
        
        // 测试图像重构
        Mat alignedImage = alignment.WarpImage(testImg, result.transformMatrix, templateImg.size());
        
        if (!alignedImage.empty()) 
        {
            cout << "图像重构成功，尺寸: " << alignedImage.cols << "x" << alignedImage.rows << endl;
            
            // 保存结果
            imwrite("../Img/aligned_result.jpg", alignedImage);
            cout << "对齐结果已保存到: ../Img/aligned_result.jpg" << endl;
        } 
        else 
        {
            cout << "图像重构失败" << endl;
        }
    } 
    else 
    {
        cout << "特征对齐失败" << endl;
        cout << "内点数量: " << result.inlierCount << endl;
    }
    
    cout << "测试完成" << endl;
    return 0;
}
