#pragma once
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <math.h>
#include "GLCM.h"
using namespace cv;
using namespace std;
class CLocation
{
public:
	CLocation(Mat img, string name);
	~CLocation();
	//! 颜色边缘提取
	bool Color_Contour();
	//! 判断蓝色
	bool Blue_Judge(int x,int y, Mat &img);
	//! 判断白色
	bool White_Judge(int x, int y, Mat &img);
	//! 轮廓查找
	int Find_contour(int x_start, int y_start);
	//! 去除内部区域
	void remove_area();
	void Drew2Back(int image[][2048]);
	//! 广搜去除内部信息
	void BFS(int x, int y);
	//! 尺寸判断
	bool Size_Judge();
	//! 尺寸判断
	bool verifySizes(RotatedRect mr);
	//! 形态学操作
	void Morphological(Mat &img); 
	//! 显示最终生成的车牌图像，便于判断是否成功进行了旋转。
	Mat showResultMat(Mat src, Size rect_size, Point2f center);
	//! 精定位
	bool FinePositioning(vector<Mat>&Result);
	//! 字符规则度判断
	double VerticalProjection(Mat src);
protected:

private:
	Mat src;
	Mat srcHSV;
	int *m_Projection;
	Mat src_contour;
	//! 尺寸常量
	static const int MAX_COLS = 1500;
	static const int MAX_ROWS = 1500;
	static const int NORM_WIDTH = 600;
	static const int NORM_HEIGHT = 800;
	static const int TYPE = CV_8UC3;
	//! 颜色标记数组
	int Color_Mark[MAX_COLS][MAX_ROWS];
	int Color_HSV[MAX_COLS][MAX_ROWS][3];
	int width;
	int height;
	//! 角度判断所用常量
	static const int DEFAULT_ANGLE = 30;

	//! 角度判断所用变量
	int m_angle;

	//! verifySize所用常量
	static const int DEFAULT_VERIFY_MIN = 1;
	static const int DEFAULT_VERIFY_MAX = 100;
	//! verifySize所用变量
	float m_error;
	float m_aspect;
	int m_verifyMin;
	int m_verifyMax;

	int m_bug;

	GLCM glcm;

	double energy; //能量
	double entropy;//熵
	double contrast;//对比度
	double idMoment;//逆方差
};

