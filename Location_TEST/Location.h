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
	bool Color_Contour();
	bool Blue_Judge(int x,int y, Mat &img);
	bool White_Judge(int x, int y, Mat &img);
	int Find_contour(int x_start, int y_start);
	void remove_area();
	void Drew2Back(int image[][2048]);
	void BFS(int x, int y);
	bool MinRect();
	bool verifySizes(RotatedRect mr);
	void Morphological(Mat &img);
	Mat showResultMat(Mat src, Size rect_size, Point2f center);
	bool calmGlcm(vector<Mat>&Result);
	double VerticalProjection(Mat src);
protected:

private:
	Mat src;
	Mat srcHSV;
	int *m_Projection;
	Mat src_contour;
	static const int MAX_COLS = 1500;
	static const int MAX_ROWS = 1500;
	static const int NORM_WIDTH = 600;
	static const int NORM_HEIGHT = 800;
	static const int TYPE = CV_8UC3;
	int Color_Mark[MAX_COLS][MAX_ROWS];
	int Color_HSV[MAX_COLS][MAX_ROWS][3];
	int width;
	int height;
	//! �Ƕ��ж����ó���
	static const int DEFAULT_ANGLE = 30;

	//! �Ƕ��ж����ñ���
	int m_angle;

	//! verifySize���ó���
	static const int DEFAULT_VERIFY_MIN = 1;
	static const int DEFAULT_VERIFY_MAX = 100;
	//! verifySize���ñ���
	float m_error;
	float m_aspect;
	int m_verifyMin;
	int m_verifyMax;
	int m_bug;
	GLCM glcm;
	double energy; //����
	double entropy;//��
	double contrast;//�Աȶ�
	double idMoment;//�淽��
};

