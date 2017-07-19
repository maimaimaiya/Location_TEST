#include "stdafx.h"
#include <queue>
#include <vector>
#include <string.h>
#include "Location.h"
#define MAX_IMAGESIZE  2048 /* 想定する縦・横の最大画素数 */  /*设定的横纵最大像素值*/
#define MAX_BRIGHTNESS  255 /* 想定する最大階調値 */   /*设定的最大灰度值*/
#define GRAYLEVEL       256 /* 想定する階調数(=最大階調値+1) */  /*最大灰度值+1*/
#define MAX_FILENAME    256 /* 想定するファイル名の最大長 */  /*设定的文件名的最大长度*/
#define MAX_BUFFERSIZE  256 /* 利用するバッファ最大長 */  /*被使用的缓存最大长度*/
#define MAX_CNTR  5000      /* 輪郭線の想定最大構成画素数 */  /*轮廓线设定的最大像素值*/
#define GRAY 128
char m_OutAddress[30];
const float DEFAULT_ERROR = 0.6;
const float DEFAULT_ASPECT = 3.75;
//车牌逆方差范围
const double IDMOMENT_MAX = 2.95;
const double IDMOMENT_MIN = 2.05;
vector<int>chain_code;
//int chain_code[MAX_CNTR];   /*轮廓链码*/
int Freeman[8][2] = {  /*链码的偏移值 8方向*/
	{ 1, 0 },{ 1, -1 },{ 0, -1 },{ -1, -1 },
	{ -1, 0 },{ -1,  1 },{ 0,  1 },{ 1,  1 } };
int image1[MAX_IMAGESIZE][MAX_IMAGESIZE],
image2[MAX_IMAGESIZE][MAX_IMAGESIZE];
int x_size2, y_size2;
int dir[4][2] = { 1,0,0,1,-1,0,0,-1 }; //4方向
Mat dst;
struct Contour
{
	Point pos;
	vector<int>chain;
};
vector<vector<Point>>_contours;
vector<Contour>contours;

CLocation::CLocation(Mat img, string name)
{
	src = img.clone();
	height = src.rows;
	width = src.cols;
	m_error = DEFAULT_ERROR;
	m_aspect = DEFAULT_ASPECT;
	m_angle = DEFAULT_ANGLE;
	m_verifyMin = DEFAULT_VERIFY_MIN;
	m_verifyMax = DEFAULT_VERIFY_MAX;
	string m_OutAddressFirst = "./src/out/";
	//地址+文件名
	strcpy(m_OutAddress, (m_OutAddressFirst + name).c_str());
	m_bug = 1;
}


CLocation::~CLocation()
{
}

bool CLocation::Color_Contour()
{
	//dst = src.clone();
	Mat temp(src.rows , src.cols, CV_8UC3, Scalar(0, 0, 0));
	dst = temp.clone();
	//cvtColor(src, srcHSV, CV_BGR2HSI);
	//int H, S, V;
	for (int i = 2; i < src.rows - 3; i++)
	{
		for (int j = 2; j < src.cols - 3; j++)
		{
			//H = (int)srcHSV.at<Vec3b>(i, j)[0];
			//S = (int)srcHSV.at<Vec3b>(i, j)[1];
			//V = (int)srcHSV.at<Vec3b>(i, j)[2];
			bool blue_status = false;
			bool white_status = false;
			for (int x = -1; x <= 1; x += 2)
			{
				int blue_count = 0;
				int white_count = 0;
			
				//for (int y = -1; y <= 1; y++)
				{//453,272
					if (Blue_Judge(i , j+x,src) && !blue_status)
						blue_count++;
					if (White_Judge(i , j+x,src) && !white_status)
						white_count++;
				}
				if (blue_count >= 1)
					blue_status = true;
				if (white_count >= 1)
					white_status = true;
			}
			/*for (int y = -1; y <= 1; y += 2)
			{
				int blue_count = 0;
				int white_count = 0;

				for (int x = -1; x <= 1; x++)
				{
					if (Blue_Judge(i+y, j+x ) && !blue_status)
						blue_count++;
					if (White_Judge(i+y, j +x) && !white_status)
						white_count++;
				}
				if (blue_count >= 1)
					blue_status = true;
				if (white_count >= 1)
					white_status = true;
			}*/
			if (blue_status&&white_status)
			{
				//for (int x = -1; x <= 1; x++)
				{
					for (int y = -1; y <= 1; y++)
					{
						image1[i + y][j] = 255;
						dst.at<Vec3b>(i + y, j)[0] = 255;
						dst.at<Vec3b>(i + y, j)[1] = 255;
						dst.at<Vec3b>(i + y, j)[2] = 255;
					}
				}
			//	cout << i << " " << j << endl;
			}
			else
			{
				for (int y = -1; y <= 1; y++)
				{
					image1[i + y][j] = 0;
				}
			}
		}
	}
	if (m_bug)
	{
		imshow("test0", dst);
		cvWaitKey(0);
	}
	cvtColor(dst, dst, CV_RGB2GRAY);
	threshold(dst, dst, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
	Morphological(dst);
	if (m_bug)
	{
		imshow("形态学", dst);
		cvWaitKey(0);
	}
	for (int i = 0; i < dst.rows; i++)
	{
		for (int j = 0; j < dst.cols; j++)
		{
			if ((int)dst.at<uchar>(i, j) == MAX_BRIGHTNESS)
				image1[i][j] = 255;
			else
				image1[i][j] = 0;
		}
	}
	remove_area();
	Drew2Back(image2);
	if (m_bug)
	{
		imshow("轮廓", dst);
		cvWaitKey(0);
	}
	return MinRect();
	vector<Contour>().swap(contours);


}

bool CLocation::Blue_Judge(int x, int y,Mat &img)
{
//	x = 453, y =271;
	/*src.at<Vec3b>(x, y)[0] = 0;
	src.at<Vec3b>(x, y)[1] = 0;
	src.at<Vec3b>(x, y)[2] = 255;*/
	double b = (double)img.at<Vec3b>(x, y)[0];
	double g = (double)img.at<Vec3b>(x, y)[1];
	double r = (double)img.at<Vec3b>(x, y)[2];
	//cout << b << " " << g << " " << r;
	/*double h;
	double numerator = (r - g + r - b) / 2;
	double denominator = sqrt( pow((r - g), 2) + (r - b)*(g - b));
	if (denominator == 0)
		h = 0;
	else
		h = acos(numerator/denominator)*180/3.14;
	if (b > g)
		h = 360 - h;
	double s = 1 - (double)3.0*min(r, min(g, b)) / (r + g + b);
	double i = (r + g + b) / 3.0;*/
	//if(h>=210&&h<240&&s>=0.25&&s<=1)
	if (b *1.0> 1.4*g*1.0&&b*1.0 > 1.4*r*1.0&&b > 50)
		return true;
	return false;
}

bool CLocation::White_Judge(int x, int y,Mat &img)
{
	double b = (double)img.at<Vec3b>(x, y)[0];
	double g = (double)img.at<Vec3b>(x, y)[1];
	double r = (double)img.at<Vec3b>(x, y)[2];
	double h;
	double numerator = (r - g + r - b) / 2;
	double denominator = sqrt(pow((r - g), 2) + (r - b)*(g - b));
	if (denominator == 0)
		h = 0;
	else
		h = acos(numerator / denominator) * 180 / 3.14;
	if (b > g)
		h = 360 - h;
	double s = 1 - (double)3.0*min(r, min(g, b)) / (r + g + b);
	double i = (r + g + b) / 3;
	if((i>=200&&i<250)||(s<=0.25&&s>=0))
//	double S = b + g + r;
	//if (b*1.0 < 0.4*S&&g*1.0 < 0.4*S&&r*1.0 < 0.4*S&&S>150)
		return true;
	return false;  
}

int CLocation::Find_contour(int x_start,int y_start)
{
	int  x, y;               /*当前的目标象素的上轮廓的坐标*/
	int xs, ys;               /*搜索点的感兴趣的像素周围的坐标*/
	int code, num;           /*轮廓点的链码，总数*/
	int i, counter, detect;   /*制约变量*/
	vector<Point>point;

	counter = 0;			 /*检查孤立点*/
	for (i = 0; i<8; i++) {
		xs = x_start + Freeman[i][1];
		ys = y_start + Freeman[i][0];
		if (xs >= 0 && xs <= height && ys >= 0 && ys <= width
			&& image1[xs][ys] == MAX_BRIGHTNESS) counter++;
	}
	if (counter == 0) num = 1;   /*起点是孤立点*/
	else {
		
		num = -1;   x = x_start;    y = y_start;    code = 0;
		do {
			detect = 0;   /*初始方向的确定*/
			code = code - 3;   if (code < 0) code = code + 8;
			do {
				xs = x + Freeman[code][1];
				ys = y + Freeman[code][0];
				if (xs >= 0 && xs <= height && ys >= 0 &&
					ys <= width &&
					image1[xs][ys] == MAX_BRIGHTNESS) {
					detect = 1;   /*下一个点的检测*/
					num++;
					//chain_code[num] = code;
					chain_code.push_back(code);
					point.push_back(Point(ys, xs));
					x = xs;  y = ys;
				//	image2[xs][ys] = MAX_BRIGHTNESS;
				}
				code++;  if (code > 7) code = 0;
			} while (detect == 0);
		} while (x != x_start || y != y_start);  /*检测起点*/
		num = num + 2; /*chain_code[]的下标偏差的修改*/
	}
	_contours.push_back(point);
	vector<Point>().swap(point);
	return(num);   /*返回轮廓总数*/
}

void CLocation::remove_area()
{
	int _threshold;              /*周长的阈值*/
	int num, x, y, xs, ys, i;
	int fill_value;

	_threshold = 150;
	 /*图像的初始化*/
	x_size2 = height;    y_size2 = width;
	for (y = 0; y < width; y++)
		for (x = 0; x < height; x++)
			image2[x][y] = 0;

	for (x = 0; x < height; x++) {
		for (y = 0; y < width; y++) {
			if (image1[x][y] == MAX_BRIGHTNESS) {  /* 开始点 */
				num = Find_contour(x, y);  /* 轮廓线跟踪 */
			
				if (chain_code.size() <_threshold)
				{
					//chain_code.clear();
					image1[x][y] = 0;
					vector<int>().swap(chain_code);
					_contours.pop_back();
					continue;
				}
				fill_value = MAX_BRIGHTNESS;
				xs = x;  ys = y;
				image1[xs][ys] = 0;
				image2[xs][ys] = fill_value; 
				BFS(x+1, y+1);
				if (num > 1) {
					for (i = 0; i < chain_code.size(); i++) {
						xs = xs + Freeman[chain_code[i]][1];
						ys = ys + Freeman[chain_code[i]][0];
						BFS(xs, ys);
						image1[xs][ys] = 0;
						image2[xs][ys] = fill_value;
					}
				}
				Contour cont;
				cont.pos.x = x;
				cont.pos.y = y;		
				cont.chain.assign(chain_code.begin(), chain_code.end());			
				contours.push_back(cont);
				vector<int>().swap(chain_code);
			}
		}
	}

	/*vector<Contour>::iterator iter;

	for (iter = contours.begin(); iter != contours.end(); iter++)
	{
		//if ((*iter).chain.size() < _threshold)
			//continue;
		cout << "轮廓起点坐标：X："<<(*iter).pos.x << " " << "Y："<<(*iter).pos.y << " "<< "轮廓周长："<< (*iter).chain.size() << endl;
	}*/
}

void CLocation::Drew2Back(int image[][2048])
{
	for (int i = 0; i < dst.rows; i++)
	{
		for (int j = 0; j < dst.cols; j++)
		{
			//dst.at<uchar>(i, j) = image[i][j];	
			*(dst.data + dst.step[0] * i + dst.step[1] * j) = image[i][j];
		}
	}
}


void CLocation::BFS(int x, int y)
{
	queue<Point>q;
	Point str, mov;
	str.x = x;
	str.y = y;
	q.push(str);
	while (!q.empty())
	{
		mov = q.front();
		q.pop();
		image1[mov.x][mov.y] = 0;
		for (int i = 0; i < 8; i++)
		{
			mov.x += Freeman[i][0];
			mov.y += Freeman[i][1];
			
			if (image1[mov.x][mov.y] == MAX_BRIGHTNESS)
			{
				image1[mov.x][mov.y] = 0;
				q.push(mov);
			}
		}
	}
}

bool CLocation::MinRect()
{
	vector<vector<Point>>::iterator iter;
	vector<RotatedRect>rects;
	for (iter = _contours.begin(); iter < _contours.end(); )
	{
		RotatedRect mr = minAreaRect(Mat(*iter));
		/*Point2f vertex[4];
		mr.points(vertex);
		for (int i = 0; i < 4; i++)
		{
			line(src, vertex[i], vertex[(i + 1) % 4], Scalar(0, 0, 255), 2, LINE_AA);
			cout << vertex[i] << endl;
		}*/
		//判断是否满足宽高比 不满足则删除
		if (!verifySizes(mr))
		{
			iter = _contours.erase(iter);
		}
		else
		{
			iter++;
			rects.push_back(mr);
		}
	}
	vector<Mat> resultVec;
	for (int i = 0; i < rects.size(); i++)
	{
		RotatedRect minRect = rects[i];

		if (verifySizes(minRect))
		{
			//倾斜校正 可以进一步筛出部分图片
			float r = (float)minRect.size.width / (float)minRect.size.height;
			float angle = minRect.angle;
			Size rect_size = minRect.size;
			if (r < 1)
			{
				angle = 90 + angle;
				swap(rect_size.width, rect_size.height);
			}
			//如果抓取的方块旋转超过m_angle角度，则不是车牌，放弃处理
			if (angle - m_angle < 0 && angle + m_angle > 0)
			{
				//Create and rotate image
				Mat rotmat = getRotationMatrix2D(minRect.center, angle, 1);
				Mat img_rotated;
				//仿射
				warpAffine(src, img_rotated, rotmat, src.size(), CV_INTER_CUBIC);

				//Mat resultMat(img_rotated, minRect);
				Mat resultMat;
				resultMat = showResultMat(img_rotated, rect_size, minRect.center);
				resultVec.push_back(resultMat);
				if (m_bug)
				{
					imshow("result", resultMat);
					cvWaitKey(0);
				}
			}
		}
	}
	return calmGlcm(resultVec);
	
}

//! 对minAreaRect获得的最小外接矩形，用纵横比进行判断
bool CLocation::verifySizes(RotatedRect mr)
{
	float error = m_error;
	//Spain car plate size: 52x11 aspect 4,7272
	//China car plate size: 440mm*140mm，aspect 3.142857
	float aspect = m_aspect;
	//Set a min and max area. All other patchs are discarded
	//int min= 1*aspect*1; // minimum area
	//int max= 2000*aspect*2000; // maximum area
	int min = 44 * 14 * m_verifyMin; // minimum area
	int max = 44 * 14 * m_verifyMax; // maximum area
									 //Get only patchs that match to a respect ratio.
	float rmin = aspect - aspect*error;
	float rmax = aspect + aspect*error;

	int area = mr.size.height * mr.size.width;
	float r = (float)mr.size.width / (float)mr.size.height;
	if (r < 1)
	{
		r = (float)mr.size.height / (float)mr.size.width;
	}

	if ((area < min || area > max) || (r < rmin || r > rmax))
	{
		return false;
	}
	else
	{
		return true;
	}
}


//！形态学处理
void CLocation::Morphological(Mat &img)
{
	//imshow("形态学前", src);
	//cvWaitKey(0);

	Mat element = getStructuringElement(MORPH_RECT, Size(25, 5)); //先横向25*2
																  //形态学处理
	morphologyEx(img, img, MORPH_CLOSE, element);

	//imshow("第一次形态学", src);
	//cvWaitKey(0);

	/*element = getStructuringElement(MORPH_RECT, Size(5, 2)); //横向5*2
															 //形态学处理
	morphologyEx(img, img, MORPH_OPEN, element);

	//imshow("第二次形态学", src);
	//cvWaitKey(0);

	element = getStructuringElement(MORPH_RECT, Size(2, 5)); //纵向2*5
															 //形态学处理
	morphologyEx(img, img, MORPH_OPEN, element);*/

	//imshow("第三次形态学", src);
	//cvWaitKey(0);


}


//! 显示最终生成的车牌图像，便于判断是否成功进行了旋转。
Mat CLocation::showResultMat(Mat src, Size rect_size, Point2f center)
{
	Mat img_crop;

	getRectSubPix(src, rect_size, center, img_crop);

	Mat resultResized;

	resultResized.create(36, 136, TYPE);

	resize(img_crop, resultResized, resultResized.size(), 0, 0, INTER_CUBIC);

	return resultResized;
}

bool CLocation::calmGlcm(vector<Mat>&Result)
{
	vector<Mat>::iterator itc;
	int k = 0;
	/*for (itc = Result.begin(); itc < Result.end(); )
	{
		VecGLCM vec;
		GLCMFeatures features;
		IplImage * img;
		Mat gray;
		cvtColor((*itc), gray, CV_RGB2GRAY);
		img = &IplImage(gray);
	//	img = cvCreateImage(cvSize((*itc).cols, (*itc).rows), IPL_DEPTH_8U, 1); //根据实际进行初始化
		//img->imageData = (char*)(*itc).data;
		glcm.initGLCM(vec);
		// 水平  
		glcm.calGLCM(img, vec, GLCM::GLCM_HORIZATION);
		glcm.getGLCMFeatures(vec, features);
		// 垂直  
		glcm.calGLCM(img, vec, GLCM::GLCM_VERTICAL);
		glcm.getGLCMFeatures(vec, features);
		// 45 度  
		glcm.calGLCM(img, vec, GLCM::GLCM_ANGLE45);
		glcm.getGLCMFeatures(vec, features);
		// 135 度  
		glcm.calGLCM(img, vec, GLCM::GLCM_ANGLE135);
		glcm.getGLCMFeatures(vec, features);
		energy = features.energy;
		entropy = features.entropy;
		contrast = features.contrast;
		idMoment = features.idMoment;
		//cout << energy << " " << entropy << " " << contrast << " " << idMoment << endl;
		if (idMoment > IDMOMENT_MAX || idMoment < IDMOMENT_MIN)
			itc = Result.erase(itc);
		else
			itc++;
	}
	/*for (itc = Result.begin(); itc < Result.end(); itc++)
	{
		imwrite("./out/"+to_string(k) + ".jpg", (*itc));
		k++;
	}*/
	double DegreeMax = -1;
	int LastKey = -1;

	for (int k = 0; k < Result.size(); k++)
	{
		int blue_conts = 0;
		int white_conts = 0;
		for (int i = 0; i < Result[k].rows; i++)
		{
			for (int j = 0; j < Result[k].cols; j++)
			{
				if (Blue_Judge(i, j,Result[k]))
				{
					blue_conts++;
				}
				if (White_Judge(i, j,Result[k]))
					white_conts++;
			}
		}
		if (blue_conts < 1500)
			continue;
		Mat TempDst;
		cvtColor(Result[k], TempDst, CV_BGR2GRAY);

		int graySum = 0;
		for (int i = 1; i < TempDst.rows; i++)
		{
			for (int j = 1; j < TempDst.cols; j++)
			{
				graySum += (int)TempDst.at<uchar>(i, j);
			}
		}

		int grayMean = graySum / (TempDst.rows* TempDst.cols);//求灰度值均值


															  //对比度增强 将灰度值小于均值的元素赋0
		for (int i = 1; i < TempDst.rows; i++)
		{
			for (int j = 1; j < TempDst.cols; j++)
			{
				if ((int)TempDst.at<uchar>(i, j) < grayMean)
					TempDst.at<uchar>(i, j) = 0;
				else
				{
					TempDst.at<uchar>(i, j) = (int)TempDst.at<uchar>(i, j)*((int)TempDst.at<uchar>(i, j) - grayMean) / (255 - grayMean);

				}
			}
		}

		threshold(TempDst, TempDst, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY); //二值化
																				//	imshow("da", TempDst);
																				//	waitKey(0);
																				// 进行车牌特征值计算
		double Degree = VerticalProjection(TempDst);

		//车牌对应最大的特征值
		if (Degree > DegreeMax)
		{
			DegreeMax = Degree;
			LastKey = k;
		}
	}
	/*if (m_deBug)
	{
		if (resultVec.size() != 0)
		{
			imshow("result", resultVec[LastKey]);
			cvWaitKey(0);
		}
	}*/
	//imshow("Last", resultVec[LastKey]);
	if (DegreeMax != -1 || LastKey != -1)
	{
		imwrite(m_OutAddress, Result[LastKey]);

		printf("成功\n");
		//vector< vector< Point> >().swap(contours);
		//vector< RotatedRect >().swap(rects);
		vector< Mat >().swap(Result);
		return true;
	}
	//vector< vector< Point> >().swap(contours);
	//vector< RotatedRect >().swap(rects);
	vector< Mat >().swap(Result);
	printf("失败\n");
	return false;

}


double CLocation::VerticalProjection(Mat img)
{
	m_Projection = new int[img.cols]();
	//memset(m_Projection, 0, sizeof(m_Projection));
	for (int j = 1; j < img.cols; j++)
	{
		//舍弃上下20% 只取中间60%做统计
		for (int i = 1 + 0.2*img.rows; i < img.rows*(1 - 0.2); i++)
		{
			if ((int)img.at<uchar>(i, j) > 0)
			{
				m_Projection[j]++;
			}
		}
		//cout << m_Projection[j]<<" ";
	}
	//cout << endl;
	bool CharState = false;
	int *m_CharWidth = new int[img.cols]();
	int m_CharNum = 0;
	int m_RuleDegree;
	//统计字符个数
	for (int i = 1; i < img.cols; i++)
	{
		if (m_Projection[i] != 0 && !CharState)
		{
			m_CharNum++;
			CharState = true;
		}
		else if (!m_Projection[i] && CharState)
		{		
			CharState = false;
		}
		if (CharState)
			m_CharWidth[m_CharNum]++;
	}
	
	double wid = img.cols / 10;
	int LastNum = m_CharNum;
	double TempNum = 0;
	for (int i = 1; i < m_CharNum; i++)
	{
		if (m_CharWidth[i] < wid / 4)
			LastNum--;
		else if (m_CharWidth[i] < wid / 3 || m_CharWidth[i] > wid * 1.5)
			TempNum++;
	}
	//cout << LastNum << " " << TempNum << endl;
	if (LastNum < 7)
		m_RuleDegree = 0;
	else
	{
		//计算待选区域规则度
		//	1 - 0.6*(abs(7 - LastNum)) / max(7, LastNum) - 0.4*(TempNum / LastNum);
		m_RuleDegree = 1 - 0.6*(abs(7 - LastNum) *1.0 / max(7, LastNum)) - (1 - 0.6)*(TempNum *1.0 / LastNum);
	}
	return m_RuleDegree;
}
