// Location_TEST.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"
#include "Location.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <io.h>
#include <opencv2/imgproc/imgproc.hpp>
#define NORM_WIDTH 600
#define NORM_HEIGHT 800

int main()
{
	ifstream ifs("./src/data.txt");
	string temp;
	string tempFirst = "./src/src_test/";
	string m_name;
	char tempname[30];
	int Sum = 0;
	int Success_Num = 0;
	time_t nowclock, overclock;
	nowclock = clock();
	while (!ifs.eof())
	{
		getline(ifs, temp);
		temp = "test12.jpg";
		if (temp.size() <= 4)
			continue;
		Sum++;
		strcpy(tempname, temp.c_str());
		printf("����⳵�ƣ�%s\n", tempname);
		m_name = tempFirst + temp;
		Mat srcImg = imread(m_name);
		//	Mat dstImage(NORM_WIDTH, NORM_HEIGHT, srcImage.type());
		Mat dstImg;
		dstImg.create(NORM_WIDTH, NORM_HEIGHT, 16);
		resize(srcImg, dstImg, dstImg.size(), 0, 0, INTER_CUBIC);
		//imshow("ԭͼ", dstImg);
		CLocation test(dstImg,temp);
		//test.Color_Contour();
		if (test.Color_Contour())
		{
			Success_Num++;
		}
	}
	overclock = clock();
	printf("ʶ��������%d�� �ɹ�ʶ��%d�� ʶ���ʣ�%.2lf%% ;ʱ�䣺%d\n", Sum, Success_Num, Success_Num*(1.0) / Sum * 100, int(overclock - nowclock));
    return 0;
}

 