#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <PGD.h>


using namespace std;

int main(int argc, char *argv[]) {
	string src_pathname = "data/Kamisato.jpeg";
	string path_project = "/Users/x-contion/Desktop/LocalCode/C++/ProgressiveGradientDescriptor/";


	cv::Mat img_origin = cv::imread(path_project + "/" + src_pathname);
	if (img_origin.empty()) printf("没有载入图像");

	cv::Mat pgd_result;
	int level_0 = sizeof(char);
	int level_1 = sizeof(unsigned short);
	int level_2 = sizeof(int);
	int level_3 = sizeof(long);

	cout << "系统中的char占用大小为：" << level_0 << "×8b"
	     << "\n系统中的short占用大小为：" << level_1 << "×8b"
	     << "\n系统中的int占用大小为：" << level_2 << "×8b"
	     << "\n系统中的long占用大小为：" << level_3 << "×8b"
	     << endl;

	PGDClass::calc_PGDFilter(img_origin, pgd_result, 1, PGDClass::PGD_SampleNums_8);




}



