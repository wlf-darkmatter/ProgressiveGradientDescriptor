#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <PGD.h>


using namespace std;

string cd(const string &_current_path, const string &str_cd);

int main(int argc, char *argv[]) {
	string path_project(*argv);
	clock_t time_count_start = 0;
	clock_t time_count_end = 0;
	//获取上一级的目录
	path_project = cd(path_project, "..");
	path_project = cd(path_project, "..");

	string src_pathname = "data/Kamisato.jpg";
	cv::Mat img_origin = cv::imread(path_project + "/" + src_pathname);
	std::cout << "输入图像原长度为 " << img_origin.rows << "行, " << img_origin.cols << " 列." << std::endl;
	if (img_origin.empty()) std::cout << "没有载入图像" << std::endl;
	else std::cout << "载入成功" << std::endl;

	time_count_start = clock();
	PGDClass_::Struct_PGD struct_PGD(img_origin.rows, img_origin.cols, PGDClass_::PGD_SampleNums_4, PGDClass_::PGD_SampleNums_4);
	time_count_end = clock();
	cout << "数据初始化耗时为： " << (double) (time_count_end - time_count_start) / CLOCKS_PER_SEC << " s" << endl;

	time_count_start = clock();
	PGDClass_::calc_PGDFilter(img_origin, struct_PGD, 5, 3);
	time_count_end = clock();
	cout << "遍历运行时间为： " << (double) (time_count_end - time_count_start) / CLOCKS_PER_SEC << " s" << endl;


	//--------------------------------------------------------------------------------------------------------
	///固化参数优化函数必须使用灰度浮点类型图像
	///******设定算子全局配置******
	int R44_1 = 5;
	int R44_2 = 3;
	PGDClass_::Struct_PGD struct_PGD_44(img_origin.rows, img_origin.cols, PGDClass_::PGD_SampleNums_4, PGDClass_::PGD_SampleNums_4);
	///*********预处理*********
	cv::Mat img_gray;
	if (img_origin.channels() == 3) {
		cv::cvtColor(img_origin, img_gray, cv::COLOR_BGR2GRAY, 0);
	} else img_gray = img_origin;
	cv::Mat img_double;
	img_gray.convertTo(img_double, CV_64FC1);
	img_double = img_double / 255;
	///*********预处理完毕*********
	time_count_start = clock();
	PGDClass_::calc_PGDFilter44_Int(img_double, struct_PGD_44, R44_1, R44_2);
	time_count_end = clock();
	cout << "优化后遍历运行时间为： " << (double) (time_count_end - time_count_start) / CLOCKS_PER_SEC << " s" << endl;
	//--------------------------------------------------------------------------------------------------------

	///读取数据
	cout << "第111行，222列的G值分别为" << endl;
	uint16_t G = 0;
	uint16_t G44 = 0;
	for (int i = 0; i < 8; ++i) {
		G = struct_PGD.PGD_read<uint16_t>(111, 222, i);
		G44 = struct_PGD_44.PGD_read<uint16_t>(111, 222, i);
		cout << G << "<=>";
		cout << G44 << " ";
	}
	cout << endl;
}

/*!
 * @brief 用来跳转路径的函数
 * @param _current_path
 * @param str_cd
 * @return dst_path 返回跳转后的路径
 * @to×do 以后要增添更多的功能，造轮子写成自己的库
 * @todo 造个屁的轮子，累死了，转python去。
 */
string cd(const string &_current_path, const string &str_cd) {
	string dst_path;
	string tmp_path = string(_current_path);
	//去掉目录的最后一个'/'
	if (tmp_path.back() == '/') tmp_path.erase(tmp_path.length() - 1);
	if (str_cd == "..") {
		unsigned int tmp_len = (unsigned int) tmp_path.length() - strlen(strrchr(tmp_path.c_str(), '/'));
		dst_path = tmp_path.substr(0, tmp_len) + '/';
	}
	return dst_path;
}

