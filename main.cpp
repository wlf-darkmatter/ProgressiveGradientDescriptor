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
	PGDClass_::Struct_PGD struct_PGD(img_origin, PGDClass_::PGD_SampleNums_8, PGDClass_::PGD_SampleNums_16);
	time_count_end = clock();
	cout << "数据初始化耗时为： " << (double) (time_count_end - time_count_start) / CLOCKS_PER_SEC << " s" << endl;

	time_count_start = clock();
	PGDClass_::calc_PGDFilter(img_origin, struct_PGD, 2, 1);
	time_count_end = clock();
	cout << "遍历运行时间为： " << (double) (time_count_end - time_count_start) / CLOCKS_PER_SEC << " s" << endl;


	///读取数据
	cout << "第111行，222列的G值分别为" << endl;
	uint16_t G = 0;
	for (int i = 0; i < 8; ++i) {
		G = struct_PGD.PGD_read<uint16_t>(111, 222, i);
		cout << G << " ";
	}
	cout << endl;
}

/*!
 * @brief 用来跳转路径的函数
 * @param _current_path
 * @param str_cd
 * @return dst_path 返回跳转后的路径
 * @todo 以后要增添更多的功能，造轮子写成自己的库
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

