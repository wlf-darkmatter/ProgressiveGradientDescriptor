#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <PGD.h>


using namespace std;

string cd(const string &_current_path, const string &str_cd);

int main(int argc, char *argv[]) {
	string path_project(*argv);
	//获取上一级的目录
	path_project = cd(path_project, "..");
	path_project = cd(path_project, "..");

	string src_pathname = "data/Kamisato.jpeg";
	cv::Mat img_origin = cv::imread(path_project + "/" + src_pathname);
	if (img_origin.empty()) std::cout << "没有载入图像" << std::endl;
	else std::cout << "载入成功" << std::endl;

	cv::Mat pgd_result;


	PGDClass::calc_PGDFilter(img_origin, pgd_result, PGDClass::PGD_SampleNums_8, 2, n_sample, 1);
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

