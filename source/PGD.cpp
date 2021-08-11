#include <opencv2/opencv.hpp>
#include <PGD.h>

#define PI 3.1415926535897932384626433832795028841971

/*!
 * @brief calc_PGDFilter()函数，根据给定的圆周大小计算n个采样点的方向不变特征
 * @param _src 输入的矩阵
 * @param _dst 返回的矩阵
 * @param radio 半径大小（浮点数）
 * @param n_sample 采样点数，一般为4的倍数
 * @return 返回值是一个矩阵，
*/
cv::Mat PGDClass::calc_PGDFilter(cv::InputArray _src, cv::OutputArray _dst, float radio, PGD_SampleNums n_sample) {
	int l_size = 1 + 2 * ceil(radio);
	int rows = _src.rows();
	int cols = _src.cols();
	cv::Mat src_gray;

	///①通道数量转换
	//如果是三通道，使用灰度图像
	if (_src.channels() == 3) {
		cv::cvtColor(_src, src_gray, cv::COLOR_BGR2GRAY, 0);
	} else src_gray = _src.getMat();

	cv::Mat dst = def_DstMat(rows, cols, n_sample);

	/*               ①→
	 *                   ↘
	 *        ④     ⭕️     ②
	 *                      ↓
	 *               ③
	 */
	///②计算样本采样坐标偏移量
	//初始化，计算圆周采样点坐标
	//返回的是double[n_sample][2]
	double **list_local_offset = calc_CircleOffset(radio, n_sample);

	/*                  _____
	 *                ①|🟥🟥|
	 *           ⑧     |🟥②|    ②号采样点位于一个田字格内
	 *         ⑦     ⭕️ ￣￣ ③   偏移坐标为(√2/2,-√2/2 )
	 *           ⑥        ④     因此需要二次插值，为了优化算法，
	 *                ⑤          这里通过列表事先计算好插值的权重值
	 *
	 */
	///③计算每一个采样点的二次插值需要的参考比例
	//初始化，把结果放到一个表里
	//返回的是double[n_sample][4]
	double **list_interp_

	///④遍历全图



	_dst.assign(dst);
	return dst;
}


/*!
 * @brief 私有函数，创建输出的空矩阵
 *  @brief rows 矩阵的行数
 *  @brief cols 矩阵的列数
 *  @brief n_sample 采样数
 */
cv::Mat PGDClass::def_DstMat(int rows, int cols, PGD_SampleNums n_sample) {
	int level_0 = 8 * sizeof(char);
	int level_1 = 8 * sizeof(short);
	int level_2 = 8 * sizeof(int);
	int level_3 = 8 * sizeof(long);
	if (n_sample <= level_0) return cv::Mat_<uchar>(rows, cols);
	else if (n_sample <= level_1) return cv::Mat_<unsigned short>(rows, cols);
	else if (n_sample <= level_2) return cv::Mat_<int>(rows, cols);//32位有符号（位操作时可以忽略符号位）
	else if (n_sample <= level_3) return cv::Mat_<double>(rows, cols);//虽然是double，但是读写的时候使用的是64位数的性质
}


/*!
 * @brief 计算在目标区域中邻域的n_sample个采样点相对于中心点的偏移量
 *  @param n_sample 采样点个数，有几个采样点就有几个需要计算的偏移量
 *  @param radio 偏移量半径
 *  @return 返回类内结构体 Struct_SampleOffsetList（专门存放每个采样点相对于中心处偏移量的结构体）
 *  @note offset[ i ][ 0 ] 表示第i个元素的x偏移量;
 *  offset[ i ][ 1 ] 表示第i个元素的y偏移量
 */
PGDClass::Struct_SampleOffsetList PGDClass::calc_CircleOffset(double radio, int n_sample) {
	double **offset;
	double theta = 0;

	offset = new double *[n_sample];
	for (int i = 0; i < n_sample; i++) {
		offset[i] = new double[2];
		theta = i * 2 * PI / n_sample;
		offset[i][0] = sin(theta);//表示x偏移量
		offset[i][1] = -cos(theta);//表示y偏移量
	}
	return offset;
}

/*!
 * @brief 内联函数，N4法二次插值。根据偏移量计算二次线性插值对邻域的权重值，插值参考值来源于样本点
 * 散落的田字格内，即最近的4个像素点。
 * @param offset_x x方向偏移量
 * @param offset_y y方向偏移量
 * @return 返回四个参考权重值
 */
double **PGDClass::calc_N4_QuadraticInterpolation(double offset_x, double offset_y) {

	return nullptr;
}









