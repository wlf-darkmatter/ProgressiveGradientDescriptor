#include <opencv2/opencv.hpp>
#include <PGD.h>

#define PI 3.1415926535897932384626433832795028841971

/*!
 * @brief calc_PGDFilter()函数，根据给定的圆周大小计算n个采样点的方向不变特征
 * @param _src 输入的矩阵
 * @param _dst 返回的矩阵
 * @param radius 半径大小（浮点数）
 * @param n_sample 采样点数，一般为4的倍数
 * @return 返回值是一个矩阵，
*/
cv::Mat PGDClass::calc_PGDFilter(cv::InputArray _src, cv::OutputArray _dst, float radius, PGD_SampleNums n_sample) {
	//这个是采样时候以中心点为圆心，radius为半径的采样圆的最小外接正四边形框的尺寸
	int l_size = 1 + 2 * int(ceil(radius));
	//采样正四边形矩形框后，还有一个步骤就是对采样圆上的点进行二次采样，二次采样的
	int rows = _src.rows();
	int cols = _src.cols();
	cv::Mat src_gray;

	///①通道数量转换
	//如果是三通道，使用灰度图像
	if (_src.channels() == 3) {
		cv::cvtColor(_src, src_gray, cv::COLOR_BGR2GRAY, 0);
	} else src_gray = _src.getMat();

	///一律使用double类型，同时对边缘进行填充



	cv::Mat dst = def_DstMat(rows, cols, n_sample);

	/*               ①→
	 *                   ↘
	 *        ④     ⭕️     ②
	 *                      ↓
	 *               ③
	 */
	///②计算样本采样坐标偏移量
	//初始化，计算圆周采样点坐标
	//返回的是Struct_SampleOffsetList结构体
	Struct_SampleOffsetList struct_sampleOffset{};
	struct_sampleOffset.arr_SampleOffsetX = new double[n_sample];
	struct_sampleOffset.arr_SampleOffsetY = new double[n_sample];
	calc_CircleOffset(struct_sampleOffset, radius, n_sample);


	/*                  _____
	 *                ①|🟥🟥|
	 *           ⑧     |🟥②|    ②号采样点位于一个田字格内
	 *         ⑦     ⭕️ ￣￣ ③   偏移坐标为(√2/2,-√2/2 )
	 *           ⑥        ④     因此需要二次插值，为了优化算法，
	 *                ⑤          这里通过列表事先计算好插值的权重值
	 *
	 */
	///③计算每一个采样点的二次插值需要的参考权重（这里是N4方法）
	//初始化，把结果放到一个表里
	//返回的是Struct_N4InterpList
	Struct_N4InterpList struct_n4Interp(struct_sampleOffset);
	calc_N4_QuadraticInterpolationInit(struct_n4Interp, n_sample);

	///④遍历全图
	//这里使用速度稍微快一些的`.ptr<Type>(i)[j]`方法，而且比较安全




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
 *  @param struct_sampleOffset 类内结构体的**引用** Struct_SampleOffsetList（专门存放每个采样点相对于中心处偏移量的结构体）
 *  @return
 */
void PGDClass::calc_CircleOffset(Struct_SampleOffsetList &struct_sampleOffset, double radio, int n_sample) {
	double theta = 0;
	int quarter = n_sample / 4;
	for (int i = 0; i < n_sample; i++) {

		theta = i * 2 * PI / n_sample;
		struct_sampleOffset.arr_SampleOffsetX[i] = radio * sin(theta);//表示x偏移量
		struct_sampleOffset.arr_SampleOffsetY[i] = -radio * cos(theta);//表示y偏移量
		//调整一下本来就位于x'轴和y'轴上点的坐标，否则会出现非常小的浮点数
		if (i % quarter == 0) {
			switch (i / quarter) {
				case 0:
					struct_sampleOffset.arr_SampleOffsetX[i] = 0;
				case 1:
					struct_sampleOffset.arr_SampleOffsetY[i] = 0;
				case 2:
					struct_sampleOffset.arr_SampleOffsetX[i] = 0;
				case 3:
					struct_sampleOffset.arr_SampleOffsetY[i] = 0;
			}
		}
	}
	return;
}


/*!
 * @brief 内联函数，N4法二次插值。根据偏移量计算二次线性插值对邻域的权重值，插值参考值来源于样本点
 * 散落的田字格内，即最近的4个像素点。
 * @param struct_n4Interp N4插值法初始信息结构体的**引用**
 * @param n_sample 样本数
 * @return
 */
void
PGDClass::calc_N4_QuadraticInterpolationInit(Struct_N4InterpList &struct_n4Interp, int n_sample) {
	ushort x_1 = 0, x_2 = 0, y_1 = 0, y_2 = 0;
	double x_i = 0, y_i = 0;
	double dx_1 = 0, dx_2 = 0, dy_1 = 0, dy_2 = 0;
	double offset_xi = 0, offset_yi = 0;
	int quarter = n_sample / 4;
	for (int i = 0; i < n_sample; ++i) {
		x_i = struct_n4Interp.arr_SampleOffsetX[i];
		y_i = struct_n4Interp.arr_SampleOffsetY[i];
		x_1 = (u_short) floor(struct_n4Interp.arr_SampleOffsetX[i]);
		x_2 = (u_short) ceil(struct_n4Interp.arr_SampleOffsetX[i]);
		y_1 = (u_short) floor(struct_n4Interp.arr_SampleOffsetY[i]);
		y_2 = (u_short) ceil(struct_n4Interp.arr_SampleOffsetY[i]);
		/// 如果恰好在x'或y'直线上，那么调制位置，反正计算采样权重的时候其他的都为0
		if (x_1==x_2) x_1=x_2-1;
		if (y_1==y_2) y_1=y_2-1;

		offset_xi = struct_n4Interp.arr_SampleOffsetX[i];
		offset_yi = struct_n4Interp.arr_SampleOffsetY[i];
		/// 设置样本点周边的四个参考点的位置，放入Struct_N4InterpList中的【arr_InterpOffsetX】和【arr_InterpOffsetY】
		//设置第一个点
		struct_n4Interp.arr_InterpOffsetX[i][0] = x_1;
		struct_n4Interp.arr_InterpOffsetY[i][0] = y_1;
		//设置第二个点
		struct_n4Interp.arr_InterpOffsetX[i][1] = x_2;
		struct_n4Interp.arr_InterpOffsetY[i][1] = y_1;
		//设置第三个点
		struct_n4Interp.arr_InterpOffsetX[i][2] = x_2;
		struct_n4Interp.arr_InterpOffsetY[i][2] = y_2;
		//设置第四个点
		struct_n4Interp.arr_InterpOffsetX[i][3] = x_1;
		struct_n4Interp.arr_InterpOffsetY[i][3] = y_2;

		///设置样本周边点的二次插值比重，放入放入Struct_N4InterpList中的【arr_InterpWeight】
		//      ①  ↑             ②
		//          |dy_1
		//  dx_1←—®——————→dx_2
		//          |
		//          |dy_2
		//      ④  ↓             ③
		//
		dx_1 = x_i - x_1;
		dx_2 = x_2 - x_i;
		dy_1 = y_i - y_1;
		dy_2 = y_2 - y_i;
		struct_n4Interp.arr_InterpWeight[i][0] = dx_2 * dy_2;
		struct_n4Interp.arr_InterpWeight[i][1] = dx_1 * dy_2;
		struct_n4Interp.arr_InterpWeight[i][2] = dx_1 * dy_1;
		struct_n4Interp.arr_InterpWeight[i][3] = dx_2 * dy_1;

	}
	return;
}

void
PGDClass::calc_N9_QuadraticInterpolationInit(PGDClass::Struct_N9InterpList &struct_n9Interp, int n_sample) {

}
/*!
 * @brief calc_N4PGD_Traverse 通过N4方法插值遍历全图
 * @param src 输入图像（必须是单通道）
 * @param dst 输出图像（本质上不是图像，而是二进制矩阵）
 * @param struct_n4Interp 输入的带权重的参数
 * @param n_sample 要获取的样本点数
 * @note 这里采用了指针索引法，速度可能不是很快，但是比at<Type>(x,y)随机读写的速度快
 */
void PGDClass::calc_N4PGD_Traverse(const cv::Mat &src, cv::Mat &dst, PGDClass::Struct_N9InterpList struct_n4Interp,
                                   int n_sample) {

}


/*!
 * @brief 结构体Struct_N4InterpList的初始化函数，通过继承上一个实例来获取样本点偏移量属性
 * @param list 父类实例
 */
PGDClass::Struct_N4InterpList::Struct_N4InterpList(PGDClass::Struct_SampleOffsetList list) : Struct_SampleOffsetList() {
	this->arr_SampleOffsetX = list.arr_SampleOffsetX;
	this->arr_SampleOffsetY = list.arr_SampleOffsetY;
}
