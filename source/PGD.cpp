#include <PGD.h>


#define PI 3.1415926535897932384626433832795028841971
#define __PGD_DEBUG 0

/*!
 * @brief calc_PGDFilter()函数，根据给定的圆周大小计算n_sample个【环点】的方向不变特征
 * @param _src 输入的矩阵
 * @param _dst 返回的矩阵
 * @param n_sample 计算的【环点】数，一般为4的倍数，由枚举值决定
 * @param radius 【环点】半径大小（浮点数）
 * @param n2_sample 计算的【子环点】个数，一般等于n_sample
 * @param radius_2 【环点】周围的【子环点】计算范围，默认值等于radius
 * @return 返回值是一个矩阵
 */
cv::Mat PGDClass::calc_PGDFilter(const cv::_InputArray &_src,
                                 const cv::_OutputArray &_dst,
                                 PGDClass::PGD_SampleNums n_sample,
                                 double radius,
                                 PGDClass::PGD_SampleNums n2_sample,
                                 double radius_2) {


	//这个是采样时候以中心点为圆心，radius为半径的采样圆的最小外接正四边形框的尺寸
	//采样正四边形矩形框后，还有一个步骤就是对采样圆上的点进行二次采样，二次采样的大小也需要再次指定
	//因此需要对原图像的边缘进行填充，填充的大小由radius和radius_2决定
	if (radius_2 == 0) radius_2 = radius;
	int R = (int) ceil(radius + radius_2);
	int l_size = 1 + 2 * R;
	if (n2_sample == PGD_SampleNums_SameAs_N_Sample) n2_sample = n_sample;

	int rows = _src.rows();
	int cols = _src.cols();
	cv::Mat src_gray;

	///①通道数量转换
	//如果是三通道，使用灰度图像
	if (_src.channels() == 3) {
		cv::cvtColor(_src, src_gray, cv::COLOR_BGR2GRAY, 0);
	} else
		src_gray = _src.getMat();
	///一律使用double类型，同时对边缘进行填充
	cv::Mat src_double;
	src_gray.convertTo(src_double, CV_64FC1);
	src_double = src_double / 255;
	///这里姑且使用边缘复制法，安全起见再多加1个像素点
	cv::copyMakeBorder(src_double, src_double, R + save_copyBorder,
	                   R + save_copyBorder, R + save_copyBorder,
	                   R + save_copyBorder, cv::BORDER_REPLICATE);
	cv::Mat dst = def_DstMat(rows, cols, n_sample);




	/*               ①→
	 *                   ↘
	 *        ④     ⭕️     ②
	 *                      ↓
	 *               ③
	 */
	///②计算样本采样坐标偏移量
	//初始化，计算【环点】坐标
	//返回的是Struct_SampleOffsetList结构体
	Struct_SampleOffsetList struct_sampleOffset(n_sample);
	calc_CircleOffset(struct_sampleOffset, n_sample, radius);

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
	Struct_N4InterpList struct_n4Interp(struct_sampleOffset, n_sample, n2_sample);
	calc_N4_QuadraticInterpolationInit(struct_n4Interp, n_sample, n2_sample, radius_2);

	///④遍历全图
	//这里使用速度稍微快一些的`.ptr<Type>(i)[j]`方法，而且比较安全
	calc_N4PGD_Traverse(src_double, dst, struct_n4Interp, n_sample,
	                    radius, 0,
	                    radius_2);

	_dst.assign(dst);
	return dst;
}

/*!
 * @brief 私有函数，创建输出的空矩阵
 *  @param rows 矩阵的行数
 *  @param cols 矩阵的列数
 *  @param n_sample 采样数
 *  @note 其实可以定义一个n_bit位的数来帮助减少内存的占用量，但是这不符合CPU的运算逻辑，并且进过调研后发现会极大影响运算速度，因此弃用
 */
cv::Mat PGDClass::def_DstMat(int rows, int cols, PGD_SampleNums n_sample) {
	int level_0 = 8 * sizeof(char);
	int level_1 = 8 * sizeof(short);
	int level_2 = 8 * sizeof(int);
	int level_3 = 8 * sizeof(long);
	if (n_sample <= level_0)
		return cv::Mat_<uchar>(rows, cols);
	else if (n_sample <= level_1)
		return cv::Mat_<unsigned short>(rows, cols);
	else if (n_sample <= level_2)
		return cv::Mat_<int>(rows, cols); // 32位有符号（位操作时可以忽略符号位）
	else if (n_sample <= level_3)
		return cv::Mat_<double>(
				rows, cols); //虽然是double，但是读写的时候使用的是64位数的性质
}

/*!
 * @brief 计算在目标区域中邻域的n_sample个采样点相对于中心点的偏移量
 *  @param n_sample 采样点个数，有几个采样点就有几个需要计算的偏移量
 *  @param radius 偏移量半径
 *  @param struct_sampleOffset 类内结构体的**引用**
 * Struct_SampleOffsetList（专门存放每个采样点相对于中心处偏移量的结构体）
 *  @return
 */
void PGDClass::calc_CircleOffset(Struct_SampleOffsetList &struct_sampleOffset, int n_sample, double radius) {
	double theta = 0;
	int quarter = n_sample / 4;
	for (int i = 0; i < n_sample; i++) {

		theta = i * 2 * PI / n_sample;
		struct_sampleOffset.arr_SampleOffsetX[i] = radius * sin(theta); //表示x偏移量
		struct_sampleOffset.arr_SampleOffsetY[i] = -radius * cos(theta); //表示y偏移量
		//调整一下本来就位于x'轴和y'轴上点的坐标，否则会出现非常小的浮点数
		if (i % quarter == 0) {
			switch (i / quarter) {
				case 0:
					struct_sampleOffset.arr_SampleOffsetX[i] = 0;
					break;
				case 1:
					struct_sampleOffset.arr_SampleOffsetY[i] = 0;
					break;
				case 2:
					struct_sampleOffset.arr_SampleOffsetX[i] = 0;
					break;
				case 3:
					struct_sampleOffset.arr_SampleOffsetY[i] = 0;
					break;
			}
		}
	}
}

/*!
 * @brief N4法插值结构体分配函数。
 * @details 根据偏移量计算【子环点】二次线性插值所需的邻域参考点权重值及邻域参考点相对中心点的偏移量，散落的田字格内，即最近的4个像素点。
 * @param struct_n4Interp N4插值法初始信息结构体的引用
 * @param n_sample 【环点】数
 * @param n2_sample 【子环点】数
 * @note theta是【中心点】指向【环点】的矢量方向与 ↑ 构成的角度（↑开始的顺时针方向为正）\n
 * phi是【环点】指向【子环点】的矢量方向与theta构成的角度（theta角开始的顺时针方向为正）
 * @see calc_N4PGD_Traverse() 在函数calc_N4PGD_Traverse中，采样框是 \f$ (1+2\cdot R) \times (1+2\cdot R) \f$ 大小的矩形框，因此这里的偏移量需要调制，但是调制这一步骤放在后面的遍历函数中
 */
void PGDClass::calc_N4_QuadraticInterpolationInit(Struct_N4InterpList &struct_n4Interp, int n_sample, int n2_sample, double radius_2) {
	//有n_sample个【环点】，每个【环点】周围有n2_sample个【子环点】
	short subsample_x_1 = 0, subsample_x_2 = 0, subsample_y_1 = 0, subsample_y_2 = 0;
	double theta = 0, phi = 0;//theta是【中心点】指向【环点】的矢量角度（↑开始的顺时针方向为正）
	double step_theta = 2 * PI / n_sample;
	double step_phi = 2 * PI / n2_sample;
	double x_i = 0, y_i = 0;
	double sub_x_ij = 0, sub_y_ij = 0;
	double dx_1 = 0, dx_2 = 0, dy_1 = 0, dy_2 = 0;
	double offset_xi = 0, offset_yi = 0;
	int quarter = n_sample / 4;

	for (int i = 0; i < n_sample; ++i) {
		x_i = struct_n4Interp.arr_SampleOffsetX[i];//【环点i】相对于【中心点】的偏移量x
		y_i = struct_n4Interp.arr_SampleOffsetY[i];//【环点i】相对于【中心点】的偏移量y
		theta = i * step_theta;

		for (int j = 0; j < n2_sample; ++j) {
			phi = theta + j * step_phi;
			sub_x_ij = x_i + radius_2 * sin(phi + theta);//【子环点i,j】相对于【中心点】的偏移量x
			sub_y_ij = y_i - radius_2 * cos(phi + theta);//【子环点i,j】相对于【中心点】的偏移量y
			///计算子环点附近的四个采样参考点
			subsample_x_1 = (short) floor(sub_x_ij);
			subsample_x_2 = (short) ceil(sub_x_ij);
			subsample_y_1 = (short) floor(sub_y_ij);
			subsample_y_2 = (short) ceil(sub_y_ij);
			/// 如果恰好在x'或y'直线上，那么调制位置，反正计算采样权重的时候其他的都为0，而且填充过了不会有问题

			/// 设置【子环点】周边的四个插值参考点的位置，放入Struct_N4InterpList中的【arr_InterpOffsetX】和【arr_InterpOffsetY】
			struct_n4Interp.arr_InterpOffsetX[i][j][0] = subsample_x_1;//第一个点，左上↖ [1,1]
			struct_n4Interp.arr_InterpOffsetY[i][j][0] = subsample_y_1;//第一个点，左上↖
			struct_n4Interp.arr_InterpOffsetX[i][j][1] = subsample_x_2;//第二个点，右上↗ [2,1]
			struct_n4Interp.arr_InterpOffsetY[i][j][1] = subsample_y_1;//第二个点，右上↗
			struct_n4Interp.arr_InterpOffsetX[i][j][2] = subsample_x_2;//第三个点，右下↘ [2,2]
			struct_n4Interp.arr_InterpOffsetY[i][j][2] = subsample_y_2;//第三个点，右下↘
			struct_n4Interp.arr_InterpOffsetX[i][j][3] = subsample_x_1;//第四个点，左下↙ [1,2]
			struct_n4Interp.arr_InterpOffsetY[i][j][3] = subsample_y_2;//第四个点，左下↙

			///设置【子环点】周边插值参考点的二次插值比重，放入Struct_N4InterpList中的【arr_InterpWeight】
			//      ①  ↑             ②
			//          |dy_1
			//  dx_1←—®——————→dx_2
			//          |
			//          |dy_2
			//      ④  ↓             ③
			dx_1 = sub_x_ij - subsample_x_1;
			dx_2 = subsample_x_2 - sub_x_ij;
			dy_1 = sub_y_ij - subsample_y_1;
			dy_2 = subsample_y_2 - sub_y_ij;
			//优先保持①号地位
			if (subsample_x_1 == subsample_x_2) dx_2 = 1;
			if (subsample_y_1 == subsample_y_2) dy_2 = 1;
			struct_n4Interp.arr_InterpWeight[i][j][0] = dx_2 * dy_2;
			struct_n4Interp.arr_InterpWeight[i][j][1] = dx_1 * dy_2;
			struct_n4Interp.arr_InterpWeight[i][j][2] = dx_1 * dy_1;
			struct_n4Interp.arr_InterpWeight[i][j][3] = dx_2 * dy_1;
#if __PGD_DEBUG
			if (i == 0) {
				std::cout << "当前采样点坐标:";
				std::cout << "(" << subsample_x_1 << "," << subsample_y_1 << ")";
				std::cout << "(" << subsample_x_2 << "," << subsample_y_1 << ")";
				std::cout << "(" << subsample_x_2 << "," << subsample_y_2 << ")";
				std::cout << "(" << subsample_x_1 << "," << subsample_y_2 << ")" << std::endl;
			}
#endif
		}
	}
}

void PGDClass::calc_N9_QuadraticInterpolationInit(PGDClass::Struct_N9InterpList &struct_n9Interp, int n_sample) {}

/*!
 * @brief calc_N4PGD_Traverse 通过N4方法插值遍历全图
 * @param src 输入图像（必须是单通道）
 * @param PGD_Data 输出图像（本质上不是图像，而是二进制矩阵）
 * @param struct_n4Interp 输入的带权重的参数
 * @param n_sample 要获取的样本点数
 * @param r1 采样圆的半径
 * @param r2 邻域样本点周围的LBP计算范围
 * @note
 * 这里采用了指针索引法，速度可能不是很快，但是比at<Type>(x,y)随机读写的速度快
 * @todo 数据类型必须是double类型，之前还存在准备措施不够的情况，需要严加规范。
 */
void PGDClass::calc_N4PGD_Traverse(const cv::Mat &src, cv::Mat &PGD_Data,
                                   Struct_N4InterpList struct_n4Interp,
                                   int n_sample, double r1,
                                   int n2_sample, double r2) {
	//输入的图像一般是拓展过的图像，因此可以直接从初始的（0，0）开始遍历
	int rows = src.rows;
	int cols = src.cols;
	int R = (int) ceil(r1 + r2); // R 是偏移量，[0. R-1]以及[rows-R,rows-1]行都不是，列同理
	int len_win = 1 + 2 * R; //滑框窗口大小
	//这里使用了行指针，因此没有必要检查Mat变量是否连续。
	//并且这里一定是double类型的数据，数据类型在前面需要做好规范措施
	double *row_ptr[1 + 2 * R];
#if __PGD_DEBUG
	double *test_row_ptr[1 + 2 * R];
	cv::Mat test = src.clone();
#endif
	for (int i = R + save_copyBorder; i < rows - R - save_copyBorder; ++i) { //[R,rows-1-R]
		///放置采样的行指针
		for (int t = 0; t < len_win; ++t) row_ptr[t] = (double *) src.ptr(i + t - R);
		//row_ptr[0]是当前行上方R行
		//row_ptr[len_win]是当前行下方R行
#if __PGD_DEBUG
		for (int t = 0; t < len_win; ++t) test_row_ptr[t] = (double *) test.ptr(i + t - R);
#endif
		///遍历当前行，同时提取周边 2*R 个行的信息
		//每一行的列范围是[R , cols -R -1]
		for (int j = R + save_copyBorder; j < cols - R - save_copyBorder; ++j) {

			///这里开始是每一个像素点的运算，由事先建立好的索引值计算
			//row_ptr[0][j-R]是最左上角的像素; row_ptr[R][j]是当前像素
			//当前中心像素的位置是 src.at<double>(i,j)
#if __PGD_DEBUG
			std::cout << "\n当前中心点（绝对坐标-行,列）：" << "(" << i << "," << j << ")" << std::endl;
			std::cout << "————————————————————————" << std::endl;
			short count_main_point = 0;//debug下记录是第几个环点
			test.at<double>(i, j) = 0.5;//当做一次中心点就设置0.5
#endif
			///遍历n_sample个【环点】，计算每一个【环点】的G值
			int64 main_G[n_sample];

			for (int k = 0; k < n_sample; ++k) {
				main_G[k] = 0;
				//计算每个【环点】的 G ,需要获取【子环点】的插值
				//针对不同个数的n2_sample，可以采用不同的长度的变量存放 G 结果，
				//直接使用64位的数作为temp
				int64 temp_G = 0;
				double InterpValue[n2_sample + 1];//记录子环点的插值结果

#if __PGD_DEBUG
				++count_main_point;
				std::cout << "\t当前环点数：" << count_main_point << std::endl;
				short count_second_point = 0;
#endif
				///进行插值
				for (int l = 0; l < n2_sample; ++l) {
					//获取当前【子环点】的4个采样样本相对于【中心点】的偏移量
					short dx1 = struct_n4Interp.arr_InterpOffsetX[k][l][0];
					short dy1 = struct_n4Interp.arr_InterpOffsetY[k][l][0];
					short dx2 = struct_n4Interp.arr_InterpOffsetX[k][l][1];
					short dy2 = struct_n4Interp.arr_InterpOffsetY[k][l][1];
					short dx3 = struct_n4Interp.arr_InterpOffsetX[k][l][2];
					short dy3 = struct_n4Interp.arr_InterpOffsetY[k][l][2];
					short dx4 = struct_n4Interp.arr_InterpOffsetX[k][l][3];
					short dy4 = struct_n4Interp.arr_InterpOffsetY[k][l][3];

					//计算子环点插值，每个子环点有四个采样参考点，
					// 当前行是row_ptr[R]，因此相对偏移行是row_ptr[R + dy]
					// 当前行列像素是row_ptr[R][j]，因此相对偏移行列像素是row_ptr[R + dy][j + dx]
					InterpValue[l] = struct_n4Interp.arr_InterpWeight[k][l][0] * row_ptr[R + dy1][j + dx1]
					                 + struct_n4Interp.arr_InterpWeight[k][l][1] * row_ptr[R + dy2][j + dx2]
					                 + struct_n4Interp.arr_InterpWeight[k][l][2] * row_ptr[R + dy3][j + dx3]
					                 + struct_n4Interp.arr_InterpWeight[k][l][3] * row_ptr[R + dy4][j + dx4];
#if __PGD_DEBUG
					++count_second_point;
					std::cout << "\t\t当前子环点数：" << count_second_point << std::endl;
					//test_row_ptr[t] = (double *) test.ptr(i + t - R);
					test_row_ptr[R + dy1][j + dx1] = 0;//表示绝对坐标-行，列(i + dy1,j + dx1)
					test_row_ptr[R + dy2][j + dx2] = 0;
					test_row_ptr[R + dy3][j + dx3] = 0;
					test_row_ptr[R + dy4][j + dx4] = 0;
					std::cout << "\t\t\t当前处理插值参考点位置（绝对坐标-行,列）：" << std::endl;
					std::cout << "\t\t\t\t(" << i + dy1 << "," << j + dx1 << ")";
					std::cout << "\t\t\t\t(" << i + dy2 << "," << j + dx2 << ")";
					std::cout << "\t\t\t\t(" << i + dy3 << "," << j + dx3 << ")";
					std::cout << "\t\t\t\t(" << i + dy4 << "," << j + dx4 << ")";
					double interp_value;
					double w1 = struct_n4Interp.arr_InterpWeight[k][l][0];
					double w2 = struct_n4Interp.arr_InterpWeight[k][l][1];
					double w3 = struct_n4Interp.arr_InterpWeight[k][l][2];
					double w4 = struct_n4Interp.arr_InterpWeight[k][l][3];
					interp_value = InterpValue[l];
					std::cout << "=====插值结果：" << interp_value << std::endl;
#endif
				}
				///计算当前【环点】的G值
				InterpValue[n2_sample] = InterpValue[0];//调制最后一位，规避if判断是否为最后一位
				for (int l = 0; l < n2_sample; ++l) {
					if (InterpValue[l] > InterpValue[l + 1])
						temp_G |= 1 << l;
				}
			}
		}
	}
}


/*!
 * @brief Struct_SampleOffsetList构造函数
 * @param n_sample 【环点】数
 */
PGDClass::Struct_SampleOffsetList::Struct_SampleOffsetList(int n_sample) {
	this->arr_SampleOffsetX = new double[n_sample];
	this->arr_SampleOffsetY = new double[n_sample];
}

PGDClass::Struct_SampleOffsetList::~Struct_SampleOffsetList() {
	delete[] this->arr_SampleOffsetX;
	delete[] this->arr_SampleOffsetY;
}

/*!
 * @brief 默认构造函数
 */
PGDClass::Struct_SampleOffsetList::Struct_SampleOffsetList() = default;

/*!
 * @brief Struct_N4InterpList构造函数，分配Struct_N4InterpList结构体
 * 结构体Struct_N4InterpList的初始化函数，通过继承上一个实例来获取样本点偏移量属性
 * @param list 父类实例
 */
PGDClass::Struct_N4InterpList::Struct_N4InterpList(Struct_SampleOffsetList list, int n_sample, int n2_sample) : Struct_SampleOffsetList(n_sample) {
	//把父类存放进来
	this->n_sample = n_sample;
	this->n2_sample = n2_sample;
	this->arr_SampleOffsetX = list.arr_SampleOffsetX;
	this->arr_SampleOffsetY = list.arr_SampleOffsetY;
	//根据n_sample的个数以及n2_sample的个数初始化数组
	this->arr_InterpWeight = new double **[n_sample];
	this->arr_InterpOffsetX = new short **[n_sample];
	this->arr_InterpOffsetY = new short **[n_sample];
	for (int i = 0; i < n_sample; ++i) {
		this->arr_InterpWeight[i] = new double *[n2_sample];
		this->arr_InterpOffsetX[i] = new short *[n2_sample];
		this->arr_InterpOffsetY[i] = new short *[n2_sample];
		for (int j = 0; j < n2_sample; ++j) {
			this->arr_InterpWeight[i][j] = new double[4];
			this->arr_InterpOffsetX[i][j] = new short[4];
			this->arr_InterpOffsetY[i][j] = new short[4];
		}
	}
}

PGDClass::Struct_N4InterpList::~Struct_N4InterpList() {
	//不释放基类
	for (int i = 0; i < n_sample; ++i) {
		for (int j = 0; j < n2_sample; ++j) {
			delete[] this->arr_InterpWeight[i][j];
			delete[] this->arr_InterpOffsetX[i][j];
			delete[] this->arr_InterpOffsetY[i][j];
		}
		delete[] this->arr_InterpWeight[i];
		delete[] this->arr_InterpOffsetX[i];
		delete[] this->arr_InterpOffsetY[i];
	}
	delete[] this->arr_InterpWeight;
	delete[] this->arr_InterpOffsetX;
	delete[] this->arr_InterpOffsetY;
}


PGDClass::Struct_N9InterpList::Struct_N9InterpList(const int nSample) : Struct_SampleOffsetList(nSample) {}

PGDClass::Struct_N9InterpList::~Struct_N9InterpList() {}
