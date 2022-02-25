
#include <PGD.h>

#define PI 3.1415926535897932384626433832795028841971


int id = 0;


/*!
 * @brief calc_PGDFilter()函数，根据给定的圆周大小计算n_sample个【环点】的方向不变特征
 * @param _src 输入的矩阵
 * @param _struct_dst 算子配置结构体(同时存放输出)
 * @param radius 【环点】半径大小（浮点数）
 * @param n2_sample 计算的【子环点】个数，一般等于n_sample
 * @param radius_2 【环点】周围的【子环点】计算范围，默认值等于radius
 * @return 返回值是一个矩阵
 */
PGDClass_::Struct_PGD PGDClass_::calc_PGDFilter(const cv::_InputArray &_src,
                                                Struct_PGD &_struct_dst,
                                                double radius,
                                                double radius_2) {
	int n_sample = _struct_dst.n_sample;
	int n2_sample = _struct_dst.n2_sample;
	cv::Mat temp_dst = _struct_dst.PGD;
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
	} else src_gray = _src.getMat();
	///一律使用double类型，同时对边缘进行填充
	cv::Mat src_double;
	src_gray.convertTo(src_double, CV_64FC1);
	src_double = src_double / 255;
	///这里姑且使用边缘复制法，安全起见再多加1个像素点
	cv::copyMakeBorder(src_double, src_double, R,
	                   R, R,
	                   R, cv::BORDER_REPLICATE);


	/*               ①→
	 *                   ↘
	 *        ④     ⭕️     ②
	 *                      ↓
	 *               ③
	 */
	///②计算样本采样坐标偏移量
	//初始化，计算【环点】坐标
	//返回的是Struct_SampleOffsetList结构体
	Struct_SampleOffsetList struct_sampleOffset(n_sample, radius);
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
	Struct_N4InterpList struct_n4Interp(std::move(struct_sampleOffset), n2_sample, radius_2);
	calc_N4_QuadraticInterpolationInit(struct_n4Interp);

	///④遍历全图
	//这里使用速度稍微快一些的`.ptr<Type>(i)[j]`方法，而且比较安全
	calc_N4PGD_Traverse(src_double, temp_dst, struct_n4Interp);
	return _struct_dst;
}

/*!
 * @brief calc_PGDFilter44Int()函数
 * @param _src 输入的矩阵 注意，这里进行了进一步优化，将通道转换的步骤移到函数外部了
 * @param _struct_dst 算子配置结构体(同时存放输出)
 * @param radius 【环点】半径大小（整数）
 * @param radius_2 【环点】周围的【子环点】计算范围，默认值等于radius（整数）
 * @return 返回值是一个 cv::Mat 类型的数据
 * @note ① 针对固化参数进行优化的函数 n1和n2都是4！
 * ② radius 和 radius_2 都是整数
 * ③ 必须是使用灰度图像
 */
cv::Mat PGDClass_::calc_PGDFilter44_Int(const cv::_InputArray &_src, Struct_PGD &_struct_dst, int radius, int radius_2) {
	const int n_sample = 4;
	const int n2_sample = 4;
	//这个是采样时候以中心点为圆心，radius为半径的采样圆的最小外接正四边形框的尺寸
	//采样正四边形矩形框后，还有一个步骤就是对采样圆上的点进行二次采样，二次采样的大小也需要再次指定
	//因此需要对原图像的边缘进行填充，填充的大小由radius和radius_2决定
	if (radius_2 == 0) radius_2 = radius;
	int R = radius + radius_2;
	int l_size = 1 + 2 * R;

	int rows = _src.rows();
	int cols = _src.cols();
	cv::Mat temp_dst = _struct_dst.PGD;

	cv::Mat src_double;

	///①通道数量转换 已被忽略，放到函数外面执行

	///这里姑且使用边缘复制法
	cv::copyMakeBorder(_src, src_double, R, R, R, R, cv::BORDER_REPLICATE);

	/*               ①→
	 *                   ↘
	 *        ④     ⭕️     ②
	 *                      ↓
	 *               ③
	 */
	///②计算样本采样坐标偏移量
	//初始化，计算【环点】坐标
	//返回的是Struct_SampleOffsetList结构体
	Struct_SampleOffsetList struct_sampleOffset(n_sample, radius);
	calc_CircleOffset(struct_sampleOffset, n_sample, radius);

	/*                  _____
	 *                ①|🟥🟥|
	 *           ⑧     |🟥②|    ②号采样点位于一个田字格内
	 *         ⑦     ⭕️ ￣￣ ③   偏移坐标为(√2/2,-√2/2 )
	 *           ⑥        ④     因此需要二次插值，为了优化算法，
	 *                ⑤          这里通过列表事先计算好插值的权重值
	 *
	 */
	///③不再计算插值的偏移量（虽然没有消耗多少计算量）
	//初始化，把结果放到一个表里
	//返回的是Struct_N4InterpList
	Struct_N4InterpList struct_n4Interp(std::move(struct_sampleOffset), n2_sample, radius_2);
	calc_N4_QuadraticInterpolationInit(struct_n4Interp);

	///④遍历全图
	//这里使用速度稍微快一些的`.ptr<Type>(i)[j]`方法，而且比较安全
	calc_44IntPGD_Traverse(src_double, temp_dst, struct_n4Interp);
	return src_double;
}

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
	 */
void PGDClass_::calc_N4PGD_Traverse(const cv::Mat &src, cv::Mat &PGD_Data, const Struct_N4InterpList &struct_n4Interp) {
	int n_sample = struct_n4Interp.n_sample;
	int n2_sample = struct_n4Interp.n2_sample;
	void (*ptr_WriteFun)(void *, uint64) = nullptr;
	switch ((int) ceil(log(n2_sample) / log(2))) {
		case 2://4位，直接使用8位 = 1 字节
			ptr_WriteFun = &write_PGD_uint8;
			break;
		case 3:
			ptr_WriteFun = &write_PGD_uint8;
			break;
		case 4:
			ptr_WriteFun = &write_PGD_uint16;
			break;
		case 5:
			ptr_WriteFun = &write_PGD_uint32;
			break;
		case 6:
			ptr_WriteFun = &write_PGD_uint64;
			break;
		default:
			printf("出现异常，没有正确指定的写入函数\n");
			break;
	}


	int channel_size = ceil((float) n2_sample / 8.0f);//每个通道的数据占用的字节数，位数不满8个则取8个位（1字节）
	double r1 = struct_n4Interp.r1;
	double r2 = struct_n4Interp.r2;
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
	int ii = 0;//原始图像的偏移量，ii = i - R
	for (int i = R /*扩充图像的偏移量*/; i < rows - R; ++i) { //[R,rows-1-R]
		///放置采样的行指针
		for (int t = 0; t < len_win; ++t) row_ptr[t] = (double *) src.ptr(ii + t);
		//row_ptr[0]是当前行上方R行
		//row_ptr[len_win]是当前行下方R行

#if __PGD_DEBUG
		for (int t = 0; t < len_win; ++t) test_row_ptr[t] = (double *) test.ptr(i + t - R);
#endif
///遍历当前行，同时提取周边 2*R 个行的信息
//每一行的列范围是[R , cols -R -1]
		int jj = 0;//原始图像的偏移量 jj = j - R
		for (int j = R /*扩充图像的偏移量*/; j < cols - R; ++j) {
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
			int kk = 0;//kk = k * channel_size;
			for (int k = 0; k < n_sample; ++k) {
				kk = k * channel_size;
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
				void *temp_ptr = (PGD_Data.data + PGD_Data.step[0] * ii + PGD_Data.step[1] * jj + kk);
#if __PGD_DEBUG2
				int temp = *reinterpret_cast<uint64 *>(temp_ptr);
				printf("(原始位置)：(%d , %d) 通道 %d, 地址：[%p]", ii, jj, k, temp_ptr);
				std::cout << "= " << (std::bitset<16>) temp << "B " << std::hex << temp;
#endif
				ptr_WriteFun(temp_ptr, temp_G);

#if __PGD_DEBUG2
				printf("  |||写入0x %04x|||", temp_G);
				temp = *reinterpret_cast<uint64 *>(temp_ptr);
				printf(" 再次查看 =");
				std::cout << (std::bitset<16>) temp;
				printf("  |||0x %04x|||\n", temp);
#endif
			}
			++jj;
		}
		++ii;
	}
}

//
//
//

///

//
//

/*!
	 * @brief calc_44IntPGD_Traverse 遍历全图 (不插值)
	 * @param src 输入图像（必须是单通道）
	 * @param PGD_Data 输出图像（本质上不是图像，而是二进制矩阵）
	 * @param struct_n4Interp 输入的带权重的参数
	 * @param r1 采样圆的半径
	 * @param r2 邻域样本点周围的LBP计算范围
	 * @note
	 * 这里采用了指针索引法，速度可能不是很快，但是比at<Type>(x,y)随机读写的速度快
	 */
void PGDClass_::calc_44IntPGD_Traverse(const cv::Mat &src, cv::Mat &PGD_Data, const Struct_N4InterpList &struct_n4Interp) {
	const int n_sample = 4;
	const int n2_sample = 4;

	int channel_size = (int) ceil(n2_sample / 8);//每个通道的数据占用的字节数，位数不满8个则取8个位（1字节）
	int r1 = (int) struct_n4Interp.r1;
	int r2 = (int) struct_n4Interp.r2;
	//输入的图像一般是拓展过的图像，因此可以直接从初始的（0，0）开始遍历
	int rows = src.rows;
	int cols = src.cols;
	int R = r1 + r2; // R 是偏移量，[0. R-1]以及[rows-R,rows-1]行都不是，列同理
	int len_win = 1 + 2 * R; //滑框窗口大小
	//这里使用了行指针，因此没有必要检查Mat变量是否连续。
	//并且这里一定是double类型的数据，数据类型在前面需要做好规范措施
	double *row_ptr[1 + 2 * R];

#if __PGD_DEBUG
	double *test_row_ptr[1 + 2 * R];
	cv::Mat test = src.clone();
#endif
	int ii = 0;//原始图像的偏移量，ii = i - R
	for (int i = R /*扩充图像的偏移量*/; i < rows - R; ++i) { //[R,rows-1-R]
		///放置采样的行指针
		for (int t = 0; t < len_win; ++t) row_ptr[t] = (double *) src.ptr(ii + t);
		//row_ptr[0]是当前行上方R行
		//row_ptr[len_win]是当前行下方R行

#if __PGD_DEBUG
		for (int t = 0; t < len_win; ++t) test_row_ptr[t] = (double *) test.ptr(i + t - R);
#endif
///遍历当前行，同时提取周边 2*R 个行的信息
//每一行的列范围是[R , cols -R -1]
		int jj = 0;//原始图像的偏移量 jj = j - R
		for (int j = R /*扩充图像的偏移量*/; j < cols - R; ++j) {
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
			int kk = 0;//kk = k * channel_size;
			for (int k = 0; k < n_sample; ++k) {
				kk = k;
				main_G[k] = 0;
				//计算每个【环点】的 G ,需要获取【子环点】
				//直接使用64位的数作为temp
				int64 temp_G = 0;

				double localPointValue[n2_sample + 1];//记录子环点的结果

#if __PGD_DEBUG
				++count_main_point;
				std::cout << "\t当前环点数：" << count_main_point << std::endl;
				short count_second_point = 0;
#endif
///不进行插值
				for (int l = 0; l < 4; ++l) {
					//short dx1 = struct_n4Interp.arr_InterpOffsetX[k][l][0];
					//short dy1 = struct_n4Interp.arr_InterpOffsetY[k][l][0];
					//localPointValue[l] = row_ptr[R + dy1][j + dx1];

					short dx = struct_n4Interp.arr_44IntOffsetX[k][l];
					short dy = struct_n4Interp.arr_44IntOffsetY[k][l];

					localPointValue[l] = row_ptr[R + dy][j + dx];
				}

///计算当前【环点】的G值
				localPointValue[n2_sample] = localPointValue[0];//调制最后一位，规避if判断是否为最后一位
				for (int l = 0; l < n2_sample; ++l) {
					if (localPointValue[l] > localPointValue[l + 1])
						temp_G |= 1 << l;
				}
				void *temp_ptr = (PGD_Data.data + PGD_Data.step[0] * ii + PGD_Data.step[1] * jj + kk);
#if __PGD_DEBUG2
				int temp = *reinterpret_cast<uint64 *>(temp_ptr);
				printf("(原始位置)：(%d , %d) 通道 %d, 地址：[%p]", ii, jj, k, temp_ptr);
				std::cout << "= " << (std::bitset<16>) temp << "B " << std::hex << temp;
#endif
				write_PGD_uint8(temp_ptr, temp_G);

#if __PGD_DEBUG2
				printf("  |||写入0x %04x|||", temp_G);
				temp = *reinterpret_cast<uint64 *>(temp_ptr);
				printf(" 再次查看 =");
				std::cout << (std::bitset<16>) temp;
				printf("  |||0x %04x|||\n", temp);
#endif
			}
			++jj;
		}
		++ii;
	}
}

void PGDClass_::write_PGD_uint8(void *ptr, uint64 G) {
	*reinterpret_cast<uint8_t *>(ptr) = (uint8_t) G;
}

void PGDClass_::write_PGD_uint16(void *ptr, uint64 G) {
	*reinterpret_cast<uint16_t *>(ptr) = (uint16_t) G;
}

void PGDClass_::write_PGD_uint32(void *ptr, uint64 G) {
	*reinterpret_cast<uint32_t *>(ptr) = (uint32_t) G;
}

void PGDClass_::write_PGD_uint64(void *ptr, uint64 G) {
	*reinterpret_cast<uint64 *>(ptr) = G;
}

/*!
 * @brief 私有函数，创建输出的空矩阵
 *  @param rows 矩阵的行数
 *  @param cols 矩阵的列数
 *  @param n_sample 【环点数】决定了通道个数
 *  @param n2_sample 【子环点数】 决定了每个通道占用的字节个数
 *  @note 其实可以定义一个n_bit位的数来帮助减少内存的占用量，但是这不符合CPU的运算逻辑，并且进过调研后发现会极大影响运算速度，因此弃用
 */
cv::Mat PGDClass_::def_DstMat(int rows, int cols, PGD_SampleNums n_sample, PGD_SampleNums n2_sample) {
	int level_0 = 8 * sizeof(char);
	int level_1 = 8 * sizeof(short);
	int level_2 = 8 * sizeof(int);
	int level_3 = 8 * sizeof(long);
	int print_B = 0;
	cv::Mat dst;
	//n2_sample决定了每个通道占用的字节个数
	if (n2_sample <= level_0)
		dst = cv::Mat(rows, cols, CV_8UC(n_sample));
	else if (n2_sample <= level_1)
		dst = cv::Mat(rows, cols, CV_16UC(n_sample));
	else if (n2_sample <= level_2)
		dst = cv::Mat(rows, cols, CV_32SC(n_sample)); // 32位有符号（位操作时可以忽略符号位）
	else if (n2_sample <= level_3)
		dst = cv::Mat(rows, cols, CV_64FC(n_sample)); //虽然是double，但是读写的时候使用的是64位数的性质
	printf("——————————————————————————\n");
	printf("①数据的step[0]为 %d————每行占用 %d 字节\n", (int) dst.step[0], (int) dst.step[0]);
	printf("②数据的step[1]为 %d————每个元素占用 %d 字节\n", (int) dst.step[1], (int) dst.step[1]);
	printf("③数据的step[2]为 %d————每个通道占用 %d 位\n", (int) dst.step[2], (int) dst.step[2]);
	printf("④数据单通道位数为 %d 位（非实际占用位数）,实际占用字节数为 %d 字节\n", n2_sample, (int) dst.step[1] / n_sample);
	std::cout << "【综上】，创建了 " << rows << "行, " << cols << " 列 的输出Mat\n"
	          << "含有个 rows × cols = " << rows * cols << " 个元素，\n"
	          << "每个元素有 " << n_sample << " 个通道，每个通道内是 " << n2_sample << " 位数据，实际占用 " << (int) dst.step[1] / n_sample << " 字节" << std::endl;
	double memory_size = (double) dst.step[0] * rows;
	if (memory_size < 1024) {
		std::cout << "数据变量占用内存为： " << memory_size << "  B" << std::endl;
	} else if ((memory_size /= 1024) < 1024) {
		std::cout << "数据变量占用内存为： " << memory_size << " KB" << std::endl;
	} else if ((memory_size /= 1024) < 1024) {
		std::cout << "数据变量占用内存为： " << memory_size << " MB" << std::endl;
	} else if ((memory_size /= 1024) < 1024) {
		std::cout << "数据变量占用内存为： " << memory_size << " GB" << std::endl;
	}


	printf("——————————————————————————\n");
	return dst;

}

/*!
 * @brief 计算在目标区域中邻域的n_sample个采样点相对于中心点的偏移量
 *  @param n_sample 采样点个数，有几个采样点就有几个需要计算的偏移量
 *  @param radius 偏移量半径
 *  @param struct_sampleOffset 类内结构体的**引用**
 * Struct_SampleOffsetList（专门存放每个采样点相对于中心处偏移量的结构体）
 *  @return
 */
void PGDClass_::calc_CircleOffset(Struct_SampleOffsetList &struct_sampleOffset, int n_sample, double radius) {
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
void PGDClass_::calc_N4_QuadraticInterpolationInit(Struct_N4InterpList &struct_n4Interp) {
	int n_sample = struct_n4Interp.n_sample;
	int n2_sample = struct_n4Interp.n2_sample;
	double radius_2 = struct_n4Interp.r2;

	//有n_sample个【环点】，每个【环点】周围有n2_sample个【子环点】
	short subsample_x_1 = 0, subsample_x_2 = 0, subsample_y_1 = 0, subsample_y_2 = 0;
	double theta = 0, phi = 0;//theta是【中心点】指向【环点】的矢量角度（↑开始的顺时针方向为正）
	double step_theta = 2 * PI / n_sample;
	double step_phi = 2 * PI / n2_sample;
	double x_i = 0, y_i = 0;
	double sub_x_ij = 0, sub_y_ij = 0;
	double dx_1 = 0, dx_2 = 0, dy_1 = 0, dy_2 = 0;

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
			double w1 = dx_2 * dy_2;
			double w2 = dx_1 * dy_2;
			double w3 = dx_1 * dy_1;
			double w4 = dx_2 * dy_1;
			if (w1 > 1 || w1 < 0 || w2 > 1 || w2 < 0 || w3 > 1 || w3 < 0 || w4 > 1 || w4 < 0) {
				std::cout << "初始化时，【环点号】:(" << i << "," << j << ") 出现异常" << std::endl;
			}
			if (i == 0) {
				std::cout << "当前采样点坐标:" << std::endl;
				std::cout << "(" << subsample_x_1 << "," << subsample_y_1 << ") 权重：" << w1 << std::endl;
				std::cout << "(" << subsample_x_2 << "," << subsample_y_1 << ") 权重：" << w2 << std::endl;
				std::cout << "(" << subsample_x_2 << "," << subsample_y_2 << ") 权重：" << w3 << std::endl;
				std::cout << "(" << subsample_x_1 << "," << subsample_y_2 << ") 权重：" << w4 << std::endl;
			}
#endif
		}

		//保留给44Int固化参数法的偏移量

	}

	if (n_sample == 4 && n2_sample == 4) {
		for (int i = 0; i < n_sample; ++i) {
			x_i = struct_n4Interp.arr_SampleOffsetX[i];//【环点i】相对于【中心点】的偏移量x
			y_i = struct_n4Interp.arr_SampleOffsetY[i];//【环点i】相对于【中心点】的偏移量y
			theta = i * step_theta;
			for (int j = 0; j < 4; ++j) {
				phi = theta + j * step_phi;
				///计算子环点偏移量
				struct_n4Interp.arr_44IntOffsetX[i][j] = (int) round(x_i + radius_2 * sin(phi + theta));//【子环点i,j】相对于【中心点】的偏移量x
				struct_n4Interp.arr_44IntOffsetY[i][j] = (int) round(y_i - radius_2 * cos(phi + theta));//【子环点i,j】相对于【中心点】的偏移量y
			}
		}
	}

}


/*!
 * @overload
 * @brief Struct_SampleOffsetList构造函数
 * @param _n_sample 【环点】数
 * @param _r1 【环点】半径
 */
PGDClass_::Struct_SampleOffsetList::Struct_SampleOffsetList(int _n_sample, double _r1) {
	id++;
	count = id;
#if __PGD_DEBUG
	std::cout << "Struct_SampleOffsetList被调用了,代号：" << count << std::endl;
#endif
	this->n_sample = _n_sample;
	this->r1 = _r1;
	this->arr_SampleOffsetX = new double[(unsigned long) n_sample];
	this->arr_SampleOffsetY = new double[(unsigned long) n_sample];
}

/*!
 * @overload
 * @brief Struct_SampleOffsetList移动构造函数
 * @param struct_move 要移动的结构体
 */
PGDClass_::Struct_SampleOffsetList::Struct_SampleOffsetList(Struct_SampleOffsetList &&struct_move) {
	id++;
	this->count = id;
#if __PGD_DEBUG
	std::cout << "Struct_SampleOffsetList被移动了,代号：" << this->count << "<---" << struct_move.count << std::endl;
#endif
	this->n_sample = struct_move.n_sample;
	this->r1 = struct_move.r1;
	this->arr_SampleOffsetX = struct_move.arr_SampleOffsetX;
	this->arr_SampleOffsetY = struct_move.arr_SampleOffsetY;
	struct_move.arr_SampleOffsetX = nullptr;
	struct_move.arr_SampleOffsetY = nullptr;
}

/*!
 * @overload
 * @brief Struct_SampleOffsetList复制构造函数
 * @param struct_copy 要复制的结构体
 */
PGDClass_::Struct_SampleOffsetList::Struct_SampleOffsetList(const Struct_SampleOffsetList &struct_copy) {
	id++;
	this->count = id;
#if __PGD_DEBUG
	std::cout << "Struct_SampleOffsetList被复制了，代号：" << this->count << " = " << struct_copy.count << std::endl;
#endif

	this->n_sample = struct_copy.n_sample;
	this->r1 = struct_copy.r1;
	this->arr_SampleOffsetX = new double[(unsigned long) n_sample];
	this->arr_SampleOffsetY = new double[(unsigned long) n_sample];
	for (int i = 0; i < n_sample; ++i) {
		this->arr_SampleOffsetX[i] = struct_copy.arr_SampleOffsetX[i];
		this->arr_SampleOffsetY[i] = struct_copy.arr_SampleOffsetY[i];
	}
}

/*!
 * @overload
 * @brief Struct_SampleOffsetList无参数构造函数
 */
PGDClass_::Struct_SampleOffsetList::Struct_SampleOffsetList() {
	id++;
	count = id;
	std::cout << "正在构造无参数Struct_SampleOffsetList，代号：" << count << std::endl;
};

/*!
 * @overload
 * @brief Struct_SampleOffsetList析构函数
 */
PGDClass_::Struct_SampleOffsetList::~Struct_SampleOffsetList() {
#if __PGD_DEBUG
	std::cout << "正在释放Struct_SampleOffsetList，代号：" << count << std::endl;
#endif

	delete[] this->arr_SampleOffsetX;
	delete[] this->arr_SampleOffsetY;
}


/*!
 * @overload
 * @brief Struct_N4InterpList继承派生构造函数，分配Struct_N4InterpList结构体
 * 结构体Struct_N4InterpList的初始化函数，通过继承上一个实例来获取样本点偏移量属性
 * @param _n2_sample 【子环点】个数
 * @param _r2 【子环点】半径
 */
PGDClass_::Struct_N4InterpList::Struct_N4InterpList(Struct_SampleOffsetList &&struct_base_move,
                                                    int _n2_sample,
                                                    double _r2) :
		Struct_SampleOffsetList(std::move(struct_base_move)) {
	++id;
	count2 = id;
#if __PGD_DEBUG
	std::cout << "正在调用Struct_N4InterpList的继承派生构造函数，代号：" << count2 << "。 <= " << count << " <-- " << struct_base_move.count << "  ↑" << std::endl;
#endif
	this->n2_sample = _n2_sample;
	this->r2 = _r2;

	//根据n_sample的个数以及n2_sample的个数初始化数组
	this->arr_InterpWeight = new double **[(unsigned long) this->n_sample];
	this->arr_InterpOffsetX = new short **[(unsigned long) this->n_sample];
	this->arr_InterpOffsetY = new short **[(unsigned long) this->n_sample];
	for (int i = 0; i < this->n_sample; ++i) {
		this->arr_InterpWeight[i] = new double *[(unsigned long) n2_sample];
		this->arr_InterpOffsetX[i] = new short *[(unsigned long) n2_sample];
		this->arr_InterpOffsetY[i] = new short *[(unsigned long) n2_sample];
		for (int j = 0; j < n2_sample; ++j) {
			this->arr_InterpWeight[i][j] = new double[4];
			this->arr_InterpOffsetX[i][j] = new short[4];
			this->arr_InterpOffsetY[i][j] = new short[4];
		}
	}
}

/*!
 * @brief 析构函数
 */
PGDClass_::Struct_N4InterpList::~Struct_N4InterpList() {
#if __PGD_DEBUG
	std::cout << "正在释放Struct_N4InterpList，代号：" << count2;
	std::cout << "。   该对象中包含的基类代码为：" << this->count << std::endl;
#endif
	//不释放基类
	for (int i = 0; i < this->n_sample; ++i) {
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

/*!
	* @brief Struct_PGD构造函数，创建一个外部接口，适合外部读写PGD结果
	* @param _rows 行数
	* @param _cols 列数
	* @param _n_sample 【环点】个数
	* @param _n2_sample 【子环点】个数
*/
PGDClass_::Struct_PGD::Struct_PGD(int _rows, int _cols, PGD_SampleNums _n_sample, PGD_SampleNums _n2_sample) {
	n_sample = _n_sample;
	n2_sample = _n2_sample;
	PGD = def_DstMat(_rows, _cols, _n_sample, _n2_sample);
	rows = _rows;
	cols = _cols;
	step_0 = PGD.step[0];
	step_1 = PGD.step[1];
}



