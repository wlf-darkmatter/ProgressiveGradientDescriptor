#ifndef PGD_H
#define PGD_H

#include <opencv2/opencv.hpp>

/*!
 * @brief 所有计算PGD的内部函数都整合到了一起方便后面的调试
 *
 */

class PGDClass {
public:
	enum PGD_SampleNums {
		PGD_SampleNums_4 = 4,
		PGD_SampleNums_8 = 8,
		PGD_SampleNums_12 = 12,
		PGD_SampleNums_16 = 16,
		PGD_SampleNums_24 = 24,
		PGD_SampleNums_28 = 28,
		PGD_SampleNums_32 = 32
	};
	/*!
	 * @struct Struct_SampleOffsetList
	 * @brief 存放样本点相对于参考中心偏移量的结构体
	 */
	struct Struct_SampleOffsetList {
	public:
		double* arr_SampleOffsetX;///< 表示第i个元素的x偏移量;
		double* arr_SampleOffsetY;///< 表示第i个元素的y偏移量;
	};
	/*!
	 * @struct Struct_N4InterpList
	 * @brief 继承自Struct_SampleOffsetList，包含有通过N4方法插值的必要列表，避免后续算法中不断重复计算该值
	 * @todo 根据速度的不同，后期可能不再使用arr_InterpOffsetX和arr_InterpOffsetY，因为存在多次寻址运算
	 */
	struct Struct_N4InterpList : Struct_SampleOffsetList {
		explicit Struct_N4InterpList(Struct_SampleOffsetList list);///< 利用父类来初始化该结构体
		double arr_InterpWeight[32][4]{};///<存放权重的数组
		unsigned short arr_InterpOffsetX[32][4]{};///<存放每个采样点插值所需的参考点相对于中心点的X偏移量
		unsigned short arr_InterpOffsetY[32][4]{};///<存放每个采样点插值所需的参考点相对于中心点的Y偏移量
	};
	/*!
	 * @struct Struct_N9InterpList
	 * @brief 继承自Struct_SampleOffsetList，包含有通过N9方法插值的必要列表，避免后续算法中不断重复计算该值
	 */
	struct Struct_N9InterpList : Struct_SampleOffsetList {
		double arr_InterpWeight[32][9];///<存放权重的数组
		unsigned short arr_InterpOffsetX[32][9];///<存放每个采样点插值所需的参考点相对于中心点的X偏移量
		unsigned short arr_InterpOffsetY[32][9];///<存放每个采样点插值所需的参考点相对于中心点的Y偏移量
	};

	static cv::Mat calc_PGDFilter(cv::InputArray _src, cv::OutputArray _dst, float radius, PGD_SampleNums n_sample);


private:
	static cv::Mat def_DstMat(int rows, int cols, PGD_SampleNums n_sample);

	static inline void calc_CircleOffset(Struct_SampleOffsetList &struct_sampleOffset, double radio, int n_sample);

	static inline void
	calc_N4_QuadraticInterpolationInit(Struct_N4InterpList &struct_n4Interp, int n_sample);

	static inline void
	calc_N9_QuadraticInterpolationInit(Struct_N9InterpList &struct_n9Interp, int n_sample);

	static void
	calc_N4PGD_Traverse(const cv::Mat& src,cv::Mat &dst,Struct_N9InterpList struct_n4Interp,int n_sample);
};


#endif
