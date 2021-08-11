#ifndef PGD_H
#define PGD_H

#include <opencv2/opencv.hpp>

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
	 * @brief 专门存放每个采样点相对于中心处偏移量的结构体
	 * @struct
	 */
	struct Struct_SampleOffsetList {
	public:
		double arr_SampleOffsetX[32];///< 表示第i个元素的x偏移量;
		double arr_SampleOffsetY[32];///< 表示第i个元素的y偏移量;
	};
	struct Struct_N4InterpList : Struct_SampleOffsetList {
		double arr_InterpWeight[32][4];
		double arr_InterpOffsetX[32][4];
		double arr_InterpOffsetY[32][4];
	};
	struct Struct_N9InterpList : Struct_SampleOffsetList {
		double arr_InterpWeight[32][9];
		double arr_InterpOffsetX[32][9];
		double arr_InterpOffsetY[32][9];
	};

	static cv::Mat calc_PGDFilter(cv::InputArray _src, cv::OutputArray _dst, float radio, PGD_SampleNums n_sample);


private:
	static cv::Mat def_DstMat(int rows, int cols, PGD_SampleNums n_sample);

	static Struct_SampleOffsetList calc_CircleOffset(double radio, int n_sample);

	static inline Struct_N4InterpList calc_N4_QuadraticInterpolationInit(double radio, int n_sample);

	static inline Struct_N9InterpList calc_N9_QuadraticInterpolationInit(double radio, int n_sample);
};


#endif
