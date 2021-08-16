#ifndef PGD_H
#define PGD_H

#include <opencv2/opencv.hpp>

/// @file  PGD.h
/// @brief 定义了PGD算子（实验性）
///
///
/// @version 1.1
/// @author 王凌枫
/// @date
///




/*!
 * @brief 所有计算PGD的内部函数都整合到了一起方便后面的调试和改进。\n
 * @note 遍历所有像素，当前像素记作【中心点】，周边有n_sample个采样点，记作【环点】\n
 * @note 每个【环点】周边有 n_2sample 个采样点，记作【子环点】\n
 * @note 每个子环电都是由二次线性插值来计算的，通过获取周边的田字格内四或九个像素值来插值计算，\n
 * @note 为了优化运算，事先计算子环点的四个或九个插值参考点相对于中心点的偏移量以及插值权重
 */
class PGDClass {
public:

	static const int save_copyBorder = 1;//安全起见对图像对边缘多加的一个保护壳
	enum PGD_SampleNums {
		PGD_SampleNums_SameAs_N_Sample = 0,///< 设置n2_sample的个数，默认与n_sample相等
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
	 * @brief 存放采样点相对于参考中心偏移量的结构体，由于只需要比较采样点周围邻域的最大相关排列，
	 * 因此不需要对采样点本身进行二次线性插值
	 * @details 每一组都有n_sample个元素
	 */
	struct Struct_SampleOffsetList {
		explicit Struct_SampleOffsetList(int n_sample);///<创建一个记录具有n_sample个【环点】的样本结构体
		Struct_SampleOffsetList();///<缺省构造函数，什么也不做
		~Struct_SampleOffsetList();

		int n_sample = 0;
		double *arr_SampleOffsetX{};///< double类型指针，记录第i个【环点】的**x**偏移量;
		double *arr_SampleOffsetY{};///< double类型指针，记录第i个【环点】的**y**偏移量;
	};

	/*!
	 * @struct Struct_N4InterpList
	 * @brief 继承自Struct_SampleOffsetList，包含有通过N4方法插值的必要列表，避免后续算法中不断重复计算该值
	 * @note 有n_sample个【环点】，每个环点有n2_sample个【子环点】，每个【子环点】有4个插值参考点
	 * @todo 根据速度的不同，后期可能不再使用arr_InterpOffsetX和arr_InterpOffsetY，因为存在多次寻址运算
	 * arr_InterpWeight
	 */
	struct Struct_N4InterpList : Struct_SampleOffsetList {
		explicit Struct_N4InterpList(Struct_SampleOffsetList list, int n_sample, int n2_sample);///< 利用父类来初始化该结构体
		~Struct_N4InterpList();///<析构函数
		
		int n2_sample;
		double ***arr_InterpWeight;///<存放权重的指针，指向[n_sample][n2_sample][4]的三维数组
		short ***arr_InterpOffsetX;///<存放每个采样点插值所需的参考点相对于中心点的X偏移量
		short ***arr_InterpOffsetY;///<存放每个采样点插值所需的参考点相对于中心点的Y偏移量
	};

	/*!
	 * @struct Struct_N9InterpList
	 * @brief 继承自Struct_SampleOffsetList，包含有通过N9方法插值的必要列表，避免后续算法中不断重复计算该值
	 */
	struct Struct_N9InterpList : Struct_SampleOffsetList {
		explicit Struct_N9InterpList(int nSample);

		~Struct_N9InterpList();

		double arr_InterpWeight[32][9];///<存放权重的数组
		short arr_InterpOffsetX[32][9];///<存放每个采样点插值所需的参考点相对于中心点的X偏移量
		short arr_InterpOffsetY[32][9];///<存放每个采样点插值所需的参考点相对于中心点的Y偏移量
	};


	static cv::Mat
	calc_PGDFilter(const cv::_InputArray &_src,
	               const cv::_OutputArray &_dst,
	               PGD_SampleNums n_sample,
	               double radius,
	               PGD_SampleNums n2_sample = PGD_SampleNums_SameAs_N_Sample,
	               double radius_2 = 0);


private:
	static cv::Mat def_DstMat(int rows, int cols, PGD_SampleNums n_sample);

	static inline void calc_CircleOffset(Struct_SampleOffsetList &struct_sampleOffset, int n_sample, double radius);

	static void
	calc_N4_QuadraticInterpolationInit(Struct_N4InterpList &struct_n4Interp, int n_sample, int n2_sample, double radius_2);

	static inline void
	calc_N9_QuadraticInterpolationInit(Struct_N9InterpList &struct_n9Interp, int n_sample);

	static void
	calc_N4PGD_Traverse(const cv::Mat &src, cv::Mat &PGD_Data,
	                    Struct_N4InterpList struct_n4Interp,
	                    int n_sample, double r1,
	                    int n2_sample, double r2);


};


#endif
