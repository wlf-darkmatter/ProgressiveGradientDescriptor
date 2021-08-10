#include <opencv2/opencv.hpp>
#include <PGD.h>


//!


cv::Mat calc_PGDFilter(cv::Mat &src, int radio) {
    int size = 1 + 2 * ceil(radio);
    if (src.channels() == 3) {
        cv::Mat *src_gray= new cv::Mat;
        
    }
}






