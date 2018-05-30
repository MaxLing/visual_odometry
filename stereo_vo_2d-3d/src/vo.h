#ifndef VO_H
#define VO_H

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <vector>
#include <string>
#include <iostream>
using namespace std;
using namespace cv;

// struct Frame{
// 	vector<Point2f> feature2d;
// 	vector<int> landmark_index;
// 	Mat rvec;
// 	Mat tvec;
// }; 


class VO {
public:
	VO(){}
	~VO(){}
	VO(const Mat _K, float offset, const string& _left_dir, const string& _right_dir, int _max_frame, int _opt_frame){
		K = _K; left_dir = _left_dir; right_dir=_right_dir; max_frame = _max_frame; opt_frame = _opt_frame;

		Mat M_left  = Mat::zeros(3,4, CV_64F); M_left.at<double>(0,0) =1; M_left.at<double>(1,1) =1;M_left.at<double>(2,2) =1;
        Mat M_right = Mat::zeros(3,4, CV_64F); M_right.at<double>(0,0) =1; M_right.at<double>(1,1) =1;M_right.at<double>(2,2) =1;
        M_right.at<double>(0,3) = -offset; // camera to world

        P_left  = K*M_left;
        P_right = K*M_right;
	}

	void run(const vector<vector<float>>& truth);

private:
	Mat K;
	Mat P_left;
	Mat P_right;
	string left_dir;
	string right_dir;
	int max_frame;
	int opt_frame;
    
    vector<Point3f> landmark_history;
    vector<Point3f> pose_history;
    // vector<Frame> frame_history;

    Mat load_image(const string& dir, int seq);
	void extract_feature(const Mat& img1, const Mat& img2, vector<Point3f>& landmark_history, vector<Point2f>& feature_points);
	vector<int> track_feature(const Mat& prev_img, const Mat& curr_img, vector<Point2f>& features, vector<Point3f>& landmarks);
    // vector<Point2f> update_frame(int i, const Mat& R, const Mat& T, const vector<Point2f>& features,  
    // 	                         const vector<int>& tracked, const vector<int>& inliers, const Mat& rvec, const Mat& tvec);
    void create_new_features(int i, const Mat& R, const Mat& T, vector<Point2f>& features, vector<Point3f>& landmarks);
    vector<int> remove_duplicate(const vector<Point2f>& old_features, const vector<Point2f>& new_features, const vector<int>& mask, int radius);


};


#endif