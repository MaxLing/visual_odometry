#include "vo.h"

void VO::run(const vector<vector<float>>& truth){
	int i = 0;
	vector<Point3f> pose_optimized;

	// init
	Mat left_img = load_image(left_dir, i);
	Mat right_img = load_image(right_dir, i);
	if ( !left_img.data || !right_img.data) { cerr<< "Error reading images\n"; return;}

	vector<Point2f> feature_points;
	vector<Point3f> landmark_set;
	// extract_feature(left_img, right_img, landmark_history, feature_points);
	extract_feature(left_img, right_img, landmark_set, feature_points);

	// Frame frame;
	// frame.feature2d = feature_points;
	// for (int j = 0; j < landmark_history.size(); j++) frame.landmark_index.push_back(j);
	// frame.rvec  = Mat::zeros(3,1,CV_32F);
	// frame.tvec  = Mat::zeros(3,1,CV_32F);
	//    frame_history.push_back(frame);

	// init plot
	char text[100];
	Point text1(50, 950); Point text2(50, 970);
	clock_t begin = clock();
	Mat traj = Mat::zeros(1000, 1000, CV_8UC3);

	Mat prev_img = left_img;
	Mat curr_img; 
	for(i = 1; i<max_frame; i++){
		cout << "Iteration: " << i <<endl;
		curr_img = load_image(left_dir, i);
	    
	//       // read from last left frame
	//       vector<Point3f> landmark_set;
	//       feature_points = frame_history.back().feature2d;
	//       for (auto index : frame_history.back().landmark_index) {
		// 	landmark_set.push_back(landmark_history[index]);
		// }
	    
	    // tracking
	    vector<int> tracked = track_feature(prev_img, curr_img, feature_points, landmark_set);
		if (landmark_set.size()<5) continue; 
	    
	    // PnP
	    Mat dist_coeffs = Mat::zeros(4,1,CV_64F);
	    vector<int> inliers; 
	    Mat tvec, rvec;
	    solvePnPRansac(landmark_set, feature_points, K, dist_coeffs, rvec, tvec, false, 100, 8.0, 0.99, inliers);
	    cout << "inliers: " << inliers.size() << "\tlandmarks: "<< landmark_set.size() << endl;
	    if (inliers.size()<5) continue;
	    // solvePnP(landmark_set, feature_points, K, dist_coeffs, rvec, tvec, false);
	    cout << tvec << endl;
	    
	    // relative pose, note epipolar is reversed
	    Mat R; Rodrigues(rvec, R);
	    R = R.t();
	    Mat T = -R*tvec;

	    cout <<"estm: " << T.t() << endl;
	    cout <<"true: " <<"["<<truth[i][0] << ", " << truth[i][1] << ", " << truth[i][2] <<"]"<<endl;
	    
	    // feature_points = update_frame(i, R, T, feature_points, tracked, inliers, rvec, tvec);
	    create_new_features(i, R, T, feature_points, landmark_set);
	    
	    // update for next iteration
	    prev_img = curr_img.clone();
	    
	    // TODO: add bundle adjustment
	    pose_history.push_back( Point3f(T.at<float>(0), T.at<float>(1), T.at<float>(2)) );

	    // plot
	    Point pose_true = Point(int(truth[i][0])+500, -int(truth[i][2])+500);
	    circle(traj, pose_true ,1, CV_RGB(0,255,0), 2);
	    Point pose_est = Point(int(T.at<float>(0)) + 500, -int(T.at<float>(2)) + 500);
	    circle(traj, pose_est ,1, CV_RGB(255,0,0), 2);
	    // update text
	    rectangle( traj, Point(0, 930), Point(1000, 980), CV_RGB(0,0,0), CV_FILLED);
	    sprintf(text, "x = %02fm y = %02fm z = %02fm", T.at<float>(0), T.at<float>(1), T.at<float>(2));
	    putText(traj, text, text1, FONT_HERSHEY_PLAIN, 1, Scalar::all(255), 1, 8);
	    sprintf(text, "frame: %d features: %lu landmarks: %lu time: %02fs",
	            i, feature_points.size(), landmark_history.size(), double(clock()-begin)/CLOCKS_PER_SEC);
	    putText(traj, text, text2, FONT_HERSHEY_PLAIN, 1, Scalar::all(255), 1, 8);

	    imshow( "Trajectory", traj);
	    waitKey(1);
	}
	// save result
	putText(traj, "Stereo VO with bundle adjustment", Point(250, 900), FONT_HERSHEY_PLAIN, 2, Scalar::all(255), 2, 8);
	imwrite("../result/vo.png", traj);
}

Mat VO::load_image(const string& dir, int seq){
	char filename[100];
	sprintf(filename, dir.c_str(), seq);
	return imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
}

void VO::extract_feature(const Mat& img1, const Mat& img2, vector<Point3f>& landmarks, vector<Point2f>& feature_points){
	Ptr<ORB> orb = ORB::create(300); // matching need tuning
	vector<KeyPoint> keypoints1, keypoints2;
	Mat descriptors1, descriptors2;

	// compute feature and descriptor
	orb->detect(img1, keypoints1);
	orb->detect(img2, keypoints2);
	orb->compute(img1, keypoints1, descriptors1);
	orb->compute(img2, keypoints2, descriptors2);

	// convert for flann
	descriptors1.convertTo(descriptors1, CV_32F);
	descriptors2.convertTo(descriptors2, CV_32F);
	// matching
	FlannBasedMatcher matcher;
	vector<vector<DMatch>> matches;
	matcher.knnMatch(descriptors1, descriptors2, matches, 2);

	vector<Point2f> match_points1;
	vector<Point2f> match_points2;
	// vector<DMatch> bestMatches;

	for (int i = 0; i < matches.size(); i++) {

		const DMatch& bestMatch = matches[i][0];  
		const DMatch& betterMatch = matches[i][1];  

		if (bestMatch.distance < 0.7*betterMatch.distance) { // ratio test
			match_points1.push_back(keypoints1[bestMatch.queryIdx].pt);
	        match_points2.push_back(keypoints2[bestMatch.trainIdx].pt);
	        // bestMatches.push_back(bestMatch);
		}
	}

	// project to 3d
	Mat candidates;
	triangulatePoints(P_left, P_right, match_points1, match_points2, candidates);

	for (int i = 0; i<candidates.cols; i++){
	    Point3f p;
	    p.x = candidates.at<float>(0,i)/candidates.at<float>(3,i);
	    p.y = candidates.at<float>(1,i)/candidates.at<float>(3,i);
	    p.z = candidates.at<float>(2,i)/candidates.at<float>(3,i);
	    // if (p.z<10 || p.z>50) continue;// remove too close or too far 
	    landmarks.push_back(p);
	    feature_points.push_back(match_points1[i]);
	}
	// // drawing the results
	// namedWindow("matches", 1);
	// Mat img_matches;
	// drawMatches(img1, keypoints1, img2, keypoints2, bestMatches, img_matches);
	// imshow("matches", img_matches);
	// waitKey(0);
}

vector<int> VO::track_feature(const Mat& prev_img, const Mat& curr_img, vector<Point2f>& features, vector<Point3f>& landmarks){
	vector<Point2f> nextPts;
	vector<uchar> status;
	vector<float> err;
	vector<int> tracked;

	calcOpticalFlowPyrLK(prev_img, curr_img, features, nextPts, status, err);

	vector<Point3f> old_landmarks = landmarks;
	features.clear();
	landmarks.clear();

	for (int i = 0; i<status.size(); i++){
		if (status[i]==1){
			features.push_back(nextPts[i]);
			landmarks.push_back(old_landmarks[i]);
		}
	}

	return tracked;
}

// vector<Point2f> VO::update_frame(int i, const Mat& R, const Mat& T, const vector<Point2f>& features,  
//     	                         const vector<int>& tracked, const vector<int>& inliers, const Mat& rvec, const Mat& tvec){
// 	vector<Point2f> new_2d; vector<Point3f> new_3d;
// 	create_new_features(i, R, T, new_2d, new_3d);

// 	const vector<int>& prev_index = frame_history.back().landmark_index;
// 	vector<int> new_index= remove_duplicate(features, new_2d, inliers, 5);
// 	cout << "new: " << new_index.size() << ": "<< new_2d.size() <<endl;

// 	vector<int> next_index; vector<Point2f> next_features;

// 	// tracked and inlier feature
// 	for (auto index : inliers) {
// 		next_features.push_back(features[index]);
// 		next_index.push_back(prev_index[tracked[index]]);
// 	}
// 	// new feature
// 	int start = landmark_history.size();
// 	for (auto index : new_index){
// 		landmark_history.push_back(new_3d[index]);
// 		next_features.push_back(new_2d[index]);
// 		next_index.push_back(start++);
// 	}

//     Frame frame;
//     frame.feature2d = next_features;
// 	frame.landmark_index = next_index;
// 	frame.rvec = rvec;
// 	frame.tvec = tvec;
//     frame_history.push_back(frame);

//     return next_features;
// }

void VO::create_new_features(int i, const Mat& R, const Mat& T, vector<Point2f>& features, vector<Point3f>& landmarks){
	if (features.size()!=0){
		features.clear();
		landmarks.clear();
	}
	Mat left_img = load_image(left_dir, i);
	Mat right_img = load_image(right_dir, i);

	vector<Point2f> new_features; vector<Point3f> new_landmarks;
	extract_feature(left_img, right_img, new_landmarks, new_features);

	// cout << R << "\n" << T << endl;

	for (int j = 0; j< new_landmarks.size(); j++){
		const Point3f& pc = new_landmarks[j]; // in camera frame
        
        // cout << "camera frame:" << pc << endl;
        if (pc.z<=0) continue;
		Point3f pw; // in world frame
		pw.x = R.at<double>(0, 0)*pc.x + R.at<double>(0, 1)*pc.y + R.at<double>(0, 2)*pc.z + T.at<double>(0, 3);
	   	pw.y = R.at<double>(1, 0)*pc.x + R.at<double>(1, 1)*pc.y + R.at<double>(1, 2)*pc.z + T.at<double>(1, 3);
	    pw.z = R.at<double>(2, 0)*pc.x + R.at<double>(2, 1)*pc.y + R.at<double>(2, 2)*pc.z + T.at<double>(2, 3);
	    // cout << "world frame:" << pw << endl;
	    
		landmarks.push_back(pw);
		features.push_back(new_features[j]);
	}
}

vector<int> VO::remove_duplicate(const vector<Point2f>& old_features, const vector<Point2f>& new_features, 
                             const vector<int>& mask, int radius){
	vector<int> new_index;
	for (int i=0; i<new_features.size(); i++){
		const Point2f& p2 = new_features[i];
		bool duplicate = false;
		for (auto index : mask) {
			const Point2f& p1 = old_features[index];
			if (norm(p1-p2) < radius) {
				duplicate = true;
				break;
			}
		}
		if(!duplicate) new_index.push_back(i);
	}
	return new_index;
}
