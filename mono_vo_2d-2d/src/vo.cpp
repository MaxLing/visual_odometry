#include "vo.h"

#define MIN_NUM_FEAT 2000

// global
int SEQ; // sequence id
char* DIR; // dataset directory


void getCalib(double* focal, Point2f& pp){
  char filename[100]; 
  sprintf(filename, "%s/sequences/%02d/calib.txt", DIR, SEQ); 
  ifstream myfile(filename);
  
  if (myfile.is_open()){
      string line; string camera = "P0: ";
      while (getline(myfile, line)){ // find right calib line
        if (line.compare(0, camera.length(), camera)==0) break;
      }

      double entry;
      istringstream ss(line.substr(4)); // remove camera header
      for (int i=0; i<12;i++){
        ss >> entry;
        if (i==0) *focal=entry;
        if (i==2) pp.x=entry;
        if (i==6) pp.y=entry;
      }
  } else {
    cerr << "unable to open file";
    exit(2);
  }
}


void getTruth(vector<double>& scales, vector< vector<double> >& truth){
  char filename[100]; 
  sprintf(filename, "%s/poses/%02d.txt", DIR, SEQ); 
  ifstream myfile(filename);

  double x, y, z;
  double x_prev=0, y_prev=0, z_prev=0;
  double scale;
  vector<double> pose;

  string line; double entry;
  if (myfile.is_open()){
    while(getline(myfile, line)){
      istringstream ss(line);
      for (int i=0; i<12;i++){
        ss >> entry;
        if (i==3) x=entry;
        if (i==7) y=entry;
        if (i==11) z=entry;
      }
      scale = sqrt((x-x_prev)*(x-x_prev) + (y-y_prev)*(y-y_prev) + (z-z_prev)*(z-z_prev));
      scales.push_back(scale);
      
      x_prev=x; y_prev=y; z_prev=z;

      pose.clear();
      pose.push_back(x); pose.push_back(y); pose.push_back(z);
      truth.push_back(pose);
    }
    myfile.close();
  } else {
    cerr << "unable to open file";
    exit(3);
  }
}


int main( int argc, char** argv )	{
  // parse command
  if (argc != 3){
    cerr << "Syntax: " << argv[0] << " [seq id] [dataset dir]\n";
    exit(1);
  }
  SEQ = atoi(argv[1]);
  DIR = argv[2];
  
  // load calibration
  double focal; Point2f pp; getCalib(&focal, pp);
  // load scale
  vector<double> scales; vector< vector<double> > truth; getTruth(scales, truth);

  // init
  Mat prevImage, currImage;
  vector<Point2f> prevFeatures, currFeatures; vector<uchar> status;
  double scale;
  char filename[100];

  //read the first two frames from the dataset
  sprintf(filename, "%s/sequences/%02d/image_0/%06d.png", DIR, SEQ, 0);
  prevImage = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
  sprintf(filename, "%s/sequences/%02d/image_0/%06d.png", DIR, SEQ, 1);
  currImage = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

  if ( !prevImage.data || !currImage.data) { 
    cerr<< "ERR: reading images " << endl; return -1;
  }
   
  // feature detection and tracking
  featureDetection(prevImage, prevFeatures);
  featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);

  // recovering the pose and the essential matrix
  Mat E, R, t, mask;
  E = findEssentialMat(currFeatures, prevFeatures, focal, pp, RANSAC, 0.999, 1.0, mask);
  recoverPose(E, currFeatures, prevFeatures, R, t, focal, pp, mask);
  
  // rotation and tranlation vectors in camera frame (not world frame!!!)
  Mat R_f = R.clone();
  Mat t_f = t.clone();

  // update for iteration
  prevImage = currImage.clone();
  prevFeatures = currFeatures;

  char text[100];
  Point textOrg(50, 950);

  clock_t begin = clock();

  namedWindow( "Monocular", WINDOW_AUTOSIZE );// Create a window for display.
  namedWindow( "Trajectory", WINDOW_AUTOSIZE );// Create a window for display.

  Mat traj = Mat::zeros(1000, 1000, CV_8UC3);

  for(int numFrame=2; ; numFrame++)	{
    sprintf(filename, "%s/sequences/%02d/image_0/%06d.png", DIR, SEQ, numFrame);
    currImage = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
    if (!currImage.data) break; // no frame now
  	
    featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);

    // epipolar geometry
    E = findEssentialMat(currFeatures, prevFeatures, focal, pp, RANSAC, 0.999, 1.0, mask);
    recoverPose(E, currFeatures, prevFeatures, R, t, focal, pp, mask);

  	scale = scales[numFrame];
    if (scale>0.15 && t.at<double>(2)>t.at<double>(0) && t.at<double>(2)>t.at<double>(1)) {
      t_f = t_f + scale*(R_f*t);
      R_f = R_f*R;
    } // else the frame is abandon
    
    // new feature detection is triggered if features drop below a threshold
    if (prevFeatures.size() < MIN_NUM_FEAT) {
      featureDetection(prevImage, prevFeatures);
      featureTracking(prevImage,currImage,prevFeatures,currFeatures, status);
    }

    // update for iteration
    prevImage = currImage.clone();
    prevFeatures = currFeatures;
    
    // plot
    int x_true = truth[numFrame][0]+500;
    int z_true = -truth[numFrame][2]+500;
    circle(traj, Point(x_true, z_true) ,1, CV_RGB(0,255,0), 2);
    int x = int(t_f.at<double>(0)) + 500;
    int z = -int(t_f.at<double>(2)) + 500;
    circle(traj, Point(x, z) ,1, CV_RGB(255,0,0), 2);
    
    // update text
    rectangle( traj, Point(50, 930), Point(950, 960), CV_RGB(0,0,0), CV_FILLED);
    sprintf(text, "x = %02fm y = %02fm z = %02fm frame: %d features: %lu time: %02fs",
            t_f.at<double>(0), t_f.at<double>(1), t_f.at<double>(2), numFrame, currFeatures.size(), double(clock()-begin)/CLOCKS_PER_SEC);
    putText(traj, text, textOrg, FONT_HERSHEY_PLAIN, 1, Scalar::all(255), 1, 8);

    imshow( "Monocular", currImage);
    imshow( "Trajectory", traj);
    waitKey(1);

  }
  
  // save result
  putText(traj, "Feature-based Monocular VO", Point(250, 900), FONT_HERSHEY_PLAIN, 2, Scalar::all(255), 2, 8);
  sprintf(filename, "../result/%02d.png", SEQ);
  imwrite(filename, traj);

  return 0;
}