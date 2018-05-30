#include "vo.h"
using namespace std;

// global
int SEQ; // sequence id
char* DIR; // dataset directory


void getCalib(vector<float>& calib){
	char filename[100]; 
	sprintf(filename, "%s/sequences/%02d/calib.txt", DIR, SEQ); 
	ifstream myfile(filename);
  
	if (myfile.is_open()){
		string line; string camera = "P0: ";
		while (getline(myfile, line)){ // find right calib line
			if (line.compare(0, camera.length(), camera)==0) break;
		}

		float entry;
		istringstream ss(line.substr(4)); // remove camera header
		for (int i=0; i<12;i++){
			ss >> entry;
			if (i==0) calib.push_back(entry);
			if (i==2) calib.push_back(entry);
			if (i==6) calib.push_back(entry);
	  	}
	} else {
		cerr << "unable to open file";
		exit(2);
	}
}


void getTruth(vector< vector<float> >& truth){
	char filename[100]; 
	sprintf(filename, "%s/poses/%02d.txt", DIR, SEQ); 
	ifstream myfile(filename);

	float x, y, z;
	float x_prev=0, y_prev=0, z_prev=0;
	vector<float> pose;

	string line; float entry;
	if (myfile.is_open()){
		while(getline(myfile, line)){
			istringstream ss(line);
	  		for (int i=0; i<12;i++){
	    		ss >> entry;
	    		if (i==3) x=entry;
	    		if (i==7) y=entry;
	    		if (i==11) z=entry;
	 		}
	        pose.clear();
	        pose.push_back(x); pose.push_back(y); pose.push_back(z);
	        truth.push_back(pose);

	        x_prev=x; y_prev=y; z_prev=z;
	    }
	    myfile.close();
	} else {
		cerr << "unable to open file";
		exit(3);
	}
}


int main( int argc, char** argv )	{
	// parse command
	if (argc != 5){
		cerr << "Syntax: " << argv[0] << " [seq id] [max frame] [opt frame] [dataset dir]\n";
		exit(1);
	}
	SEQ = atoi(argv[1]);
	int max_frame = atoi(argv[2]);
	int opt_frame = atoi(argv[3]);
	DIR = argv[4];

	// load calibration
	vector<float> calib; getCalib(calib);
	// load truth
	vector< vector<float> > truth; getTruth(truth);

	// init
	char filename[100]; sprintf(filename, "%s/sequences/%02d", DIR, SEQ);
	string left_dir = string(filename) + "/image_0/%06d.png";
	string right_dir = string(filename) + "/image_1/%06d.png";

	cv::Mat K = (cv::Mat_<double>(3, 3) << calib[0], 0, calib[1], 0, calib[0], calib[2], 0, 0, 1);
	float offset = 0.53716; // left right camera x offset, calculated from calib.txt

	VO vo(K, offset, left_dir, right_dir, max_frame, opt_frame);
	vo.run(truth);

    return 0;
}