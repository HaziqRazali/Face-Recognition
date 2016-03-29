#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include <algorithm> 

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

class FaceDetector {

public:

	// Database
	String face_cascade_name = "lbpcascade_frontalface.xml";
	CascadeClassifier face_cascade;

	// Rotation Matrix
	vector<Mat> rotationMatrix;
	
	// Capture resolution
	int rows, cols;
	
	//=============================================================================================
	FaceDetector(string configFileName) {

	// Open text file
	string line;
	ifstream myfile(configFileName);

	// Read in base resolution
	getline(myfile, line);
	getline(myfile, line); cols = convertToInt(line);
	getline(myfile, line); rows = convertToInt(line);

	myfile.close();
	
	// Rotation matrix is wrt to capture width and height
	rotationMatrix.push_back((Mat_<double>(2, 3) << 0.8660254037844387, 0.4999999999999999, -77.12812921102037, -0.4999999999999999, 0.8660254037844387, 192.1539030917347)); // 30
	rotationMatrix.push_back((Mat_<double>(2, 3) << 0.8660254037844387, -0.4999999999999999, 162.8718707889796, 0.4999999999999999, 0.8660254037844387, -127.8460969082653)); // -30
	rotationMatrix.push_back((Mat_<double>(2, 3) << 6.123233995736766e-017, 1, 0, -1, 6.123233995736766e-017, 0)); // 90
	rotationMatrix.push_back((Mat_<double>(2, 3) << 6.123233995736766e-017, -1, 0, 1, 6.123233995736766e-017, 0)); // -90
	
	// Load the cascades
	if (!face_cascade.load(face_cascade_name)) { printf("--(!)Error loading\n");};

	}

	//=============================================================================================
	void detect(Mat frame, vector<Mat>& candidate, vector<Rect>& candidateRect, int TID = 0) {
		
		// Rotate image
		if (TID != 0)
		{
			frame = rotateImage(frame, TID);
		}

		// Convert to grayscale
		Mat frame_gray;
		cvtColor(frame, frame_gray, CV_BGR2GRAY);

		// LBP
		vector<Rect> _haarRect;
		face_cascade.detectMultiScale(frame_gray, _haarRect, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

		// Update Faces
		for (int i = 0; i < _haarRect.size(); i++)
		{
			_haarRect[i] = Rect(_haarRect[i].tl().x + 0.17*_haarRect[i].width, _haarRect[i].tl().y + 0.17*_haarRect[i].height, 0.66*_haarRect[i].width, 0.7*_haarRect[i].height);

			Rect roi = _haarRect[i] & Rect(0, 0, 640, 480);

			// Convert to Grayscale and resize before pushing
			Mat _haarCandidate = frame_gray(roi);
			equalizeHist(_haarCandidate, _haarCandidate);

			resize(_haarCandidate, _haarCandidate, Size(cols, rows));

			// Push
			candidate.push_back(_haarCandidate);
			candidateRect.push_back(_haarRect[i]);
		}

		if (TID == 0) cout << candidate.size() << endl;
		
	}

	//=============================================================================================
	Mat rotateImage(Mat& frame, int TID) {

		Mat rotatedFrame;
		warpAffine(frame, rotatedFrame, rotationMatrix[TID - 1], frame.size(), INTER_CUBIC);
		return rotatedFrame;
	}

	//=============================================================================================
	int convertToInt(string text) {

		int result;
		istringstream convert(text);
		convert >> result;
		return result;
	}

};