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
	vector<Mat> rotationMatrix, invRotMtrx;
	
	// Face resolution
	int rows, cols;
	int captureWidth, captureHeight;
	
	//=============================================================================================
	void initialize(string configFileName, int _captureWidth, int _captureHeight) {

	captureWidth  = _captureWidth;
	captureHeight = _captureHeight;

	// Open text file
	string line;
	ifstream myfile(configFileName);

	// Read in base resolution
	getline(myfile, line);
	getline(myfile, line); cols = convertToInt(line);
	getline(myfile, line); rows = convertToInt(line);

	myfile.close();

	// Generate rotation matrices
	rotationMatrix.push_back(getRotationMatrix2D(Point2f(_captureWidth / 2, _captureHeight / 2), 30, 1));
	rotationMatrix.push_back(getRotationMatrix2D(Point2f(_captureWidth / 2, _captureHeight / 2), -30, 1));
	rotationMatrix.push_back(getRotationMatrix2D(Point2f(_captureWidth / 2, _captureHeight / 2), 60, 1));
	rotationMatrix.push_back(getRotationMatrix2D(Point2f(_captureWidth / 2, _captureHeight / 2), -60, 1));

	// Generate rotation matrices
	invRotMtrx.push_back(getRotationMatrix2D(Point2f(_captureWidth / 2, _captureHeight / 2), -30, 1));
	invRotMtrx.push_back(getRotationMatrix2D(Point2f(_captureWidth / 2, _captureHeight / 2), 30, 1));
	invRotMtrx.push_back(getRotationMatrix2D(Point2f(_captureWidth / 2, _captureHeight / 2), -60, 1));
	invRotMtrx.push_back(getRotationMatrix2D(Point2f(_captureWidth / 2, _captureHeight / 2), 60, 1));
	
	// Load the cascades
	if (!face_cascade.load(face_cascade_name)) { printf("--(!)Error loading\n");};

	}

	//=============================================================================================
	void detect(Mat frame, vector<Mat>& candidate, vector<Rect>& candidateRect, vector<Mat>& rtdCandidate, vector<RotatedRect>& rtdcandidateRect, int TID) {
				
		// Rotate image
		if (TID != 0)
		{
			frame = rotateImage(frame, TID);

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
				rtdCandidate.push_back(_haarCandidate);
				rtdcandidateRect.push_back(getRotatedRect(_haarRect[i],TID));
			}
		}

		// Standard detector
		else
		{
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
		}
	}

	//=============================================================================================
	Mat rotateImage(Mat& frame, int TID) {

		Mat rotatedFrame;
		warpAffine(frame, rotatedFrame, rotationMatrix[TID - 1], frame.size(), INTER_CUBIC);
		return rotatedFrame;
	}

	//=============================================================================================
	RotatedRect getRotatedRect(Rect& rect, int TID) {

		// Initialize rotated rect corners
		vector<Point2f> corners;
		corners.push_back(rect.tl());
		corners.push_back(rect.br());
		corners.push_back(rect.tl() + Point(0, rect.height));
		corners.push_back(rect.tl() + Point(rect.width, 0));

		// Fill up Mat
		Mat coordinates = (Mat_<double>(3, 4) << corners[0].x, corners[1].x, corners[2].x, corners[3].x,
												 corners[0].y, corners[1].y, corners[2].y, corners[3].y,
														    1,            1,            1,            1);

		// Get rotation Matrix -- Optimize
		Mat r = getRotationMatrix2D(Point(captureWidth / 2, captureHeight / 2), -30, 1.0);

		// Rotate
		Mat result = invRotMtrx[TID-1] * coordinates;

		// Assign new coordinates
		corners[0].x = (int)result.at<double>(0, 0);
		corners[0].y = (int)result.at<double>(1, 0);

		corners[1].x = (int)result.at<double>(0, 1);
		corners[1].y = (int)result.at<double>(1, 1);

		corners[2].x = (int)result.at<double>(0, 2);
		corners[2].y = (int)result.at<double>(1, 2);

		corners[3].x = (int)result.at<double>(0, 3);
		corners[3].y = (int)result.at<double>(1, 3);

		RotatedRect rotRect = minAreaRect(Mat(corners));
		
		return rotRect;
	}

	//=============================================================================================
	int convertToInt(string text) {

		int result;
		istringstream convert(text);
		convert >> result;
		return result;
	}

};