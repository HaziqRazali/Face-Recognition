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
#include "ml.h"

using namespace cv;
using namespace std;

class Detector {

	// Database
	String face_cascade_name = "lbpcascade_frontalface.xml";
	CascadeClassifier face_cascade;

	//=============================================================================================
	void detect(Mat frame, vector<Mat>& haarFaces, vector<Rect>& haarRect) {

		// Convert to grayscale
		Mat frame_gray;
		Mat color = frame.clone();
		cvtColor(frame, frame_gray, CV_BGR2GRAY);

		// LBP
		vector<Rect> _haarRect;
		face_cascade.detectMultiScale(frame_gray, _haarRect, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

		// Update haarFaces
		for (int i = 0; i < _haarRect.size(); i++)
		{
			_haarRect[i] = Rect(_haarRect[i].tl().x + 0.17*_haarRect[i].width, _haarRect[i].tl().y + 0.17*_haarRect[i].height, 0.66*_haarRect[i].width, 0.7*_haarRect[i].height);

			Rect roi = _haarRect[i] & Rect(0, 0, 640, 480);

			// Convert to Grayscale and resize before pushing
			Mat _haarCandidate = frame_gray(roi);
			equalizeHist(_haarCandidate, _haarCandidate);
			Mat _colorFace = color(roi);

			resize(_haarCandidate, _haarCandidate, Size(92, 112));

			// Push
			haarFaces.push_back(_haarCandidate);
			haarRect.push_back(_haarRect[i]);
		}
		
	}

};