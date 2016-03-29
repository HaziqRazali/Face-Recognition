#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include <algorithm> 

#include <iostream>
#include <fstream>

#include "settings.h"
#include "PCA.h"
#include "FaceDetector.h"
#include "omp.h"

using namespace std;
using namespace cv;

// Display
void initializeDisplay();
void CallBackFunc(int event, int x, int y, int flags, void* userdata);
void updateDisplay(Mat& frame, const vector<Rect>& candidate, const vector<string>& candidateName);

// Extra unsused functions
string type2str(int type);

bool viola_called = false;

int main(int argc, const char** argv)
{
	// ============== Initialize ================
	
	//initializeDisplay();										// Display
	//setMouseCallback("Face detection", CallBackFunc, NULL);	// Buttons

	PrincipalComponentsAnalysis PCA("databasetest.txt", 0.95);    // PCA
	FaceDetector Detector("databasetest.txt");				    // Face Detector

	// =============== Begin =====================
	
	// Initialize camera
	capture.open(0);
	capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

	// Merge data
	

	// Begin multi thread
	vector<Rect> combinedCandidateRect;
	vector<string> combinedCandidateName;
	omp_set_num_threads(3);
    #pragma omp parallel shared(combinedCandidateRect, combinedCandidateName)
	{
		Mat frame;

		// Get thread ID
		int TID = omp_get_thread_num();			
		while (true)
		{
			// Read in frame
			capture >> frame;
			flip(frame, frame, 1);

			// Image, Bounding box and name of detected face
			vector<Mat> candidate;
			vector<Rect> candidateRect;
			vector<string> candidateName;
			
			// Run default detector on thread 0
			if (TID == 0)
			{					
				if (true)
				{
					Detector.detect(frame, candidate, candidateRect);	// Detect faces
					PCA.classify(candidate, candidateName);				// Classify faces
				}
			}

			// Rotate images and run detector on remaining threads
			if (TID == 1 || TID == 2)
			{
				if (true)
				{
					Detector.detect(frame, candidate, candidateRect, TID);	// Detect faces
					PCA.classify(candidate, candidateName);					// Classify faces
				}
			}

			// Merge data
			#pragma omp critical
			{
				combinedCandidateRect.insert(combinedCandidateRect.end(), candidateRect.begin(), candidateRect.end());
				combinedCandidateName.insert(combinedCandidateName.end(), candidateName.begin(), candidateName.end());
			}

			// Wait for all threads
			#pragma omp barrier

			// Show result -- Why name change = hang ?
			if (TID == 0)
			{
				cout << combinedCandidateRect.size() << " " << combinedCandidateName.size() << endl;
				updateDisplay(frame, combinedCandidateRect, combinedCandidateName);
				imshow("Result", frame);
				waitKey(1);
			}

			// Wait for all threads
            #pragma omp barrier

			combinedCandidateRect.clear();
			combinedCandidateName.clear();
		}
	}
}

void updateDisplay(Mat& frame, const vector<Rect>& candidate, const vector<string>& candidateName) {

	// Loop through all candidates
	for (int i = 0; i < candidate.size(); i++)
	{
		// Draw bounding box
		rectangle(frame, candidate[i], CV_RGB(0, 0, 0));

		// Display ID
		putText(frame, candidateName[i], candidate[i].tl(), CV_FONT_HERSHEY_SIMPLEX, 2, CV_RGB(255, 0, 0));
	}
}

void initializeDisplay() {

//	display_color = imread("Background.jpg");
//	namedWindow("Face detection", CV_WINDOW_AUTOSIZE);

//	rectangle(display_color, face_detect_button, Scalar(0, 0, 255), -1);
//	rectangle(display_color, exit_button, Scalar(0, 255, 0), -1);
}

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		if (face_detect_button.contains(Point(x, y)))	viola_called = true;
		if (exit_button.contains(Point(x, y)))			exit(0);
	}

	else if (event == EVENT_RBUTTONDOWN)
	{
		cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
	}

	else if (event == EVENT_MBUTTONDOWN)
	{
		cout << "Middle button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
	}

}