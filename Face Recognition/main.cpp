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

//=============================================================================================

// Display functions
void initializeDisplay();
void CallBackFunc(int event, int x, int y, int flags, void* userdata);
void updateDisplay(Mat& frame, const vector<Rect>& candidate, const vector<string>& candidateName, Mat& display);

// Display buttons
Rect recognition_button = Rect(680, 400, 400, 100);
Rect exit_button		= Rect(680, 520, 400, 100);

// State
bool recognition = 0;

// GUI display
Mat display;

//=============================================================================================
int main(int argc, const char** argv)
{
	// ============== Initialize ================
	
	// Display
	initializeDisplay();

	// Detector and Recognition classes
	PrincipalComponentsAnalysis PCA("databasetest.txt", 0.95);
	FaceDetector Detector("databasetest.txt");

	// Camera
	capture.open(0);
	capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

	// Shared container
	vector<Rect> combinedCandidateRect;
	vector<string> combinedCandidateName;

	// =============== Begin =====================

	// Begin multi threading
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

			// Recognition enabled - barrier ?
			if (recognition)
			{			
				// Run default detector on thread 0
				if (TID == 0)
				{					
					if (true)
					{
						Detector.detect(frame, candidate, candidateRect);	// Detect faces
						PCA.classify(candidate, candidateName);				// Classify faces
					}
				}

				// Rotate images and run detector on threads 1 and 2
				if (TID == 1 || TID == 2)
				{
					if (true)
					{
						Detector.detect(frame, candidate, candidateRect, TID);	// Detect faces
						PCA.classify(candidate, candidateName);					// Classify faces
					}
				}

				// Merge data - does it act as a barrier ?
				#pragma omp critical
				{
					combinedCandidateRect.insert(combinedCandidateRect.end(), candidateRect.begin(), candidateRect.end());
					combinedCandidateName.insert(combinedCandidateName.end(), candidateName.begin(), candidateName.end());
				}
			}

			// Wait for all threads
			#pragma omp barrier

			// Show result
			if (TID == 0)
			{
				updateDisplay(frame, combinedCandidateRect, combinedCandidateName, display);
				imshow("EE4902 - Face Recognition", display);
				waitKey(1);
			}

			// Wait for all threads
			#pragma omp barrier

			// Clear shared container for next iteration
			#pragma omp single
			{
				combinedCandidateRect.clear();
				combinedCandidateName.clear();
			}
		}
	}
}

//=============================================================================================
void initializeDisplay() {

	// Read in background image
	display = imread("Background.png");	

	// Read in buttons
	Mat recognition_button_image	= imread("recognition_button.png");
	Mat exit_button_image			= imread("exit_button.png");
	Mat NTU_logo_image				= imread("NTU_Logo.png");

	// Initialize display
	recognition_button_image.copyTo(display(recognition_button));
	exit_button_image.copyTo(display(exit_button));
	NTU_logo_image.copyTo(display(Rect(15, 20, 300, 100)));

	// Create display
	namedWindow("EE4902 - Face Recognition", CV_WINDOW_AUTOSIZE);
	imshow("EE4902 - Face Recognition", display);
	setMouseCallback("EE4902 - Face Recognition", CallBackFunc, NULL);
	waitKey(1);
}

//=============================================================================================
void updateDisplay(Mat& frame, const vector<Rect>& candidate, const vector<string>& candidateName, Mat& display) {

	// Loop through all candidates
	for (int i = 0; i < candidate.size(); i++)
	{
		// Draw bounding box
		rectangle(frame, candidate[i], CV_RGB(0, 0, 0));

		// Display ID
		putText(frame, candidateName[i], candidate[i].tl(), CV_FONT_HERSHEY_SIMPLEX, 2, CV_RGB(255, 0, 0));
	}

	// Copy to GUI
	frame.copyTo(display(Rect(15, 140, 640, 480)));
}

//=============================================================================================
void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		if (recognition_button.contains(Point(x, y)))	recognition = !recognition;
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