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
void CallBackFunc_mainWindow(int event, int x, int y, int flags, void* userdata);
void CallBackFunc_subWindow(int event, int x, int y, int flags, void* userdata);
void updateDisplay(Mat& frame, const vector<Rect>& candidate, const vector<string>& candidateName, vector<RotatedRect>& rtdCandidate, vector<string>& rtdCandidateName, Mat& display);

// Not in use
int captureWidth = 640;
int captureHeight = 480;

// Main window fake buttons
Rect detection_button		= Rect(680, 280, 400, 100);
Rect recognition_button		= Rect(680, 400, 400, 100);
Rect exit_button			= Rect(680, 520, 400, 100);

// State
bool recognition = 0;
bool detection = 0;
int globalCounter = 0;

// GUI display
Mat display, sub_display;

// Main window names
string sub_window_name = "Detected Face";
string main_window_name = "EE4902 Face Recognition";

//=============================================================================================
int main(int argc, const char** argv)
{
	// ============== Initialize ================

	// Test
	/*PrincipalComponentsAnalysis PCAtest;
	FaceDetector Detectortest;
	PCAtest.initialize("databasetest.txt", 0.95);
	Detectortest.initialize("databasetest.txt");

	vector<Mat> cand;
	vector<string> name;
	Mat testim = imread("C:\\Users\\Haziq\\Documents\\Visual Studio 2013\\Projects\\Face Recognition\\Face Recognition\\DatabaseTest\\101.pgm", CV_LOAD_IMAGE_GRAYSCALE);
	imshow("L", testim);
	waitKey(0);
	cand.push_back(testim);
	PCAtest.classify(cand, name);

	waitKey(0);*/
	
	// Display
	initializeDisplay();

	// Camera
	capture.open(0);
	capture.set(CV_CAP_PROP_FRAME_WIDTH, captureWidth);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, captureHeight);

	// Global variables
	vector<Rect> combinedCandidateRect;
	vector<RotatedRect> combinedRtdCandidateRect;
	vector<string> combinedCandidateName;
	vector<string> combinedRtdCandidateName;

	// Number of threads
	omp_set_num_threads(3);

	// ============= Multi Thread ================

	// Generate threads
	#pragma omp parallel shared(combinedCandidateRect, combinedCandidateName)
	{
		// Private variables
		Mat frame;
		PrincipalComponentsAnalysis PCA;
		FaceDetector Detector;

		// Get thread ID
		int TID = omp_get_thread_num();

		// Initialize PCA and detector class
		#pragma omp critical
		{
			PCA.initialize("test.txt", 0.95);
			Detector.initialize("test.txt", captureWidth, captureHeight);
			cout << "Thread " << TID << " initialized" << endl;
		}
				
		// Wait for all threads
		#pragma omp barrier

		// ========================= Begin ==============================
		while (true)
		{
			// Read in frame
			capture >> frame;
			flip(frame, frame, 1);

			// Image, Bounding box and name of detected face
			vector<Mat> candidate;
			vector<Rect> candidateRect;
			vector<string> candidateName;

			// Image, Bounding box and name of rotated face
			vector<Mat> rtdCandidate;
			vector<RotatedRect> rtdCandidateRect;
			vector<string> rtdCandidateName;

			// ========================= Recognition ==============================

			if (recognition)
			{			

				// Detect faces
				Detector.detect(frame, candidate, candidateRect, rtdCandidate, rtdCandidateRect, TID);

				// Classify faces
				PCA.classify(candidate, candidateName, rtdCandidate, rtdCandidateName);

				// Merge results
				#pragma omp critical
				{
					combinedCandidateRect.insert(combinedCandidateRect.end(), candidateRect.begin(), candidateRect.end());
					combinedCandidateName.insert(combinedCandidateName.end(), candidateName.begin(), candidateName.end());
					combinedRtdCandidateRect.insert(combinedRtdCandidateRect.end(), rtdCandidateRect.begin(), rtdCandidateRect.end());
					combinedRtdCandidateName.insert(combinedRtdCandidateName.end(), rtdCandidateName.begin(), rtdCandidateName.end());
				}
			}

			// ========================= For Training ============================

			if (detection && TID == 0)
			{
				// Detect faces
				Detector.detect(frame, candidate, candidateRect, rtdCandidate, rtdCandidateRect, TID);
				
				// Show only 1 Face
				if (candidate.size() != 0)
				{
					// Initialize display
					namedWindow(sub_window_name, CV_WINDOW_NORMAL);
					setMouseCallback(sub_window_name, CallBackFunc_subWindow);
					sub_display = candidate[0];
					imshow(sub_window_name, sub_display);
					char newFace[40];
					sprintf(newFace, "Face%d.png", globalCounter);
					imwrite(newFace, sub_display);
					globalCounter++;
					destroyWindow(sub_window_name);
				}

				// Disable detector
				detection = 0;
			}

			// =========================== Merge Results ===========================
			
			// Wait for all threads
			#pragma omp barrier

			// Show result
			if (TID == 0)
			{
				updateDisplay(frame, combinedCandidateRect, combinedCandidateName, combinedRtdCandidateRect, combinedRtdCandidateName, display);
				imshow(main_window_name, display);
				waitKey(1);
			}

			// Wait for all threads
			#pragma omp barrier

			// Clear shared container for next iteration -- Change to TID since single lags ?
			#pragma omp single
			{
				combinedCandidateRect.clear();
				combinedRtdCandidateRect.clear();
				combinedCandidateName.clear();
				combinedRtdCandidateName.clear();
			}
		}
	}
}

//=============================================================================================
void initializeDisplay() {

	// Read in background image
	display = imread("Background.png");	

	// Read in images for fake buttons
	Mat recognition_button_image	= imread("recognition_button.png");
	Mat exit_button_image			= imread("exit_button.png");
	Mat NTU_logo_image				= imread("NTU_Logo.png");
	Mat detect_button_image			= imread("recognition_button.png");

	// Position fake buttons
	detect_button_image.copyTo(display(detection_button));
	recognition_button_image.copyTo(display(recognition_button));
	exit_button_image.copyTo(display(exit_button));
	NTU_logo_image.copyTo(display(Rect(15, 20, 300, 100)));

	// Create display
	namedWindow(main_window_name, CV_WINDOW_AUTOSIZE);
	imshow(main_window_name, display);

	// Set mouse callback
	setMouseCallback(main_window_name, CallBackFunc_mainWindow, NULL);
	waitKey(1);
}

//=============================================================================================
void updateDisplay(Mat& frame, const vector<Rect>& candidate, const vector<string>& candidateName, vector<RotatedRect>& rtdCandidate, vector<string>& rtdCandidateName, Mat& display) {

	// Display all candidates
	for (int i = 0; i < candidate.size(); i++)
	{
		// Draw bounding box
		rectangle(frame, candidate[i], CV_RGB(0, 0, 0));

		// Display ID
		putText(frame, candidateName[i], candidate[i].tl(), CV_FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255, 0, 0));
	}

	// Display all candidates
	for (int i = 0; i < rtdCandidate.size(); i++)
	{
		// Get rtdRect coordinates
		Point2f rect_points[4];
		rtdCandidate[i].points(rect_points);

		// Draw rotated bounding box
		for (int j = 0; j < 4; j++)
			line(frame, rect_points[j], rect_points[(j + 1) % 4], CV_RGB(255, 0, 0));
		
		//rtdCandidate[i].
		// Display ID
		putText(frame, rtdCandidateName[i], rect_points[2], CV_FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255, 0, 0));
	}

	// Copy to GUI
	frame.copyTo(display(Rect(15, 140, 640, 480)));
}

//=============================================================================================
void CallBackFunc_mainWindow(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		if (recognition_button.contains(Point(x, y)))	recognition = !recognition;
		if (detection_button.contains(Point(x, y)))	    detection = 1;
		if (exit_button.contains(Point(x, y)))			exit(0);
	}
}

//=============================================================================================
void CallBackFunc_subWindow(int event, int x, int y, int flags, void* input)
{	
	if (event == EVENT_LBUTTONDOWN)
	{
		// Save image
		if (event == EVENT_LBUTTONDOWN)
		{
			char newFace[40];
			sprintf(newFace, "Face%d.png", globalCounter);
			imwrite(newFace, sub_display);
			globalCounter++;
			destroyWindow(sub_window_name);
		}
	}
}