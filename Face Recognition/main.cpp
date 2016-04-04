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
Rect detection_button		= Rect(680, 280, 400, 100);
Rect recognition_button		= Rect(680, 400, 400, 100);
Rect exit_button			= Rect(680, 520, 400, 100);
Rect exit_subwindow_button	= Rect(0, 300, 100, 100);

Mat save_button_image = imread("save_button.png");

// State
bool recognition = 0;
bool detection = 0;

// GUI display
Mat display;

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
	capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

	// Global variables
	vector<Rect> combinedCandidateRect;
	vector<string> combinedCandidateName;

	// Initialize number of threads
	omp_set_num_threads(3);

	// ========= Multi Thread Section ===========

	// Generate threads
	#pragma omp parallel shared(combinedCandidateRect, combinedCandidateName)
	{
		// Private variables
		Mat frame;
		PrincipalComponentsAnalysis PCA;
		FaceDetector Detector;

		// Get thread ID
		int TID = omp_get_thread_num();

		// Initialize
		#pragma omp critical
		{
			PCA.initialize("databasetest.txt", 0.95);
			Detector.initialize("databasetest.txt");
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

			// ========================= Recognition ==============================
			if (recognition)
			{			
				// Run default detector on thread 0
				if (TID == 0)
				{	
					// Detect faces
					Detector.detect(frame, candidate, candidateRect, TID);

					// Classify faces
					PCA.classify(candidate, candidateName);					
				}

				// Rotate images and run detector on threads 1 and 2
				if (TID == 1 || TID == 2)
				{
					// Detect faces
					Detector.detect(frame, candidate, candidateRect, TID);

					// Classify faces
					PCA.classify(candidate, candidateName);					
				}

				// Merge results - implicit barrier ?
				#pragma omp critical
				{
					combinedCandidateRect.insert(combinedCandidateRect.end(), candidateRect.begin(), candidateRect.end());
					combinedCandidateName.insert(combinedCandidateName.end(), candidateName.begin(), candidateName.end());
				}
			}

			// ================= Manual Detection on Thread 0 ======================
			if (detection && TID == 0)
			{
				// Detect faces
				Detector.detect(frame, candidate, candidateRect, TID);
				
				int window_count = candidate.size();
				char window_name[40];
				
				// Show faces
				for (int i = 0; i < window_count; i++)
				{
					Mat display;
					sprintf(window_name, "Face %d", window_count);
					cout << candidate[i].size() << endl;
					vconcat(candidate[i], save_button_image, display);
					imshow(window_name, display);
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
				updateDisplay(frame, combinedCandidateRect, combinedCandidateName, display);
				imshow("EE4902 - Face Recognition", display);
				waitKey(1);
			}

			// Wait for all threads
			#pragma omp barrier

			// Clear shared container for next iteration -- Change to TID since single lags ?
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

	// Read in button images
	Mat recognition_button_image	= imread("recognition_button.png");
	Mat exit_button_image			= imread("exit_button.png");
	Mat NTU_logo_image				= imread("NTU_Logo.png");
	Mat detect_button_image			= imread("recognition_button.png");

	// Initialize display
	detect_button_image.copyTo(display(detection_button));
	recognition_button_image.copyTo(display(recognition_button));
	exit_button_image.copyTo(display(exit_button));
	NTU_logo_image.copyTo(display(Rect(15, 20, 300, 100)));

	// Create display
	namedWindow("EE4902 - Face Recognition", CV_WINDOW_AUTOSIZE);
	imshow("EE4902 - Face Recognition", display);

	// Set mouse callback
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
		// Main window
		if (recognition_button.contains(Point(x, y)))	recognition = !recognition;
		if (detection_button.contains(Point(x, y)))	    detection = 1;
		if (exit_button.contains(Point(x, y)))			exit(0);

		// Sub window
	}
}