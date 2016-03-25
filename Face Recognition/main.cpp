/*

[✓] Stabilize bounding box
[] Clean
[] Get .exe

Impt
- Reshape: OpenCV traverses from top left to bottom right (so transpose image before reshaping)

Check
-- All Eigenvectors correct

*/

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
#include "time.h"
#include "omp.h"

using namespace std;
using namespace cv;

struct database {

public:

	Mat Image;
	Mat Image_compressed;
	String Name;
	int Label;

	database(Mat _image, String _name) : Image(_image), Name(_name) {}

};

// Display
void initializeDisplay();
void CallBackFunc(int event, int x, int y, int flags, void* userdata);
void updateDisplay(Mat& frame, const vector<Rect>& haarRect, const vector<int>& matchID, const vector<pair<double, double>>& distanceToBestMatch, vector<database> Face); // Optimize

// Detection
void detectFaces(Mat& frame, vector<Mat> &haarFaces, vector<Rect>& haarRect);

// Recognition
void PrincipalComponentsAnalysis(vector<database>& Face, int principalComponents, Mat& eigenFace, Mat& u);
void projectToEigenspace(vector<Mat>& input);
void projectToEigenspace(vector<database>& face);
void getMatches(const vector<database>& Faces, vector<Mat>& haarFaces_compressed, vector<int>& matchID, vector<pair<double, double>>& distanceToBestMatch);
void mahalanobis(Mat& input);

// Extra unsused functions
string type2str(int type);

// Global variables
String face_cascade_name = "lbpcascade_frontalface.xml";
CascadeClassifier face_cascade;
string window_name = "Capture - Face detection";
RNG rng(12345);

Mat eigenFace, Mean, Eigenvalues;
Mat display_color = Mat(Size(1920, 1080), CV_8UC3, Scalar(150, 100, 0));

bool viola_called = false;

int main(int argc, const char** argv)
{
	/*********************************************************************************
									Initialize
	**********************************************************************************/

	// Initialize display
	initializeDisplay();
	setMouseCallback(window_name, CallBackFunc, NULL);

	// Load the cascades
	if (!face_cascade.load(face_cascade_name)) { printf("--(!)Error loading\n"); return -1; };

	// Initialize openmp

	// STORE OFFLINE
	vector<Mat> rotationMatrix;
	rotationMatrix.push_back((Mat_<double>(2, 3) << 0.8660254037844387, 0.4999999999999999, -77.12812921102037, -0.4999999999999999, 0.8660254037844387, 192.1539030917347)); // 30
	rotationMatrix.push_back((Mat_<double>(2, 3) << 0.8660254037844387, -0.4999999999999999, 162.8718707889796,	0.4999999999999999, 0.8660254037844387, -127.8460969082653)); // -45
	rotationMatrix.push_back((Mat_<double>(2, 3) << 6.123233995736766e-017,  1, 0, -1, 6.123233995736766e-017, 0)); // 90
	rotationMatrix.push_back((Mat_<double>(2, 3) << 6.123233995736766e-017, -1, 0,	1, 6.123233995736766e-017, 0)); // -90

	/*********************************************************************************
								Initialize
	**********************************************************************************/

	// Read all in folder
	String folder = "DatabaseTest";
	vector<String> files;
	glob(folder, files);

	// Read in Training Data
	vector<database> Face;

	//string line;
	//ifstream myfile("databasetest.txt");

	//int currentCount = 0;
	//int Label = 0;

	//while (!myfile.eof())
	//{
	//	// read line
	//	getline(myfile, line);

	//	// Make sure its not empty
	//	if (line.empty()) continue;

	//	// Store as name
	//	string name = line.substr(6);

	//	// read size of class
	//	getline(myfile, line);
	//	string size = line.substr(6);
	//	istringstream iss(size);
	//	int classSize;
	//	iss >> classSize;

	//	// Store training data
	//	for (int i = currentCount; i < currentCount + classSize; i++)
	//	{
	//		Mat src = imread(files[i], CV_LOAD_IMAGE_GRAYSCALE);
	//		resize(src, src, Size(92, 112));
	//		src.convertTo(src, CV_32FC1);
	//		Face.push_back(database(src, name));
	//	}

	//	currentCount = currentCount + classSize;
	//	Label++;
	//}

	_PrincipalComponentsAnalysis PCA;
	PCA.train("databasetest.txt",50);
	
	cin.ignore();

	/*********************************************************************************
									Train
	**********************************************************************************/

	// PCA
	//PrincipalComponentsAnalysis(Face, 50, eigenFace, Mean);

	// Project to Eigenspace
	//projectToEigenspace(Face);

	/*********************************************************************************
									BEGIN
	**********************************************************************************/
		
	Mat frame;

	// Initialize camera
	capture.open(0);
	capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

	// Begin multi thread
	omp_set_num_threads(3);
    #pragma omp parallel shared(frame)
	{
		// Get thread ID
		int TID = omp_get_thread_num();			
		while (true)
		{
			vector<Mat> haarFaces;
			vector<Rect> haarRect;
			vector<int> matchID;
			vector<pair<double, double>> distanceToBestMatch;

			// Read in frame
			#pragma omp single
			{
				capture >> frame;
				flip(frame, frame, 1);
			}

			// Run default detector on thread 0
			if (TID == 0)
			{					
				if (true)
				{
					/*vector<Mat> test;
					Mat ttest;
					Face[11].Image.convertTo(ttest, CV_8UC1);
					test.push_back(Face[11].Image);
					imshow("O", ttest);*/
					detectFaces(frame, haarFaces, haarRect);					// Detect faces
					for (int i = 0; i < haarFaces.size(); i++)
					{
						char windname[40];
						sprintf(windname, "haha %d", i);
						namedWindow(windname, CV_WINDOW_NORMAL);
						imshow(windname, haarFaces[i]);
					}

					projectToEigenspace(haarFaces);								    // Project to Eigenspace
					//mahalanobis(test[0]);
					//getMatches(Face, haarFaces, matchID, distanceToBestMatch);	// Get ID of best matches		

				}
			}

			// Rotate images and run detector on remaining threads
			//if (TID == 1 || TID == 2)
			//{
			//	// Rotate input frame
			//	Mat rotated;
			//	warpAffine(frame, rotated, rotationMatrix[TID-1], frame.size(), INTER_CUBIC);

			//	if (true)
			//	{
			//		detectFaces(rotated, haarFaces, haarRect);					// Detect faces
			//		projectToEigenspace(haarFaces);								// Project to Eigenspace
			//		getMatches(Face, haarFaces, matchID, distanceToBestMatch);	// Get ID of best matches
			//	}
			//}
			
			// Wait for all threads
			#pragma omp barrier

			if (TID == 0)
			{
				imshow("L", frame);
				waitKey(1);
			}

			// Draw bounding boxes
			#pragma omp critical
			{
				updateDisplay(frame, haarRect, matchID, distanceToBestMatch, Face);
			}

			//#pragma omp flush(frame)

			// Show result -- Why name change = hang ?
			if (TID == 0)
			{
				//frame.copyTo(display_color(input_video));
				imshow("Result", frame);
				waitKey(1);
			}
		}
	}
}

void PrincipalComponentsAnalysis(vector<database>& Face, int principalComponents, Mat& eigenFace, Mat& mean) {

	// Initialize dimension of column vector
	int faceDimension = Face[0].Image.rows* Face[0].Image.cols;

	// Initialize size of database
	int N = Face.size();

	// Compute mean
	mean = Mat::zeros(Face[0].Image.size(), CV_32FC1);
	for (int i = 0; i < N; i++) mean = mean + Face[i].Image;
	mean = mean / Face.size();

	// Centralize data, reshape to column vector, accumulate matrix D
	Mat D = Mat::zeros(Size(N, faceDimension), CV_32FC1);
	
	// D -- correct means subtraction all correct
	for (int i = 0; i < N; i++)
	{
		Face[i].Image_compressed = Face[i].Image - mean;							// Subtract mean
		Face[i].Image_compressed = Face[i].Image_compressed.t();					// Transpose before reshaping
		Face[i].Image_compressed = Face[i].Image_compressed.reshape(0, 92 * 112);	
		Face[i].Image_compressed.copyTo(D.col(i));									// Fill up Matrix D
	}

	// Compute small covariance matrix
	Mat S = (D.t() * D) / N;
	
	// Decompose - Eigenvectors and values correct
	Mat eigenvalues, s_eigenvectors, eigenvectors;
	eigen(S, eigenvalues, s_eigenvectors); // Eigenvectors stored in rows
	s_eigenvectors = s_eigenvectors.t();   // Transpose so Eigenvectors are stored in columns
	eigenvectors   = D * s_eigenvectors;   // Sign different compared to Matlab

	// Retain top principal components -- Correct
	Mat eigenvectors_retained = eigenvectors(Rect(0, 0, principalComponents, faceDimension)).clone();
	Eigenvalues = eigenvalues(Rect(0, 0, 1, principalComponents));

	// Normalize all eigenvectors -- Correct
	for (int i = 0; i < eigenvectors_retained.cols; i++)
	{
		normalize(eigenvectors_retained.col(i), eigenvectors_retained.col(i), 1.0, 0.0, NORM_L2);
		/*cout << eigenvectors_retained.col(i) << endl;
		Mat temp;
		pow(eigenvectors_retained.col(i), 2, temp);
		cout << sum(temp)(0) << endl;
		waitKey(0);*/
	}
	
	// Project onto Eigenspace
	for (int i = 0; i < N; i++)
	{
		Face[i].Image_compressed = eigenvectors_retained.t() * Face[i].Image_compressed.reshape(0, faceDimension);
	}

	eigenFace = eigenvectors_retained;
}

void projectToEigenspace(vector<Mat>& input) {

	// Convert to 32 bit for operation
	for (int i = 0; i < input.size(); i++)
	{
		if (type2str(input[i].type()).compare("8UC1") == 0)	input[i].convertTo(input[i], CV_32FC1);
		Mat temp_input = input[i].clone();
		temp_input = temp_input - Mean;
		temp_input = temp_input.t();
		input[i] = eigenFace.t() * temp_input.reshape(0, temp_input.rows * temp_input.cols);
	}
}

void projectToEigenspace(vector<database>& face) {

	// Convert to 32 bit for operation
	for (int i = 0; i < face.size(); i++)
	{
		if (type2str(face[i].Image.type()).compare("8UC1") == 0)	face[i].Image.convertTo(face[i].Image, CV_32FC1);

		Mat temp_input = face[i].Image.clone();
		temp_input = temp_input - Mean;
		face[i].Image_compressed = eigenFace.t() * temp_input.reshape(0, temp_input.rows * temp_input.cols);
	}
}

void detectFaces(Mat& frame, vector<Mat> &haarFaces, vector<Rect>& haarRect) {
	
	static int j = 0;

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

void updateDisplay(Mat& frame, const vector<Rect>& haarRect, const vector<int>& matchID, const vector<pair<double, double>>& distanceToBestMatch, vector<database> Face) {

	// Loop through all candidates
	for (int i = 0; i < haarRect.size(); i++)
	{
		// Convert for display
		//Face[matchID[i]].Image.convertTo(Face[matchID[i]].Image, CV_8U);

		// Draw bounding box
		rectangle(frame, haarRect[i], CV_RGB(0, 0, 0));
		//putText(frame, Face[matchID[i]].Name, haarRect[i].tl(), FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255, 0, 0));

		//cout << distanceToBestMatch[i].first << " " << distanceToBestMatch[i].second << endl;
	}

}

void getMatches(const vector<database>& Faces, vector<Mat>& haarFaces_compressed, vector<int>& matchID, vector<pair<double, double>>& distanceToBestMatch) {

	//for (int i = 0; i < haarFaces_compressed.size(); i++)
	//{
	//	mahalanobis(haarFaces_compressed[i]);
	//	cout << endl;
	//}

	int numberOfCandidates = haarFaces_compressed.size();
	int numberOfFaces = Faces.size();
	
	// Loop through Candidates
	for (int i = 0; i < numberOfCandidates; i++)
	{
		vector<float> euclideanDist;

		// Compute distance to every face in database
		for (int j = 0; j < numberOfFaces; j++)
		{
			euclideanDist.push_back(norm(haarFaces_compressed[i] - Faces[j].Image_compressed));
		}

		// Get ID of best match
		matchID.push_back(min_element(euclideanDist.begin(), euclideanDist.end()) - euclideanDist.begin());

		// Get distance to best and 2nd best match
		nth_element(euclideanDist.begin(), euclideanDist.begin() + 2, euclideanDist.end());
		distanceToBestMatch.push_back(make_pair(euclideanDist[0], euclideanDist[1]));
	}

}

void mahalanobis(Mat& input) {

	//cout << input << endl;

	// Square
	Mat squared = input.clone();
	pow(input, 2, squared);

	// Compute distance
	Mat out;
	divide(squared, Eigenvalues, out);

	//cout << out << endl;

	//cout << "maha " << sum(out) << endl;
	//waitKey(0);
}

void initializeDisplay() {

	display_color = imread("Background.jpg");
	namedWindow(window_name, CV_WINDOW_AUTOSIZE);

	rectangle(display_color, face_detect_button, Scalar(0, 0, 255), -1);
	rectangle(display_color, exit_button, Scalar(0, 255, 0), -1);
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

string type2str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}

/*string ty = type2str(frame.type());
printf("Matrix: %s %D%d \n", ty.c_str(), frame.cols, frame.rows);*/