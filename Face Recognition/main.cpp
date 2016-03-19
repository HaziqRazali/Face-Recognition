/*

[✓] Face Detector
[✓]	Principal Components Analysis
[] Use color histogram to eliminate skin matches due to ZNCC
[] T Matching to eliminate region
[] Enable openmp
[] NCC vs NNC distance plot
[] Scatter Matrix
[] Integrate Everything

[] Stabilize/Quantize Bounding box

Others
Wrapper Function to project data onto Eigenspace
Wrapper Function to convert file to 32FC1
Wrapper Function for imshow
Standardize (row,col) of images

*/

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include <algorithm> 

#include <iostream>
#include <fstream>

#include "Settings.h"
#include "time.h"

using namespace std;
using namespace cv;

#define AUTOMATIC
#define MANUAL

//[DebuggerDisplay("{Class}")]
struct database {

public:

	Mat Image;
	Mat Image_compressed;
	String Name;

	database(Mat _image, String _name) : Image(_image), Name(_name) {}

};

// Display functions
void initializeDisplay();
void CallBackFunc(int event, int x, int y, int flags, void* userdata);

// Recognition functions
void PrincipalComponentsAnalysis(vector<database>& Face, int principalComponents, Mat& eigenFace, Mat& meanFace);
void projectToEigenspace(Mat input, Mat& output);
bool myfunction(int i, int j);

// Detection functions
void detectFaces(Mat frame, vector<Mat> &haarFaces, vector<Rect>& haarRect);

void detectRotatedFaces(Mat frame, vector<Rect> haarRect, vector<Mat> &rotatedFaces);
RotatedRect formRotatedRect(Mat left, Mat right, Mat frame);
Mat cropRotatedRect(RotatedRect boundingRect, Mat src);
void visit(int i, vector<vector<Point>> contours, vector<int>& group);
RotatedRect formRotatedRect2(vector<vector<Point>> contour);
void updateDisplay(Mat& frame, vector<Rect> haarRect, vector<int> matchID, vector<pair<double,double>> distanceToBestMatch, vector<database> Face);
void getMatches(vector<database> Faces, vector<Mat> haarFaces_compressed, vector<int>& matchID, vector<pair<double,double>>& distanceToBestMatch);

void processContours(vector<vector<Point>> contours, vector<Point>& matches, vector<vector<Point>>& test);
void updateMatches(vector<Point>& matches, vector<Rect> haarRect = vector<Rect>(), vector<Rect> rotatedRect = vector<Rect>());

// Extra unsused functions
string type2str(int type);

// Global variables
String face_cascade_name = "lbpcascade_frontalface.xml";
CascadeClassifier face_cascade;
string window_name = "Capture - Face detection";
RNG rng(12345);

Mat eigenFace, meanFace;
Mat display_color = Mat(Size(1920, 1080), CV_8UC3, Scalar(150,100,0));

bool viola_called = false;

Rect boundary = Rect(0, 0, 601, 453); // 601 453

vector<Mat> tmplates;
vector<Point> offsets;

int main(int argc, const char** argv)
{
	/*********************************************************************************
								INITIALIZE DISPLAY
	**********************************************************************************/

	// Create display window
	initializeDisplay();

	// Attach mouse call back to display
	setMouseCallback(window_name, CallBackFunc, NULL);

	/*********************************************************************************
								INITIALIZE DATABASE
	**********************************************************************************/

	// Read all in folder
	String folder = "Database"; 
	vector<String> files;
	glob(folder, files);

	// Read in Training Data
	vector<database> Face;

	string line;
	ifstream myfile("database.txt");

	int currentCount = 0;

	while (!myfile.eof())
	{
		// read line
		getline(myfile, line);

		// Make sure its not empty
		if (line.empty()) continue;

		// Store as name
		string name = line.substr(6);

		// read size of class
		getline(myfile, line);
		string size = line.substr(6);
		istringstream iss(size);
		int classSize;
		iss >> classSize;

		// Store training data
		for (int i = currentCount; i < currentCount + classSize; ++i)
		{
			Mat src = imread(files[i], CV_LOAD_IMAGE_GRAYSCALE);
			resize(src, src, Size(92, 112));
			src.convertTo(src, CV_32FC1);
			Face.push_back(database(src, name));
		}
		
		currentCount = currentCount + classSize;
	}

	/*********************************************************************************
										TRAIN
	**********************************************************************************/
	
	// PCA
	PrincipalComponentsAnalysis(Face, 50, eigenFace, meanFace);

	// Project training data onto Eigenspace
	for (int i = 0; i < Face.size(); i++) projectToEigenspace(Face[i].Image, Face[i].Image_compressed);
	
	/*********************************************************************************
	
										BEGIN
	
	**********************************************************************************/

	// Video frame
	Mat frame;

	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name)){ printf("--(!)Error loading\n"); return -1; };

	//-- 2. Read the video stream
	if (capture.open(0))
	{
		// Set capture width and height
		capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
		capture.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

		while (true)
		{
			// read in frame and flip
			capture >> frame;

			flip(frame, frame, 1);
			
			if (!frame.empty())
			{
				// Detect Faces
				if (viola_called) 
				{	
					/********************************************************
					
											DETECTION

					*********************************************************/

					// Detect haar faces
					vector<Mat> haarFaces;
					vector<Rect> haarRect;
					detectFaces(frame, haarFaces, haarRect);
					
					// Detect rotated faces 
					/*vector<Mat> rotatedFaces;
					vector<Rect> rotatedRect;
					detectRotatedFaces(frame, haarRect, rotatedFaces);*/

					/********************************************************

											RECOGNITION

					*********************************************************/
					
					int numberOfCandidates = haarFaces.size();

					// Project to Eigenspace
					vector<Mat> haarFaces_compressed(numberOfCandidates);
					for (int i = 0; i < numberOfCandidates; i++)
						projectToEigenspace(haarFaces[i], haarFaces_compressed[i]);

					// Get ID of best matches
					vector<int> matchID;
					vector<pair<double,double>> distanceToBestMatch;
					getMatches(Face, haarFaces_compressed, matchID, distanceToBestMatch);

					// Draw result on frame
					updateDisplay(frame, haarRect, matchID, distanceToBestMatch, Face);
					viola_called = true;
				}

				// Display video stream
				frame.copyTo(display_color(input_video));
				imshow(window_name, display_color);
				waitKey(1);
			}

			else
			{
				printf(" --(!) No captured frame -- Break!"); break;
			}

		}
	}

	else
	{
		std::cerr << "ERROR: Could not open camera" << std::endl;
		return 1;
	}

	return 0;
}

RotatedRect formRotatedRect(Point lefteye, Point righteye, Mat src) {

	// 4 corners of the rect
	Point topLeft, topRight, bottomLeft, bottomRight;

	float theta = atan(((float)(righteye.y - lefteye.y) / (righteye.x - lefteye.x)));
	float pd	= norm(lefteye - righteye);

	// Compute the 4 corners
	topLeft.x = lefteye.x - 1 * pd*cos(theta) + 1 * pd*sin(theta);
	topLeft.y = lefteye.y - 1 * pd*sin(theta) - 1 * pd*cos(theta);

	topRight.x = righteye.x + 1 * pd*cos(theta) + 1 * pd*sin(theta);
	topRight.y = righteye.y + 1 * pd*sin(theta) - 1 * pd*cos(theta);

	bottomLeft.x = lefteye.x - pd * cos(theta) - 1.5 * pd * sin(theta);
	bottomLeft.y = lefteye.y - pd * sin(theta) + 1.5 * pd * cos(theta);

	bottomRight.x = righteye.x + pd*cos(theta) - 1.5 * pd * sin(theta);
	bottomRight.y = righteye.y + pd*sin(theta) + 1.5 * pd * cos(theta);

	// Push
	vector<Point> rectCorners;
	rectCorners.push_back(topLeft);
	rectCorners.push_back(topRight);
	rectCorners.push_back(bottomLeft);
	rectCorners.push_back(bottomRight);

	// Generate the rotated rectangle
	RotatedRect boundingRect;
	boundingRect = minAreaRect(Mat(rectCorners));

	return boundingRect;
}

void updateMatches(vector<Point>& matches, vector<Rect> haarRect, vector<Rect> rotatedRect) {

	if (haarRect.size() != 0)
	for (auto it = matches.begin(); it != matches.end();)
	{
		bool erase = 0;

		// Matches cannot be enclosed by haarRect ( already detected faces )
		for (int j = 0; j < haarRect.size(); j++) if (haarRect[j].contains(*it)) erase = 1;

		if (erase) it = matches.erase(it);
		else	   it++;
	}

	if (rotatedRect.size() != 0)
	for (auto it = matches.begin(); it != matches.end();)
	{
		bool erase = 0;

		// Matches cannot be enclosed by rotatedRect ( newly detected faces )
		for (int k = 0; k < rotatedRect.size(); k++) if (rotatedRect[k].contains(*it)) erase = 1;
		
		if (erase)	it = matches.erase(it);
		else		it++;		
	}
}

void processContours(vector<vector<Point>> contours, vector<Point>& matches, vector<vector<Point>>& test) {

	for (int i = 0; i < contours.size(); i++)
	{
		// Get parameters
		float perimeter = arcLength(contours[i], false);
		float area = contourArea(contours[i], false);
		float roundness = pow(perimeter, 2) / (4 * 3.142 * area);

		Moments mu = moments(contours[i]);
		Point center = Point(mu.m10 / mu.m00, mu.m01 / mu.m00);

		// Push if parameters are reasonable
		if (area > 50)
		{
			matches.push_back(center);
			test.push_back(contours[i]);
		}
	}
}

void detectRotatedFaces(Mat frame, vector<Rect> haarRect, vector<Mat> &rotatedFaces) {

	Mat drawing = frame.clone();
	Mat drawing2 = Mat::zeros(frame.size(), CV_8UC3);

	// Negative transformation
	Mat negative_im = frame.clone();
	cvtColor(negative_im, negative_im, CV_BGR2GRAY);
	negative_im = 255 - negative_im;

	// Threshold to extract high intensity regions
	Mat binary_im;
	threshold(negative_im, binary_im, 200, 255, CV_THRESH_BINARY);

	// Form contours from high intensity regions
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(binary_im, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point());

	// Determine center of desired contours
	vector<Point> matches;
	vector<vector<Point>> test;
	processContours(contours, matches, test);

	for (int i = 0; i < test.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		//drawContours(drawing, test, i, color, 2, 8, hierarchy, 0, Point());
		drawContours(drawing2, test, i, color, 2, 8, hierarchy, 0, Point());

		Moments mu = moments(test[i]);
		Point ctr = Point(mu.m10 / mu.m00, mu.m01 / mu.m00);

		//circle(drawing2, ctr, 100, CV_RGB(255, 0, 0));
	}


	vector<vector<vector<Point>>> connectedNodes;
	while (test.size() > 0)
	{
		//cout << "contour size " << test.size() << endl;
		// Reinitialize group
		vector<int> group;

		// Start from the beginning
		Moments mu1 = moments(test[0]);
		Point ctr1 = Point(mu1.m10 / mu1.m00, mu1.m01 / mu1.m00);

		// Update group
		group.push_back(0);

		// Search all remaining contours
		if (test.size() > 1)
		for (int j = 1; j < test.size(); j++)
		{
			Moments mu2 = moments(test[j]);
			Point ctr2 = Point(mu2.m10 / mu2.m00, mu2.m01 / mu2.m00);

			// Distance constraint satisfied and contour not yet visited
			if (norm(ctr1 - ctr2) < 75 && find(group.begin(), group.end(), j) == group.end())
			{
				group.push_back(j);
				visit(j, test, group);
			}
		}

		// Update group and remove element from contour
		vector<vector<Point>> temp;

		sort(group.begin(), group.end(), myfunction);

		for (int i = (group.size() - 1); i >= 0; i--)
		{
			// Update temporary container
			temp.push_back(test[group[i]]);

			// Erase contour
			//test.erase(test.begin() + group[i]);

			test[group[i]] = test.back();
			test.pop_back();

		}
		connectedNodes.push_back(temp);
	}

	/*for (auto cn : connectedNodes)
	{
		if (cn.size() > 1)
		for (int i = 0; i < cn.size() - 1; i++)
		{
			Moments mu1 = moments(cn[i]);
			Point ctr1 = Point(mu1.m10 / mu1.m00, mu1.m01 / mu1.m00);

			Moments mu2 = moments(cn[i + 1]);
			Point ctr2 = Point(mu2.m10 / mu2.m00, mu2.m01 / mu2.m00);

			line(drawing, ctr1, ctr2, CV_RGB(255, 0, 0), 2, 8);
		}
	}*/
	
	for (int i = 0; i < connectedNodes.size(); i++)
	{

		RotatedRect boundingRect;
		boundingRect = formRotatedRect2(connectedNodes[i]);

		Point2f rect_points[4];
		boundingRect.points(rect_points);
		for (int j = 0; j < 4; j++)
		{
			cout << rect_points[j];
			line(drawing, rect_points[j], rect_points[(j + 1) % 4], CV_RGB(0, 0, 0), 1, 8);
		}

	}

	imshow("drawing", drawing);
	imshow("drawing2", drawing2);
	// Update matches
	//updateMatches(matches, haarRect);

	//vector<Rect> rotatedRect;
	//restart:;
	//// Loop through vector of matches and form pairs
	//if (matches.size() > 1)
	//for (int i = 0; i < matches.size() - 1; i++) 
	//	for (int j = i + 1; j < matches.size(); j++)
	//	{
	//		// Determine L / R eye
	//		Point left, right;
	//		if (matches[i].x < matches[j].x)
	//		{
	//			left = matches[i];
	//			right = matches[j];
	//		}

	//		else
	//		{
	//			left = matches[j];
	//			right = matches[i];
	//		}

	//		// Compute similarity between the 2 images
	//		Mat Leye = frame(Rect(left.x - 20, left.y - 20, 40, 40) & Rect(0, 0, 640, 480));
	//		Mat Reye = frame(Rect(right.x - 20, right.y - 20, 40, 40) & Rect(0, 0, 640, 480));

	//		// Eyes should not be located near borders of image (haar does not work near borders)
	//		if (Leye.size() != Reye.size()) continue;

	//		// Convert to grayscale
	//		cvtColor(Leye, Leye, CV_BGR2GRAY);
	//		cvtColor(Reye, Reye, CV_BGR2GRAY);

	//		Mat corrMtrx;
	//		matchTemplate(Leye, Reye, corrMtrx, 5);

	//		Point min_loc, max_loc; double min, max;
	//		minMaxLoc(corrMtrx, &min, &max, &min_loc, &max_loc);

	//		// Proceed if image patches are similar
	//		if (max > 0.8)
	//		{
	//			//Form rotated rect
	//			RotatedRect rotRect = formRotatedRect(left, right, frame);

	//			Point2f rect_points[4];
	//			rotRect.points(rect_points);
	//			for (int j = 0; j < 4; j++)
	//				line(frame, rect_points[j], rect_points[(j + 1) % 4], CV_RGB(255, 0, 0), 1, 8);

	//			//Crop image and unrotate
	//			Mat croppedIm = cropRotatedRect(rotRect, frame);

	//			//imshow("croppedIm", croppedIm);

	//			//Continue if croppedIm not empty
	//			if (!croppedIm.empty())
	//			{
	//				// Convert to grayscale for LBP
	//				cvtColor(croppedIm, croppedIm, CV_BGR2GRAY);

	//				// Min size to be detected
	//				Size minSize = Size(croppedIm.cols*0.5, croppedIm.rows*0.5);

	//				// LBP
	//				vector<Rect> _haarRect;
	//				face_cascade.detectMultiScale(croppedIm, _haarRect, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, minSize);

	//				// Rotated Face found, update
	//				if (_haarRect.size() != 0)
	//				{
	//					Rect roi = _haarRect[0] & Rect(0, 0, croppedIm.cols, croppedIm.rows);

	//					// Crop and resize before pushing
	//					Mat rotatedFace = croppedIm(roi);
	//					resize(rotatedFace, rotatedFace, Size(91, 112));

	//					// Push
	//					rotatedFace.push_back(rotatedFace);
	//					rotatedRect.push_back(rotRect.boundingRect());

	//					updateMatches(matches, rotatedRect);		
	//					goto restart;
	//				}
	//			}
	//		}
	//	}	
}

RotatedRect formRotatedRect2(vector<vector<Point>> contour) {

	vector<Point> points;

	for (int i = 0; i < contour.size(); i++)
		for (int j = 0; j < contour[i].size(); j++)
		{
			points.push_back(contour[i][j]);
		}

	RotatedRect boundingRect;
	boundingRect = minAreaRect(Mat(points));

	return boundingRect;

}

bool myfunction(int i, int j) { return (i<j); }

void visit(int i, vector<vector<Point>> contours, vector<int>& group) {

	// Start from current contour
	Moments mu1 = moments(contours[i]);
	Point center1 = Point(mu1.m10 / mu1.m00, mu1.m01 / mu1.m00);

	// Search for nearest neighbour
	for (int j = i + 1; j < contours.size(); j++)
	{
		Moments mu2 = moments(contours[j]);
		Point center2 = Point(mu2.m10 / mu2.m00, mu2.m01 / mu2.m00);

		if (norm(center1 - center2) < 75 && find(group.begin(), group.end(), j) == group.end())
		{
			group.push_back(j);
			visit(j, contours, group);
		}
	}

}

void PrincipalComponentsAnalysis(vector<database>& Face, int principalComponents, Mat& eigenFace, Mat& meanFace) {
	
	// Size of column vector
	int faceDimension	= Face[0].Image.rows* Face[0].Image.cols;

	cout << Face[0].Image.rows << " " << Face[0].Image.cols;

	// Size of database
	int sizeOfDatabase  = Face.size();

	meanFace			= Mat::zeros(Face[0].Image.size(), CV_32FC1);
	Mat meanshiftedData = Mat::zeros(Size(sizeOfDatabase, faceDimension), CV_32FC1);

	// Compute mean
	for (int i = 0; i < sizeOfDatabase; i++)	meanFace = meanFace + Face[i].Image;
	meanFace = meanFace / Face.size();
	
	// Normalize data and reshape to column vector
	for (int i = 0; i < sizeOfDatabase; i++)
	{
		Face[i].Image_compressed = Face[i].Image - meanFace;
		Face[i].Image_compressed.reshape(0, faceDimension).copyTo(meanshiftedData.col(i));
	}

	// Compute covariance (small) matrix 
	Mat small_CovMat = meanshiftedData.t() * meanshiftedData / sizeOfDatabase;

	// Eigendecompose
	Mat eigenvalues, s_eigenvectors, eigenvectors;
	eigen(small_CovMat, eigenvalues, s_eigenvectors);
	eigenvectors = meanshiftedData * s_eigenvectors;

	// Retain top principal components
	Mat eigenvectors_retained = eigenvectors(Rect(0, 0, principalComponents, faceDimension)).clone();

	// Compress database
	for (int i = 0; i < sizeOfDatabase; i++)	Face[i].Image_compressed = eigenvectors_retained.t() * Face[i].Image_compressed.reshape(0, faceDimension);

	eigenFace = eigenvectors_retained;
}

void projectToEigenspace(Mat input, Mat& output) {

	// Convert to 32 bit for operation
	if (type2str(input.type()).compare("8UC1") == 0)	input.convertTo(input, CV_32FC1);

	Mat temp_input = input.clone();
	temp_input = temp_input - meanFace;
	output = eigenFace.t() * temp_input.reshape(0, temp_input.rows * temp_input.cols);

}

void detectFaces(Mat frame, vector<Mat> &haarFaces, vector<Rect>& haarRect) {

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
		_haarRect[i] = Rect(_haarRect[i].tl().x + 0.1*_haarRect[i].width, _haarRect[i].tl().y + 0.1*_haarRect[i].height, 0.8*_haarRect[i].width, 0.8*_haarRect[i].height);

		Rect roi = _haarRect[i] & Rect(0, 0, 640, 480);
		
		// Convert to Grayscale and resize before pushing
		Mat _haarCandidate = frame_gray(roi);
		Mat _colorFace = color(roi);

		resize(_haarCandidate, _haarCandidate, Size(92, 112));

		// Push
		haarFaces.push_back(_haarCandidate);
		haarRect.push_back(_haarRect[i]);
	}
}

void updateDisplay(Mat& frame, vector<Rect> haarRect, vector<int> matchID, vector<pair<double,double>> distanceToBestMatch, vector<database> Face) {
	
	// Loop through all candidates
	for (int i = 0; i < haarRect.size(); i++)
	{
		// Convert for display
		Face[matchID[i]].Image.convertTo(Face[matchID[i]].Image, CV_8U);	

		// Draw bounding box
		rectangle(frame, haarRect[i], CV_RGB(0, 0, 0));
		putText(frame, Face[matchID[i]].Name, haarRect[i].tl(), FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255, 0, 0));

		cout << distanceToBestMatch[i].first << " " << distanceToBestMatch[i].second << endl;
	}

}

void getMatches(vector<database> Faces, vector<Mat> haarFaces_compressed, vector<int>& matchID, vector<pair<double,double>>& distanceToBestMatch) {

	int numberOfCandidates = haarFaces_compressed.size();
	int numberOfFaces      = Faces.size();
		
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

Mat cropRotatedRect(RotatedRect boundingRect, Mat src) {

	Point2f rect_points[4];
	boundingRect.points(rect_points);

	// if Rect does not exceed image boundaries
	/*if (boundary.contains(rect_points[0]) && boundary.contains(rect_points[1]) && boundary.contains(rect_points[2]) && boundary.contains(rect_points[3]))
	{*/

		Mat M, rotated, cropped;

		// get angle and size from the bounding box
		float angle = boundingRect.angle;
		Size rect_size = boundingRect.size;

		// thanks to http://felix.abecassis.me/2011/10/opencv-rotation-deskewing/
		if (angle < -45.) {
			angle += 90.0;
			swap(rect_size.width, rect_size.height);
		}

		// get the rotation matrix
		M = getRotationMatrix2D(boundingRect.center, angle, 1.0);

		// perform the affine transformation
		warpAffine(src, rotated, M, src.size(), INTER_CUBIC);

		// crop the resulting image
		getRectSubPix(rotated, rect_size, boundingRect.center, cropped);

		return cropped;
	//}
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
printf("Matrix: %s %dx%d \n", ty.c_str(), frame.cols, frame.rows);*/