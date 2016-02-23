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

#include "settings.h"
#include "time.h"

using namespace std;
using namespace cv;

#define AUTOMATIC
#define MANUAL

//[DebuggerDisplay("{Class}")]
struct face {

public:

	Mat Image;
	Mat Image_compressed;
	int Class;

	face(Mat _image, int _class) : Image(_image), Class(_class) {}

};

// Display functions
void initializeDisplay();
void CallBackFunc(int event, int x, int y, int flags, void* userdata);

// Recognition functions
void PrincipalComponentsAnalysis(vector<face>& Face, int principalComponents, Mat& eigenFace, Mat& meanFace);
void projectToEigenspace(Mat input, Mat& output);

// Detection functions
void detectFaces(Mat frame, vector<Mat>& candidates);
void locateRotatedFaces(Mat frame, vector<Rect> haarCandidates);
RotatedRect formRotatedRect(Mat left, Mat right, Mat angle, Mat frame);
Mat cropRotatedRect(RotatedRect boundingRect, Mat src);

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

Rect boundary = Rect(0, 0, 601, 453);

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

										READ IN DATA

	**********************************************************************************/

	/*********************************************************************************
								Generate vector of templates
	*********************************************************************************/
	Mat tmplate = imread("Eye.png");

	vector<Mat> tmplates;
	vector<Mat> masks;
	vector<Point> offsets;

	for (int i = 0; i < 5; i++)
	{
		// Push template, mask and offset
		tmplates.push_back(tmplate);
		masks.push_back(Mat::zeros(tmplate.rows, tmplate.cols, CV_32FC1));
		offsets.push_back(Point(tmplate.rows / 2, tmplate.cols / 2));

		// Resize
		resize(tmplate, tmplate, Size(), 0.8, 0.8, CV_INTER_CUBIC);
	}

	/*********************************************************************************
								Read in training data
	*********************************************************************************/
	vector<face> Face;
	vector<String> files;
	
	String folder = "C:\\Users\\Haziq\\Documents\\MATLAB\\att_faces(8x20)"; 

	// Read training data
	glob(folder, files);

	// Store training data
	for (int i = 0; i < files.size(); ++i)
	{
		Mat src = imread(files[i], CV_LOAD_IMAGE_GRAYSCALE);
		src.convertTo(src, CV_32FC1);
		Face.push_back(face(src, i / 8 + 1));
	}

	if (Face.size() == 0) { cout << "Database is empty"; return 0; }

	// -- Read in test data
	//Mat testImage = imread("C:\\Users\\Haziq\\Documents\\MATLAB\\att_faces\\29.pgm");
	//testImage.copyTo(display_color(Rect(1200, 0, testImage.cols, testImage.rows)));
	//cvtColor(testImage, testImage, CV_BGR2GRAY);
	//testImage.convertTo(testImage, CV_32FC1);


	///*********************************************************************************

	//									TRAIN

	//**********************************************************************************/
	
	// PCA
	PrincipalComponentsAnalysis(Face, 30, eigenFace, meanFace);

	// Project training data onto Eigenspace
	for (int i = 0; i < Face.size(); i++) projectToEigenspace(Face[i].Image, Face[i].Image_compressed);
	
	//// Project test data onto the Eigenspace
	/*Mat testImage_compressed;
	projectToEigenspace(testImage, testImage_compressed);*/

	///*********************************************************************************

	//									CLASSIFY

	//**********************************************************************************/
	/*
	// Compute Nearest Neighbour
	//vector<float> euclideanDist;
	//int result;
	//for (int i = 0; i < Face.size(); i++)	euclideanDist.push_back(norm(testImage_compressed - Face[i].Image_compressed));

	//// Get index of best match
	//result = min_element(begin(euclideanDist), end(euclideanDist)) - euclideanDist.begin();

	//// Display best match
	//Face[result].Image.convertTo(Face[result].Image, CV_8U);
	//cvtColor(Face[result].Image, Face[result].Image, CV_GRAY2BGR);
	//Face[result].Image.copyTo(display_color(Rect(1400, 0, Face[result].Image.cols, Face[result].Image.rows)));
	*/
	/*********************************************************************************
	
										BEGIN
	
	**********************************************************************************/

	// Video frame
	Mat frame;

	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name)){ printf("--(!)Error loading\n"); return -1; };

	//-- 2. Read the video stream
	if (capture.open(1))
	{
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
					
					// Vector of candidates (images and rectangles)
					vector<Mat> haarCandidates;
					vector<Rect> haarRect;

					// Detect faces -----------------------------------------
					detectFaces(frame, haarCandidates, haarRect);
					
					// Propose oriented faces 
					vector<Point> matches;
					locateRotatedFaces(frame, haarRect, matches);


					/********************************************************

											RECOGNITION

					*********************************************************/
					// Project candidates to Eigenspace
					vector<Mat> candidates_compressed(haarCandidates.size());
					for (int i = 0; i < haarCandidates.size(); i++)
						projectToEigenspace(haarCandidates[i], candidates_compressed[i]);

					// Nearest Neighbour Classifier
					vector<vector<float>> euclideanDist(haarCandidates.size());
					vector<int> result(haarCandidates.size());
					
					for (int i = 0; i < haarCandidates.size(); i++)
					{
						for (int j = 0; j < Face.size(); j++)
							euclideanDist[i].push_back(norm(candidates_compressed[i] - Face[j].Image_compressed));

						// Classify
						result[i] = min_element(begin(euclideanDist), end(euclideanDist)) - euclideanDist.begin();
						Face[result[i]].Image.convertTo(Face[result[i]].Image, CV_8U);
						imshow("L", Face[result[i]].Image);
					}
									
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

RotatedRect formRotatedRect(Point lefteye, Point righteye, float _angle, Mat src) {

	/*******************************************************
						Form boundingRect
	*******************************************************/

	Point topLeft, topRight, bottomLeft, bottomRight;

	float d = 0.3*norm(Mat(lefteye), Mat(righteye), NORM_L2);
	float theta = _angle;

	// Compute the 4 corners
	topLeft.x = lefteye.x - d*cos(theta) + 1.5* d*sin(theta);
	topLeft.y = lefteye.y - d*sin(theta) - 1.5* d*cos(theta);

	topRight.x = righteye.x + d*cos(theta) + 1.5 * d*sin(theta);
	topRight.y = righteye.y + d*sin(theta) - 1.5 * d*cos(theta);

	bottomLeft.x = topLeft.x - 3.5 * d*sin(theta);
	bottomLeft.y = topLeft.y + 3.5 * d*cos(theta);

	bottomRight.x = topRight.x - 3.5 * d*sin(theta);
	bottomRight.y = topRight.y + 3.5 * d*cos(theta);

	// Push
	vector<Point> rectCorners;
	rectCorners.push_back(topLeft);
	rectCorners.push_back(topRight);
	rectCorners.push_back(bottomRight);
	rectCorners.push_back(bottomLeft);

	// Generate the rotated rectangle
	RotatedRect boundingRect;
	boundingRect = minAreaRect(Mat(rectCorners));

	/*Point2f rect_points[4];
	boundingRect.points(rect_points);
	for (int j = 0; j < 4; j++)
	line(src, rect_points[j], rect_points[(j + 1) % 4], CV_RGB(0,0,0), 1, 8);*/

	return boundingRect;
}

Mat cropRotatedRect(RotatedRect boundingRect, Mat src) {

	Point2f rect_points[4];
	boundingRect.points(rect_points);
	for (int j = 0; j < 4; j++)
		line(src, rect_points[j], rect_points[(j + 1) % 4], CV_RGB(0, 0, 0), 1, 8);

	imshow("Rect", src);

	// if Rect does not exceed image boundaries
	if (boundary.contains(rect_points[0]) && boundary.contains(rect_points[1]) && boundary.contains(rect_points[2]) && boundary.contains(rect_points[3]))
	{
		/*******************************************************
		Rotate
		*******************************************************/

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
	}
}

void locateRotatedFaces(Mat frame, vector<Rect> haarCandidates, vector<Point>& matches) {

	Mat eyeball;
	Point offset;
	Mat mask;

	Mat negative_im = frame.clone();
	cvtColor(negative_im, negative_im, CV_BGR2GRAY);
	negative_im = 255 - negative_im;

	// ZNCC
	Mat corrMtrx;
	matchTemplate(negative_im, eyeball, corrMtrx, 5);

	double minVal, maxVal; 
	Point minLoc, maxLoc;
	vector<Point> matches;

	// Locate best matches
	for (int i = 0; i < 5; i++)
	{
		// locate best match
		minMaxLoc(corrMtrx, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
		
		// Mask out best match
		Rect roi	= Rect(maxLoc.x - offset.x/2, maxLoc.y - offset.y/2, eyeball.cols, eyeball.rows) & boundary;
		Mat crop	= negative_im(roi);
		crop.setTo(0);

		// best match should not be inside haarCandidates
		for (unsigned int j = 0; j < haarCandidates.size(); j++)
			if (haarCandidates[j].contains(maxLoc)) 
				goto end;

		// push
		matches.push_back(maxLoc);

		// End, locate 2nd best match
		end:
	}

	// Loop through matches and form pairs
	for (int i = 0; i < matches.size(); i++)
	{
		for (int j = 0; j < matches.size(); j++)
		{
			// Determine L / R eye
			Point left, right;
			if (matches[i].x < matches[j].x)
			{
				left = matches[i];
				right = matches[j];
			}

			else
			{
				left = matches[j];
				right = matches[i];
			}

			// Determine angle of rotation
			float angle = ((float)(right.y - left.y) / (right.x - left.x));

			// Form rotated rect
			RotatedRect rotRect = formRotatedRect(left, right, angle, frame);
			Mat croppedIm = cropRotatedRect(rotRect, frame);
		}
	}
}

void PrincipalComponentsAnalysis(vector<face>& Face, int principalComponents, Mat& eigenFace, Mat& meanFace)
{
	// Initialize constants
	int faceDimension	= Face[0].Image.rows* Face[0].Image.cols;
	int size			= Face.size();

	meanFace			= Mat::zeros(Face[0].Image.size(), CV_32FC1);
	Mat meanshiftedData	= Mat::zeros(Size(size, faceDimension), CV_32FC1);

	// Compute Mean Face 
	for (int i = 0; i < size; i++)	meanFace = meanFace + Face[i].Image;
	meanFace = meanFace / Face.size();

	// Compute covariance (small) matrix 	
	for (int i = 0; i < size; i++)
	{
		Face[i].Image_compressed = Face[i].Image - meanFace;
		Face[i].Image_compressed.reshape(0, faceDimension).copyTo(meanshiftedData.col(i));
	}
	Mat small_CovMat = meanshiftedData.t() * meanshiftedData / size;

	// Eigendecomposition ______________________________________________________________
	Mat eigenvalues, s_eigenvectors, eigenvectors;
	eigen(small_CovMat, eigenvalues, s_eigenvectors);
	eigenvectors = meanshiftedData * s_eigenvectors;

	// Retain top principal components
	Mat eigenvectors_retained = eigenvectors(Rect(0, 0, principalComponents, faceDimension)).clone();

	// Reduced face space
	for (int i = 0; i < size; i++)	Face[i].Image_compressed = eigenvectors_retained.t() * Face[i].Image_compressed.reshape(0, faceDimension);

	eigenFace = eigenvectors_retained;
}

void projectToEigenspace(Mat input, Mat& output) {

	// Convert to 32 bit for operation
	if (type2str(input.type()).compare("8UC1") == 0)	input.convertTo(input, CV_32FC1);

	Mat temp_input = input.clone();
	temp_input = temp_input - meanFace;
	output = eigenFace.t() * temp_input.reshape(0, temp_input.rows * temp_input.cols);

}

void detectFaces(Mat frame, vector<Mat> &haarCandidates, vector<Rect> haarRect) {

	// Convert to grayscale
	Mat frame_gray;
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	//equalizeHist(frame_gray, frame_gray);

	// LBP
	vector<Rect> faces_boundingBox;
	face_cascade.detectMultiScale(frame_gray, faces_boundingBox, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

	// Update haarCandidates
	for (int i = 0; i < faces_boundingBox.size(); i++)
	{
		// Convert to Grayscale and resize before pushing
		Mat face = frame_gray(faces_boundingBox[i]);
		resize(face, face, Size(92, 112));
		haarCandidates.push_back(face);

		// Push
		haarRect.push_back(faces_boundingBox[i]);
	}

	// Draw
	for (int i = 0; i < faces_boundingBox.size(); i++)
	{
		rectangle(frame, faces_boundingBox[i], CV_RGB(0, 0, 0));
		putText(frame, "Unidentified", faces_boundingBox[i].tl(), FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255, 0, 0));
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


RotatedRect formRotatedRect(Point lefteye, Point righteye, float _angle, Mat src) {

	/*******************************************************
						Form boundingRect
	*******************************************************/

	Point topLeft, topRight, bottomLeft, bottomRight;

	float d = 0.3*norm(Mat(lefteye), Mat(righteye), NORM_L2);
	float theta = _angle;

	topLeft.x = lefteye.x - d*cos(theta) + 1.5* d*sin(theta);
	topLeft.y = lefteye.y - d*sin(theta) - 1.5* d*cos(theta);

	topRight.x = righteye.x + d*cos(theta) + 1.5 * d*sin(theta);
	topRight.y = righteye.y + d*sin(theta) - 1.5 * d*cos(theta);

	bottomLeft.x = topLeft.x - 3.5 * d*sin(theta);
	bottomLeft.y = topLeft.y + 3.5 * d*cos(theta);

	bottomRight.x = topRight.x - 3.5 * d*sin(theta);
	bottomRight.y = topRight.y + 3.5 * d*cos(theta);

	vector<Point> rectCorners;
	rectCorners.push_back(topLeft);
	rectCorners.push_back(topRight);
	rectCorners.push_back(bottomRight);
	rectCorners.push_back(bottomLeft);

	RotatedRect boundingRect;
	boundingRect = minAreaRect(Mat(rectCorners));

	return boundingRect;
}

Mat cropRotatedRect(RotatedRect boundingRect, Mat src) {

	Point2f rect_points[4];
	boundingRect.points(rect_points);
	for (int j = 0; j < 4; j++)
		line(src, rect_points[j], rect_points[(j + 1) % 4], CV_RGB(0, 0, 0), 1, 8);

	imshow("Rect", src);

	// if Rect does not exceed image boundaries
	if (boundary.contains(rect_points[0]) && boundary.contains(rect_points[1]) && boundary.contains(rect_points[2]) && boundary.contains(rect_points[3]))
	{
		/*******************************************************
									Rotate
		*******************************************************/

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
printf("Matrix: %s %dx%d \n", ty.c_str(), frame.cols, frame.rows);*/