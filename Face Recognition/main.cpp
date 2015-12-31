/*

[✓] Face Detector
[✓]	Principal Components Analysis
[]	Integrate Everything
[]	Confusion Matrix
[]	RoC curve
[]	Train
[]  Cross-validate ?

Others
Wrapper Function to project data onto Eigenspace
Wrapper Function to convert file to 32FC1
Wrapper Function for imshow

*/

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

//[DebuggerDisplay("{Class}")]
struct face {

public:
	Mat Image;
	Mat Image_compressed;
	int Class;

	face(Mat _image, int _class) : Image(_image), Class(_class) {}

};

// Function Headers
void detectAndDisplay(Mat frame);
string type2str(int type);
void PrincipalComponentsAnalysis(vector<face>& Face, int principalComponents, Mat& eigenFace, Mat& meanFace);
void projectToEigenspace(Mat input, Mat& output);
void show(string window_name, Mat image, Mat& display);
void detectFace(Mat frame);

// Global variables
String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";
RNG rng(12345);

Mat eigenFace, meanFace;
Mat display = Mat(Size(1920, 1080), CV_8U, Scalar::all(127));


int main(int argc, const char** argv)
{

	/*********************************************************************************

									READ IN DATA

	**********************************************************************************/

	vector<face> Face;
	vector<String> files;

	String folder = "C:\\Users\\Haziq\\Documents\\MATLAB\\att_faces(8x20)"; 

	// Read all files in folder
	glob(folder, files);

	// Store files in Face database
	for (int i = 0; i < files.size(); ++i)
	{
		Mat src = imread(files[i], CV_LOAD_IMAGE_GRAYSCALE);
		src.convertTo(src, CV_32FC1);
		Face.push_back(face(src, i / 8 + 1));
	}

	if (Face.size() == 0) { cout << "Database is empty"; return 0; }

	/*********************************************************************************

										TRAINING

	**********************************************************************************/
	
	// Input  : Faces , no. of eigenvectors to retain
	// Output : projectedFaces , eigenFaces , meanFace
	PrincipalComponentsAnalysis(Face, 30, eigenFace, meanFace);

	// Read in test Image
	Mat testImage = imread("C:\\Users\\Haziq\\Documents\\MATLAB\\att_faces\\9.pgm", CV_LOAD_IMAGE_GRAYSCALE);
	testImage.copyTo(display(Rect(1200, 0, testImage.cols, testImage.rows)));
	testImage.convertTo(testImage, CV_32FC1);
	
	// Project onto the Eigenspace
	Mat testImage_compressed;
	projectToEigenspace(testImage, testImage_compressed);
	
	// Compute Nearest Neighbour
	vector<float> euclideanDist;
	int result;
	for (int i = 0; i < Face.size(); i++)	euclideanDist.push_back(norm(testImage_compressed - Face[i].Image_compressed));
	result = min_element(std::begin(euclideanDist), std::end(euclideanDist)) - euclideanDist.begin();

	// Display best match
	Face[result].Image.convertTo(Face[result].Image, CV_8U);
	Face[result].Image.copyTo(display(Rect(1000, 0, Face[result].Image.cols, Face[result].Image.rows)));

	/*********************************************************************************
	
									FACE DETECTION
	
	**********************************************************************************/

	VideoCapture capture;
	Mat frame;

	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name)){ printf("--(!)Error loading\n"); return -1; };
	if (!eyes_cascade.load(eyes_cascade_name)){ printf("--(!)Error loading\n"); return -1; };

	//-- 2. Read the video stream
	if (capture.open(0))
	{
		while (true)
		{

			capture >> frame;

			//-- 3. Apply the classifier to the frame
			if (!frame.empty())
			{
				// Clear display
				display = Mat(Size(1920, 1080), CV_8U, Scalar::all(127));

				// Detect Faces Input - frame Output - Faces
				detectFace(frame);

				// Project onto Eigenspace

				cvtColor(frame, frame, CV_BGR2GRAY);
				frame.copyTo(display(Rect(0, 0, frame.cols, frame.rows)));

				imshow(window_name, display);
				waitKey(100);
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

void show(string window_name, Mat image, Mat& display)
{
	namedWindow(window_name, WINDOW_FULLSCREEN);
	image.convertTo(image, CV_8U);
	image.copyTo(display(Rect(10, 10, image.cols, image.rows)));
	imshow(window_name, display);
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
	Mat eigenvectors_retained = eigenvectors(cv::Rect(0, 0, principalComponents, faceDimension)).clone();

	// Reduced face space
	for (int i = 0; i < size; i++)	Face[i].Image_compressed = eigenvectors_retained.t() * Face[i].Image_compressed.reshape(0, faceDimension);

	eigenFace = eigenvectors_retained;
}

void projectToEigenspace(Mat input, Mat& output) {

	input = input - meanFace;
	output = eigenFace.t() * input.reshape(0, input.rows * input.cols);

}

void detectFace(Mat frame) {

	// Image processing
	Mat frame_gray;
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	// Detect faces
	vector<Rect> faces_boundingBox;
	face_cascade.detectMultiScale(frame_gray, faces_boundingBox, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

	cout << faces_boundingBox.size(); 

	for (int i = 0; i < faces_boundingBox.size(); i++)
	{
		Mat faceROI = frame_gray(faces_boundingBox[i]);
		resize(faceROI, faceROI, Size(100, 100));
		faceROI.copyTo(display(Rect(800, 0, faceROI.cols, faceROI.rows)));
	}
}

void detectAndDisplay(Mat frame)
{
	std::vector<Rect> faces;
	Mat frame_gray;

	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

	for (size_t i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
		//ellipse(frame, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
		rectangle(frame, Rect(faces[i].tl(), faces[i].br()), (0, 0, 0), 3);

		Mat faceROI = frame_gray(faces[i]);
		std::vector<Rect> eyes;

		// In each face, detect eyes
		eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

		for (size_t j = 0; j < eyes.size(); j++)
		{
			Point center(faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5);
			int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
			circle(frame, center, radius, Scalar(255, 0, 0), 4, 8, 0);
		}
	}
	// Show what you got
	imshow(window_name, frame);
	waitKey(1);
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