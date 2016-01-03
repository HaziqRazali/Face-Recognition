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
void PrincipalComponentsAnalysis(vector<face>& Face, int principalComponents, Mat& eigenFace, Mat& meanFace);
void projectToEigenspace(Mat input, Mat& output);
void detectFaces(Mat frame, vector<Mat>& candidates);

void fakeButtons();
void fakeButtons_update();

string type2str(int type);

// Unused functions
void detectAndDisplay(Mat frame);

void CallBackFunc(int event, int x, int y, int flags, void* userdata);

// Global variables
String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";
RNG rng(12345);

Mat eigenFace, meanFace;
Mat display_color = Mat(Size(1920, 1080), CV_8UC3, Scalar::all(127));

Rect Video = Rect(0, 0, 1280, 780);
Rect detectedFaces = Rect(1400, 0, 100, 0);

bool detect = false;

int main(int argc, const char** argv)
{
	namedWindow(window_name, CV_WINDOW_NORMAL);
	setMouseCallback(window_name, CallBackFunc, NULL);

	/*********************************************************************************

								READ IN TRAINING DATA

	**********************************************************************************/

	//vector<face> Face;
	//vector<String> files;

	//String folder = "C:\\Users\\Haziq\\Documents\\MATLAB\\att_faces(8x20)"; 

	//// Read training data
	//glob(folder, files);

	//// Store training data
	//for (int i = 0; i < files.size(); ++i)
	//{
	//	Mat src = imread(files[i], CV_LOAD_IMAGE_GRAYSCALE);
	//	src.convertTo(src, CV_32FC1);
	//	Face.push_back(face(src, i / 8 + 1));
	//}

	//if (Face.size() == 0) { cout << "Database is empty"; return 0; }

	//// -- Read in test data
	//Mat testImage = imread("C:\\Users\\Haziq\\Documents\\MATLAB\\att_faces\\29.pgm", CV_LOAD_IMAGE_GRAYSCALE);
	//testImage.copyTo(display(Rect(1200, 0, testImage.cols, testImage.rows)));
	//testImage.convertTo(testImage, CV_32FC1);


	///*********************************************************************************

	//									TRAIN

	//**********************************************************************************/
	//
	//// PCA
	//PrincipalComponentsAnalysis(Face, 30, eigenFace, meanFace);

	//// Project training data onto Eigenspace
	///*for (int i = 0; i < Face.size(); i++) projectToEigenspace(Face[i].Image, Face[i].Image_compressed);*/
	//
	//// Project test data onto the Eigenspace
	//Mat testImage_compressed;
	//projectToEigenspace(testImage, testImage_compressed);

	///*********************************************************************************

	//									CLASSIFY

	//**********************************************************************************/
	//
	//// Compute Nearest Neighbour
	//vector<float> euclideanDist;
	//int result;
	//for (int i = 0; i < Face.size(); i++)	euclideanDist.push_back(norm(testImage_compressed - Face[i].Image_compressed));

	//// Get index of best match
	//result = min_element(std::begin(euclideanDist), std::end(euclideanDist)) - euclideanDist.begin();

	//// Display best match
	//Face[result].Image.convertTo(Face[result].Image, CV_8U);
	//Face[result].Image.copyTo(display(Rect(1400, 0, Face[result].Image.cols, Face[result].Image.rows)));

	//imshow("lol", display);
	//waitKey(0);

	/*********************************************************************************
	
									FACE DETECTION
	
	**********************************************************************************/

	VideoCapture capture;
	Mat frame;

	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name)){ printf("--(!)Error loading\n"); return -1; };
	if (!eyes_cascade.load(eyes_cascade_name)){ printf("--(!)Error loading\n"); return -1; };

	//-- 2. Read the video stream
	if (capture.open(1))
	{

		capture.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
		capture.set(CV_CAP_PROP_FRAME_HEIGHT, 720);

		while (true)
		{
			// read in frame
			capture >> frame;

			if (!frame.empty())
			{
				// Detect Faces
				if (detect == true /*&& mode == recognition*/) 
				{					
					// Re initialize candidates and clear display
					vector<Mat> candidates;
					display_color = Mat(Size(1920, 1080), CV_8UC3, Scalar::all(127));

					// Detect faces
					detectFaces(frame, candidates);

					// Display all candidates
					for (int i = 0; i < candidates.size(); i++) 
						candidates[i].copyTo(display_color(Rect(0, 0, candidates[i].cols, candidates[i].rows)));

					// Project candidates to Eigenspace
					Mat candidates_compressed;
					for (int i = 0; i < candidates.size(); i++) projectToEigenspace(candidates[i], candidates_compressed);

					// Classify candidates

					// Display best match

					detect = false;
				}

				else if (detect == true /*&& mode == training*/)
				{
					// Re initialize candidates
					vector<Mat> candidates;

					// Detect faces
					detectFaces(frame, candidates);

					// Save to file

					// Display - title (captured)

				}

				//cvtColor(frame, frame, CV_BGR2GRAY);
				frame.copyTo(display_color(Rect(0, 0, frame.cols, frame.rows)));

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

void detectFaces(Mat frame, vector<Mat> &candidates) {

	// Image processing
	Mat frame_gray;
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	// Detect faces
	vector<Rect> faces_boundingBox;
	face_cascade.detectMultiScale(frame_gray, faces_boundingBox, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

	// Update candidates
	for (int i = 0; i < faces_boundingBox.size(); i++)
	{
		Mat face = frame_gray(faces_boundingBox[i]);
		resize(face, face, Size(100, 100));
		candidates.push_back(face);
		//face.copyTo(display_color(Rect(800, 0, faceROI.cols, faceROI.rows)));
	}
}

void detectAndDisplay(Mat frame)
{
	vector<Rect> faces;
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

void fakeButtons()
{
	rectangle(display_color, Rect(0, 800, 200, 100), Scalar(0, 0, 255));
	putText(display_color, "haha", Point(10, 810), FONT_HERSHEY_SIMPLEX, 3, Scalar(0, 0, 0), 3);
}

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
	}

	else if (event == EVENT_RBUTTONDOWN)
	{
		cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
		detect = true;

	}

	else if (event == EVENT_MBUTTONDOWN)
	{
		cout << "Middle button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
	}

	else if (event == EVENT_MOUSEMOVE)
	{
		cout << "Mouse move over the window - position (" << x << ", " << y << ")" << endl;

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