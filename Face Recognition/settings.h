#include <cv.h>
using namespace cv;

Rect input_video		= Rect(50, 100, 640, 480);


// Button Position
Rect candidate			= Rect(1280, 0, 100, 100);
Rect face_detect_button	= Rect(100, 790, 100, 100);

// Video Resolution
VideoCapture capture;

// Image Resolution

/*clock_t t1, t2;
t1 = clock();*/
/*t2 = clock();
float diff((float)t2 - (float)t1);
cout << diff << endl;*/