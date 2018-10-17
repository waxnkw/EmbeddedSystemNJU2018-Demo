#include <cstdlib>
#include <iostream>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define _DEBUG
#define PI 3.1415926

const string CAM_PATH="/dev/video0";
const string MAIN_WINDOW_NAME="Processed Image";
const string CANNY_WINDOW_NAME="Canny";

const int CANNY_LOWER_BOUND=50;
const int CANNY_UPPER_BOUND=250;

const int HOUGH_THRESHOLD=150;

const int Resized_Width = 512;
const int Resized_Height = 512;

const double Threshold = 180;
const int Max_Binary_Val = 255;
const int Binary_Threshold_Type = 0;



const Mat Conv_Kernel = getStructuringElement(MORPH_RECT, Size(15, 15));

void my_shrink(Mat& src, Mat& tar){
    resize(src, tar, Size(Resized_Width, Resized_Height), 0, 0,  INTER_LINEAR);
}

void my_crop(Mat& src, Mat& tar){
    //crop
    Rect roi = Rect(0, src.rows/3, src.cols, src.rows/3);
    tar = Mat(src, roi);
}

void my_grey(Mat& src, Mat& tar){
    cvtColor(src, tar, COLOR_BGR2GRAY);
    threshold(tar, tar, Threshold, Max_Binary_Val, Binary_Threshold_Type);
}

void my_dilate_and_erode(Mat& src, Mat& tar){
    dilate(src, tar, Conv_Kernel);
    erode(tar, tar, Conv_Kernel);
}

void my_inverse(Mat& src, Mat& tar){
    tar = 255-src;
}

void my_smooth(Mat& src, Mat& tar){
    medianBlur(src, tar, 7);
}

void getPoints(Mat& x,Mat& img, Point2f src[], Point2f dst[]){
    //hough
    Mat contours;
    Canny(img,contours,CANNY_LOWER_BOUND,CANNY_UPPER_BOUND);
#ifdef _DEBUG
    //imshow(CANNY_WINDOW_NAME,contours);
#endif

    vector<Vec2f> lines;
    HoughLines(contours,lines,1,PI/180,HOUGH_THRESHOLD);
    Mat result(img.size(),CV_8U,Scalar(255));
    img.copyTo(result);

#ifdef _DEBUG
    cout<<"lines: ";
    cout<<lines.size()<<endl;
#endif

    Point2f left_down;
    Point2f right_down;
    Point2f left_up;
    Point2f right_up;
    bool  is_left = false;
    bool is_right = false;
    //Draw the lines and judge the slope
    for(vector<Vec2f>::const_iterator it=lines.begin();it!=lines.end();++it)
    {
        float rho=(*it)[0];			//First element is distance rho
        float theta=(*it)[1];		//Second element is angle theta
        cout<<rho<<" "<<theta<<endl;

        if(!is_left&&theta>0&&theta<1.0)
        {
            //point of intersection of the line with first row
            Point pt1(rho/cos(theta),0);
            left_up = pt1;
            //point of intersection of the line with last row
            Point pt2((rho-result.rows*sin(theta))/cos(theta),result.rows);
            left_down = pt2;
            //Draw a line
#ifdef _DEBUG
            line(x,pt1,pt2,Scalar(0,255,255),3,CV_AA);
            cout<<pt1<<pt2<<endl;
#endif
            is_left= true;
        }
        if (!is_right&&theta>2&&theta<3.14){
            //point of intersection of the line with first row
            Point pt1(rho/cos(theta),0);
            right_up = pt1;
            //point of intersection of the line with last row
            Point pt2((rho-result.rows*sin(theta))/cos(theta),result.rows);
            right_down = pt2;
            //Draw a line
#ifdef _DEBUG
            line(x,pt1,pt2,Scalar(0,255,255),3,CV_AA);
            cout<<pt1<<pt2<<endl;
#endif
            is_right= true;
        }
    }

    src[0] = left_down;
    src[1] = right_down;
    src[2] = left_up;
    src[3] = right_up;
    dst[0] = Point2f(left_down.x, right_down.y);
    dst[1] = Point2f(right_down.x, right_down.y);
    dst[2] = Point2f(left_down.x, left_up.y);
    dst[3] = Point2f(right_down.x, right_up.y);

    imshow(MAIN_WINDOW_NAME,x);
}

void my_transform(Mat& img, Mat& tar){
    Point2f src[4];
    Point2f dst[4];
    getPoints(img, img, src, dst);
    Mat trans = getPerspectiveTransform(src, dst);
    warpPerspective(img, tar, trans, Size(img.cols, img.rows));
}

void process_img(Mat& img, double ret[]){
    Mat img_grey;
    my_crop(img, img_grey);

//    my_shrink(img_grey, img_grey);

    Mat x = img_grey;
    my_grey(img_grey, img_grey);
    my_dilate_and_erode(img_grey, img_grey);
    my_inverse(img_grey, img_grey);
    my_smooth(img_grey, img_grey);
    imshow("crop", img_grey);

    Point2f src[4];
    Point2f dst[4];
    getPoints(x, img_grey, src, dst);

//    my_transform(img_grey, img_grey);
    //imshow("",img_grey);
    waitKey();
}


int main()
{
	VideoCapture capture(CAM_PATH);
	//If this fails, try to open as a video camera, through the use of an integer param
	if (!capture.isOpened())
	{
		capture.open(atoi(CAM_PATH.c_str()));
	}

	double dWidth=capture.get(CV_CAP_PROP_FRAME_WIDTH);			//the width of frames of the video
	double dHeight=capture.get(CV_CAP_PROP_FRAME_HEIGHT);		//the height of frames of the video
	clog<<"Frame Size: "<<dWidth<<"x"<<dHeight<<endl;

	Mat image;
	while(true)
	{
		capture>>image;
		if(image.empty())
			break;
		process_img(image, NULL);
		waitKey(1);
	}
	return 0;
}

