

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "IP/GetMatches.h"
#include <time.h> 
#include <fstream>
//#include "Mathematic.h"
//#define GEOGRAPHICLIB_SHARED_LIB 1
//#include <GeographicLib/Geocentric.hpp>
//#include <GeographicLib/LocalCartesian.hpp>
#include <filesystem>
#include "common.h"
#include "IP/SimpleNet.h"

//#include "SuperFilter.h"

using namespace std;
using namespace GeographicLib;
namespace fs = std::filesystem;

int main()
{
	SimpleNet model;
	model.load("net_data/model8-5.net");
	std::vector <float> f, out, out_2;
	out.resize(model.getOutputsCount());
	
	//SuperVelFilter SVF(0.1);

	/////////////////
	vector<cv::Point2d> gps_traj;
	vector<uint64_t> gps_ts;
	vector<float> h;
	vector<cv::Point3d> ang_v;
	vector<uint64_t> pack_ts;
	vector<uint64_t> frame_ts;
	load_pack_traj("data/pack.csv", h, ang_v, pack_ts);
	load_frame_ts("data/frame_data2.csv", frame_ts);
	load_gps_traj("data/gps.csv", gps_traj, gps_ts);
	vector<cv::Point2d> vel;
	make_hor_v(gps_ts, gps_traj, vel);
	vector<cv::Point3d> ang_vel;
	vector<float> vh;
	make_ang_v(pack_ts, ang_v, ang_vel, h, vh); 
	//save_traj_gps(gps_traj, "gps_meters.csv");
	vector<vector<float> > features, features_mirr;

	GetMatches get_matches("distors.txt", "intrincics.txt");
	uint64_t ts = 0;

	std::vector<Camera> cams;

	int ii = 0;
	int bad_fr = 0;

	double all_tm = 0;
 
	//Зададим начальное положение
	Camera drone;
	drone.set_start_pos(360, 240);

	double k = 0.05;
	
	static fs::directory_iterator di(".\\data\\frame2");
	static auto it = begin(di);
	//for (size_t i=0; i<250;i++)
	cv::VideoWriter video("demo.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 25, cv::Size(720, 480*2));

	//карта в реалтайм
	cv::Mat field(480,720, CV_8UC3, cv::Scalar(0, 0, 0));
	cv::circle(field, cv::Point2d(360, 240), 3, cv::Scalar(100, 100), 2);
	cv::Point3d c1(360, 240, 0);


	float cur_az = 0;
	float cur_v = 0;
	ofstream fout("res.csv");
	ofstream fout2("vel_debug.csv");
	float v_base = 0.0f;
	float az = 0;
	float v = 0;
	float hh = 0;
	float fvh = 0;
	while (true)
	{
		ii++;
		cv::Mat frame, grey;
	
		if (it == end(di)) break;
		cv::Mat cropped_image = cv::imread(it->path().u8string());
		it++;
		if (cropped_image.empty())break;
		cv::cvtColor(cropped_image, grey, cv::COLOR_BGRA2GRAY);
		ts += 40000;
		std::vector<cv::Point2f> v1, v2;
		std::vector <float> err1, err2;
		clock_t tm_start = clock();
		get_matches.get_matches(grey, ts, v1, v2, 9, err1, err2);
		clock_t tm_end = clock();
		double seconds = (double)(tm_end - tm_start) / CLOCKS_PER_SEC;
		all_tm += seconds;

		crop_props(v1, v2);
		draw_matches(cropped_image, v1, v2);
	
		//прямой проход Нейросети
		if (v1.size() > 15)
		{
			matchFeaturing(v1, v2, f);
			model.compute(&f[0], &out[0]);		
			az =out[2]; 
			v = out[4];
			hh = hh + out[3] * 0.025f;
			const float af = 0.01f;
			fvh = fvh * (1 - af) + out[3] * af;
			if (fabsf(fvh) > 0.05f) v = 0;		
		}
		
		const float ka = 0.03f;
		cur_az += az * ((float)(frame_ts[ii] - frame_ts[ii - 1])) / 1000000.f;//угол из угловой скорости
		cv::Point3d p(c1);
		const float a = 0.1f;
		v_base = 6.f;
		cur_v = cur_v * (1.f - a) + a * v;

		p.x+=cos(cur_az)*cur_v*ka;
		p.y+=sin(cur_az)*cur_v*ka;

		const float scale = 3.f;
		fout << out[0] << ";" << out[1] << ";" << out[2] << ";" << out[3] << ";" << out[4] << ";" << p.x*scale<< ";" << p.y*scale<< "\n";

		cv::Point c1i = static_cast<cv::Point2i>(cv::Point2d(c1.x, c1.y));
		cv::Point c2i = static_cast<cv::Point2i>(cv::Point2d(p.x, p.y));
		cv::line(field, c1i, c2i, cv::Scalar(100, 20, 220), 3);
		c1 = p;
		///-----------------------------
		cv::Mat res_img;
		cv::vconcat(cropped_image, field, res_img);
		cv::imshow("Trace", res_img);
		cv::waitKey(1);
		video.write(res_img);
		///data_grab
		/*if (v1.size() > 15)
		{			
			vector<float> f , f_mirr;
			matchFeaturing(v1, v2, f);
			matchFeaturing(v1, v2, f_mirr, true);
			features.push_back(f);
			features_mirr.push_back(f_mirr);
		}
		else
		{
			//vector<float> f{ 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
			vector<float> f{ 0,0,0,0,0,0,0,0};
			features.push_back(f);
			features_mirr.push_back(f);
		}*/
	}

	//сохранение нового файла с фичами
	//save_synchronized_data("data_features2.csv", pack_ts, ang_vel, h, vh, gps_ts, vel, frame_ts, features, features_mirr);
	printf("i = %d	bad_fr = %d  %f\n",ii, bad_fr, all_tm / 300.f);
	fout2.close();
	fout.close();
	cv::waitKey(0);
	//save_traj_csv(cams, "traj.csv");
	///нарисуем 2д траекторию
	//draw_2D_trajectoru(cams);
	
}