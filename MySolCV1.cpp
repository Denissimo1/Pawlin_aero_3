

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

#include "IP/SimpleNetMy.h"

//#include <PWNNLib2_0/SimpleNet.h>
//#include <PWNGeneral/Dataset.h>

//#include "SuperFilter.h"

using namespace std;
//using namespace GeographicLib;
namespace fs = std::filesystem;


#define TEST


int main()
{
#ifdef TEST
	SimpleNet model;
	model.load("data_features_8_to_5.net");
	std::vector <float> f, out;
	out.resize(model.getOutputsCount());
	
#else

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
#endif
	//save_traj_gps(gps_traj, "gps_meters.csv");
	vector<vector<float> > features, features_mirr;

	GetMatches get_matches("distors.txt", "intrincics.txt");
	uint64_t ts = 0;

	int ii = 0;
	int bad_fr = 0;

	double all_tm = 0;
	double k = 0.05;
	
	static fs::directory_iterator di(".\\data\\frame2");
	static auto it = begin(di);

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
	std::vector<float> fl;
	fout << "wx" << ";" << "wy" << ";" << "wz" << ";" << "vh" << ";" << "v" << ";" << "pos_x" << ";" << "pos_y" << ";" << "h" << "\n";
	cv::Point3d p;
	float az_start = -0.45f;

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
	
#ifdef TEST
		//прямой проход Нейросети
		if (v1.size() > 15)
		{
			matchFeaturing(v1, v2, f);
			if (fl.size() != 0)
			{
				for (size_t j = 0; j < f.size(); j++)
				{
					if (f[j] == 0) {
						f[j] = fl[j];//заполняем от предыдущих фич
					}
				}
			}
			fl = f;
			model.compute(&f[0], &out[0]);				
			az =out[2]; 
			v = out[4];
			hh = hh + out[3] * 0.025f;
			const float af = 0.02f;
			fvh = fvh * (1 - af) + out[3] * af;
			//fout2 << out2[0] << ";" << out2[1] << ";" << out2[2] << "\n";
			if (fabsf(fvh) > 0.5f) v = 0;		
		}
		else
		{
			f = { 0,0,0,0,0,0,0,0 };
		}
		
		const float ka = 0.025f;
		const float kaj = 0.012f;
		cur_az += az * 0.033f;//угол из угловой скорости
		cv::Point3d pview(c1);
		
		const float a = 0.1f;
		cur_v = cur_v * (1.f - a) + a * v;

		pview.x+=cos(cur_az)*cur_v*ka;
		pview.y+=sin(cur_az)*cur_v*ka;
		p.x += cos(az_start + cur_az)*cur_v*kaj;
		p.y += sin(az_start + cur_az)*cur_v*kaj;

		const float scale = 3.f;
		fout << out[0] << ";" << out[1] << ";" << out[2] << ";" << fvh << ";" << out[4] << ";" << p.x*scale<< ";" << p.y*scale<< ";"<<hh<<"\n";

		cv::Point c1i = static_cast<cv::Point2i>(cv::Point2d(c1.x, c1.y));
		cv::Point c2i = static_cast<cv::Point2i>(cv::Point2d(pview.x, pview.y));
		cv::line(field, c1i, c2i, cv::Scalar(100, 20, 220), 3);
		c1 = pview;
#endif
		///-----------------------------
		cv::Mat res_img;
		cv::vconcat(cropped_image, field, res_img);
		cv::imshow("Trace", res_img);
		cv::waitKey(1);
		video.write(res_img);
#ifndef TEST	
		///data_grab
		if (v1.size() > 15)
		{			
			vector<float> f , f_mirr;
			matchFeaturing(v1, v2, f);
			matchFeaturing(v1, v2, f_mirr, true);
			features.push_back(f);
			features_mirr.push_back(f_mirr);
		}
		else
		{
			vector<float> f{ 0,0,0,0,0,0,0,0};
			features.push_back(f);
			features_mirr.push_back(f);
		}
#endif
	}

#ifndef TEST	
	//сохранение нового файла с фичами
	save_synchronized_data("out/data_features_8_to_5.csv", pack_ts, ang_vel, h, vh, gps_ts, vel, frame_ts, features, features_mirr);
#endif

	fout2.close();
	fout.close();
	cv::waitKey(0);
	//save_traj_csv(cams, "traj.csv");
	
}