#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "IP/GetMatches.h"
#include <time.h> 
//#include "Camera.h"
#include <fstream>
#include "Mathematic.h"
#define GEOGRAPHICLIB_SHARED_LIB 1
#include <GeographicLib/Geocentric.hpp>
#include <GeographicLib/LocalCartesian.hpp>
#include <filesystem>

using namespace std;
using namespace GeographicLib;

template< class T>
void get_next(std::ifstream& fl, T& d)
{
	string s;
	std::stringstream is;
	getline(fl, s, ',');
	is << s;
	is >> d;
}

uint64_t msec_from_string(std::string &str);



void load_gps_traj(const string& fn, std::vector<cv::Point2d> &coords, std::vector<uint64_t>&tsv)
{
	Geocentric earth(Constants::WGS84_a(), Constants::WGS84_f());
	const double lat0 = 48 + 50 / 60.0, lon0 = 2 + 20 / 60.0;
	//LocalCartesian proj(lat0, lon0, 0, earth);
	//proj.Forward(lat, lon, h, x, y, z);
	//proj.Reverse(x, y, z, lat, lon, h);
	std::ifstream fl0(fn);
	string s;

	getline(fl0, s);
	cv::Point2d p;
	getline(fl0, s);
	istringstream iss(s);
	getline(iss, s, ',');
	getline(iss, s, ',');
	std::stringstream is;
	is << s;
	is >> p.x;
	getline(iss, s, '\n');
	std::stringstream is2;
	is2 << s;
	is2 >> p.y;

	LocalCartesian proj(p.x, p.y, 0, earth);

	while (getline(fl0, s))// если начинаетс€ с какого то ненужного столбца!!!
	{
		istringstream iss(s);
		getline(iss, s, ',');
		uint64_t ts = msec_from_string(s);
		getline(iss, s, ',');
		std::stringstream is;
		is << s;
		is >> p.x;
		getline(iss, s, '\n');
		std::stringstream is2;
		is2 << s;
		is2 >> p.y;
		cv::Point2d gp;
		double z;
		proj.Forward(p.x, p.y, 0, gp.x, gp.y, z);
		coords.push_back(gp);
		tsv.push_back(ts);
	}
	printf("GPS_traj_readed\n");
}

void load_pack_traj(const string& fn, std::vector <float> &hv, std::vector<cv::Point3d> &ang, std::vector<uint64_t>&tsv)
{
	std::ifstream fl0(fn);
	string s;
	getline(fl0, s);

	while (getline(fl0, s))// если начинаетс€ с какого то ненужного столбца!!!
	{
		istringstream iss(s);
		getline(iss, s, ',');
		uint64_t ts = msec_from_string(s);
		getline(iss, s, ',');
		float h = atof(s.c_str());
		cv::Point3d p;
		getline(iss, s, ',');
		p.x = (double)atof(s.c_str());
		getline(iss, s, ',');
		p.y = (double)atof(s.c_str());
		getline(iss, s, '\n');
		p.z = (double)atof(s.c_str());
		
		hv.push_back(h);
		ang.push_back(p);
		tsv.push_back(ts);
	}
	printf("pack_readed\n");
}

void load_frame_ts(const string& fn, std::vector<uint64_t>&tsv)
{
	std::ifstream fl0(fn);
	string s;
	while (getline(fl0, s))// если начинаетс€ с какого то ненужного столбца!!!
	{
		istringstream iss(s);
		getline(iss, s, ',');
		getline(iss, s, '\n');
		uint64_t ts = msec_from_string(s);
		tsv.push_back(ts);
	}
}

void save_traj_gps(const std::vector<cv::Point2d>& pos, const std::string &fn)
{
	ofstream fout(fn); // создаЄм объект класса ofstream дл€ записи и св€зываем его с файлом cppstudio.txt
	for (size_t i = 0; i < pos.size(); i++)
	{
		fout << pos[i].x << ";" << pos[i].y << "\n";
	}
	fout.close();
}

void draw_matches(cv::Mat &img, const std::vector<cv::Point2f> &v1, const std::vector<cv::Point2f> &v2)
{
	for (size_t i = 0; i < v1.size(); i++)
	{
		cv::line(img, v1[i], v2[i], cv::Scalar(130, 40, 200), 2);
	}
}

void solveH(const cv::Mat& H, cv::Mat& Rr, cv::Mat &Tt)
{
	// Normalization to ensure that ||c1|| = 1
	double norm = sqrt(H.at<double>(0, 0)*H.at<double>(0, 0) +
		H.at<double>(1, 0)*H.at<double>(1, 0) +
		H.at<double>(2, 0)*H.at<double>(2, 0));
	H /= norm;
	cv::Mat c1 = H.col(0);
	cv::Mat c2 = H.col(1);
	cv::Mat c3 = c1.cross(c2);
	Tt = H.col(2);
	cv::Mat R(3, 3, CV_64F);
	for (int i = 0; i < 3; i++)
	{
		R.at<double>(i, 0) = c1.at<double>(i, 0);
		R.at<double>(i, 1) = c2.at<double>(i, 0);
		R.at<double>(i, 2) = c3.at<double>(i, 0);
	}
	//cout << "R (before polar decomposition):\n" << R << "\ndet(R): " << determinant(R) << endl;
	cv::Mat_<double> W, U, Vt;
	SVDecomp(R, W, U, Vt);
	Rr = U * Vt;
	double det = determinant(R);
	if (det < 0)
	{
		Vt.at<double>(2, 0) *= -1;
		Vt.at<double>(2, 1) *= -1;
		Vt.at<double>(2, 2) *= -1;
		Rr = U * Vt;
	}
	//cout << "R (after polar decomposition):\n" << R << "\ndet(R): " << determinant(R) << endl;
	//cv::Mat rvec;
	//Rodrigues(R, Rr);
}

/*
void draw_2D_trajectoru(const std::vector<Camera>& cams)
{
	const double *data = cams[0].T.ptr<double>(0);
	cv::Point3d c1(data[0], data[1], data[2]);
	cv::Mat field(800, 800, CV_8UC3, cv::Scalar(0, 0, 0));
	cv::circle(field, cv::Point2d(data[0], data[1]), 3, cv::Scalar(100, 100), 2);

	for (size_t i = 1; i < cams.size(); i++)
	{
		const double *data = cams[i].T.ptr<double>(0);
		cv::Point3d p(data[0], data[1], data[2]);
		//cout << p.x << "  " << p.y << endl;
		cv::Point c1i = static_cast<cv::Point2i>(cv::Point2d(c1.x, c1.y));
		cv::Point c2i = static_cast<cv::Point2i>(cv::Point2d(p.x, p.y));
		cv::line(field, c1i, c2i, cv::Scalar(100, 200, 220), 1);
		c1 = p;
	}

	cv::imshow("Trace", field);
	cv::waitKey(0);
}*/
/*
void save_traj_csv(const std::vector<Camera>& cams, const std::string &fn)
{
	ofstream fout(fn); // создаЄм объект класса ofstream дл€ записи и св€зываем его с файлом cppstudio.txt
	for (size_t i = 0; i < cams.size(); i++)
	{
		cv::Vec3d a = Mathematic::rotationMatrixToEulerAngles(cams[i].R);
		cv::Vec3d p(cams[i].T);
		fout << a[0] << ";" << a[1] << ";" << a[2] << ";" << p[0] << ";" << p[1] << ";" << p[2] << "\n";
	}
	fout.close();
}*/

void crop_props(std::vector<cv::Point2f> &v1, std::vector<cv::Point2f> &v2)
{
	for (size_t i = 0; i < v1.size(); ++i)
	{
		if (v1[i].x < 140 && v1[i].y < 140)
		{
			quick_remove_at(v1, i);
			quick_remove_at(v2, i);
			i--;
			continue;
		}
		if (v1[i].x > 600 && v1[i].y < 140)
		{
			quick_remove_at(v1, i);
			quick_remove_at(v2, i);
			i--;
			continue;
		}
		if (v1[i].y > 410)
		{
			quick_remove_at(v1, i);
			quick_remove_at(v2, i);
			i--;
			continue;
		}
	}
}


double med_tang(const vector<cv::Point2f> &v1, const vector<cv::Point2f> &v2)
{
	double r1 = 00;
	for (size_t i = 0; i < v1.size(); i++)
	{
		if (v2[i].y - v1[i].y != 0)
		{
			double a = (v2[i].x - v1[i].x) / (v2[i].y - v1[i].y);
			r1 += a;
		}
	}
	return r1 / (double)v1.size();
}

double med_len(const vector<cv::Point2f> &v1, const vector<cv::Point2f> &v2)
{
	double r1 = 00;
	for (size_t i = 0; i < v1.size(); i++)
	{
		double a = sqrtf((v2[i].x - v1[i].x)*(v2[i].x - v1[i].x) + (v2[i].y - v1[i].y)*(v2[i].y - v1[i].y));
		r1 += a;
	}
	return r1 / (double)v1.size();
}

double med_lr(const vector<cv::Point2f> &v1, const vector<cv::Point2f> &v2)
{
	double r1 = 00;
	int r2 = 0;
	for (size_t i = 0; i < v1.size(); i++)
	{
		if (v1[i].y > 200 && v1[i].y < 550)
		{
			r1 += (v2[i].x - v1[i].x);
			r2++;
		}		
	}
	return r1 / (double)r2;
}

std::pair<double, double> med_lr_pair(const vector<cv::Point2f> &v1, const vector<cv::Point2f> &v2)
{
	double r1l = 00;
	int r2l = 0;
	double r1r = 00;
	int r2r = 0;
	for (size_t i = 0; i < v1.size(); i++)
	{
		if (v1[i].y > 200 && v1[i].y < 550)
		{
			if (v1[i].x<360)
			{ 
				r1l += (v2[i].x - v1[i].x);
				r2l++;
			}
			else
			{
				r1r += (v2[i].x - v1[i].x);
				r2r++;
			}

		}
	}
	r1l= r1l / (double)r2l;
	r1r = r1r / (double)r2r;
	return std::pair<double, double>(r1l, r1r);
}

float dist(float p1, float p2)
{
	return sqrtf((p2 - p1)*(p2 - p1));
}

void matchFeaturing(const vector<cv::Point2f> &_v1, const vector<cv::Point2f> &_v2, vector<float> &features, bool mirr = false)
{
	features.clear();
	//первые 4 верх, последние низ. lx ly rx ry
	float ulx =0, uly = 0, urx = 0, ury = 0, dlx = 0, dly = 0, drx = 0, dry = 0;
	float sulx = 0, suly = 0, surx = 0, sury = 0, sdlx = 0, sdly = 0, sdrx = 0, sdry = 0;
	float qulx = 0, quly = 0, qurx = 0, qury = 0, qdlx = 0, qdly = 0, qdrx = 0, qdry = 0;
	int ulxi = 0, ulyi = 0, urxi = 0, uryi = 0, dlxi = 0, dlyi = 0, drxi = 0, dryi = 0;
	vector<cv::Point2f> v1 = _v1;
	vector<cv::Point2f> v2 = _v2;
	if (mirr)
	{
		for (size_t i = 0; i < v1.size(); i++)
		{
			v1[i].x = 720 - v1[i].x;
			v2[i].x = 720 - v2[i].x;
		}
	}
	for (size_t i = 0; i < v1.size(); i++)
	{
		if (v1[i].y < 240)
		{//верхн€€ половина
			if (v1[i].x < 360)//лева€
			{
				ulx += (v1[i].x - v2[i].x);
				sulx += v1[i].x;
				qulx += v1[i].x*v1[i].x;
				ulxi++;
				uly += (v1[i].y - v2[i].y);
				suly += v1[i].y;
				quly += v1[i].y*v1[i].y;
				ulyi++;
			}
			else
			{  //права€
				urx += (v1[i].x - v2[i].x);
				surx += v1[i].x;
				qurx += v1[i].x*v1[i].x;
				urxi++;
				ury += (v1[i].y - v2[i].y);
				sury += v1[i].y;
				qury += v1[i].y*v1[i].y;
				uryi++;
			}
		}
		else
		{//нижн€€
			if (v1[i].x < 360)//лева€
			{
				dlx += v1[i].x - v2[i].x;
				sdlx += v1[i].x;
				qdlx += v1[i].x*v1[i].x;
				dlxi++;
				dly += v1[i].y - v2[i].y;
				sdly += v1[i].y;
				qdly += v1[i].y*v1[i].y;
				dlyi++;
			}
			else
			{  //права€
				drx += v1[i].x - v2[i].x;
				sdrx += v1[i].x;
				qdrx += v1[i].x*v1[i].x;
				drxi++;
				dry += v1[i].y - v2[i].y;
				sdry += v1[i].y;
				qdry += v1[i].y*v1[i].y;
				dryi++;
			}
		}
	}//for

	if (ulxi != 0) { ulx = ulx / (float)ulxi; sulx = sulx / (float)ulxi; qulx = qulx / (float)ulxi;}
	if (ulyi != 0) { uly = uly / (float)ulyi; suly = suly / (float)ulyi; quly = quly / (float)ulyi;}
	if (urxi != 0) { urx = urx / (float)urxi; surx = surx / (float)urxi; qurx = qurx / (float)urxi;}
	if (uryi != 0) { ury = ury / (float)uryi; sury = sury / (float)uryi; qury = qury / (float)uryi;}
	if (dlxi != 0) { dlx = dlx / (float)dlxi; sdlx = sdlx / (float)dlxi; qdlx = qdlx / (float)dlxi;}
	if (dlyi != 0) { dly = dly / (float)dlyi; sdly = sdly / (float)dlyi; qdly = qdly / (float)dlyi;}
	if (drxi != 0) { drx = drx / (float)drxi; sdrx = sdrx / (float)drxi; qdrx = qdrx / (float)drxi;}
	if (dryi != 0) { dry = dry / (float)dryi; sdry = sdry / (float)dryi; qdry = qdry / (float)dryi;}

	features.push_back(ulx);
	features.push_back(uly);
	features.push_back(urx);
	features.push_back(ury);
	features.push_back(dlx);
	features.push_back(dly);
	features.push_back(drx);
	features.push_back(dry);
	/*features.push_back(sulx);
	features.push_back(suly);
	features.push_back(surx);
	features.push_back(sury);
	features.push_back(sdlx);
	features.push_back(sdly);
	features.push_back(sdrx);
	features.push_back(sdry);
	features.push_back(qulx);
	features.push_back(quly);
	features.push_back(qurx);
	features.push_back(qury);
	features.push_back(qdlx);
	features.push_back(qdly);
	features.push_back(qdrx);
	features.push_back(qdry);*/
	return;
}

uint64_t msec_from_string(std::string &str)
{
	istringstream iss(str);
	string s;
	vector<string> st;
	while (getline(iss, s, ' ')) {
		st.push_back(s);
	}
	istringstream iss2(st[1]);
	vector<string> st2;
	while (getline(iss2, s, ':')) {
		st2.push_back(s);
	}
	int val = atoi(st2[0].c_str());	
	uint64_t uv = (uint64_t)val;
	uint64_t t = uv*60*60*1000*1000;
	val = atoi(st2[1].c_str());
	t = t+ ((uint64_t)val) * 60 * 1000 * 1000;
	float valf = atof(st2[2].c_str());
	t = t + (uint64_t)(valf * 1000 * 1000);
	return t;
}

void make_hor_v(const std::vector<uint64_t> &ts, const std::vector<cv::Point2d> &p, std::vector<cv::Point2d> &vel)
{
	double dt = 0;
	double vx = 0, vy = 0;
	double a = 0.2;
	for (size_t i = 0; i < p.size() - 1; i++)
	{
		dt = ((double)(ts[i + 1] - ts[i])) / 1000000.0;
		if (dt > 0)
		{
			vx = vx * (1.0 - a) + a * (p[i + 1].x - p[i].x) / dt;
			vy = vy * (1.0 - a) + a * (p[i + 1].y - p[i].y) / dt;
		}
		vel.push_back(cv::Point2d(vx, vy));
	}
}

double rangle(double a, double b)
{
	double r = a - b;
	if (r > 3.1415) r = r - 2* 3.1415;
	if (r < -3.1415) r = r + 2* 3.1415;
	return r;
}

void make_ang_v(const std::vector<uint64_t> &ts, const std::vector<cv::Point3d> &p, std::vector<cv::Point3d> &vel, const std::vector<float> &h, std::vector<float> &h_vel)
{
	double dt = 0;
	double vx = 0, vy = 0, vz = 0, vh = 0;
	double a = 0.1;
	for (size_t i = 0; i < p.size() - 1; i++)
	{
		dt = ((double)(ts[i + 1] - ts[i])) / 1000000.0;
		if (dt > 0)
		{
			vx = vx * (1.0 - a) + a * rangle(p[i + 1].x , p[i].x) / dt;
			vy = vy * (1.0 - a) + a * rangle(p[i + 1].y , p[i].y) / dt;
			vz = vz * (1.0 - a) + a * rangle(p[i + 1].z , p[i].z) / dt;
			vh = vh * (1.0 - a) + a * (h[i + 1] - h[i]) / dt;			
		}
		vel.push_back(cv::Point3d(vx, vy, vz));
		h_vel.push_back(vh);
	}
}

void save_synchronized_data(const std::string& fn, const std::vector<uint64_t> &ts_pack, std::vector<cv::Point3d> &ang_vel, std::vector <float> &h, const vector<float> & vh, const std::vector<uint64_t> &ts_vel, std::vector<cv::Point2d> &vel, const std::vector<uint64_t> &ts_frame, vector<vector<float> > &features, vector<vector<float> > &features_mirr)
{
	ofstream fout(fn); 
	//каждому набору фич сопоставим набор данных
	size_t i_av = 0;
	size_t i_v = 0;
	fout <<"ts"<<";"<<"f[0]"<< ";" << "f[1]" << ";" << "f[2]" << ";" << "f[3]" << ";" << "f[4]" << ";" << "f[5]" << ";" << "f[6]" << ";" << "f[7]"<<
		//";"<< "f[8]" << ";" << "f[9]" << ";" << "f[10]" << ";" << "f[11]" << ";" << "f[12]" << ";" << "f[13]" << ";" << "f[14]" << ";" << "f[15]" <<
		//";"<<"f[16]" << ";" << "f[17]" << ";" << "f[18]" << ";" << "f[19]" << ";" << "f[20]" << ";" << "f[21]" << ";" << "f[22]" << ";" << "f[23]"<<
		";" << "ang_vel.x" << ";" << "ang_vel.y" << ";"<< "ang_vel.z" << ";" << "hv"<<";"<< "vel" << "\n";
		//";" << "hv" << "\n";
		//";" << "ang_vel.x" << ";" << "ang_vel.y" << ";" << "ang_vel.z" << ";" << "vel" << "\n";
	for (size_t i = 0; i < ts_frame.size(); i++)
	{
		std::vector<float > f = features[i];
		for (size_t j = 0; j < f.size(); j++)
		{
			if (i == 0) break;
			if (f[j] == 0) {
				f[j] = features[i - 1][j];//заполн€ем от предыдущих фич
				features[i][j] = features[i - 1][j];
			}
		}
		uint64_t cur_t = ts_frame[i];
		while (ts_pack[i_av] < cur_t)
		{
			i_av++;
			if (i_av > ts_pack.size() - 1)
			{
				i_av = ts_pack.size() - 1;
				break;
			}
		}
		cv::Point3d cur_ang_vel = ang_vel[i_av];
		float cur_h = h[i_av];
		float cur_hv = vh[i_av];
		while (ts_vel[i_v] < cur_t) 
		{
			i_v++;
			if (i_v > ts_vel.size() - 1)
			{
				i_v = ts_vel.size() - 1;
				break;
			}
		}
		cv::Point2d cur_vel = vel[i_v];		
		float velf = sqrt(cur_vel.x*cur_vel.x + cur_vel.y*cur_vel.y);

		fout << cur_t << ";" << f[0] << ";" << f[1] << ";" << f[2] << ";" << f[3] << ";" << f[4] << ";" << f[5] << ";" << f[6] << ";" << f[7] << ";" <<
			// f[8] << ";" << f[9] << ";" << f[10] << ";" << f[11] << ";" << f[12] << ";" << f[13] << ";" << f[14] << ";" << f[15] << ";" <<
			// f[16] << ";" << f[17] << ";" << f[18] << ";" << f[19] << ";" << f[20] << ";" << f[21] << ";" << f[22] << ";" << f[23] << ";" <<
			cur_ang_vel.x << ";" << cur_ang_vel.y << ";" << cur_ang_vel.z << ";" << cur_hv << ";" << velf <<  "\n";
			//cur_hv << "\n";
			//cur_ang_vel.x << ";" << cur_ang_vel.y << ";" << cur_ang_vel.z << ";" << velf << "\n";
	}
	///зеркалированные данные 

	i_av = 0;
	i_v = 0;
	for (size_t i = 0; i < ts_frame.size(); i++)
	{
		std::vector<float > f = features_mirr[i];
		for (size_t j = 0; j < f.size(); j++)
		{
			if (i == 0) break;
			if (f[j] == 0) {
				f[j] = features_mirr[i - 1][j];//заполн€ем от предыдущих фич
				features_mirr[i][j] = features_mirr[i - 1][j];
			}
		}
		uint64_t cur_t = ts_frame[i];
		while (ts_pack[i_av] < cur_t)
		{
			i_av++;
			if (i_av > ts_pack.size() - 1)
			{
				i_av = ts_pack.size() - 1;
				break;
			}
		}
		cv::Point3d cur_ang_vel = ang_vel[i_av];
		cur_ang_vel.z = -cur_ang_vel.z;
		cur_ang_vel.y = -cur_ang_vel.y;
		float cur_h = h[i_av];
		float cur_hv = vh[i_av];
		while (ts_vel[i_v] < cur_t)
		{
			i_v++;
			if (i_v > ts_vel.size() - 1)
			{
				i_v = ts_vel.size() - 1;
				break;
			}
		}
		cv::Point2d cur_vel = vel[i_v];
		float velf = sqrt(cur_vel.x*cur_vel.x + cur_vel.y*cur_vel.y);

		fout << cur_t << ";" << f[0] << ";" << f[1] << ";" << f[2] << ";" << f[3] << ";" << f[4] << ";" << f[5] << ";" << f[6] << ";" << f[7] << ";" <<
			// f[8] << ";" << f[9] << ";" << f[10] << ";" << f[11] << ";" << f[12] << ";" << f[13] << ";" << f[14] << ";" << f[15] << ";" <<
			// f[16] << ";" << f[17] << ";" << f[18] << ";" << f[19] << ";" << f[20] << ";" << f[21] << ";" << f[22] << ";" << f[23] << ";" <<
			cur_ang_vel.x << ";" << cur_ang_vel.y << ";" << cur_ang_vel.z << ";" << cur_hv << ";" << velf << "\n";
			//cur_hv << "\n";
			//cur_ang_vel.x << ";" << cur_ang_vel.y << ";" << cur_ang_vel.z << ";" << velf << "\n";
	}



	fout.close();
}




