#include "GetMatches.h"
#include <numeric>

#include <fstream>

GetMatches::GetMatches(const std::string &dist_file ,const std::string &intr_file )
{
	timeout = 0.1f; // 0.1 sec , time after which we generate matches anyway even if they are shorter than needed
	minMatchesLengthPixels = 0.0f; // minimum length of matches to report matches
	qualityLevel = 0.1; ;// 0.1f;// 001;//0.1;
	termcrit = cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 20, 0.01);
	subPixWinSize = cv::Size(5, 5);
	winSize = cv::Size(21, 21);
	method = 1;
	max_pts = 200;
	num_v = 0;
	num_mtch = 0;
	num_seed = 0;
	load_dist_k(dist_file,intr_file);
	for (size_t i = 0; i < 5; i++)distort.push_back(dist[i]);
}

void GetMatches::do_full_undistort(
	int dec,int cam_width, int cam_height, 
	const std::vector<cv::Point2f>& v1, const std::vector<cv::Point2f>& v2, 
	std::vector<cv::Point2f>& vp1, std::vector<cv::Point2f>& vp2
)const
{
	
	vp1.clear();
	vp2.clear();
	undistort(v1, vp1);
	undistort(v2, vp2);			
	const int side_gap = 10;

	filtering(vp1, vp2, cam_width, cam_height, side_gap);
}

void GetMatches::do_full_undistort(
	int dec, int cam_width, int cam_height, 
	const std::vector<cv::Point2f>& v1, const std::vector<cv::Point2f>& v2,
	std::vector<cv::Point2f>& vp1, std::vector<cv::Point2f>& vp2
	, std::vector<float>& matches_err, std::vector<float>& inv_err)const
{
	vp1.clear();
	vp2.clear();
	undistort(v1, vp1);
	undistort(v2, vp2);
	const int side_gap = 10;//эта ветка рабочая
	filtering(vp1, vp2,  cam_width, cam_height, side_gap);
}

void GetMatches::undistort(const std::vector<cv::Point2f> & in, std::vector<cv::Point2f> & out) const
{
	const size_t size = in.size();	
	out.resize(size);
	if (size == 0) return;

	float cam_matrix_1[9] = { 
		cam_matrix[0],	0.0f,			cam_matrix[2], 
		0.0f,			cam_matrix[4],	cam_matrix[5], 
		0.0f,			0.0f,			1.0f };
	cv::Mat cam_matrix_mat(3, 3, CV_32FC1, cam_matrix_1);
	cv::undistortPoints(cv::Mat(in), cv::Mat(out), cam_matrix_mat, distort, cv::noArray(), cam_matrix_mat);
	return;
}

void GetMatches::load_dist_k(const std::string &f_name_k, const std::string &f_name_m)
{
	printf("try_intrinsic_parameters_load\n");
	std::ifstream file(f_name_k, std::ios::in);  //открыли файл
	char str[16384];
	float temp;
	int i = 0;
	while (file.good()) // перебор строк до конца файла
		{
			file.getline(str, 200, '\n');
			std::sscanf(str, "%f", &temp);
			dist[i] = temp;
			i++;
			if (i >= 5) break;
		}
	std::ifstream file2(f_name_m, std::ios::in);   //открыли файл

	i = 0;
	while (file2.good()) // перебор строк до конца файла
	{
		file2.getline(str, 200, '\t');
		std::sscanf(str, "%f", &temp);
		cam_matrix[i] = temp;
		i++;
		file2.getline(str, 200, '\t');
		std::sscanf(str, "%f", &temp);
		cam_matrix[i] = temp;
		i++;
		file2.getline(str, 200, '\n');
		std::sscanf(str, "%f", &temp);
		cam_matrix[i] = temp;
		i++;
		if (i >= 9) break;
	}
	
	printf("intrinsic_parameters_loaded\n");

}

void GetMatches::back_check(
	const std::vector<cv::Point2f> &v1, 
	const std::vector<cv::Point2f> &v2, 
	std::vector<uchar> &status)
{
	float delta = 0.2f;
	for (size_t i = 0; i < v1.size(); i++)
	{
		if (status[i] == 1) if (fabsf(v1[i].x - v2[i].x) > delta || fabsf(v1[i].y - v2[i].y) > delta) {
				status[i] = 0;
			}
	}
}

void GetMatches::point_generate(const cv::Mat &img, const size_t num_pt, std::vector<cv::Point2f> &points2)
{
	cv::Mat mask;
	std::vector<cv::Point2f> points;

	cv::goodFeaturesToTrack(img, points, (int)num_pt, qualityLevel, 20, mask, 3, false, 0.04); //false
	if (points.size() > 0)	cv::cornerSubPix(img, points, subPixWinSize, cv::Size(-1, -1), termcrit);
	for (size_t i = 0; i < points.size(); i++)
	{
		if (points[i].x < 0 || points[i].x>img.cols || points[i].y < 0 || points[i].y>img.rows)
		{
			//printf("MAtch out fild!!!\n");
		}	
		else points2.push_back(points[i]);			
	}
}

void GetMatches::init(const cv::Mat & frame2, uint64_t ts2)
{
	//cv::cvtColor(frame2, img1, cv::COLOR_BGR2GRAY);
	img1 = frame2.clone();
	ts1 = ts2;
}


void GetMatches::generate_matches_on_fullframe(cv::Mat img2,
	std::vector<cv::Point2f> pts_cleared, std::vector<cv::Point2f>& points)
{
	std::vector<cv::Point2f> pts_candidate;
	points = pts_cleared;

	size_t ll = points.size();
	if (ll < max_pts) //генерим точки
	{
		point_generate(img2, max_pts - ll, pts_candidate);
	}
	for (size_t i = 0; i < pts_candidate.size(); i++) points.push_back(pts_candidate[i]);

	std::vector<size_t> matches_counter;
	size_t n = 3;
	size_t width = img1.cols / n;
	size_t height = img1.rows / n;

	for (size_t col = 0; col < n; col++)
	{
		for (size_t row = 0; row < n; row++)
		{
			size_t x = col * width;
			size_t y = row * height;
			size_t count = 0;
			for (size_t i = 0; i < points.size(); i++)
			{
				if ((points[i].x >= x) && (points[i].x <= x + width)
					&& (points[i].y >= y) && (points[i].y <= y + height))
					count++;
			}
			matches_counter.push_back(count);
		}
	}

	num_seed += points.size();
	num_mtch += ll;
}

void GetMatches::generate_matches_on_cropedframes(cv::Mat img2,
	std::vector<cv::Point2f> &pts_cleared, std::vector<cv::Point2f>& points)
{
	std::vector<cv::Point2f> tmp_points;
	std::vector<size_t> matches_counter;
	const size_t n = 3;
	const int mp = int(max_pts / (n*n));
	points = pts_cleared;
	const size_t width = img1.cols / n;
	const size_t height = img1.rows / n;
	std::vector<cv::Point2f> pts_candidate;
	
	for (size_t i = 0; i < points.size(); i++)
	{
		if ((points[i].x >= img1.cols) || (points[i].x < 0) || 
			(points[i].y >= img1.rows) || (points[i].y < 0))
			quick_remove_at<cv::Point2f>(points, i); 			
	}
	for (size_t col = 0; col < n; col++)
	{
		for (size_t row = 0; row < n; row++)
		{
			const int x = (int)(col * width);
			const int y = (int)(row * height);
			size_t count = 0;
		
			for (size_t i = 0; i < points.size(); i++)
			{
				if ((points[i].x >= (float) x) && (points[i].x < (float)(x + width))
					&& (points[i].y >= (float)y) && (points[i].y < (float)(y + height)))
					count++;
				
				if (count > mp)
				{
					quick_remove_at<cv::Point2f>(points, i);
					count--;
				}
			}
			matches_counter.push_back(count);
		}
	}
	//	printf("mtch ");
	//	for (size_t i = 0; i < 9; i++) printf(" %d ", matches_counter[i]);
	//	printf("   total %d pts_sz %d \n", std::accumulate(matches_counter.begin(),
	//	matches_counter.end(), 0), points.size());

	//cv::imshow("sdsdf", img2);
	//cv::waitKey(0);
	
	for (size_t col = 0; col < n; col++)
	{
		for (size_t row = 0; row < n; row++)
		{
			int x = (int)(col * width);
			int y = (int)(row * height);
			cv::Mat cropedImage = img2(cv::Rect(x, y, (int)width, (int)height));
			size_t ll = matches_counter[n * col + row];
		//	printf("ll %d |",ll);
			
			if (ll < mp) //генерим точки
			{
				//printf("p_gen %d ", n * col + row);
				point_generate(cropedImage, mp - ll, pts_candidate);
			//	printf(" g_%d \n", pts_candidate.size());
			}

			for (size_t i = 0; i < pts_candidate.size(); i++)
			{
				pts_candidate[i].x += (float)x;
				pts_candidate[i].y += (float)y;
				points.push_back(pts_candidate[i]);
			//	printf(" add_pts \n");
			}
			num_seed += points.size();
			num_mtch += ll;
			pts_candidate.clear();
		}
	}
	//printf(" pts %d\n", points.size());
}

std::pair<bool, uint64_t /*base frame timestamp*/> GetMatches::get_matches2(
	const cv::Mat &img2,
	uint64_t ts2,
	std::vector<cv::Point2f> &v1,
	std::vector<cv::Point2f> &v2,
	size_t num_seg,
	std::vector<float> &matches_err, 
	std::vector<float> &inv_err
		)
{
	if (img1.empty())
	{
		init(img2, ts2);
		return {false, 0};
	}

	v1.clear();
	v2.clear();
	status.clear();
	err.clear();
	re_status.clear();
	re_err.clear();
	re_points[0].clear();
	re_points[1].clear();
	points[1].clear();
	points[0].clear();
	pts_cleared[0].clear();
	pts_cleared[1].clear();

	//vector<cv::Point2f> pts_candidate;
	point_generate(img1, max_pts, points[0]);
	cv::calcOpticalFlowPyrLK(img1, img2, points[0], points[1], status, err, winSize, 3, termcrit, cv::OPTFLOW_LK_GET_MIN_EIGENVALS, 0.0001);
	cv::calcOpticalFlowPyrLK(img2, img1, points[1], re_points[0], re_status, re_err, winSize, 3, termcrit, cv::OPTFLOW_LK_GET_MIN_EIGENVALS, 0.00001);
	back_check(re_points[0], points[0], status);
	for (size_t i = 0; i < points[0].size(); i++)
	{
		if (status[i])
		{
			//pts_cleared[1].push_back(points[1][i]);
			v1.push_back(points[0][i]);
			v2.push_back(points[1][i]);
			//matches_err.push_back(err[i]);
			//inv_err.push_back(re_err[i]);
		}
	}

	auto base_ts = ts1;
	if (v1.size()>10)
	{
		ts1 = ts2;
		img1 = img2.clone();
		return {true, base_ts};
	}
	if ((ts2 - ts1) > timeout * 1000)
	{
		ts1 = ts2;
		img1 = img2.clone();
		return {true, base_ts};
	}
	else
	{
		v1.clear();
		v2.clear();
		matches_err.clear();
		inv_err.clear();
		return {false, 0};
	}
	
}

std::pair<bool, uint64_t/*base frame timestamp*/> GetMatches::get_matches(
	const cv::Mat & img2, 
	uint64_t ts2, 
	std::vector<cv::Point2f> &v1, 
	std::vector<cv::Point2f> &v2,
	size_t num_seg,
	std::vector<float>& matches_err, 
	std::vector<float>& inv_err)
{
	if (img1.empty())
	{
		init(img2, ts2);
		return { false, 0 };		
	}
	
	v1.clear(); v2.clear();
	status.clear(); err.clear();
	re_status.clear(); re_err.clear();
	re_points[0].clear();
	points[1].clear();
	pts_cleared[0].clear();
	pts_cleared[1].clear();
	//printf("met = %d\n", method);
	if (method == 1)
	{
		if (num_seg == 1)
		{
			generate_matches_on_fullframe(img2, points[0], points[0]);
		}
		else
			generate_matches_on_cropedframes(img2, points[0], points[0]);

		if (points[0].size() > 0)
		{
			/*cv::meanStdDev(img1, mean1, stdev1);
			cv::meanStdDev(img2, mean2, stdev2);

			if (mean2[0] > 0 && stdev2[0] > 0)
			{			
				koef = float(mean1[0] / mean2[0]);
				if (fabsf(1 - koef) > 0.05f)
				{
					//printf("mean1 = %f, mean2 = %f, stdev1 = %f, stdev2 = %f\n", mean1[0], mean2[0], stdev1[0], stdev2[0]);
					img1 = img1 / koef;
				}
			}*/

			cv::calcOpticalFlowPyrLK(
				img1, img2, points[0], points[1],
				status, err, winSize, 3, termcrit, cv::OPTFLOW_LK_GET_MIN_EIGENVALS, 0.001); //
			cv::calcOpticalFlowPyrLK(
				img2, img1,  points[1], re_points[0],
				re_status, re_err, winSize, 3, termcrit, cv::OPTFLOW_LK_GET_MIN_EIGENVALS, 0.001);
			for (size_t i = 0; i < status.size(); i++) if (re_status[i] == 0) status[i] = 0;
		
			back_check(re_points[0], points[0], status);
			//draw(img1, points[0], points[1], cv::Point2f(), CV_RGB(55, 55, 250), 0, 0, 0);
			//cv::Mat tmp = img2;
			//draw(tmp, points[1], re_points[0], cv::Point2f(), CV_RGB(55, 55, 250), 0, 0, 0);
			//cv::imshow("img1", img1);
			//cv::imshow("img2", tmp);
			//cv::waitKey(0);
		}

		int match_counter = 0;
		double distance = 0;
		
		for (size_t i = 0; i < points[0].size(); i++)
		{
			if (status[i]) 
			{
				distance += sqrt((points[0][i].x - points[1][i].x)*(points[0][i].x - points[1][i].x)
					+ (points[0][i].y - points[1][i].y)*(points[0][i].y - points[1][i].y));
				
				pts_cleared[1].push_back(points[1][i]);
				v1.push_back(points[0][i]);
				v2.push_back(points[1][i]);
				matches_err.push_back(err[i]);
				inv_err.push_back(re_err[i]);
				
				match_counter++;
			}
		}
		/*
		if (num_seg == 1)
		{
			generate_matches_on_fullframe(img2, points[1], points[0]);
		}
		else
			generate_matches_on_cropedframes(img2, points[1], points[0]);
*/
		if (match_counter != 0)
		{
			distance /= match_counter;
		}
		match_counter = 0;

		/////1111
		auto base_ts = ts1;
		ts1 = ts2;
		img1 = img2.clone();
		points[0] = v2;
		return { true, base_ts };

		/////2222
		/*auto base_ts = ts1;
		if (distance >= minMatchesLengthPixels)
		{
			ts1 = ts2;
			img1 = img2.clone();
			points[0] = v2;
			return { true, base_ts };
		}
		else if((ts2 - ts1) > timeout * 1000000) // it was 1000 
		{
			ts1 = ts2;
			img1 = img2.clone();
			points[0] = v2;
			return { true, base_ts };
		}
		else
		{
			v1.clear();
			v2.clear();
			matches_err.clear();
			inv_err.clear();
			return { false, 0 };
		}*/
		//////--2222
	}
	if (method == 2)
	{
		if (points[0].size() > 0)
		{
			cv::calcOpticalFlowPyrLK(img1, img2, points[0], points[1], 
				status, err, winSize, 3, termcrit, cv::OPTFLOW_LK_GET_MIN_EIGENVALS, 0.0001);
		}
		for (size_t i = 0; i < points[0].size(); i++)
		{
			if (status[i])
			{
				pts_cleared[1].push_back(points[1][i]);
				//pts_cleared[0].push_back(points[0][i]);
				//заполним выдачу
				v1.push_back(points[0][i]);
				v2.push_back(points[1][i]);
			}
		}

		points[0] = pts_cleared[1];

		std::vector<cv::Point2f> pts_candidate;
		size_t ll = points[0].size();
		if (ll < max_pts) //генерим точки
			{
				point_generate(img2, max_pts - ll, pts_candidate);
			}
		for (size_t i = 0; i < pts_candidate.size(); i++) points[0].push_back(pts_candidate[i]);
		num_seed += points[0].size();
		num_mtch += ll;
		img2.copyTo(img1);		
		return { true, ts1 };
	}
	return { false, 0 };
}


void GetMatches::draw(cv::Mat &img,
	const std::vector<cv::Point2f> &v1,
	const std::vector<cv::Point2f> &v2, 
	cv::Point2f offset,
	cv::Scalar colorLine, 
	cv::Scalar colorCircleStart,
	cv::Scalar colorCircleEnd,
	int circleR
	) const
{
	if (v2.size() != v1.size()) throw std::runtime_error("GetMaches::draw non same size of matches points");
	for (size_t j = 0; j < v1.size(); j++)
	{
		if (circleR) {
			cv::circle(img, v1[j] + offset, circleR, colorCircleStart, -1);
			cv::circle(img, v2[j] + offset, circleR, colorCircleEnd, -1);
		}
		cv::arrowedLine(img, v1[j]+offset, v2[j]+offset, colorLine, 1, cv::LINE_AA);
		/*if (sqrtf((v1[j].x - v2[j].x)*(v1[j].x - v2[j].x) + (v1[j].y - v2[j].y)*(v1[j].y - v2[j].y)) > 100.f)
			std::cout << "AHTUNG_long_matches!!\n";*/
	}
}
	
void GetMatches::print_test_result()
{
	float pp = ((float)num_mtch) / ((float)num_seed);
	printf(" survivability %f \n", pp);

}

void GetMatches::filtering(
	std::vector <cv::Point2f> &v1, std::vector <cv::Point2f> &v2, 
		int width, int height, int gap)
{
	for (int i = 0; i < (int)v1.size(); i++)
	{
		if ((v1[i].x < gap || v1[i].x>width - gap
			|| v1[i].y < gap || v1[i].y>height - gap
			|| v2[i].x < gap || v2[i].x>width - gap
			|| v2[i].y < gap || v2[i].y>height - gap)/*||(sqrtf((v1[i].x - v2[i].x)*(v1[i].x - v2[i].x) + (v1[i].y - v2[i].y)*(v1[i].y - v2[i].y)) > 300.f)*/)
		{
			quick_remove_at<cv::Point2f>(v1, i);
			quick_remove_at<cv::Point2f>(v2, i);
			--i;
		}
	}
	
}

