#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

template <typename T>
void quick_remove_at(std::vector<T> &v, std::size_t idx)
{
	if (idx < v.size()) {
		v[idx] = std::move(v.back());
		v.pop_back();
	}
}

class GetMatches
{
	int method;
	double qualityLevel;
	cv::TermCriteria termcrit;
	cv::Size subPixWinSize;
	cv::Size winSize;
	std::vector<cv::Point2f> points[2], re_points[2];
	std::vector<cv::Point2f> pts_cleared[2];
	std::vector<uchar> re_status, status;
	std::vector<float> re_err, err;
	cv::Mat img1;// , img2;
	uint64_t ts1;
	//cv::Mat frame, frame2;
	size_t max_pts ;
	size_t num_v;
	size_t num_mutches;
	size_t num_seed;
	size_t num_mtch;
	//bool initi;
	float timeout; // time after which we generate matches anyway even if they are shorter than needed
	float minMatchesLengthPixels; // minimum length of matches to report matches
	float dist[5];
	std::vector<float> distort;
	float cam_matrix[9];
	float /*cv::Scalar*/ koef;
	cv::Scalar mean1, stdev1, mean2, stdev2;
	void back_check(const std::vector<cv::Point2f> &v1, const std::vector<cv::Point2f> &v2, std::vector<uchar> &status);
	void point_generate(const cv::Mat &img, const size_t num_pt, std::vector<cv::Point2f> &points);
	void init(const cv::Mat & img, uint64_t ts2);
	void load_dist_k(const std::string &f_name_k, const std::string &f_name_m);

	bool write_log_file_status;
	std::string logpath;
	

public:
  cv::Mat get_img1() {
	  return img1;
	  }
  void do_full_undistort(
	  int dec, int cam_width, int cam_height, 
	  const std::vector<cv::Point2f> &v1, const std::vector<cv::Point2f> &v2, 
	  std::vector<cv::Point2f> &vp1, std::vector<cv::Point2f> &vp2) const;
  void do_full_undistort(
	  int dec, int cam_width, int cam_height, 
	  const std::vector<cv::Point2f> &v1, const std::vector<cv::Point2f> &v2, 
	  std::vector<cv::Point2f> &vp1, std::vector<cv::Point2f> &vp2, 
	  std::vector<float> &matches_err, std::vector<float> &inv_err) const;

  const cv::Mat &getBaseImage() const { return img1; }
  void setMinMatchesLength(float pixels) { minMatchesLengthPixels = pixels; }
  float getMinMatchesLength() const { return minMatchesLengthPixels; }
  void setTimeOut(float t_sec) { timeout = t_sec; }
  float getTimeOut() const { return timeout; }
  void setQualityLevel(double q) { qualityLevel = q; }
  double getQualityLevel() const { return qualityLevel; }
  void undistort(const std::vector<cv::Point2f> &distorted, std::vector<cv::Point2f> &undistorted) const;
  GetMatches(const std::string &dist_file , const std::string &intr_file);
  void set_max_pts(const size_t _max_pts)
  {
	  max_pts = _max_pts;
	}
	void set_method(int i)
	{
		method = i;
	}
//	bool get_matches(const cv::Mat & img, uint64_t ts2, std::vector<Point2Df> &v1, 
//		std::vector<Point2Df> &v2, size_t num_seg);
	std::pair<bool, uint64_t/*base frame timestamp*/> get_matches(
		const cv::Mat & img, uint64_t ts2, 
		std::vector<cv::Point2f> &v1,std::vector<cv::Point2f> &v2, 
		size_t num_seg, std::vector<float>& matches_err, std::vector<float>& inv_err);
	void print_test_result();

	std::pair<bool, uint64_t /*base frame timestamp*/> get_matches2(
		const cv::Mat &img, uint64_t ts2, 
		std::vector<cv::Point2f> &v1, std::vector<cv::Point2f> &v2, 
		size_t num_seg, std::vector<float> &matches_err, std::vector<float> &inv_err);

	void generate_matches_on_fullframe(cv::Mat img2, std::vector<cv::Point2f> pts_cleared,
		std::vector<cv::Point2f>& points);
	void generate_matches_on_cropedframes(cv::Mat img2, std::vector<cv::Point2f> &pts_cleared,
		std::vector<cv::Point2f>& points);
	static void filtering(
		std::vector <cv::Point2f> &v1, 
		std::vector <cv::Point2f> &v2,
	//	std::vector <float> &matches_err,
	//	std::vector <float> &inv_err,
		int width, int height, int gap = 0);
	void draw(cv::Mat &img,
		const std::vector<cv::Point2f> &v1,
		const std::vector<cv::Point2f> &v2, 
		cv::Point2f offset = cv::Point2f(0, 0),
		cv::Scalar colorLine = CV_RGB(255,150,50), 
		cv::Scalar colorCircleStart = CV_RGB(0,0,255),
		cv::Scalar colorCircleEnd = CV_RGB(255, 0, 0),
		int circleR = 3
		) const;
	void setCxy(int cx, int cy) { setCxy(float(cx), float(cy)); }
	void setCxy(float cx, float cy) { cam_matrix[2] = cx; cam_matrix[5] = cy; }
	std::pair<float, float> getCxy() { return std::make_pair(cam_matrix[2], cam_matrix[5]); }

	void setFx(float fx) { cam_matrix[0] = fx; cam_matrix[4] = fx; }
	float getFx() { return cam_matrix[0]; }

	void setDistCoefs(std::vector<float> dist) { for(size_t i=0; i<dist.size();++i) distort[i] = dist[i];	}
	std::vector<float> getDistCoefs() const { return distort; }


};





