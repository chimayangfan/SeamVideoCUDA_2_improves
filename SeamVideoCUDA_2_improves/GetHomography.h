#ifndef GETHOMOGRAPHY_H
#define GETHOMOGRAPHY_H




#include <opencv2\core\core.hpp>
#include <vector>
//���㵥Ӧ�Ծ���
cv::Mat computeHomography(std::vector<cv::Point2f> src, std::vector<cv::Point2f> dst, double err_tol, bool& flag, std::vector<std::vector<cv::Point2f> >& consensus);
//�����ĽǶ���
void calcFourCorners(cv::Mat& H, cv::Point& leftTop, cv::Point& leftBottom, cv::Point& rightTop, cv::Point& rightBottom, cv::Mat& img2);
//����Ҫ������ٵ�һ���Լ���Ԫ�ظ���
int calc_min_inliers(int n, int m, double p_badsupp, double p_badxform);
//����n�Ľ׳˵���Ȼ����
double log_factorial(int n);
// ���ݼ�point, ��������n, ������samp, ���ѡȡ������
void getSample(std::vector< std::vector<cv::Point2f> >& point, int n, std::vector< std::vector<cv::Point2f> >& samp);
//���������㵥Ӧ����
cv::Mat getHomoMatrix(std::vector<std::vector<cv::Point2f> >& sample);
//�ɵ�Ӧ�Ծ����ҳ�һ���Լ�
int findConsensus(std::vector<std::vector<cv::Point2f> >& point, std::vector<std::vector<cv::Point2f> >& consensus, cv::Mat H, double err_tol);

#endif