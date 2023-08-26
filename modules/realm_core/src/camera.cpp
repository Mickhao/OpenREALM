
#include <cassert>

#include <opencv2/imgproc/imgproc.hpp>

// TODO: Update this to be conditional on OpenCV4
#include <opencv2/calib3d.hpp>

#include <realm_core/camera.h>

namespace realm
{
namespace camera
{

// CONSTRUCTION（构造函数）初始化
// 使用基础的摄像机参数构造函数

/*函数接受焦距(fx和fy)、主点(cx和cy)、图像的宽度和高度(img_width和img_height)作为参数
设置了成员变量m_do_undistort为false，表示是否执行畸变矫正
构建了内参矩阵m_K
设置了主对角线上的焦距和主点，其余元素为零
*/
Pinhole::Pinhole(double fx,
                 double fy,
                 double cx,
                 double cy,
                 uint32_t img_width,
                 uint32_t img_height)
    : m_do_undistort(false),
      m_width(img_width),
      m_height(img_height)
{
  // 设置内参矩阵 K
  m_K = cv::Mat_<double>(3, 3);
  m_K.at<double>(0, 0) = fx;
  m_K.at<double>(0, 1) = 0;
  m_K.at<double>(0, 2) = cx;
  m_K.at<double>(1, 0) = 0;
  m_K.at<double>(1, 1) = fy;
  m_K.at<double>(1, 2) = cy;
  m_K.at<double>(2, 0) = 0;
  m_K.at<double>(2, 1) = 0;
  m_K.at<double>(2, 2) = 1.0;
}
// 使用现有内参矩阵 K 的构造函数
Pinhole::Pinhole(const cv::Mat &K,
                 uint32_t img_width,
                 uint32_t img_height)
    : m_do_undistort(false),
      m_K(K),
      m_width(img_width),
      m_height(img_height)
{
}
// 使用内参矩阵 K 和畸变系数构造函数
Pinhole::Pinhole(const cv::Mat &K,
                 const cv::Mat &dist_coeffs,
                 uint32_t img_width,
                 uint32_t img_height)
    : m_do_undistort(false),
      m_K(K),
      m_width(img_width),
      m_height(img_height)
{
  //调用setDistortionMap函数设置畸变参数
  setDistortionMap(dist_coeffs);
}
// 拷贝构造函数
Pinhole::Pinhole(const Pinhole &that)
: m_do_undistort(that.m_do_undistort),
  m_R(that.m_R.clone()),
  m_t(that.m_t.clone()),
  m_K(that.m_K.clone()),
  m_width(that.m_width),
  m_height(that.m_height)
{
  //如果需要进行畸变矫正，克隆畸变系数和校正映射
  if (m_do_undistort)
  {
    m_dist_coeffs = that.m_dist_coeffs.clone();
    m_undistortion_map1 = that.m_undistortion_map1.clone();
    m_undistortion_map2 = that.m_undistortion_map2.clone();
  }
}
/*用于将一个Pinhole实例赋值给另一个实例
首先检查是否为自我赋值
然后，它将that实例的属性赋值给当前实例
*/
Pinhole& Pinhole::operator=(const Pinhole &that)
{
  if (this != &that)
  {
    m_do_undistort = that.m_do_undistort;
    m_R = that.m_R.clone();
    m_t = that.m_t.clone();
    m_K = that.m_K.clone();
    m_width = that.m_width;
    m_height = that.m_height;

//如果需要进行畸变矫正，克隆畸变系数和校正映射
    if (m_do_undistort) {
      m_dist_coeffs = that.m_dist_coeffs.clone();
      m_undistortion_map1 = that.m_undistortion_map1.clone();
      m_undistortion_map2 = that.m_undistortion_map2.clone();
    }
  }
  return *this;
}
//查询是否有畸变
bool Pinhole::hasDistortion() const
{
  return m_do_undistort;
}
//查询图像宽度
uint32_t Pinhole::width() const
{
  return m_width;
}
//查询图像高度
uint32_t Pinhole::height() const
{
  return m_height;
}
//查询焦距 fx
double Pinhole::fx() const
{
  assert(!m_K.empty() && m_K.type() == CV_64F);
  return m_K.at<double>(0, 0);
}
//查询焦距 fy
double Pinhole::fy() const
{
  assert(!m_K.empty() && m_K.type() == CV_64F);
  return m_K.at<double>(1, 1);
}
//查询主点 cx
double Pinhole::cx() const
{
  assert(!m_K.empty() && m_K.type() == CV_64F);
  return m_K.at<double>(0, 2);
}
//查询主点 cy
double Pinhole::cy() const
{
  assert(!m_K.empty() && m_K.type() == CV_64F);
  return m_K.at<double>(1, 2);
}
//查询畸变系数 k1
double Pinhole::k1() const
{
  assert(!m_dist_coeffs.empty() && m_dist_coeffs.type() == CV_64F);
  return m_dist_coeffs.at<double>(0);
}
//查询畸变系数 k2
double Pinhole::k2() const
{
  assert(!m_dist_coeffs.empty() && m_dist_coeffs.type() == CV_64F);
  return m_dist_coeffs.at<double>(1);
}
//查询畸变系数 p1
double Pinhole::p1() const
{
  assert(!m_dist_coeffs.empty() && m_dist_coeffs.type() == CV_64F);
  return m_dist_coeffs.at<double>(2);
}
//查询畸变系数 p2
double Pinhole::p2() const
{
  assert(!m_dist_coeffs.empty() && m_dist_coeffs.type() == CV_64F);
  return m_dist_coeffs.at<double>(3);
}
//查询畸变系数 k3
double Pinhole::k3() const
{
  assert(!m_dist_coeffs.empty() && m_dist_coeffs.type() == CV_64F);
  return m_dist_coeffs.at<double>(4);
}
//查询相机内参矩阵 K
cv::Mat Pinhole::K() const
{
  assert(!m_K.empty() && m_K.type() == CV_64F);
  return m_K.clone();
}
//查询投影矩阵 P
/*首先，检查外部旋转矩阵和平移向量是否都已设置
然后，计算从世界坐标系到相机坐标系的变换矩阵 T_w2c
最后，使用内参矩阵 K 和变换矩阵 T_w2c 计算投影矩阵 P
*/
cv::Mat Pinhole::P() const
{
  if (m_R.empty())
    throw(std::runtime_error("Error: Projection matrix could not be computed. Exterior rotation not set!"));
  if (m_t.empty())
    throw(std::runtime_error("Error: Projection matrix could not be computed. Exterior translation not set!"));

  cv::Mat T_w2c = Tw2c();

  cv::Mat P;
  hconcat(m_K * T_w2c.rowRange(0, 3).colRange(0, 3), m_K * T_w2c.rowRange(0, 3).col(3), P);
  return P;
}
//查询畸变系数矩阵
cv::Mat Pinhole::distCoeffs() const
{
  assert(!m_dist_coeffs.empty() && m_K.type() == CV_64F);

  return m_dist_coeffs.clone();
}
//查询外部旋转矩阵 R
cv::Mat Pinhole::R() const
{
  assert(!m_R.empty() && m_R.type() == CV_64F);

  return m_R.clone();
}
//查询外部平移向量 t
cv::Mat Pinhole::t() const
{
  assert(!m_t.empty() && m_t.type() == CV_64F);
  return m_t.clone();
}
//查询外部位姿矩阵 pose
cv::Mat Pinhole::pose() const
{
  //检查外部旋转矩阵和平移向量是否都已设置
  if (m_R.empty() || m_t.empty())
    return cv::Mat();
//合并成一个位姿矩阵 pose，并返回
  cv::Mat pose = cv::Mat_<double>(3, 4);
  m_R.copyTo(pose.rowRange(0, 3).colRange(0, 3));
  m_t.copyTo(pose.col(3));
  return std::move(pose);
}
//查询从世界坐标系到相机坐标系的变换矩阵 Tw2c
cv::Mat Pinhole::Tw2c() const
{
  //检查外部旋转矩阵和平移向量是否都已设置
  if (m_R.empty() || m_t.empty())
    return cv::Mat();
    //通过旋转矩阵 R 和平移向量 t 计算出变换矩阵 Tw2c，并返回
  cv::Mat T_w2c = cv::Mat::eye(4, 4, CV_64F);
  cv::Mat R_w2c = m_R.t();
  cv::Mat t_w2c = -R_w2c * m_t;
  R_w2c.copyTo(T_w2c.rowRange(0, 3).colRange(0, 3));
  t_w2c.copyTo(T_w2c.rowRange(0, 3).col(3));
  return T_w2c;
}

// Pose of camera in the world reference（相机在世界参考系中的姿态）
//用于生成从相机坐标系到世界坐标系的变换矩阵
cv::Mat Pinhole::Tc2w() const
{
  if (m_R.empty() || m_t.empty())
    return cv::Mat();

    //创建一个4x4的单位矩阵 T_c2w
  cv::Mat T_c2w = cv::Mat::eye(4, 4, CV_64F);
  //将旋转矩阵 m_R 复制到 T_c2w 的左上角的3x3子矩阵区域
  m_R.copyTo(T_c2w.rowRange(0, 3).colRange(0, 3));
  //将平移向量 m_t 复制到 T_c2w 的第四列的前三行
  m_t.copyTo(T_c2w.rowRange(0, 3).col(3));
  return T_c2w;
}

//用于设置镜头失真参数，并根据这些参数生成畸变矫正映射
void Pinhole::setDistortionMap(const double &k1,
                               const double &k2,
                               const double &p1,
                               const double &p2,
                               const double &k3)
{/*创建一个大小为 1x5 的双精度浮点数类型的矩阵
将输入的失真参数分别赋值给矩阵的相应位置
用 setDistortionMap(dist_coeffs) 函数
将上述生成的失真参数矩阵传递给另一个同名的函数
*/
  cv::Mat dist_coeffs(1, 5, CV_64F);
  dist_coeffs.at<double>(0) = k1;
  dist_coeffs.at<double>(1) = k2;
  dist_coeffs.at<double>(2) = p1;
  dist_coeffs.at<double>(3) = p2;
  dist_coeffs.at<double>(4) = k3;
  // Set undistortion map
  setDistortionMap(dist_coeffs);
}

//用于根据传入的失真参数矩阵生成畸变矫正映射
//并将映射设置到类的成员变量中
void Pinhole::setDistortionMap(const cv::Mat &dist_coeffs)
{
  assert(!dist_coeffs.empty() && dist_coeffs.type() == CV_64F);
  m_dist_coeffs = dist_coeffs;
  //调用 OpenCV 的函数 cv::initUndistortRectifyMap 来生成畸变矫正映射
  cv::initUndistortRectifyMap(m_K,
                              m_dist_coeffs,
                              cv::Mat_<double>::eye(3, 3),
                              m_K,
                              cv::Size(m_width, m_height),
                              CV_16SC2,
                              m_undistortion_map1,
                              m_undistortion_map2);
  m_do_undistort = true;//表示畸变矫正映射已经生成并设置
}

//用于设置相机的姿态信息
void Pinhole::setPose(const cv::Mat &pose)
{
  assert(!pose.empty() && pose.type() == CV_64F);
  assert(pose.rows == 3 && pose.cols == 4);
  m_t = pose.col(3).rowRange(0, 3);
  m_R = pose.colRange(0, 3).rowRange(0, 3);
}

//用于将相机模型按比例调整尺寸
Pinhole Pinhole::resize(double factor) const
{
  assert(!m_K.empty());
  cv::Mat K = m_K.clone();
  //通过乘以 factor 来调整内参矩阵的元素，实现缩放
  K.at<double>(0, 0) *= factor;
  K.at<double>(1, 1) *= factor;
  K.at<double>(0, 2) *= factor;
  K.at<double>(1, 2) *= factor;
  cv::Mat dist_coeffs = m_dist_coeffs.clone();
  //计算缩放后的图像宽度和高度
  auto width = static_cast<uint32_t>(std::round((double)m_width * factor));
  auto height = static_cast<uint32_t>(std::round((double)m_height * factor));
//使用缩放后的内参矩阵、畸变系数矩阵、以及新的图像宽度和高度，创建一个新的 Pinhole 实例 cam_resized
  Pinhole cam_resized(K, dist_coeffs, width, height);
  if (!m_R.empty() && !m_t.empty())
  {
    cam_resized.m_R = m_R.clone();
    cam_resized.m_t = m_t.clone();
  }

  return cam_resized;
}

//用于将相机模型调整为指定的图像尺寸
Pinhole Pinhole::resize(uint32_t width, uint32_t height)
{
  if (width == 0 || height == 0)
    throw(std::invalid_argument("Error resizing camera: Resizing to zero dimensions not permitted."));

    //计算宽度和高度的缩放比例因子 factor_x 和 factor_y，分别将目标尺寸除以原始尺寸
  double factor_x = static_cast<double>(width) / m_width;
  double factor_y = static_cast<double>(height) / m_height;
  //确保目标尺寸只能是原始尺寸的整数倍
  if (fabs(factor_x - factor_y) > 10e-2)
    throw(std::invalid_argument("Error resizing camera: Only multiples of the original image size are supported."));

  return resize(factor_x);
}

// FUNCTIONALITIES

//计算图像的畸变后的边界坐标，返回一个大小为 4x2 的矩阵，表示图像的四个边界点坐标
cv::Mat Pinhole::computeImageBounds2Ddistorted() const
{
  return (cv::Mat_<double>(4, 2)
      << 0, 0, 0, (double) m_height, (double) m_width, (double) m_height, (double) m_width, 0);
}

//计算图像的去畸变边界坐标
cv::Mat Pinhole::computeImageBounds2D() const
{
  cv::Mat img_bounds = computeImageBounds2Ddistorted();

  if (m_do_undistort)
  {
    img_bounds = img_bounds.reshape(2);
    cv::undistortPoints(img_bounds, img_bounds, m_K, m_dist_coeffs, cv::Mat(), m_K);
    img_bounds = img_bounds.reshape(1);
  }

  return img_bounds;
}

//将图像边界投影到世界坐标平面上
cv::Mat Pinhole::projectImageBoundsToPlane(const cv::Mat &pt, const cv::Mat &n) const
{
  assert(!m_R.empty() && !m_t.empty() && !m_K.empty());
  cv::Mat img_bounds = computeImageBounds2D();
  cv::Mat plane_points;
  for (uint i = 0; i < 4; i++)
  {
    cv::Mat u = (cv::Mat_<double>(3, 1) << img_bounds.at<double>(i, 0), img_bounds.at<double>(i, 1), 1.0);
    double s = (pt - m_t).dot(n) / (m_R * m_K.inv() * u).dot(n);
    cv::Mat p = m_R * (s * m_K.inv() * u) + m_t;
    cv::Mat world_point = (cv::Mat_<double>(1, 3) << p.at<double>(0), p.at<double>(1), p.at<double>(2));
    plane_points.push_back(world_point);
  }
  return plane_points;
}

//用于对输入图像进行去畸变操作
cv::Mat Pinhole::undistort(const cv::Mat &src, int interpolation) const
{
  // If undistortion is not neccessary, just return input img
  // Elsewise undistort img
  cv::Mat img_undistorted;
  if (m_do_undistort)
    cv::remap(src, img_undistorted, m_undistortion_map1, m_undistortion_map2, interpolation);
  else
    img_undistorted = src;
  return img_undistorted;
}

//计算投影到平面后的边界矩形区域
cv::Rect2d Pinhole::projectImageBoundsToPlaneRoi(const cv::Mat &pt, const cv::Mat &n) const
{
  cv::Mat bounds_in_plane = projectImageBoundsToPlane(pt, n);
  double roi_l, roi_r;
  double roi_u, roi_d;
  cv::minMaxLoc(bounds_in_plane.col(0), &roi_l, &roi_r);
  cv::minMaxLoc(bounds_in_plane.col(1), &roi_d, &roi_u);
  return cv::Rect2d(roi_l, roi_d, roi_r-roi_l, roi_u-roi_d);
}

//将图像上的一个点投影到世界坐标系中
cv::Mat Pinhole::projectPointToWorld(double x, double y, double depth) const
{
  if (depth <= 0.0)
    return cv::Mat();
  cv::Mat point(4, 1, CV_64F);
  point.at<double>(0) = (x - m_K.at<double>(0, 2)) * depth / m_K.at<double>(0, 0);
  point.at<double>(1) = (y - m_K.at<double>(1, 2)) * depth / m_K.at<double>(1, 1);
  point.at<double>(2) = depth;
  point.at<double>(3) = 1;
  return Tc2w()*point;
}

} // namespace camera

} // namespace realm