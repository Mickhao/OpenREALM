

#include <realm_core/loguru.h>
#include <realm_ortho/rectification.h>

/*图像校正*/
using namespace realm;


//对给定的帧进行图像校正
CvGridMap::Ptr ortho::rectify(const Frame::Ptr &frame)
{
  // Check if all relevant layers are in the observed map
  //从输入的帧中获取表面模型的指针
  CvGridMap::Ptr surface_model = frame->getSurfaceModel();

  //检查表面模型中是否包含名为 "elevation" 的图层，并且图层的类型是否为 CV_32F
  if (!surface_model->exists("elevation") || (*surface_model)["elevation"].type() != CV_32F)
    throw(std::invalid_argument("Error: Layer 'elevation' does not exist or type is wrong."));

  // Apply rectification using the backprojection from grid
  //图像校正
  CvGridMap::Ptr rectification =
      backprojectFromGrid(
        //获取去畸变后的图像
          frame->getImageUndistorted(),
         //获取相机参数
          *frame->getCamera(),
          //获取表面高度数据
          surface_model->get("elevation"),
          //获取表面模型的感兴趣区域
          surface_model->roi(),
          //获取表面模型的分辨率
          surface_model->resolution(),
          //获取表面模型的假设
          frame->getSurfaceAssumption() == SurfaceAssumption::ELEVATION
          );

  return rectification;
}

//将高度图数据从地面坐标系反投影到图像坐标系，生成校正后的图像
CvGridMap::Ptr ortho::backprojectFromGrid(
    const cv::Mat &img,
    const camera::Pinhole &cam,
    cv::Mat &surface,
    const cv::Rect2d &roi,
    double GSD,
    bool is_elevated,
    bool verbose)
{
  // Implementation details:
  // Implementation is chosen as a compromise between readability and performance. Especially the raw array operations
  // could be implemented in opencv. However depending on the resolution of the surface grid and the image the loop
  // iterations can go up to several millions. To keep the computation time as low as possible for this performance sink,
  // the style is as follows

  // Prepare projection, use raw arrays for performance
  //获取投影矩阵 P
  cv::Mat cv_P = cam.P();
  double P[3][4] = {cv_P.at<double>(0, 0), cv_P.at<double>(0, 1), cv_P.at<double>(0, 2), cv_P.at<double>(0, 3),
                    cv_P.at<double>(1, 0), cv_P.at<double>(1, 1), cv_P.at<double>(1, 2), cv_P.at<double>(1, 3),
                    cv_P.at<double>(2, 0), cv_P.at<double>(2, 1), cv_P.at<double>(2, 2), cv_P.at<double>(2, 3)};

  // Prepare elevation angle calculation
  //获取相机的位置 t
  cv::Mat t_pose = cam.t();
  double t[3] = {t_pose.at<double>(0), t_pose.at<double>(1), t_pose.at<double>(2)};

  uchar is_elevated_val    = (is_elevated ? (uchar)0 : (uchar)255);
  cv::Mat valid            = (surface == surface);                      // Check for NaN values in the elevation(用于检查 surface 中是否包含 NaN)
  cv::Mat color_data       = cv::Mat::zeros(surface.size(), CV_8UC4);   // contains BGRA color data(用于保存校正后的图像颜色数据)
  cv::Mat elevation_angle  = cv::Mat::zeros(surface.size(), CV_32FC1);  // contains the observed elevation angle(用于保存观测到的地面高度角度数据)
  cv::Mat elevated         = cv::Mat::zeros(surface.size(), CV_8UC1);   // flag to set wether the surface has elevation info or not(表示地面是否提升)
  cv::Mat num_observations = cv::Mat::zeros(surface.size(), CV_16UC1);  // number of observations, should be one if it's a valid surface point(观测数目的计数)

  LOG_IF_F(INFO, verbose, "Processing rectification:");
  LOG_IF_F(INFO, verbose, "- ROI (%f, %f, %f, %f)", roi.x, roi.y, roi.width, roi.height);
  LOG_IF_F(INFO, verbose, "- Dimensions: %i x %i", surface.rows, surface.cols);

  // Iterate through surface and project every cell to the image
  //迭代地面高度数据 surface
  for (uint32_t r = 0; r < surface.rows; ++r)
    for (uint32_t c = 0; c < surface.cols; ++c)
    {
      if (!valid.at<uchar>(r, c))
      {
        continue;
      }
      //计算地面点在图像中的投影坐标 (x, y)
      auto elevation_val = static_cast<double>(surface.at<float>(r, c));

      double pt[3]{roi.x+(double)c*GSD, roi.y+roi.height-(double)r*GSD, elevation_val};
      double z = P[2][0]*pt[0]+P[2][1]*pt[1]+P[2][2]*pt[2]+P[2][3]*1.0;
      double x = (P[0][0]*pt[0]+P[0][1]*pt[1]+P[0][2]*pt[2]+P[0][3]*1.0)/z;
      double y = (P[1][0]*pt[0]+P[1][1]*pt[1]+P[1][2]*pt[2]+P[1][3]*1.0)/z;

      //检查 (x, y) 是否在图像边界内
      if (x > 0.0 && x < img.cols && y > 0.0 && y < img.rows)
      {
        color_data.at<cv::Vec4b>(r, c)      = img.at<cv::Vec4b>((int)y, (int)x);
        elevation_angle.at<float>(r, c)     = static_cast<float>(ortho::internal::computeElevationAngle(t, pt));
        elevated.at<uchar>(r, c)            = is_elevated_val;
        num_observations.at<uint16_t>(r, c) = 1;
      }
      else
      {
        //将相应的 surface 值设置为 NaN，表示无效值
        surface.at<float>(r, c)             = std::numeric_limits<float>::quiet_NaN();
      }
    }

  LOG_IF_F(INFO, verbose, "Image successfully rectified.");

  //存储校正后的图像数据
  auto rectification = std::make_shared<CvGridMap>(roi, GSD);
  rectification->add("color_rgb", color_data);
  rectification->add("elevation_angle", elevation_angle);
  rectification->add("elevated", elevated);
  rectification->add("num_observations", num_observations);
  return rectification;
}

//计算两个点之间的高度角度
double ortho::internal::computeElevationAngle(double *t, double *p)
{
  //v表示从点 p 到点 t 的方向向量
  double v[3]{t[0]-p[0], t[1]-p[1], t[2]-p[2]};\
  //计算向量 v 的长度 v_length
  double v_length = sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
  /*使用 sqrt(v[0]*v[0] + v[2]*v[2])计算向量 v 在 x-z 平面上的投影长度
  使用 acos 函数计算投影长度与向量长度之比的弧度值
  将弧度值转换为度数，乘以 180/3.1415
  */
  return acos(sqrt(v[0]*v[0]+v[1]*v[1])/v_length)*180/3.1415;
}