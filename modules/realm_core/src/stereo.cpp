

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc/imgproc_c.h>

#include <realm_core/stereo.h>

//用于计算双目立体视觉中的校正参数
void realm::stereo::computeRectification(const Frame::Ptr &frame_left,
                                         const Frame::Ptr &frame_right,
                                         cv::Mat &R1,
                                         cv::Mat &P1,
                                         cv::Mat &R2,
                                         cv::Mat &P2,
                                         cv::Mat &Q)
{
  //断言输入的帧对象已经设置了图像大小的缩放（resize）信息
  assert(frame_left->isImageResizeSet() && frame_right->isImageResizeSet());
  //从帧对象中获取左右相机的内部标定矩阵（内参）K以及畸变参数D，外部标定矩阵（外参）T
  // inner calib
  cv::Mat K_l = frame_left->getResizedCalibration();
  cv::Mat K_r = frame_right->getResizedCalibration();
  // distortion
  cv::Mat D_l = frame_left->getCamera()->distCoeffs();
  cv::Mat D_r = frame_right->getCamera()->distCoeffs();
  // exterior calib
  cv::Mat T_l_w2c = frame_left->getCamera()->Tw2c();
  cv::Mat T_r_c2w = frame_right->getCamera()->Tc2w();
  cv::Mat T_lr = T_l_w2c*T_r_c2w;
  // Now calculate transformation between the cameras
  // Formula: R = R_1^T * R_2
  //          t = R_1^T * t_2 - R_1^T*t_1
  //使用外参矩阵计算左右相机之间的变换矩阵R和t
  cv::Mat R = T_lr.rowRange(0, 3).colRange(0, 3);
  cv::Mat t = T_lr.rowRange(0, 3).col(3);
  // Compute rectification parameters
  //使用OpenCV的stereoRectify函数来执行立体校正
  //校正的目标是使得两个相机的成像平面在同一个平面上，从而简化之后的视差计算和点云重建
  cv::Rect* roi_left = nullptr;
  cv::Rect* roi_right = nullptr;
  cv::stereoRectify(K_l, D_l, K_r, D_r, frame_left->getResizedImageSize(), R, t, R1, R2, P1, P2, Q,
                    cv::CALIB_ZERO_DISPARITY, -1, frame_left->getResizedImageSize(), roi_left, roi_right);
}

//执行图像的畸变校正
void realm::stereo::remap(const Frame::Ptr &frame,
                          const cv::Mat &R,
                          const cv::Mat &P,
                          cv::Mat &img_remapped)
{
  //获取相机的内部标定矩阵K和畸变系数D
  // inner calib
  cv::Mat K = frame->getResizedCalibration();
  // distortion
  cv::Mat D = frame->getCamera()->distCoeffs();
  // remapping
  //调用initUndistortRectifyMap函数来计算畸变校正的映射表map11和map12
  cv::Mat map11, map12;
  initUndistortRectifyMap(K, D, R, P, frame->getResizedImageSize(), CV_16SC2, map11, map12);
  // Get image and convert to grayscale
  //从Frame对象中获取校正后的图像，并将其转换为灰度图像
  cv::Mat img = frame->getResizedImageUndistorted();
  cvtColor(img, img, CV_BGR2GRAY);
  // Compute remapping
  //使用cv::remap函数和之前计算得到的映射表，对校正后的灰度图像进行畸变校正
  cv::remap(img, img_remapped, map11, map12, cv::INTER_LINEAR);
}

//用于重新投影深度图，将深度图中的每个深度值转换为世界坐标中的三维点
cv::Mat realm::stereo::reprojectDepthMap(const camera::Pinhole::ConstPtr &cam,
                                         const cv::Mat &depthmap)
{
  // Chosen formula for reprojection follows the linear projection model:
  // x = K*(R|t)*X
  // R^T*K^-1*x-R^T*t = X
  // If pose is defined as "camera to world", then this formula simplifies to
  // R*K^-1*x+t=X
  //检查输入的深度图尺寸和数据类型是否符合要求
  if (depthmap.rows != cam->height() || depthmap.cols != cam->width())
    throw(std::invalid_argument("Error: Reprojecting depth map failed. Dimension mismatch!"));
  if (depthmap.type() != CV_32F)
    throw(std::invalid_argument("Error: Reprojecting depth map failed. Matrix has wrong type. It is expected to have type CV_32F."));

  // Implementation is chosen to be raw array, because it saves round about 30% computation time
  //从相机参数对象（cam）中获取相机内部参数
  double fx = cam->fx();
  double fy = cam->fy();
  double cx = cam->cx();
  double cy = cam->cy();
  //将旋转矩阵（R_c2w）和平移向量（t_c2w）转换为原始的C数组格式
  cv::Mat R_c2w = cam->R();
  cv::Mat t_c2w = cam->t();

  if (fabs(fx) < 10e-6 || fabs(fy) < 10-6 || fabs(cx) < 10e-6 || fabs(cy) < 10e-6)
    throw(std::invalid_argument("Error: Reprojecting depth map failed. Camera model invalid!"));

  if (R_c2w.empty() || t_c2w.empty())
    throw(std::invalid_argument("Error: Reprojecting depth map failed. Pose matrix is empty!"));

  // Array preparation
  double ar_R_c2w[3][3];
  for (uint8_t r = 0; r < 3; ++r)
    for (uint8_t c = 0; c < 3; ++c)
      ar_R_c2w[r][c] = R_c2w.at<double>(r, c);

  double ar_t_c2w[3];
  for (uint8_t r = 0; r < 3; ++r)
      ar_t_c2w[r] = t_c2w.at<double>(r);

  // Iteration
  //存储重新投影后的三维点
  cv::Mat img3d(depthmap.rows, depthmap.cols, CV_64FC3);
  //遍历深度图中的每个像素，计算其对应的三维点
  for (int r = 0; r < depthmap.rows; ++r)
    for (int c = 0; c < depthmap.cols; ++c)
    {
      cv::Vec3d pt(0, 0, 0);

      auto depth = static_cast<double>(depthmap.at<float>(r, c));

      //如果深度值大于0，则使用相机内部参数、深度值和像素坐标来计算三维点在世界坐标中的位置
      if (depth > 0)
      {
        // K^-1*(dc,dr,d)
        double u = (c - cx)*depth/fx;
        double v = (r - cy)*depth/fy;

        pt[0] = ar_R_c2w[0][0]*u + ar_R_c2w[0][1]*v + ar_R_c2w[0][2]*depth + ar_t_c2w[0];
        pt[1] = ar_R_c2w[1][0]*u + ar_R_c2w[1][1]*v + ar_R_c2w[1][2]*depth + ar_t_c2w[1];
        pt[2] = ar_R_c2w[2][0]*u + ar_R_c2w[2][1]*v + ar_R_c2w[2][2]*depth + ar_t_c2w[2];
      }
      //计算得到的三维点存储在 img3d 矩阵的相应位置
      img3d.at<cv::Vec3d>(r, c) = pt;
    }
  return img3d;
}

//用于根据点云数据计算深度图
cv::Mat realm::stereo::computeDepthMapFromPointCloud(const camera::Pinhole::ConstPtr &cam, const cv::Mat &points)
{
  /*
   * Depth computation according to [Hartley2004] "Multiple View Geometry in Computer Vision", S.162 for normalized
   * camera matrix
   */
  //检查输入的点云矩阵类型是否为 CV_64F
  if (points.type() != CV_64F)
    throw(std::invalid_argument("Error: Computing depth map from point cloud failed. Point matrix type should be CV_64F!"));

  // Prepare depthmap dimensions
  //获取相机的图像尺寸
  uint32_t width = cam->width();
  uint32_t height = cam->height();

  // Prepare output data
  //创建一个新的 cv::Mat 对象（depth_map）用于存储深度图，并初始化所有像素的深度值为 -1
  cv::Mat depth_map = cv::Mat(height, width, CV_32F, -1.0);

  // Prepare extrinsics
  //从相机参数对象（cam）中获取相机到世界坐标的变换矩阵（T_w2c）
  cv::Mat T_w2c = cam->Tw2c();
  //从变换矩阵（T_w2c）中提取出旋转矩阵的第三行，并将其转换为 C 数组格式
  cv::Mat cv_R_w2c = T_w2c.row(2).colRange(0, 3).t();
  double R_w2c[3] = { cv_R_w2c.at<double>(0), cv_R_w2c.at<double>(1), cv_R_w2c.at<double>(2) };
  double zwc = T_w2c.at<double>(2, 3);

  // Prepare projection
  //从相机参数对象（cam）中获取相机到世界坐标的投影矩阵（P_cv）
  cv::Mat P_cv = cam->P();
  //将投影矩阵（P_cv）中的元素提取出来，存储在数组 P 中
  double P[3][4]{P_cv.at<double>(0, 0), P_cv.at<double>(0, 1), P_cv.at<double>(0, 2), P_cv.at<double>(0, 3),
                 P_cv.at<double>(1, 0), P_cv.at<double>(1, 1), P_cv.at<double>(1, 2), P_cv.at<double>(1, 3),
                 P_cv.at<double>(2, 0), P_cv.at<double>(2, 1), P_cv.at<double>(2, 2), P_cv.at<double>(2, 3)};
  //遍历点云矩阵的每一行，每一行表示一个点的坐标
  for (int i = 0; i < points.rows; ++i)
  {
    //从当前点的坐标中提取 x、y、z 分量
    auto pixel = points.ptr<double>(i);
    double pt_x = pixel[0];
    double pt_y = pixel[1];
    double pt_z = pixel[2];

    // Depth calculation
    //计算深度值
    double depth = R_w2c[0]*pt_x + R_w2c[1]*pt_y + R_w2c[2]*pt_z + zwc;

    // Projection to image with x = P * X
    //使用投影矩阵计算点在图像平面上的像素坐标（u、v）
    double w =        P[2][0]*pt_x + P[2][1]*pt_y + P[2][2]*pt_z + P[2][3]*1.0;
    auto   u = (int)((P[0][0]*pt_x + P[0][1]*pt_y + P[0][2]*pt_z + P[0][3]*1.0)/w);
    auto   v = (int)((P[1][0]*pt_x + P[1][1]*pt_y + P[1][2]*pt_z + P[1][3]*1.0)/w);
    
    //如果像素坐标在图像范围内，并且深度值大于0，则将深度值存储在深度图中相应的位置
    if (u >= 0 && u < width && v >= 0 && v < height)
    {
      if (depth > 0)
        depth_map.at<float>(v, u) = static_cast<float>(depth);
      else
        depth_map.at<float>(v, u) = -1.0f;
    }
  }
  return depth_map;
}

//用于从深度图计算表面法线
cv::Mat realm::stereo::computeNormalsFromDepthMap(const cv::Mat& depth)
{
  // We have to shrink the normal mat by two, because otherwise we would run into border issues as the Kernel as size 3x3
  //存储计算得到的法线向量
  cv::Mat normals(depth.rows-2, depth.cols-2, CV_32FC3);

  //使用双重循环遍历深度图内部像素，跳过边界像素
  for(int r = 1; r < depth.rows-1; ++r)
    for(int c = 1; c < depth.cols-1; ++c)
    {
      //计算中心像素周围的深度梯度，分别在 x 和 y 方向上计算 dzdx 和 dzdy
      float dzdx = (depth.at<float>(r, c+1) - depth.at<float>(r, c-1)) / 2.0f;
      float dzdy = (depth.at<float>(r+1, c) - depth.at<float>(r-1, c)) / 2.0f;

      //据深度梯度计算法线方向向量 d，其中 x 和 y 分量分别是 -dzdx 和 -dzdy，z 分量是常数 1.0
      cv::Vec3f d(-dzdx, -dzdy, 1.0f);
      //将法线方向向量 d 归一化，并将结果存储在 normals 的相应位置（r-1, c-1）
      normals.at<cv::Vec3f>(r-1, c-1) = d / cv::norm(d);
    }
    //对 normals 进行边界扩展，使用反射边界模式
  cv::copyMakeBorder(normals, normals, 1, 1, 1, 1, cv::BORDER_REFLECT);
  return normals;
}

//用于从两个相机的投影矩阵计算基线长度
double realm::stereo::computeBaselineFromPose(const cv::Mat &p1, const cv::Mat &p2)
{
  //从第一个相机的投影矩阵 p1 中提取平移向量，即矩阵的第 4 列，存储在 t1 中
  cv::Mat t1 = p1.col(3);
  //从第二个相机的投影矩阵 p2 中提取平移向量，即矩阵的第 4 列，存储在 t2 中
  cv::Mat t2 = p2.col(3);
  return cv::norm(t1-t2);
}