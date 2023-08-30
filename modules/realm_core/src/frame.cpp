

#include <iostream>
#include <realm_core/frame.h>
/*表示无人机拍摄的图像帧
包含了图像数据，时间戳，位姿等信息
提供了一些函数来获取图像的元数据
*/

namespace realm
{

// CONSTRUCTION
//用于初始化帧对象
Frame::Frame(const std::string &camera_id,
             const uint32_t &frame_id,
             const uint64_t &timestamp,
             const cv::Mat &img,
             const UTMPose &utm,
             const camera::Pinhole::Ptr &cam,
             const cv::Mat &orientation)
    : m_camera_id(camera_id),
      m_frame_id(frame_id),
      m_is_keyframe(false),
      m_is_georeferenced(false),
      m_is_img_resizing_set(false),
      m_is_depth_computed(false),
      m_has_accurate_pose(false),
      m_surface_assumption(SurfaceAssumption::PLANAR),
      m_surface_model(nullptr),
      m_orthophoto(nullptr),
      m_timestamp(timestamp),
      m_img(img),
      m_utm(utm),
      m_camera_model(cam),
      m_orientation(orientation.empty() ? cv::Mat::eye(3, 3, CV_64F) : orientation),
      m_img_resize_factor(0.0),
      m_min_depth(0.0),
      m_max_depth(0.0),
      m_med_depth(0.0),
      m_depthmap(nullptr),
      m_sparse_cloud(nullptr)
{
  //确保相机 ID、相机模型和图像数据不为空
  if (m_camera_id.empty())
    throw(std::invalid_argument("Error creating frame: Camera Id not provided!"));
  if (!m_camera_model)
    throw(std::invalid_argument("Error creating frame: Camera model not provided!"));
  if (m_img.empty())
    throw(std::invalid_argument("Error creating frame: Image data empty!"));

  //将默认位姿设置为相机模型的位姿  
  m_camera_model->setPose(getDefaultPose());
}

// GETTER

// 返回m_camera_id
std::string Frame::getCameraId() const
{
  return m_camera_id;
}

//返回m_frame_id
uint32_t Frame::getFrameId() const
{
  return m_frame_id;
}
//返回调整大小后图像的宽度
uint32_t Frame::getResizedImageWidth() const
{
  std::lock_guard<std::mutex> lock(m_mutex_cam);
  if (isImageResizeSet())
  //如果已设置图像的调整大小因子，则将当前相机模型的宽度乘以该因子以获取调整后的宽度
    return (uint32_t)((double) m_camera_model->width() * m_img_resize_factor);
  else
    throw(std::runtime_error("Error returning resized image width: Image resize factor not set!"));
}
// 返回调整大小后图像的高度
uint32_t Frame::getResizedImageHeight() const
{
  std::lock_guard<std::mutex> lock(m_mutex_cam);
  if (isImageResizeSet())
  //如果已设置图像的调整大小因子，则将当前相机模型的高度乘以该因子以获取调整后的高度
    return (uint32_t)((double) m_camera_model->height() * m_img_resize_factor);
  else
    throw(std::runtime_error("Error returning resized image height: Image resize factor not set!"));
}
//返回场景中的最小深度
double Frame::getMinSceneDepth() const
{
  if (isDepthComputed())
    return m_min_depth;
  else
    throw(std::runtime_error("Error: Depth was not computed!"));
}
//返回场景中的最大深度
double Frame::getMaxSceneDepth() const
{
  if (isDepthComputed())
    return m_max_depth;
  else
    throw(std::runtime_error("Error: Depth was not computed!"));
}
//返回场景中的中值深度
double Frame::getMedianSceneDepth() const
{
  if (isDepthComputed())
    return m_med_depth;
  else
    throw(std::runtime_error("Error: Depth was not computed!"));
}
//返回深度图的指针
Depthmap::Ptr Frame::getDepthmap() const
{
  return m_depthmap;
}
//返回调整大小后图像的大小
cv::Size Frame::getResizedImageSize() const
{
  if (isImageResizeSet())
  {
    auto width = (uint32_t)((double) m_camera_model->width() * m_img_resize_factor);
    auto height = (uint32_t)((double) m_camera_model->height() * m_img_resize_factor);
    return cv::Size(width, height);
  }
  else
    throw(std::runtime_error("Error: Image resize factor not set!"));
}
//返回去畸变后的图像
cv::Mat Frame::getImageUndistorted() const
{
  std::lock_guard<std::mutex> lock(m_mutex_cam);
  cv::Mat img_undistorted;

  if(m_camera_model->hasDistortion())
    img_undistorted = m_camera_model->undistort(m_img, cv::InterpolationFlags::INTER_LINEAR);
  else
    img_undistorted = m_img;
  return std::move(img_undistorted);
}
//返回原始图像
cv::Mat Frame::getImageRaw() const
{
  // - No deep copy
  return m_img;
}

//根据提供的 UTM 坐标和朝向信息（如果有）创建一个默认的相机位姿矩阵
cv::Mat Frame::getDefaultPose() const
{
  // Translation set to measured utm coordinates
  cv::Mat t = cv::Mat::zeros(3, 1, CV_64F);
  t.at<double>(0) = m_utm.easting;
  t.at<double>(1) = m_utm.northing;
  t.at<double>(2) = m_utm.altitude;

  // Create default pose
  cv::Mat default_pose = cv::Mat::eye(3, 4, CV_64F);
  m_orientation.copyTo(default_pose.rowRange(0, 3).colRange(0, 3));
  t.col(0).copyTo(default_pose.col(3));
  return default_pose;
}

//用于获取经过调整大小和去畸变处理后的图像
cv::Mat Frame::getResizedImageUndistorted() const
{
  // - Resized image will be calculated and set with first
  // access to avoid multiple costly resizing procedures
  // - Check also if resize factor has not changed in the
  // meantime
  // - No deep copy
  if (isImageResizeSet())
  {
    camera::Pinhole cam_resized = m_camera_model->resize(m_img_resize_factor);
    return cam_resized.undistort(m_img_resized, cv::InterpolationFlags::INTER_LINEAR);
  }
  else
    throw(std::invalid_argument("Error: Image resize factor not set!"));
}

//用于获取经过调整大小的图像的深拷贝
cv::Mat Frame::getResizedImageRaw() const
{
  // deep copy, as it might be painted or modified
  return m_img_resized.clone();
}

//用于获取经过调整大小的相机内参矩阵的深拷贝
cv::Mat Frame::getResizedCalibration() const
{
  std::lock_guard<std::mutex> lock(m_mutex_cam);
  if (isImageResizeSet())
    return m_camera_model->resize(m_img_resize_factor).K();
  else
    throw(std::runtime_error("Error resizing camera: Image resizing was not set!"));
}

//获取稀疏点云数据的指针
PointCloud::Ptr Frame::getSparseCloud() const
{
  std::lock_guard<std::mutex> lock(m_mutex_sparse_points);
  return m_sparse_cloud;
}
//设置深度图数据
void Frame::setDepthmap(const Depthmap::Ptr &depthmap)
{
  m_depthmap = depthmap;
}
//获取相机位姿
cv::Mat Frame::getPose() const
{
  // Camera pose is a 3x4 motion matrix. However, it depends on the current state of information how it exactly looks like.
  // If frame pose is accurate, then two options exist:
  // 1) Pose is accurate and georeference was computed -> return motion in the geographic frame
  // 2) Pose is accurate but georeference was not computed -> return motion in visual world frame
  // If frame pose is not accurate, then return the default pose based on GNSS and heading
  
  //位姿准确且地理参考已计算：返回在地理坐标系中的位姿
  //位姿准确但地理参考未计算：返回在视觉世界坐标系中的位姿
  //如果相机位姿不准确，则返回基于 GNSS 和方向的默认位姿
  if (hasAccuratePose())
  {
    // Option 1+2: Cameras pose is always uptodate
    return m_camera_model->pose();
  }

  // Default:
  return getDefaultPose();
}
//获取相机位姿在地理坐标系中的表示
cv::Mat Frame::getVisualPose() const
{
  return m_motion_c2w.clone();
}
//获取地理坐标系下的相机位姿
cv::Mat Frame::getGeographicPose() const
{
  return m_motion_c2g.clone();
}
//获取地理参考变换矩阵
cv::Mat Frame::getGeoreference() const
{
  return m_transformation_w2g.clone();
}
//获取地表假设信息
SurfaceAssumption Frame::getSurfaceAssumption() const
{
  std::lock_guard<std::mutex> lock(m_mutex_flags);
  return m_surface_assumption;
}
//获取地表模型的指针
CvGridMap::Ptr Frame::getSurfaceModel() const
{
  std::lock_guard<std::mutex> lock(m_mutex_surface_model);
  return m_surface_model;
}
//获取正射影像地图的指针
CvGridMap::Ptr Frame::getOrthophoto() const
{
  std::lock_guard<std::mutex> lock(m_mutex_orthophoto);
  return m_orthophoto;
}
//获取GNSS测量得到的UTM坐标信息
UTMPose Frame::getGnssUtm() const
{
  return m_utm;
}

//获取指向相机模型的常量指针
camera::Pinhole::ConstPtr Frame::getCamera() const
{
  std::lock_guard<std::mutex> lock(m_mutex_cam);
  return m_camera_model;
}
// 获取帧的时间戳
uint64_t Frame::getTimestamp() const
{
  // in [nanosec]
  return m_timestamp;
}
//获取调整大小后的相机模型的指针
camera::Pinhole::Ptr Frame::getResizedCamera() const
{
  assert(m_is_img_resizing_set);
  return std::make_shared<camera::Pinhole>(m_camera_model->resize(m_img_resize_factor));
}

// SETTER
//设置可视姿态
void Frame::setVisualPose(const cv::Mat &pose)
{
  m_motion_c2w = pose;

  // If frame is already georeferenced, then set camera pose as geographic pose. Otherwise use visual pose
  if (m_is_georeferenced)
  {
    //通过 applyTransformationToVisualPose 函数计算地理坐标系中的姿态，然后调用 setGeographicPose 函数设置地理姿态
    cv::Mat M_c2g = applyTransformationToVisualPose(m_transformation_w2g);
    setGeographicPose(M_c2g);
  }
  else
  {
    //更新相机模型的姿态为传入的可视姿态，并将 m_motion_c2w 设置为传入的姿态
    std::lock_guard<std::mutex> lock(m_mutex_cam);
    m_camera_model->setPose(pose);
  }
  //将姿态精确标志 setPoseAccurate 设置为真
  setPoseAccurate(true);
}
//设置地理姿态
void Frame::setGeographicPose(const cv::Mat &pose)
{
  if (!pose.empty())
  {
    std::lock_guard<std::mutex> lock(m_mutex_cam);
    m_camera_model->setPose(pose);
    m_motion_c2g = pose;
    setPoseAccurate(true);
  }
}
//设置地理参考变换
void Frame::setGeoreference(const cv::Mat &T_w2g)
{
  if (T_w2g.empty())
    throw(std::invalid_argument("Error setting georeference: Transformation is empty!"));

  std::lock_guard<std::mutex> lock(m_mutex_T_w2g);
  //函数会将传入的变换矩阵克隆存储在 m_transformation_w2g 中
  m_transformation_w2g = T_w2g.clone();
  m_is_georeferenced = true;
}

//设置稀疏点云
void Frame::setSparseCloud(const PointCloud::Ptr &sparse_cloud, bool in_visual_coordinates)
{
  if (sparse_cloud->empty())
    return;

  m_mutex_sparse_points.lock();
  //传入的稀疏点云赋值给 m_sparse_cloud
  m_sparse_cloud = sparse_cloud;
  m_mutex_sparse_points.unlock();

  m_mutex_flags.lock();
  //是否需要将稀疏点云进行坐标系变换
  if (in_visual_coordinates && m_is_georeferenced)
  {
    m_mutex_T_w2g.lock();
    //应用地理参考的变换到稀疏点云
    applyTransformationToSparseCloud(m_transformation_w2g);
    m_mutex_T_w2g.unlock();
  }
  m_mutex_flags.unlock();
  //计算场景深度
  computeSceneDepth(1000);
}

//设置表面模型
void Frame::setSurfaceModel(const CvGridMap::Ptr &surface_model)
{
  std::lock_guard<std::mutex> lock(m_mutex_surface_model);
  m_surface_model = surface_model;
}
//设置正射影像
void Frame::setOrthophoto(const CvGridMap::Ptr &orthophoto)
{
  std::lock_guard<std::mutex> lock(m_mutex_orthophoto);
  m_orthophoto = orthophoto;
}
//设置关键帧标志
void Frame::setKeyframe(bool flag)
{
  std::lock_guard<std::mutex> lock(m_mutex_flags);
  m_is_keyframe = flag;
}
//设置姿态准确性标志
void Frame::setPoseAccurate(bool flag)
{
  std::lock_guard<std::mutex> lock(m_mutex_flags);
  // If we are clearing the flag, clear the georeferenced pose data as well
  if (!flag) {
    m_camera_model->setPose(getDefaultPose());
  }
  m_has_accurate_pose = flag;
}
//设置表面假设
void Frame::setSurfaceAssumption(SurfaceAssumption assumption)
{
  std::lock_guard<std::mutex> lock(m_mutex_flags);
  m_surface_assumption = assumption;
}
//设置图像缩放因子
void Frame::setImageResizeFactor(const double &value)
{
  std::lock_guard<std::mutex> lock(m_mutex_img_resized);
  m_img_resize_factor = value;
  cv::resize(m_img, m_img_resized, cv::Size(), m_img_resize_factor, m_img_resize_factor);
  m_is_img_resizing_set = true;
}


// FUNCTIONALITY

//初始化地理参考
void Frame::initGeoreference(const cv::Mat &T)
{
  assert(!T.empty());

  m_transformation_w2g = cv::Mat::eye(4, 4, CV_64F);
  updateGeoreference(T, true);
}
//判断帧是否为关键帧
bool Frame::isKeyframe() const
{
  std::lock_guard<std::mutex> lock(m_mutex_flags);
  return m_is_keyframe;
}
//判断帧是否已地理参考
bool Frame::isGeoreferenced() const
{
  std::lock_guard<std::mutex> lock(m_mutex_flags);
  return m_is_georeferenced;
}
//判断图像是否设置了缩放
bool Frame::isImageResizeSet() const
{
  std::lock_guard<std::mutex> lock(m_mutex_flags);
  return m_is_img_resizing_set;
}
//判断深度是否已计算
bool Frame::isDepthComputed() const
{
  std::lock_guard<std::mutex> lock(m_mutex_flags);
  return m_is_depth_computed;
}
//判断帧的姿态是否准确
bool Frame::hasAccuratePose() const
{
  std::lock_guard<std::mutex> lock(m_mutex_flags);
  return m_has_accurate_pose;
}

//用于打印帧的信息
std::string Frame::print()
{
  //使用互斥锁 m_mutex_flags 锁定以保护对帧信息的读取
  std::lock_guard<std::mutex> lock(m_mutex_flags);
  char buffer[5000];//使用字符缓冲区 buffer 存储打印信息
  sprintf(buffer, "### FRAME INFO ###\n");
  sprintf(buffer + strlen(buffer), "Stamp: %lu \n", m_timestamp);
  sprintf(buffer + strlen(buffer), "Image: [%i x %i]\n", m_img.cols, m_img.rows);
  sprintf(buffer + strlen(buffer), "GNSS: [%4.02f E, %4.02f N, %4.02f Alt, %4.02f Head]\n",
          m_utm.easting, m_utm.northing, m_utm.altitude, m_utm.heading);
  sprintf(buffer + strlen(buffer), "Is key frame: %s\n", (m_is_keyframe ? "yes" : "no"));
  sprintf(buffer + strlen(buffer), "Has accurate pose: %s\n", (m_has_accurate_pose ? "yes" : "no"));
  //使用互斥锁 m_mutex_sparse_points 锁定以保护对稀疏点云的读取
  std::lock_guard<std::mutex> lock1(m_mutex_cam);
  cv::Mat pose = getPose();
  if (!pose.empty())
    sprintf(buffer + strlen(buffer), "Pose: Exists [%i x %i]\n", pose.rows, pose.cols);
  std::lock_guard<std::mutex> lock2(m_mutex_sparse_points);
  if (!m_sparse_cloud->empty())
    sprintf(buffer + strlen(buffer), "Mappoints: %i\n", m_sparse_cloud->data().rows);

  return std::string(buffer);
}

// TODO: move this into sparse cloud?
//用于计算稀疏点云中的场景深度
void Frame::computeSceneDepth(int max_nrof_points)
{
  /*
   * Depth computation according to [Hartley2004] "Multiple View Geometry in Computer Vision", S.162:
   * Projection formula P*X = x with P=K(R|t), X=(x,y,z,1) and x=w*(u,v,1)
   * For the last row therefore (p31,p32,p33,p34)*(x,y,z,1)^T=w. If p3=(p31,p32,p33) is the direction of the principal
   * axis and det(p3)>0,||p3||=1 then w can be interpreted as projected depth. Therefore the final formula:
   * w=depth=(r31,r32,r33,t_z)*(x,y,z,1)
   */
  //如果稀疏点云为空，则直接返回
  if (!m_sparse_cloud || m_sparse_cloud->empty())
    return;
    
  //使用互斥锁 m_mutex_sparse_points 锁定以保护对稀疏点云数据的读取
  std::lock_guard<std::mutex> lock(m_mutex_sparse_points);

  // The user can limit the number of points on which the depth is computed. The increment for the iteration later on
  // is changed accordingly.
  //据用户设定的最大点数限制 max_nrof_points，计算实际要计算深度的点数 n，以及迭代时的增量 inc
  int n = 0;
  int inc = 1;

  cv::Mat sparse_data = m_sparse_cloud->data();
  if (max_nrof_points == 0 || sparse_data.rows < max_nrof_points)
  {
    n = sparse_data.rows;
  }
  else
  {
    n = max_nrof_points;
    inc = sparse_data.rows * (max_nrof_points / sparse_data.rows);

    // Just to make sure we don't run into an infinite loop
    if (inc <= 0)
      inc = 1;
  }

  std::vector<double> depths;
  depths.reserve(n);

  m_mutex_cam.lock();
  cv::Mat P = m_camera_model->P();
  m_mutex_cam.unlock();

  // Prepare extrinsics
  //获取稀疏点云的数据，并根据 n 和 inc 的值来计算深度
  cv::Mat T_w2c = m_camera_model->Tw2c();
  cv::Mat R_wc2 = T_w2c.row(2).colRange(0, 3).t();
  double z_wc = T_w2c.at<double>(2, 3);

  for (int i = 0; i < n; i += inc)
  {
    cv::Mat pt = sparse_data.row(i).colRange(0, 3).t();

    // Depth calculation
    double depth = R_wc2.dot(pt) + z_wc;
    depths.push_back(depth);
  }
  //对深度值进行排序,将最小、最大和中位数深度值分别存储
  sort(depths.begin(), depths.end());
  m_min_depth = depths[0];
  m_max_depth = depths[depths.size() - 1];
  m_med_depth = depths[(depths.size() - 1) / 2];
  m_is_depth_computed = true;
}
//用于更新地理参考框架
void Frame::updateGeoreference(const cv::Mat &T, bool do_update_sparse_cloud)
{
  // Update the visual pose with the new georeference
  //更新视觉姿态
  cv::Mat M_c2g = applyTransformationToVisualPose(T);
  setGeographicPose(M_c2g);

  // In case we want to update the surface points as well, we have to compute the difference of old and new transformation.
  //如果指定了要更新稀疏点云
  if (do_update_sparse_cloud && !m_transformation_w2g.empty())
  {
    //计算新旧地理参考框架之间的变换差异 T_diff
    cv::Mat T_diff = computeGeoreferenceDifference(m_transformation_w2g, T);
    applyTransformationToSparseCloud(T_diff);
  }

  // Pose and / or surface points have changed. So update scene depth accordingly
  //调用 computeSceneDepth 函数，重新计算最小、最大和中位数深度值
  computeSceneDepth(1000);
  //将输入的地理参考框架变换矩阵 T 设置为地理参考矩阵
  setGeoreference(T);
}
//用于返回相机帧的方向矩阵 m_orientation 的副本
cv::Mat Frame::getOrientation() const
{
  return m_orientation.clone();
}

//获取相机帧的方向矩阵以及对稀疏点云进行坐标变换
void Frame::applyTransformationToSparseCloud(const cv::Mat &T)
{
  //检查稀疏点云是否为空
  if (m_sparse_cloud && !m_sparse_cloud->empty())
  {
    //获取稀疏点云的数据
    cv::Mat sparse_data = m_sparse_cloud->data();
    m_mutex_sparse_points.lock();
    //对每个点进行循环迭代
    for (uint32_t i = 0; i < sparse_data.rows; ++i)
    {
      cv::Mat pt = sparse_data.row(i).colRange(0, 3).t();
      pt.push_back(1.0);
      cv::Mat pt_hom = T * pt;
      pt_hom.pop_back();
      sparse_data.row(i) = pt_hom.t();
    }
    m_mutex_sparse_points.unlock();
  }
}

//用于将输入的变换矩阵 T 应用于当前相机帧的视觉姿态
cv::Mat Frame::applyTransformationToVisualPose(const cv::Mat &T)
{
  //检查当前相机帧的视觉姿态矩阵是否为空
  if (!m_motion_c2w.empty())
  {
    //创建一个单位矩阵 T_c2w，并将当前视觉姿态的旋转和平移部分复制到其中
    cv::Mat T_c2w = cv::Mat::eye(4, 4, CV_64F);
    m_motion_c2w.copyTo(T_c2w.rowRange(0, 3).colRange(0, 4));
    //将输入的变换矩阵 T 与 T_c2w 相乘
    cv::Mat T_c2g = T * T_c2w;
    //从 T_c2g 矩阵中提取前三行和前四列
    cv::Mat M_c2g = T_c2g.rowRange(0, 3).colRange(0, 4);

    // Remove scale
    //对 M_c2g 中的列向量进行归一化
    M_c2g.col(0) /= cv::norm(M_c2g.col(0));
    M_c2g.col(1) /= cv::norm(M_c2g.col(1));
    M_c2g.col(2) /= cv::norm(M_c2g.col(2));

    return M_c2g;
  }
  return cv::Mat();
}

//用于计算两个地理参考矩阵 T_old 和 T_new 之间的差异
cv::Mat Frame::computeGeoreferenceDifference(const cv::Mat &T_old, const cv::Mat &T_new)
{
  //从输入的矩阵 T_old 和 T_new 中提取旋转矩阵，并计算各个方向上的缩放因子
  // Remove scale from old georeference
  cv::Mat T1 = T_old.clone();
  double sx_old = cv::norm(T_old.rowRange(0, 3).col(0));
  double sy_old = cv::norm(T_old.rowRange(0, 3).col(1));
  double sz_old = cv::norm(T_old.rowRange(0, 3).col(2));
  //进行缩放归一化，将其缩放部分除以相应的缩放因子
  T1.rowRange(0, 3).col(0) /= sx_old;
  T1.rowRange(0, 3).col(1) /= sy_old;
  T1.rowRange(0, 3).col(2) /= sz_old;

  // Remove scale from old georeference
  cv::Mat T2 = T_new.clone();
  double sx_new = cv::norm(T_new.rowRange(0, 3).col(0));
  double sy_new = cv::norm(T_new.rowRange(0, 3).col(1));
  double sz_new = cv::norm(T_new.rowRange(0, 3).col(2));
  //进行缩放归一化，将其缩放部分除以相应的缩放因子
  T2.rowRange(0, 3).col(0) /= sx_new;
  T2.rowRange(0, 3).col(1) /= sy_new;
  T2.rowRange(0, 3).col(2) /= sz_new;

  //计算 T_old 的逆矩阵 T_old_inv，并从中提取旋转矩阵 R_old_inv 和平移向量 t_old_inv
  cv::Mat T_old_inv = cv::Mat::eye(4, 4, CV_64F);
  cv::Mat R_old_inv = (T1.rowRange(0, 3).colRange(0, 3)).t();
  cv::Mat t_old_inv = -R_old_inv * T1.rowRange(0, 3).col(3);
  //将 R_old_inv 和 t_old_inv 放入 T_old_inv 中的适当位置
  R_old_inv.copyTo(T_old_inv.rowRange(0, 3).colRange(0, 3));
  t_old_inv.copyTo(T_old_inv.rowRange(0, 3).col(3));

  cv::Mat T_diff = T_old_inv * T2;
  //计算从 T_old 到 T_new 的变换矩阵 T_diff
  T_diff.rowRange(0, 3).col(0) *= sx_new / sx_old;
  T_diff.rowRange(0, 3).col(1) *= sy_new / sx_old;
  T_diff.rowRange(0, 3).col(2) *= sz_new / sx_old;

  return T_diff;
}

} // namespace realm