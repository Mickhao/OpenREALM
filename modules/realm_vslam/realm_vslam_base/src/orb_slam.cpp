

#include <realm_vslam_base/orb_slam.h>
#include <realm_core/loguru.h>

/*将 ORB-SLAM 框架集成在一起，用于执行视觉 SLAM*/
using namespace realm;

//初始化 ORB-SLAM 框架的设置和参数
OrbSlam::OrbSlam(const VisualSlamSettings::Ptr &vslam_set, const CameraSettings::Ptr &cam_set, const ImuSettings::Ptr &imu_set)
: m_prev_keyid(-1),
  m_resizing((*vslam_set)["resizing"].toDouble()),
  m_timestamp_reference(0),
  m_path_vocabulary((*vslam_set)["path_vocabulary"].toString())
{
  // Read the settings files
  //读取相机和视觉 SLAM 设置
  cv::Mat K_32f = cv::Mat::eye(3, 3, CV_32F);
  K_32f.at<float>(0, 0) = (*cam_set)["fx"].toFloat() * static_cast<float>(m_resizing);
  K_32f.at<float>(1, 1) = (*cam_set)["fy"].toFloat() * static_cast<float>(m_resizing);
  K_32f.at<float>(0, 2) = (*cam_set)["cx"].toFloat() * static_cast<float>(m_resizing);
  K_32f.at<float>(1, 2) = (*cam_set)["cy"].toFloat() * static_cast<float>(m_resizing);

  cv::Mat dist_coeffs_32f = cv::Mat::zeros(1, 5, CV_32F);
  dist_coeffs_32f.at<float>(0) = (*cam_set)["k1"].toFloat();
  dist_coeffs_32f.at<float>(1) = (*cam_set)["k2"].toFloat();
  dist_coeffs_32f.at<float>(2) = (*cam_set)["p1"].toFloat();
  dist_coeffs_32f.at<float>(3) = (*cam_set)["p2"].toFloat();
  dist_coeffs_32f.at<float>(4) = (*cam_set)["k3"].toFloat();
  
  //构建相机参数和特征提取参数
  ORB_SLAM::CameraParameters cam{};
  cam.K = K_32f;
  cam.distCoeffs = dist_coeffs_32f;
  cam.fps        = (*cam_set)["fps"].toFloat();
  cam.width      = (*cam_set)["width"].toInt();
  cam.height     = (*cam_set)["height"].toInt();
  cam.isRGB      = false; // BGR

  ORB_SLAM::OrbParameters orb{};
  orb.nFeatures   = (*vslam_set)["nrof_features"].toInt();
  orb.nLevels     = (*vslam_set)["n_pyr_levels"].toInt();
  orb.scaleFactor = (*vslam_set)["scale_factor"].toFloat();
  orb.minThFast   = (*vslam_set)["min_th_FAST"].toInt();
  orb.iniThFast   = (*vslam_set)["ini_th_FAST"].toInt();

//根据编译选项选择初始化方式
#ifdef USE_ORB_SLAM2
  m_slam = new ORB_SLAM::System(m_path_vocabulary, cam, orb, ORB_SLAM::System::MONOCULAR);
#endif

#ifdef USE_ORB_SLAM3
  ORB_SLAM::ImuParameters imu{};
  if (imu_set != nullptr)
  {
    LOG_F(INFO, "Detected IMU settings. Loading ORB SLAM3 with IMU support.");

    imu.accelWalk  = (*imu_set)["gyro_noise_density"].toFloat();
    imu.gyroWalk   = (*imu_set)["gyro_bias_random_walk_noise_density"].toFloat();
    imu.noiseAccel = (*imu_set)["acc_noise_density"].toFloat();
    imu.noiseGyro  = (*imu_set)["acc_bias_random_walk_noise_density"].toFloat();
    imu.Tbc        = (*imu_set)["T_cam_imu"].toMat();
    imu.freq       = (*imu_set)["freq"].toFloat();
  }

  if (imu_set != nullptr)
    m_slam = new ORB_SLAM::System(m_path_vocabulary, cam, imu, orb, ORB_SLAM3::System::IMU_MONOCULAR);
  else
    m_slam = new ORB_SLAM::System(m_path_vocabulary, cam, imu, orb, ORB_SLAM3::System::MONOCULAR);
#endif

  //namespace ph = std::placeholders;
  //std::function<void(ORB_SLAM2::KeyFrame*)> kf_update = std::bind(&OrbSlam::keyframeUpdateCb, this, ph::_1);
  //_slam->RegisterKeyTransport(kf_update);
}

OrbSlam::~OrbSlam()
{
  m_slam->Shutdown();//关闭 ORB-SLAM 框架
  delete m_slam;//释放通过 new 运算符分配的 ORB-SLAM 框架实例的内存
}

//执行单目视觉 SLAM 跟踪操作
VisualSlamIF::State OrbSlam::track(Frame::Ptr &frame, const cv::Mat &T_c2w_initial)
{
  //检查是否是第一帧
  if (m_timestamp_reference == 0)
  {
    //设置为当前帧的时间戳
    m_timestamp_reference = frame->getTimestamp();
    return State::LOST;
  }

  // Set image resizing accoring to settings
  //根据设置调整图像大小
  frame->setImageResizeFactor(m_resizing);

  //计算时间戳差
  double timestamp = static_cast<double>(frame->getTimestamp() - m_timestamp_reference)/10e3;
  LOG_IF_F(INFO, true, "Time elapsed since first frame: %4.2f [s]", timestamp);

  // ORB SLAM returns a transformation from the world to the camera frame (T_w2c). In case we provide an initial guess
  // of the current pose, we have to invert this before, because in OpenREALM the standard is defined as T_c2w.

  //调用 ORB-SLAM 进行跟踪：根据使用的 ORB-SLAM 版本，调用适当的函数来执行单目跟踪
  cv::Mat T_w2c;
#ifdef USE_ORB_SLAM2
  T_w2c = m_slam->TrackMonocular(frame->getResizedImageRaw(), timestamp);
#endif

#ifdef USE_ORB_SLAM3
  T_w2c = m_slam->TrackMonocular(frame->getResizedImageRaw(), timestamp, m_imu_queue);
  m_imu_queue.clear();
#endif
  //如果跟踪成功：解析 T_w2c 并将其转换为相机到世界坐标系的变换矩阵 T_c2w
  // In case tracking was successfull and slam not lost
  if (!T_w2c.empty())
  {
    // Pose definition as 3x4 matrix, calculated as 4x4 with last row (0, 0, 0, 1)
    // ORB SLAM 2 pose is defined as T_w2c, however the more intuitive way to describe
    // it for mapping is T_c2w (camera to world) therefore invert the pose matrix
    cv::Mat T_c2w = invertPose(T_w2c);

    // Also convert to double precision
    T_c2w.convertTo(T_c2w, CV_64F);//将 T_c2w 转换为双精度
    T_c2w.pop_back();
    frame->setVisualPose(T_c2w);//设置帧的视觉姿态

    //std::cout << "Soll:\n" << T_c2w << std::endl;
    //std::cout << "Schätz:\n" << T_c2w_initial << std::endl;
    //获取当前跟踪到的地图点云
    frame->setSparseCloud(getTrackedMapPoints(), true);

    // Check if new frame is keyframe by comparing current keyid with last keyid
    auto keyid = static_cast<int32_t>(m_slam->GetLastKeyFrameId());

    // Check current state of the slam
    //确定当前帧的状态
    if (m_prev_keyid == -1)
    {
      m_prev_keyid = keyid;
      return State::INITIALIZED;
    }
    else if (m_prev_keyid != keyid)
    {
      m_prev_keyid = keyid;
      m_orb_to_frame_ids.insert({keyid, frame->getFrameId()});
      return State::KEYFRAME_INSERT;
    }
    else
    {
      return State::FRAME_INSERT;
    }
  }
  return State::LOST;
}

//重置整个 OrbSlam 实例，将其状态回复到初始状态
void OrbSlam::reset()
{
  m_slam->Reset();
  m_timestamp_reference = 0;
}

//关闭 OrbSlam 实例，释放其内部资源
void OrbSlam::close()
{
  m_slam->Shutdown();
}

//获取已跟踪的地图点信息
PointCloud::Ptr OrbSlam::getTrackedMapPoints()
{
  
  std::vector<ORB_SLAM::MapPoint*> mappoints;
  //获取被 ORB-SLAM 跟踪的所有地图点的指针
  mappoints = m_slam->GetTrackedMapPoints();

  size_t n = mappoints.size();

  cv::Mat points;
  points.reserve(n);

  std::vector<uint32_t> point_ids;
  point_ids.reserve(n);
  //遍历获取到的地图点指针数组 mappoints
  for (size_t i = 0; i < n; ++i)
  {
    if (mappoints[i] != nullptr)
    {
      //获取地图点在世界坐标系中的位置
      cv::Mat p = mappoints[i]->GetWorldPos().t();
      points.push_back(p);//将该位置矩阵添加到 points 中
      //将地图点的唯一标识（mnId）添加到 point_ids 中
      point_ids.push_back(mappoints[i]->mnId);
    }
  }
  //转换为双精度浮点数
  points.convertTo(points, CV_64F);
  return std::make_shared<PointCloud>(point_ids, points);
}

//绘制已跟踪的图像
bool OrbSlam::drawTrackedImage(cv::Mat &img) const
{
  img = m_slam->DrawTrackedImage();
  return true;
}

//注册一个回调函数，用于在 SLAM 更新后传递更新的姿态信息
void OrbSlam::registerUpdateTransport(const PoseUpdateFuncCb &func)
{
  m_pose_update_func_cb = func;
}

//注册一个回调函数，用于在 SLAM 重置时执行特定的操作
void OrbSlam::registerResetCallback(const ResetFuncCb &func)
{
  if (func)
  {
    //_slam->RegisterResetCallback(func);
  }
}

//在每次关键帧被更新时，将更新的姿态信息和地图点坐标传递给已注册的姿态更新回调函数
void OrbSlam::keyframeUpdateCb(ORB_SLAM::KeyFrame* kf)
{
  
  if (kf != nullptr && m_pose_update_func_cb)
  {
    //从关键帧对象中获取帧的 ID，并将其转换为 uint32_t 类型
    auto id = (uint32_t) kf->mnFrameId;

    // Get update on pose
    //从关键帧对象中获取世界到相机的变换矩阵 T_w2c
    cv::Mat T_w2c = kf->GetPose();
    //计算其逆矩阵得到 T_c2w
    cv::Mat T_c2w = invertPose(T_w2c);
    //转换为双精度 CV_64F 格式
    T_c2w.convertTo(T_c2w, CV_64F);
    //去除最后一行
    T_c2w.pop_back();

    // Get update on map points
    //从关键帧对象中获取与该关键帧相关的地图点集合
    //然后将每个地图点的世界坐标添加到 points 中
    std::set<ORB_SLAM::MapPoint*> map_points = kf->GetMapPoints();
    cv::Mat points;
    points.reserve(map_points.size());
    for (const auto &pt : map_points)
      if (pt != nullptr)
        points.push_back(pt->GetWorldPos().t());
    //转换为双精度 CV_64F 格式
    points.convertTo(points, CV_64F);

    // Transport to update function
    //更新回调函数 m_pose_update_func_cb
    m_pose_update_func_cb(m_orb_to_frame_ids[id], T_c2w, points);
  }
}

//计算给定姿态矩阵的逆矩阵
cv::Mat OrbSlam::invertPose(const cv::Mat &pose) const
{
  cv::Mat pose_inv = cv::Mat::eye(4, 4, pose.type());
  //计算姿态矩阵 pose 的旋转部分的转置矩阵 R_t
  cv::Mat R_t = (pose.rowRange(0, 3).colRange(0, 3)).t();
  //计算矩阵 pose 的平移向量 t
  cv::Mat t = -R_t*pose.rowRange(0, 3).col(3);
  //将旋转部分 R_t 复制到 pose_inv 的前三行前三列
  R_t.copyTo(pose_inv.rowRange(0, 3).colRange(0, 3));
  //将平移向量 t 复制到 pose_inv 的前三行第四列
  t.copyTo(pose_inv.rowRange(0, 3).col(3));
  return pose_inv;
}

//将 OrbSlam2的设置信息输出到日志中
void OrbSlam::printSettingsToLog()
{
  LOG_F(INFO, "### OrbSlam2 general settings ###");
  LOG_F(INFO, "- use_viewer: %i", m_use_viewer);
  LOG_F(INFO, "- resizing: %4.2f", m_resizing);
  LOG_F(INFO, "- path settings: %s", m_path_settings.c_str());
  LOG_F(INFO, "- path vocabulary: %s", m_path_vocabulary.c_str());
}

//将IMU数据加入到IMU数据队列中，以供ORB SLAM3处理
#ifdef USE_ORB_SLAM3
void OrbSlam::queueImuData(const ImuData &data)
{
  m_imu_queue.emplace_back(
      ORB_SLAM::IMU::Point(static_cast<float>(data.acceleration.x), static_cast<float>(data.acceleration.y), static_cast<float>(data.acceleration.z),
                           static_cast<float>(data.gyroscope.x),    static_cast<float>(data.gyroscope.y),    static_cast<float>(data.gyroscope.z),
                           data.timestamp/10e9)
  );
}
#endif
