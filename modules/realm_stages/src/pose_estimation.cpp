

#define LOGURU_WITH_STREAMS 1

#include <realm_stages/pose_estimation.h>

using namespace realm;
using namespace stages;

PoseEstimation::PoseEstimation(const StageSettings::Ptr &stage_set,
                               const VisualSlamSettings::Ptr &vslam_set,
                               const CameraSettings::Ptr &cam_set,
                               const ImuSettings::Ptr &imu_set,
                               double rate)
    : StageBase("pose_estimation", (*stage_set)["path_output"].toString(), rate, (*stage_set)["queue_size"].toInt(), bool((*stage_set)["log_to_file"].toInt())),
      m_is_georef_initialized(false),
      m_use_vslam((*stage_set)["use_vslam"].toInt() > 0),
      m_use_imu((*stage_set)["use_imu"].toInt() > 0),
      m_set_all_frames_keyframes((*stage_set)["set_all_frames_keyframes"].toInt() > 0),
      m_strategy_fallback(PoseEstimation::FallbackStrategy((*stage_set)["fallback_strategy"].toInt())),
      m_use_fallback(false),
      m_init_lost_frames_reset_count((*stage_set)["init_lost_frames_reset_count"].toInt()),
      m_init_lost_frames(0),
      m_use_initial_guess((*stage_set)["use_initial_guess"].toInt() > 0),
      m_do_update_georef((*stage_set)["update_georef"].toInt() > 0),
      m_do_delay_keyframes((*stage_set)["do_delay_keyframes"].toInt() > 0),
      m_do_suppress_outdated_pose_pub((*stage_set)["suppress_outdated_pose_pub"].toInt() > 0),
      m_th_error_georef((*stage_set)["th_error_georef"].toDouble()),
      m_min_nrof_frames_georef((*stage_set)["min_nrof_frames_georef"].toInt()),
      m_do_auto_reset(false),
      m_th_scale_change(20.0),
      m_overlap_max((*stage_set)["overlap_max"].toDouble()),
      m_overlap_max_fallback((*stage_set)["overlap_max_fallback"].toDouble()),
      m_settings_save({(*stage_set)["save_trajectory_gnss"].toInt() > 0,
                      (*stage_set)["save_trajectory_visual"].toInt() > 0,
                      (*stage_set)["save_frames"].toInt() > 0,
                      (*stage_set)["save_keyframes"].toInt() > 0,
                      (*stage_set)["save_keyframes_full"].toInt() > 0})
{
  LOG_S(INFO) << "Stage [" << m_stage_name << "]: Created Stage with Settings:\n";
  stage_set->print();
  //注册为异步数据准备的回调函数
  registerAsyncDataReadyFunctor([=]{ return !m_buffer_no_pose.empty(); });

  if (m_use_vslam)//否应该使用视觉SLAM
  {
    if (m_use_imu)//是否使用IMU
    {
      if (imu_set == nullptr)//检查是否提供了IMU的设置信息
        throw(std::runtime_error("Error creating SLAM with IMU support. No IMU settings provided!"));
      //根据提供的SLAM设置、相机设置和IMU设置，创建一个视觉SLAM实例
      m_vslam = VisualSlamFactory::create(vslam_set, cam_set, imu_set);
    }
    else
      m_vslam = VisualSlamFactory::create(vslam_set, cam_set);

    // Set reset callback from vSLAM to this node
    // therefore if SLAM resets itself, node is being informed
    //将一个 lambda 函数注册为视觉SLAM的重置回调函数
    m_vslam->registerResetCallback([=](){ reset(); });

    // Set pose update callback
    namespace ph = std::placeholders;
    VisualSlamIF::PoseUpdateFuncCb update_func = std::bind(&PoseEstimation::updateKeyframeCb, this, ph::_1, ph::_2, ph::_3);
    //将刚刚创建的 update_func 注册为视觉SLAM的位姿更新回调函数
    m_vslam->registerUpdateTransport(update_func);

    // Create geo reference initializer
    //创建一个地理参考（Georeferencer）的实例，并传递一些初始化参数
    m_georeferencer = std::make_shared<GeometricReferencer>(m_th_error_georef, m_min_nrof_frames_georef);
  }
  //根据传入的策略参数 m_strategy_fallback 来评估回退策略
  evaluateFallbackStrategy(m_strategy_fallback);

  // Create Pose Estimation publisher
  //创建一个 PoseEstimationIO 实例，作为位姿估计（SLAM）的发布者
  m_stage_publisher.reset(new PoseEstimationIO(this, 2*rate, m_do_delay_keyframes));
  //启动位姿估计发布者，开始发布位姿相关信息
  m_stage_publisher->start();

  // Creation of reference plane, currently only the one below is supported
  //设置参考平面的点和法线
  m_plane_ref.pt = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 0.0);
  m_plane_ref.n = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 1.0);

  // Previous roi initialization
  //初始化上一个感兴趣区域
  m_roi_prev = cv::Rect2d(0.0, 0.0, 0.0, 0.0);
}

PoseEstimation::~PoseEstimation()
{
}

void PoseEstimation::finishCallback()
{
  if (m_use_vslam) {
    //关闭视觉SLAM模块
    m_vslam->close();
  }
  //请求位姿估计发布者停止发布数据
  m_stage_publisher->requestFinish();
  //等待位姿估计发布者完成其工
  m_stage_publisher->join();
}

//根据传入的参数值来决定是否使用图像投影作为后备策略，并在日志中记录所选择的策略
void PoseEstimation::evaluateFallbackStrategy(PoseEstimation::FallbackStrategy strategy)
{
  LOG_F(INFO, "Evaluating fallback strategy...");
  //根据传入的 strategy 参数的值，使用 switch 语句进行不同的情况判断
  switch (strategy)
  {
    case FallbackStrategy::ALWAYS:
      LOG_F(INFO, "Selected: ALWAYS - Images will be projected whenever tracking is lost.");
      m_use_fallback = true;
      break;
    case FallbackStrategy::NEVER:
      m_use_fallback = false;
      LOG_F(INFO, "Selected: NEVER - Image projection will not be used.");
      break;
    default:
      throw(std::invalid_argument("Error: Unknown fallback strategy!"));
  }
}

void PoseEstimation::addFrame(const Frame::Ptr &frame)
{
  // First update statistics about incoming frame rate
  updateStatisticsIncoming();

  // The user can provide a-priori georeferencing. Check if this is the case
  //检查是否存在先验地理参考
  if (!m_is_georef_initialized && frame->isGeoreferenced())
  {
    LOG_F(INFO, "Detected a-priori georeference for frame %u. Assuming all frames are georeferenced.", frame->getFrameId());
    //  创建一个DummyReferencer实例并用输入帧的地理参考信息进行初始化 
    m_georeferencer = std::make_shared<DummyReferencer>(frame->getGeoreference());
  }

  // Push to buffer for visual tracking
  if (m_use_vslam)
    pushToBufferNoPose(frame);
  else
    pushToBufferPublish(frame);
  notify();

  // Ringbuffer implementation for buffer with no pose
  if (m_buffer_no_pose.size() > m_queue_size)
  {
    std::unique_lock<std::mutex> lock(m_mutex_buffer_no_pose);
    m_buffer_no_pose.pop_front();
    updateStatisticsSkippedFrame();
  }
  //使用m_transport_pose函数将某些帧的姿态信息传输至输出
  m_transport_pose(frame->getDefaultPose(), frame->getGnssUtm().zone, frame->getGnssUtm().band, "output/pose/gnss");
}

//获取队列的深度
uint32_t PoseEstimation::getQueueDepth() {
  // If no vslam is used, only the publish buffer has elements
  // If vslam is in use, then frames can be added to no_pose as well as do_publish
  // To get an accurate read of the backlog, it should include both elements.
  return m_buffer_no_pose.size() + m_buffer_do_publish.size();
}

bool PoseEstimation::process()
{
  // Trigger; true if there happened any processing in this cycle.
  bool has_processed = false;

  // Grab georeference flag once at the beginning, to avoid multithreading problems
  if (m_use_vslam)
  //检查地理参考是否已初始化,将结果存储在 m_is_georef_initialized 中
    m_is_georef_initialized = m_georeferencer->isInitialized();

  // Process new frames without a visual pose currently
  if (!m_buffer_no_pose.empty())
  {
    // Grab frame from buffer with no poses
    //从没有姿态的缓冲区中获取一个帧
    Frame::Ptr frame = getNewFrameTracking();

    LOG_IF_F(INFO, m_stage_statistics.frames_processed % 10 == 0, "Buffer [all, init, publish]: %lu, %lu, %lu",
             m_buffer_pose_all.size(), m_buffer_pose_init.size(), m_buffer_do_publish.size());

    // Track current frame -> compute visual accurate pose
    //对当前帧进行跟踪
    track(frame);

    // Identify buffer for push
    if (frame->hasAccuratePose())
    {
      // Branch accurate pose and georef initialized
      if (m_is_georef_initialized)
      {
        //计算当前帧的尺度变化并记录信息
        double scale_change = m_georeferencer->computeScaleChange(frame);
        LOG_F(INFO, "Info [Scale Drift]: Scale change of current frame: %4.2f%%", scale_change);

        bool is_scale_consistent = (scale_change < m_th_scale_change);
        if (!is_scale_consistent)//检查尺度变化是否在阈值范围内
        {
          LOG_F(WARNING, "Detected scale divergence.");

          // For now, just warn that our scale may be off until we resolve the issues discussed here:
          // https://github.com/laxnpander/OpenREALM/pull/59

          // frame->setPoseAccurate(false);
          // frame->setKeyframe(false);
          //
          // if (m_do_auto_reset)
          // {
          //   LOG_F(WARNING, "Resetting.");
          //   m_reset_requested = true;
          // reset();
          // }
        }

        // if (is_scale_consistent && m_do_update_georef && !m_georeferencer->isBuisy())
        if (m_do_update_georef && !m_georeferencer->isBuisy())
        {
          //创建一个新线程来执行地理参考更新操作
          std::thread th(std::bind(&GeospatialReferencerIF::update, m_georeferencer, frame));
          th.detach();
        }
        //将当前帧推送到包含所有帧的缓冲区中
        pushToBufferAll(frame);
      }
    }
    //检查当前帧是否被标记为关键帧并且地理参考尚未初始化
    if (frame->isKeyframe() && !m_is_georef_initialized)
    {
      //将当前帧推送到包含所有帧的缓冲区
      pushToBufferAll(frame);
      //将当前帧推送到包含初始化帧的缓冲区
      pushToBufferInit(frame);
    }

    // Data was processed during this loop
    has_processed = true;
  }

  // Handles georeference initialization and georeferencing of frame poses
  // but only starts, if a new frame was processed during this loop
  if (m_use_vslam && has_processed)
  {
    if (!m_is_georef_initialized && !m_buffer_pose_init.empty() && !m_georeferencer->isBuisy())
    {
      // Branch: Georef is not calculated yet
      //记录初始化帧缓冲区的大小
      LOG_F(INFO, "Size of init buffer: %lu", m_buffer_pose_init.size());
      //创建一个新线程来执行地理参考初始化操作
      std::thread th(std::bind(&GeospatialReferencerIF::init, m_georeferencer, m_buffer_pose_init));
      //分离线程
      th.detach();
      has_processed = true;
    }
    else if (m_is_georef_initialized && !m_buffer_pose_all.empty())
    {
      // Branch: Georef was successfully initialized and data waits to be georeferenced
      // Process all measurements in seperate thread
      if (!m_buffer_pose_init.empty())
      //清空
        m_buffer_pose_init.clear();
      //创建一个新线程来执行将地理参考应用于缓冲区的操作
      std::thread th(std::bind(&PoseEstimation::applyGeoreferenceToBuffer, this));
      th.detach();
      has_processed = true;
    }
  }
  return has_processed;
}

void PoseEstimation::track(Frame::Ptr &frame)
{
  LOG_SCOPE_FUNCTION(INFO);//记录函数执行的日志范围
  LOG_F(INFO, "Frame id: #%i, timestamp: %lu", frame->getFrameId(), frame->getTimestamp());

  // Check if initial guess should be computed
  cv::Mat T_c2w_initial;
  //检查是否需要计算初始猜测
  if (m_use_initial_guess && m_georeferencer->isInitialized())
  {
    LOG_F(INFO, "Computing initial guess of current pose...");
    //计算初始姿态猜测
    T_c2w_initial = computeInitialPoseGuess(frame);
  }

  // Compute visual pose
  //计算视觉姿态
  m_mutex_vslam.lock();
  //进行视觉姿态跟踪
  VisualSlamIF::State state = m_vslam->track(frame, T_c2w_initial);
  m_mutex_vslam.unlock();

  if (m_set_all_frames_keyframes && (state == VisualSlamIF::State::FRAME_INSERT))
  {
    state = VisualSlamIF::State::KEYFRAME_INSERT;
  }

  // Identify SLAM state
  //根据视觉SLAM的状态进行不同的操作
  switch (state)
  {
    case VisualSlamIF::State::LOST:

      // If we haven't yet initialized, make sure we haven't completely lost tracking.  If we have, force a reset
      // to hopefully recover.
      if (!m_is_georef_initialized) {
        m_init_lost_frames++;

        if (m_init_lost_frames > m_init_lost_frames_reset_count) {
          LOG_F(WARNING, "Lost tracking while initializing georeferencing, resetting VSLAM to reacquire reference...");
          std::unique_lock<std::mutex> lock(m_mutex_vslam);
          m_vslam->reset();
          m_init_lost_frames = 0;
        }
      }

      if (estimatePercOverlap(frame) < m_overlap_max_fallback)
        pushToBufferPublish(frame);
      LOG_F(WARNING, "No tracking.");
      break;
    case VisualSlamIF::State::INITIALIZED:
      LOG_F(INFO, "Visual SLAM initialized.");
      break;
    case VisualSlamIF::State::FRAME_INSERT:
      LOG_F(INFO, "Frame insertion.");
      break;
    case VisualSlamIF::State::KEYFRAME_INSERT:
      frame->setKeyframe(true);
      LOG_F(INFO, "Key frame insertion.");
      break;
  }
  // Save tracked img with features in member
  std::unique_lock<std::mutex> lock(m_mutex_img_debug);
  //将视觉SLAM跟踪结果绘制到调试图像中m_img_debug
  m_vslam->drawTrackedImage(m_img_debug);
}

void PoseEstimation::reset()
{
  std::unique_lock<std::mutex> lock(m_mutex_reset_requested);
  std::unique_lock<std::mutex> lock1(m_mutex_buffer_no_pose);
  std::unique_lock<std::mutex> lock2(m_mutex_buffer_pose_init);
  std::unique_lock<std::mutex> lock3(m_mutex_buffer_pose_all);
  //检查是否已请求重置
  if (m_reset_requested)  // User reset
  {
    //输出日志，指示重置已被请求
    LOG_F(INFO, "Reset has been requested!");
    // reset visual slam
    m_mutex_vslam.lock();
    //重置视觉SLAM
    m_vslam->reset();
    m_mutex_vslam.unlock();
  }
  else  // visual slam reset
  {
    //输出日志，指示视觉SLAM 触发了重置
    LOG_F(INFO, "Visual SLAM has triggered reset!");
    // In case of reset by vslam, publish all images in buffer with
    // default pose if overlap is less than defined
    for (auto const &frame : m_buffer_pose_all)
    {
      //将准确姿态和关键帧标志设置为假
      frame->setPoseAccurate(false);
      frame->setKeyframe(false);
      //推送到发布缓冲区
      pushToBufferPublish(frame);
    }
  }
  // Clear all buffers except the publisher buffer
  //清空所有缓冲区
  m_buffer_pose_all.clear();
  m_buffer_no_pose.clear();
  m_buffer_pose_init.clear();
  //重置初始化丢失的帧计数器为0
  m_init_lost_frames = 0;

  // Reset georeferencing
  if (m_use_vslam)
  //重新创建一个新的地理参考器实例
    m_georeferencer.reset(new GeometricReferencer(m_th_error_georef, m_min_nrof_frames_georef));
  m_stage_publisher->requestReset();//请求发布器阶段进行重置
  m_is_georef_initialized = false;
  m_reset_requested = false;
}
//将IMU数据加入到视觉SLAM中进行处理
void PoseEstimation::queueImuData(const VisualSlamIF::ImuData &imu) const
{
  m_vslam->queueImuData(imu);
}

//更改参数设置
bool PoseEstimation::changeParam(const std::string& name, const std::string &val)
{
  std::unique_lock<std::mutex> lock;
  if (name == "use_vslam")
  {
    m_use_vslam = (val == "true" || val == "1");
    return true;
  }
  return false;
}

//
void PoseEstimation::initStageCallback()
{
  // The publisher has to decide on it's own to create folders
  // So inform it, even if we don't log ourselves
  //通知发布器设置输出路径为 m_stage_path
  m_stage_publisher->setOutputPath(m_stage_path);

  // If we aren't saving any information, skip directory creation
  //如果不需要保存任何信息，则直接返回
  if (!(m_log_to_file || m_settings_save.save_required()))
  {
    return;
  }

  // Stage directory first
  //如果需要保存信息，首先检查主文件夹是否存在，如果不存在则创建
  if (!io::dirExists(m_stage_path))
    io::createDir(m_stage_path);

  // Then sub directories
  //根据不同的保存设置，创建子文件夹
  if (!io::dirExists(m_stage_path + "/trajectory") && m_settings_save.save_trajectory())
    io::createDir(m_stage_path + "/trajectory");
  if (!io::dirExists(m_stage_path + "/keyframes") && m_settings_save.save_keyframes)
    io::createDir(m_stage_path + "/keyframes");
  if (!io::dirExists(m_stage_path + "/keyframes_full") && m_settings_save.save_keyframes_full)
    io::createDir(m_stage_path + "/keyframes_full");
  if (!io::dirExists(m_stage_path + "/frames") && m_settings_save.save_frames)
    io::createDir(m_stage_path + "/frames");

}

//将阶段处理的设置打印到日志中
void PoseEstimation::printSettingsToLog()
{
  LOG_F(INFO, "### Stage process settings ###");
  LOG_F(INFO, "- use_vslam: %i", m_use_vslam);
  LOG_F(INFO, "- use_fallback: %i", m_use_fallback);
  LOG_F(INFO, "- do_update_georef: %i", m_do_update_georef);
  LOG_F(INFO, "- do_suppress_outdated_pose_pub: %i", m_do_suppress_outdated_pose_pub);
  LOG_F(INFO, "- th_error_georef: %4.2f", m_th_error_georef);
  LOG_F(INFO, "- min_nrof_frames_georef: %d", m_min_nrof_frames_georef);
  LOG_F(INFO, "- init_lost_frames_reset_count: %df", m_init_lost_frames_reset_count);
  LOG_F(INFO, "- overlap_max: %4.2f", m_overlap_max);
  LOG_F(INFO, "- overlap_max_fallback: %4.2f", m_overlap_max_fallback);

  LOG_F(INFO, "### Stage save settings ###");
  LOG_F(INFO, "- save_trajectory_gnss: %i", m_settings_save.save_trajectory_gnss);
  LOG_F(INFO, "- save_trajectory_visual: %i", m_settings_save.save_trajectory_visual);
  LOG_F(INFO, "- save_frames: %i", m_settings_save.save_frames);
  LOG_F(INFO, "- save_keyframes: %i", m_settings_save.save_keyframes);
  LOG_F(INFO, "- save_keyframes_full: %i", m_settings_save.save_keyframes_full);

  if (m_use_vslam)
    m_vslam->printSettingsToLog();
}

//用于将帧数据推送到没有姿态的缓冲区 
void PoseEstimation::pushToBufferNoPose(const Frame::Ptr &frame)
{
  std::unique_lock<std::mutex> lock(m_mutex_buffer_no_pose);
  m_buffer_no_pose.push_back(frame);
}

//用于将帧数据推送到初始化帧的缓冲区
void PoseEstimation::pushToBufferInit(const Frame::Ptr &frame)
{
  std::unique_lock<std::mutex> lock(m_mutex_buffer_pose_init);
  m_buffer_pose_init.push_back(frame);
}

//用于将帧数据推送到包含所有帧的缓冲区
void PoseEstimation::pushToBufferAll(const Frame::Ptr &frame)
{
  std::unique_lock<std::mutex> lock(m_mutex_buffer_pose_all);
  m_buffer_pose_all.push_back(frame);
}

//用于将帧数据推送到用于发布的缓冲区
void PoseEstimation::pushToBufferPublish(const Frame::Ptr &frame)
{
  std::unique_lock<std::mutex> lock(m_mutex_buffer_do_publish);
  m_buffer_do_publish.push_back(frame);
}

//用于更新先前感兴趣区域的信息
void PoseEstimation::updatePreviousRoi(const Frame::Ptr &frame)
{
  m_roi_prev = frame->getCamera()->projectImageBoundsToPlaneRoi(m_plane_ref.pt, m_plane_ref.n);
}

void PoseEstimation::updateKeyframeCb(int id, const cv::Mat &pose, const cv::Mat &points)
{
//  std::unique_lock<std::mutex> lock(m_mutex_buffer_pose_all);
//
//  // Find frame in "all" buffer for updating
//  for (auto &frame : m_buffer_pose_all)
//    if (frame->getFrameId() == (uint32_t)id)
//    {
//      if (!pose.empty())
//        frame->setVisualPose(pose);
//      if (!points.empty())
//        frame->setSparseCloud(points, true);
//      frame->setKeyframe(true);
//    }
//
//  for (auto &frame : m_buffer_do_publish)
//    if (frame->getFrameId() == (uint32_t)id)
//    {
//      if (!pose.empty())
//        frame->setVisualPose(pose);
//      //if (!points.empty())
//      //  frame->setSurfacePoints(points);
//      //frame->setKeyframe(true);
//    }
}

//估计当前帧与先前感兴趣区域的重叠百分比
double PoseEstimation::estimatePercOverlap(const Frame::Ptr &frame)
{
  cv::Rect2d roi_curr = frame->getCamera()->projectImageBoundsToPlaneRoi(m_plane_ref.pt, m_plane_ref.n);
  return ((roi_curr & m_roi_prev).area() / roi_curr.area()) * 100;
}

//从没有姿态的帧缓冲区获取新的帧以进行跟踪
Frame::Ptr PoseEstimation::getNewFrameTracking()
{
  //算帧缓冲区中帧的占用百分比
  double perc_queue = static_cast<double>(m_buffer_no_pose.size()) / m_queue_size * 100.0;
  LOG_IF_F(INFO,    perc_queue <  80, "Input frame buffer at %4.2f%%", perc_queue);
  LOG_IF_F(WARNING, perc_queue >= 80, "Input frame buffer at %4.2f%%", perc_queue);

  std::unique_lock<std::mutex> lock(m_mutex_buffer_no_pose);
  Frame::Ptr frame = m_buffer_no_pose.front();
  m_buffer_no_pose.pop_front();

  updateStatisticsProcessedFrame();

  return std::move(frame);
}
//从发布帧缓冲区获取新的帧以进行发布
Frame::Ptr PoseEstimation::getNewFramePublish()
{
  std::unique_lock<std::mutex> lock(m_mutex_buffer_do_publish);
  Frame::Ptr frame = m_buffer_do_publish.front();
  m_buffer_do_publish.pop_front();
  return std::move(frame);
}

//将估计的地理参考应用于缓冲区中的所有帧
void PoseEstimation::applyGeoreferenceToBuffer()
{
  // Grab estimated georeference
  m_mutex_t_w2g.lock();
  //获取估计的地理参考矩阵 m_T_w2g
  m_T_w2g = m_georeferencer->getTransformation();
  m_mutex_t_w2g.unlock();

  // Apply estimated georeference to all measurements in the buffer
  //循环处理缓冲区中的每个帧
  while(!m_buffer_pose_all.empty())
  {
    m_mutex_buffer_pose_all.lock();
    Frame::Ptr frame = m_buffer_pose_all.front();
    m_buffer_pose_all.pop_front();
    m_mutex_buffer_pose_all.unlock();

    // But check first, if frame has actually a visually estimated pose information
    // In case of default GNSS pose generated from lat/lon/alt/heading, pose is already in world frame
    if (frame->hasAccuratePose())
    {
      frame->initGeoreference(m_T_w2g);
    }
    //将帧推送到发布缓冲区
    pushToBufferPublish(frame);
  }
}

//计算初始姿态猜测
cv::Mat PoseEstimation::computeInitialPoseGuess(const Frame::Ptr &frame)
{
  //获取帧的默认姿态矩阵 default_pose 和地理参考矩阵 T_w2g
  cv::Mat default_pose = frame->getDefaultPose();
  cv::Mat T_w2g = m_georeferencer->getTransformation();
  T_w2g.pop_back();

  // Compute scale of georeference
  //计算地理参考的尺度
  double sx = cv::norm(T_w2g.col(0));
  double sy = cv::norm(T_w2g.col(1));
  double sz = cv::norm(T_w2g.col(2));

  // Remove scale from georeference
  //前三列进行归一化
  T_w2g.col(0) /= sx;
  T_w2g.col(1) /= sy;
  T_w2g.col(2) /= sz;

  // Invert transformation from world to global frame
  //构建从全局到世界坐标系的变换矩阵 T_g2w
  cv::Mat T_g2w = cv::Mat::eye(4, 4, T_w2g.type());
  cv::Mat R_t = (T_w2g.rowRange(0, 3).colRange(0, 3)).t();
  cv::Mat t = -R_t*T_w2g.rowRange(0, 3).col(3);
  R_t.copyTo(T_g2w.rowRange(0, 3).colRange(0, 3));
  t.copyTo(T_g2w.rowRange(0, 3).col(3));

  // Add row for homogenous coordinates
  cv::Mat hom = (cv::Mat_<double>(1, 4) << 0.0, 0.0, 0.0, 1.0);
  default_pose.push_back(hom);

  // Finally transform default pose from geographic into the world coordinate frame
  cv::Mat default_pose_in_world = T_g2w * default_pose;

  // Apply scale change
  //对计算得到的姿态矩阵进行尺度的逆缩放操作
  default_pose_in_world.at<double>(0, 3) = default_pose_in_world.at<double>(0, 3) / sx;
  default_pose_in_world.at<double>(1, 3) = default_pose_in_world.at<double>(1, 3) / sy;
  default_pose_in_world.at<double>(2, 3) = default_pose_in_world.at<double>(2, 3) / sz;

  return default_pose_in_world.rowRange(0, 3);
}

//将地理参考信息和视觉估计的信息打印到日志中
void PoseEstimation::printGeoReferenceInfo(const Frame::Ptr &frame)
{
  UTMPose utm = frame->getGnssUtm();
  cv::Mat t = frame->getCamera()->t();

  LOG_F(INFO, "Georeferenced pose:");
  LOG_F(INFO, "GNSS: [%10.2f, %10.2f, %4.2f]", utm.easting, utm.northing, utm.altitude);
  LOG_F(INFO, "Visual: [%10.2f, %10.2f, %4.2f]", t.at<double>(0), t.at<double>(1), t.at<double>(2));
  LOG_F(INFO, "Diff: [%10.2f, %10.2f, %4.2f]", utm.easting-t.at<double>(0), utm.northing-t.at<double>(1), utm.altitude-t.at<double>(2));
}

//初始化姿态估计的输入输出
PoseEstimationIO::PoseEstimationIO(PoseEstimation* stage, double rate, bool do_delay_keyframes)
    : WorkerThreadBase("Publisher [pose_estimation]", static_cast<int64_t>(1/rate*1000.0), false),
      m_is_time_ref_set(false),
      m_is_new_output_path_set(false),
      m_do_delay_keyframes(do_delay_keyframes),
      m_t_ref({0, 0}),
      m_stage_handle(stage)
{
  //检查 stage 是否为有效指针，如果为 NULL，抛出异常
  if (!m_stage_handle)
    throw(std::invalid_argument("Error: Could not create PoseEstimationIO. Stage handle points to NULL."));
}

//设置输出路径
void PoseEstimationIO::setOutputPath(const std::string &path)
{
  //m_path_output 更新为传入的路径
  m_path_output = path;
  m_is_new_output_path_set = true;
}

//处理姿态估计的输入输出
bool PoseEstimationIO::process()
{
  //检查是否有新的输出路径被设置
  if (m_is_new_output_path_set)
  {
    m_is_new_output_path_set = false;
  }
  //如果发布帧的缓冲区不为空
  if (!m_stage_handle->m_buffer_do_publish.empty())
  {
    // Grab frame from pose estimation geoereferenced mmts
    //从姿态估计的发布帧缓冲区中获取一个帧
    Frame::Ptr frame = m_stage_handle->getNewFramePublish();

    // Data to be published for every georeferenced frame (usually small data packages).
    // Poses get only published, if suppress flag was not set (old poses might crash state estimate filter)
    //如果不满足抑制标志条件，发布姿态信息
    if (!(m_stage_handle->m_do_suppress_outdated_pose_pub && m_stage_handle->m_buffer_do_publish.size() > 1))
      publishPose(frame);

    // Check what type of frame
    bool is_gnss_frame = !m_stage_handle->m_use_vslam && m_stage_handle->estimatePercOverlap(frame) < m_stage_handle->m_overlap_max_fallback;
    bool is_vslam_frame = frame->isKeyframe() && m_stage_handle->estimatePercOverlap(frame) < m_stage_handle->m_overlap_max;
    bool is_vslam_fallback_frame = m_stage_handle->m_use_fallback && !frame->hasAccuratePose() && m_stage_handle->estimatePercOverlap(frame) < m_stage_handle->m_overlap_max_fallback;

    ////根据不同类型的帧进行判断，然后调用相应的处理函数
    // Keyframes to be published (big data packages -> publish only if needed)
    if (is_gnss_frame || is_vslam_frame || is_vslam_fallback_frame)
    {
      if (is_vslam_fallback_frame)
      {
        LOG_F(INFO, "VSLAM tracking lost, using GNSS fallback for frame %d!", frame->getFrameId());
      }
      m_stage_handle->updatePreviousRoi(frame);
      if (m_do_delay_keyframes)
        scheduleFrame(frame);
      else
        publishFrame(frame);
    }
  }
  if (!m_stage_handle->m_img_debug.empty())
  {
    std::unique_lock<std::mutex> lock(m_stage_handle->m_mutex_img_debug);
    m_stage_handle->m_transport_img(m_stage_handle->m_img_debug, "debug/tracked");
    m_stage_handle->m_img_debug.release();
  }
  //发布已调度的帧
  publishScheduled();
  return false;
}

//重置姿态估计的输入输出状态
void PoseEstimationIO::reset()
{
  std::unique_lock<std::mutex> lock(m_mutex_reset_requested);
  //将时间参考和相关的标志重置为初始值
  m_is_time_ref_set = false;
  m_t_ref = TimeReference{0, 0};
  m_reset_requested = false;
}

//发布帧的姿态信息
void PoseEstimationIO::publishPose(const Frame::Ptr &frame)
{
  //打印日志，表示正在发布帧的姿态信息
  LOG_F(INFO, "Publishing pose of frame #%u...", frame->getFrameId());

  // Save trajectories
  //根据设置，保存轨迹信息到文件中
  if (m_stage_handle->m_settings_save.save_trajectory_gnss || m_stage_handle->m_settings_save.save_trajectory_visual)
    io::saveTimestamp(frame->getTimestamp(), frame->getFrameId(), m_stage_handle->m_stage_path + "/trajectory/timestamps.txt");
  if (m_stage_handle->m_settings_save.save_trajectory_gnss)
    io::saveTrajectory(frame->getTimestamp(), frame->getDefaultPose(), m_stage_handle->m_stage_path + "/trajectory/gnss_traj_TUM.txt");
  if (m_stage_handle->m_settings_save.save_trajectory_visual && frame->hasAccuratePose())
    io::saveTrajectory(frame->getTimestamp(), frame->getPose(), m_stage_handle->m_stage_path + "/trajectory/f_traj_TUM.txt");
  if (m_stage_handle->m_settings_save.save_trajectory_visual && frame->isKeyframe())
    io::saveTrajectory(frame->getTimestamp(), frame->getPose(), m_stage_handle->m_stage_path + "/trajectory/kf_traj_TUM.txt");
    //将帧的姿态信息发布出去
  m_stage_handle->m_transport_pose(frame->getPose(), frame->getGnssUtm().zone, frame->getGnssUtm().band, "output/pose/visual");
}

//发布帧的稀疏点云信息
void PoseEstimationIO::publishSparseCloud(const Frame::Ptr &frame)
{
  //获取帧中的稀疏点云数据
  PointCloud::Ptr sparse_cloud = frame->getSparseCloud();
  if (!sparse_cloud || sparse_cloud->empty())
    return;
    //将稀疏点云数据发布出去
  m_stage_handle->m_transport_pointcloud(sparse_cloud, "output/pointcloud");
}

//发布帧的信息
void PoseEstimationIO::publishFrame(const Frame::Ptr &frame)
{
  // First update statistics about outgoing frame rate
  m_stage_handle->updateStatisticsOutgoing();//更新有关输出帧速率的统计信息

  // Two situation can occure, when publishing a frame is triggered
  // 1) Frame is marked as keyframe by the SLAM -> publish directly
  // 2) Frame is not marked as keyframe -> mostly in GNSS only situations.
  //根据帧的类型，打印日志
  LOG_IF_F(INFO, frame->isKeyframe(), "Publishing keyframe #%u...", frame->getFrameId());
  LOG_IF_F(INFO, !frame->isKeyframe(), "Publishing frame #%u...", frame->getFrameId());
  //发布稀疏点云数据
  publishSparseCloud(frame);
  //将帧数据发布出去
  m_stage_handle->m_transport_frame(frame, "output/frame");
  //打印地理参考信息
  m_stage_handle->printGeoReferenceInfo(frame);

/*如果支持 Exiv2 库，则根据设置，保存图像相关的数据到文件中
包括普通帧和关键帧的图像数据。如果不支持 Exiv2 库，则打印警告信息
*/
#ifdef WITH_EXIV2
  // Save image related data only if Exiv2 is enabled
  if (m_stage_handle->m_settings_save.save_frames && !frame->isKeyframe())
    io::saveExifImage(frame, m_stage_handle->m_stage_path + "/frames", "frames", frame->getFrameId(), true);
  if (m_stage_handle->m_settings_save.save_keyframes && frame->isKeyframe())
    io::saveExifImage(frame, m_stage_handle->m_stage_path + "/keyframes", "keyframe", frame->getFrameId(), true);
  if (m_stage_handle->m_settings_save.save_keyframes_full && frame->isKeyframe())
    io::saveExifImage(frame, m_stage_handle->m_stage_path + "/keyframes_full", "keyframe_full", frame->getFrameId(), false);
#else
  if (m_stage_handle->m_settings_save.save_frames && !frame->isKeyframe())
    LOG_F(WARNING, "Exiv2 Library required for save frames from pose_estimation!");
  if (m_stage_handle->m_settings_save.save_keyframes && frame->isKeyframe())
    LOG_F(WARNING, "Exiv2 Library required for save keyframes from pose_estimation!");
  if (m_stage_handle->m_settings_save.save_keyframes_full && frame->isKeyframe())
    LOG_F(WARNING, "Exiv2 Library required for save full keyframes from pose_estimation!");
#endif
}

//将帧数据添加到计划队列中
void PoseEstimationIO::scheduleFrame(const Frame::Ptr &frame)
{
  //获取当前时间戳，用于计算时间间隔
  long t_world = Timer::getCurrentTimeMilliseconds();       // millisec
  uint64_t t_frame = frame->getTimestamp();  	              // millisec

  // Check if either first measurement ever (does not have to be keyframe)
  // Or if first keyframe
  //如果是第一次添加计划或第一次添加关键帧到计划中，设置时间参考
  if ((frame->isKeyframe() && !m_is_time_ref_set)
      || (m_t_ref.first == 0 && m_t_ref.second == 0))
  {
    m_t_ref.first = t_world;
    m_t_ref.second = t_frame;
    if (frame->isKeyframe())
      m_is_time_ref_set = true;
  }

  // Create task for publish
  //计算时间间隔 dt
  uint64_t dt = (t_frame - m_t_ref.second);
  std::unique_lock<std::mutex> lock(m_mutex_schedule);
  m_schedule.emplace_back(Task{(long)dt, frame});

  // Time until schedule
  //计算距离任务执行的剩余时间
  long t_remain = ((long)dt) - (getCurrentTimeMilliseconds() - m_t_ref.first);

  LOG_F(INFO, "Scheduled publish frame #%u in %4.2fs", frame->getFrameId(), (double)t_remain/10e3);
}

//检查计划中的任务是否需要发布
void PoseEstimationIO::publishScheduled()
{
  if (m_schedule.empty())
    return;
  Task task = m_schedule.front();//获取计划队列中的第一个任务
  if (task.first < (getCurrentTimeMilliseconds() - m_t_ref.first))
  {
    m_mutex_schedule.lock();
    m_schedule.pop_front();
    m_mutex_schedule.unlock();
    publishFrame(task.second);
  }
}

//发布计划队列中的所有任务
void PoseEstimationIO::publishAll()
{
  //获取计划队列中的所有任务，并逐个调用 publishFrame 函数，发布帧数据
  m_mutex_schedule.lock();
  for (const auto &task : m_schedule)
  {
    publishFrame(task.second);
  }
  //清空计划队列
  m_schedule.clear();
  m_mutex_schedule.unlock();
}
