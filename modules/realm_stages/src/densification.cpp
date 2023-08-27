

#include <realm_stages/densification.h>

using namespace realm;
using namespace stages;

Densification::Densification(const StageSettings::Ptr &stage_set,
                             const DensifierSettings::Ptr &densifier_set,
                             double rate)
: StageBase("densification", (*stage_set)["path_output"].toString(), rate, (*stage_set)["queue_size"].toInt(), bool((*stage_set)["log_to_file"].toInt())),
  m_use_filter_bilat((*stage_set)["use_filter_bilat"].toInt() > 0),
  m_use_filter_guided((*stage_set)["use_filter_guided"].toInt() > 0),
  m_depth_min_current(0.0),
  m_depth_max_current(0.0),
  m_do_drop_planar((*stage_set)["compute_normals"].toInt() > 0),
  m_compute_normals((*stage_set)["compute_normals"].toInt() > 0),
  m_rcvd_frames(0),
  m_settings_save({(*stage_set)["save_bilat"].toInt() > 0,
                  (*stage_set)["save_dense"].toInt() > 0,
                  (*stage_set)["save_guided"].toInt() > 0,
                  (*stage_set)["save_imgs"].toInt() > 0,
                  (*stage_set)["save_sparse"].toInt() > 0,
                  (*stage_set)["save_thumb"].toInt() > 0,
                  (*stage_set)["save_normals"].toInt() > 0})
{
  //注册异步数据就绪函数，该函数将在缓冲区不为空时返回 true
  registerAsyncDataReadyFunctor([=]{ return !m_buffer_reco.empty(); });
  //建一个稠密化器实例，并存储在 m_densifier 成员变量中
  m_densifier = densifier::DensifierFactory::create(densifier_set);
  //获取输入帧的数量
  m_n_frames = m_densifier->getNrofInputFrames();

  // Creation of reference plane, currently only the one below is supported
  //创建一个参考平面，其中点位于原点，法线指向 Z 轴正方向
  m_plane_ref.pt = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 0.0);
  m_plane_ref.n = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 1.0);
}

//将一个帧添加到稠密化处理队列中进行处理
void Densification::addFrame(const Frame::Ptr &frame)
{
  // First update statistics about incoming frame rate
  //更新关于传入帧速率的统计信息
  updateStatisticsIncoming();

  // Increment received valid frames
  //增加接收到的有效帧数
  m_rcvd_frames++;

  // Check if frame and settings are fulfilled to process/densify incoming frames
  // if not, redirect to next stage
  //检查帧是否满足进行处理/稠密化的条件，如果不满足，则将帧传递到下一个阶段并记录日志
  if (   !frame->isKeyframe()
      || !frame->hasAccuratePose()
      || !frame->isDepthComputed())
  {
    LOG_F(INFO, "Frame #%u:", frame->getFrameId());
    LOG_F(INFO, "Keyframe? %s", frame->isKeyframe() ? "Yes" : "No");
    LOG_F(INFO, "Accurate Pose? %s", frame->hasAccuratePose() ? "Yes" : "No");
    LOG_F(INFO, "Surface? %i Points", frame->getSparseCloud() != nullptr ? frame->getSparseCloud()->size() : 0);

    LOG_F(INFO, "Frame #%u not suited for dense reconstruction. Passing through...", frame->getFrameId());
    if (!m_do_drop_planar)
      m_transport_frame(frame, "output/frame");
    updateStatisticsBadFrame();
    return;
  }
  //满足稠密化条件，将帧添加到稠密化处理缓冲区
  pushToBufferReco(frame);
  notify();
}

bool Densification::process()
{
  // NOTE: All depthmap maps are CV_32F except they are explicitly casted

  // First check if buffer has enough frames already, else don't do anything
  //检查稠密化缓冲区中是否有足够的帧用于进行稠密化处理
  if (m_buffer_reco.size() < m_n_frames)
  {
    return false;
  }

  // Densification step using stereo
  //执行稠密重建
  //获取了当前时间的毫秒表示
  long t = getCurrentTimeMilliseconds();
  //执行了稠密化的实际处理
  Frame::Ptr frame_processed;
  Depthmap::Ptr depthmap = processStereoReconstruction(m_buffer_reco, frame_processed);
  //将已经处理过的帧从稠密化缓冲区中移除
  popFromBufferReco();
  updateStatisticsProcessedFrame();

  LOG_IF_F(INFO, m_verbose, "Timing [Dense Reconstruction]: %lu ms", getCurrentTimeMilliseconds() - t);
  if (!depthmap)
    return true;

  // Compute normals if desired
  t = getCurrentTimeMilliseconds();
  cv::Mat normals;
  //如果需要计算法线信息，就从深度图中计算法线信息
  if (m_compute_normals)
    normals = stereo::computeNormalsFromDepthMap(depthmap->data());
  LOG_IF_F(INFO, m_verbose, "Timing [Computing Normals]: %lu ms", getCurrentTimeMilliseconds() - t);

  // Remove outliers
  //强制将深度图中的深度值限制在特定范围内
  double depth_min = frame_processed->getMedianSceneDepth()*0.25;
  double depth_max = frame_processed->getMedianSceneDepth()*1.75;
  depthmap = forceInRange(depthmap, depth_min, depth_max);
  LOG_F(INFO, "Scene depthmap forced in range %4.2f ... %4.2f", depth_min, depth_max);

  // Set data in the frame
  //将生成的深度图设置到已处理的帧对象中
  t = getCurrentTimeMilliseconds();
  frame_processed->setDepthmap(depthmap);
  LOG_IF_F(INFO, m_verbose, "Timing [Setting]: %lu ms", getCurrentTimeMilliseconds() - t);

  // Creating dense cloud
  //利用深度图重新投影生成稠密点云
  cv::Mat img3d = stereo::reprojectDepthMap(depthmap->getCamera(), depthmap->data());
  cv::Mat dense_cloud = img3d.reshape(1, img3d.rows*img3d.cols);

  // Denoising
  //去除噪声
  t = getCurrentTimeMilliseconds();
  m_buffer_consistency.emplace_back(std::make_pair(frame_processed, dense_cloud));
  if (m_buffer_consistency.size() >= 4)
  {
    frame_processed = consistencyFilter(&m_buffer_consistency);
    m_buffer_consistency.pop_front();
  }
  else
  {
    LOG_IF_F(INFO, m_verbose, "Consistency filter is activated. Waiting for more frames for denoising...");
    return true;
  }
  LOG_IF_F(INFO, m_verbose, "Timing [Denoising]: %lu ms", getCurrentTimeMilliseconds() - t);

  // Last check if frame still has valid depthmap
  if (!frame_processed->getDepthmap())
  {
    //_transport_frame(frame_processed, "output/frame");
    return true;
  }

  // Post processing
  //对深度图应用后处理操作
  depthmap->data() = applyDepthMapPostProcessing(depthmap->data());

  // Savings
  //保存结果，包括帧和深度图
  t = getCurrentTimeMilliseconds();
  saveIter(frame_processed, normals);
  LOG_IF_F(INFO, m_verbose, "Timing [Saving]: %lu ms", getCurrentTimeMilliseconds() - t);

  // Republish frame to next stage
  //将处理过的帧和深度图发布到下一个处理阶段
  t = getCurrentTimeMilliseconds();
  publish(frame_processed, depthmap->data());
  LOG_IF_F(INFO, m_verbose, "Timing [Publish]: %lu ms", getCurrentTimeMilliseconds() - t);

  return true;
}

//在稠密重建过程中对深度图进行进一步的去噪和校正
Frame::Ptr Densification::consistencyFilter(std::deque<std::pair<Frame::Ptr, cv::Mat>>* buffer_denoise)
{
  //选择队列中的中间帧作为参考帧进行一致性滤波
  Frame::Ptr frame = (*buffer_denoise)[buffer_denoise->size()/2].first;

  Depthmap::Ptr depthmap_ii = frame->getDepthmap();

  cv::Mat depthmap_ii_data = depthmap_ii->data();
  int rows = depthmap_ii_data.rows;
  int cols = depthmap_ii_data.cols;
  //创建一个空白的 votes 矩阵
  cv::Mat votes = cv::Mat::zeros(rows, cols, CV_8UC1);

  float th_depth = 0.1;
//通过遍历队列中的其他帧，比较参考帧的深度图和其他帧重建的深度图，来计算一致性投票
  for (const auto &f : *buffer_denoise)
  {
    if (f.first == frame)
      continue;

    cv::Mat dense_cloud = f.second;
    cv::Mat depthmap_ij_data = stereo::computeDepthMapFromPointCloud(depthmap_ii->getCamera(), dense_cloud);

    for (int r = 0; r < rows; ++r)
      for (int c =0; c < cols; ++c)
      {
        float d_ii = depthmap_ii_data.at<float>(r, c);
        float d_ij = depthmap_ij_data.at<float>(r, c);

        if (d_ii <= 0 || d_ij <= 0)
          continue;

        if (fabsf(d_ij-d_ii)/d_ii < th_depth)
          votes.at<uchar>(r, c) = votes.at<uchar>(r, c) + 1;
      }
  }
  //根据一致性投票，将不一致的像素位置标记为无效
  cv::Mat mask = (votes < 2);
  depthmap_ii_data.setTo(-1.0, mask);

  //计算深度图覆盖率,即被保留的像素占总像素的百分比
  double perc_coverage = 100 - (static_cast<double>(cv::countNonZero(mask)) / (mask.rows*mask.cols) * 100.0);
  /*如果深度图覆盖率低于30%，认为这是一个平面区域
  在这种情况下，将帧的深度图设置为 nullptr
  并从队列中移除这个帧
  */
  if (perc_coverage < 30.0)
  {
    LOG_IF_F(WARNING, m_verbose, "Depthmap coverage too low (%3.1f%%). Assuming plane surface.", perc_coverage);
    frame->setDepthmap(nullptr);

    for (auto it = buffer_denoise->begin(); it != buffer_denoise->end(); ) {
      if (it->first->getFrameId() == frame->getFrameId())
      {
        it = buffer_denoise->erase(it);
        break;
      }
      else
        ++it;
    }
  }
  else
  {
    //保留帧的深度图，但在日志中记录覆盖率
    LOG_IF_F(INFO, m_verbose, "Depthmap coverage left after denoising: %3.1f%%", perc_coverage);
  }

  return frame;
}

//实现密集重建过程中的立体视觉重建
Depthmap::Ptr Densification::processStereoReconstruction(const std::deque<Frame::Ptr> &buffer, Frame::Ptr &frame_processed)
{
  LOG_F(INFO, "Performing stereo reconstruction...");

  // Reference frame is the one in the middle (if more than two)
  //选择中间帧作为参考帧，用于后续的稠密重建
  int ref_idx = (int)buffer.size()/2;
  frame_processed = buffer[ref_idx];

  // Compute baseline information for all frames
  //计算每个帧到参考帧的基线信息
  std::vector<double> baselines;
  baselines.reserve(buffer.size());
  
  //循环遍历每个帧，计算它们到参考帧的基线
  //并将基线信息存储在 baselines 向量中
  std::string stringbuffer;
  for (auto &f : buffer)
  {
    if (f == frame_processed)
      continue;

    baselines.push_back(stereo::computeBaselineFromPose(frame_processed->getPose(), f->getPose()));
    stringbuffer += std::to_string(baselines.back()) + "m ";
  }
  LOG_F(INFO, "Baselines to reference frame: %s", stringbuffer.c_str());

  LOG_F(INFO, "Reconstructing frame #%u...", frame_processed->getFrameId());
  //重建参考帧的深度图
  Depthmap::Ptr depthmap = m_densifier->densify(buffer, (uint8_t)ref_idx);

  LOG_IF_F(INFO, depthmap != nullptr, "Successfully reconstructed frame!");
  LOG_IF_F(WARNING, depthmap == nullptr, "Reconstruction failed!");

  return depthmap;
}

//强制将深度图中深度值限制在指定范围内
Depthmap::Ptr Densification::forceInRange(const Depthmap::Ptr &depthmap, double min_depth, double max_depth)
{
  cv::Mat mask;
  //获取深度图的数据矩阵
  cv::Mat data = depthmap->data();

  //创建一个二值掩码
  cv::inRange(data, min_depth, max_depth, mask);
  //掩码进行逐位取反操作
  cv::bitwise_not(mask, mask);
  //将深度图中深度值不在指定范围内的像素设置为 -1.0
  data.setTo(-1.0f, mask);

  return depthmap;
}

//对深度图进行后处理
cv::Mat Densification::applyDepthMapPostProcessing(const cv::Mat &depthmap)
{
  cv::Mat depthmap_filtered;
  //判断是否要应用双边滤波
  if (m_use_filter_bilat)
      //对深度图进行双边滤波处理
      cv::bilateralFilter(depthmap, depthmap_filtered, 5, 25, 25);

  /*if (_settings_save.save_bilat)
    io::saveImageColorMap(depthmap_filtered, _depth_min_current, _depth_max_current, _stage_path + "/bilat", "bilat",
                          _frame_current->getFrameId(), io::ColormapType::DEPTH);*/

  return depthmap_filtered;
}

//生成深度图遮罩
cv::Mat Densification::computeDepthMapMask(const cv::Mat &depth_map, bool use_sparse_mask)
{
  cv::Mat mask1, mask2, mask3;

  /*if (use_sparse_mask)
    densifier::computeSparseMask(_frame_current->getSparseCloud(), _frame_current->getResizedCamera(), mask1);
  else
    mask1 = cv::Mat::ones(depth_map.rows, depth_map.cols, CV_8UC1)*255;

  cv::inRange(depth_map, _depth_min_current, _depth_max_current, mask2);
  cv::bitwise_and(mask1, mask2, mask3);*/
  return mask3;
}

void Densification::publish(const Frame::Ptr &frame, const cv::Mat &depthmap)
{
  // First update statistics about outgoing frame rate
  //更新关于发送帧率的统计信息
  updateStatisticsOutgoing();

  //将帧数据传输到输出通道 "output/frame"
  m_transport_frame(frame, "output/frame");
  //将帧的位姿信息传输到输出通道 "output/pose"
  m_transport_pose(frame->getPose(), frame->getGnssUtm().zone, frame->getGnssUtm().band, "output/pose");
  //将帧的矫正后图像传输到输出通道 "output/img_rectified"
  m_transport_img(frame->getResizedImageUndistorted(), "output/img_rectified");
  //将深度图数据传输到输出通道 "output/depth"
  m_transport_depth_map(depthmap, "output/depth");
  //将稀疏点云数据传输到输出通道 "output/pointcloud
  m_transport_pointcloud(frame->getSparseCloud(), "output/pointcloud");
  
  //将深度图范围规范化到 [0, 65535] 的范围内，并设置深度大于0的像素为有效值
  cv::Mat depthmap_display;
  cv::normalize(depthmap, depthmap_display, 0, 65535, cv::NormTypes::NORM_MINMAX, CV_16UC1, (depthmap > 0));
 //将规范化后的深度图传输到输出通道 "output/depth_display"
  m_transport_img(depthmap_display, "output/depth_display");
}

//根据设置，保存处理后的数据
void Densification::saveIter(const Frame::Ptr &frame, const cv::Mat &normals)
{
  //从帧数据中获取深度图数据
  cv::Mat depthmap_data = frame->getDepthmap()->data();

  if (m_settings_save.save_imgs)
  //保存矫正后的图像
    io::saveImage(frame->getResizedImageUndistorted(), io::createFilename(m_stage_path + "/imgs/imgs_", frame->getFrameId(), ".png"));
  if (m_settings_save.save_normals && m_compute_normals && !normals.empty())
    //保存法线图
    io::saveImageColorMap(normals, (depthmap_data > 0), m_stage_path + "/normals", "normals", frame->getFrameId(), io::ColormapType::NORMALS);
  if (m_settings_save.save_sparse)
  {
    //cv::Mat depthmap_sparse = stereo::computeDepthMapFromPointCloud(frame->getResizedCamera(), frame->getSparseCloud()->data().colRange(0, 3));
    //io::saveDepthMap(depthmap_sparse, m_stage_path + "/sparse/sparse_%06i.tif", frame->getFrameId());
  }
  if (m_settings_save.save_dense)
  //保存深度图数据
    io::saveDepthMap(frame->getDepthmap(), m_stage_path + "/dense/dense_%06i.tif", frame->getFrameId());
}

void Densification::pushToBufferReco(const Frame::Ptr &frame)
{
  std::unique_lock<std::mutex> lock(m_mutex_buffer_reco);
  //将帧数据推入缓冲区
  m_buffer_reco.push_back(frame);
    //如果缓冲区的大小超过了限制
  if (m_buffer_reco.size() > m_queue_size)
  {
    //最早的帧数据从缓冲区中移除，并更新跳过的帧统计
    m_buffer_reco.pop_front();
    updateStatisticsSkippedFrame();
  }
}

void Densification::popFromBufferReco()
{
  std::unique_lock<std::mutex> lock(m_mutex_buffer_reco);
  //将最早的帧数据从缓冲区中移除
  m_buffer_reco.pop_front();
}

void Densification::reset()
{
  // TODO: Reset in _densifier
  //将收到的帧数 m_rcvd_frames 重置为零
  m_rcvd_frames = 0;
  //向日志输出信息表明重构阶段已被重置
  LOG_F(INFO, "Densification Stage: RESETED!");
}

//初始化保存阶段所需的目录结构
void Densification::initStageCallback()
{
  // If we aren't saving any information, skip directory creation
  //检查是否需要保存信息
  if (!(m_log_to_file || m_settings_save.save_required()))
  {
    return;
  }

  // Stage directory first
  //首先检查主目录是否存在
  if (!io::dirExists(m_stage_path))
      //不存在则创建主目录 m_stage_path
    io::createDir(m_stage_path);

  // Then sub directories
  //根据不同的保存选项，分别检查并创建子目录
  if (!io::dirExists(m_stage_path + "/sparse") && m_settings_save.save_sparse)
    io::createDir(m_stage_path + "/sparse");
  if (!io::dirExists(m_stage_path + "/dense") && m_settings_save.save_dense)
    io::createDir(m_stage_path + "/dense");
  if (!io::dirExists(m_stage_path + "/bilat") && m_settings_save.save_bilat)
    io::createDir(m_stage_path + "/bilat");
  if (!io::dirExists(m_stage_path + "/guided") && m_settings_save.save_guided)
    io::createDir(m_stage_path + "/guided");
  if (!io::dirExists(m_stage_path + "/normals") && m_settings_save.save_normals)
    io::createDir(m_stage_path + "/normals");
  if (!io::dirExists(m_stage_path + "/imgs") && m_settings_save.save_imgs)
    io::createDir(m_stage_path + "/imgs");
  if (!io::dirExists(m_stage_path + "/thumb") && m_settings_save.save_thumb)
    io::createDir(m_stage_path + "/thumb");
}

//将当前阶段的设置信息输出到日志中
void Densification::printSettingsToLog()
{
  //输出处理阶段的设置信息
  LOG_F(INFO, "### Stage process settings ###");
  LOG_F(INFO, "- use_filter_bilat: %i", m_use_filter_bilat);
  LOG_F(INFO, "- use_filter_guided: %i", m_use_filter_guided);
  LOG_F(INFO, "- compute_normals: %i", m_compute_normals);

  //输出保存阶段的设置信息
  LOG_F(INFO, "### Stage save settings ###");
  LOG_F(INFO, "- save_bilat: %i", m_settings_save.save_bilat);
  LOG_F(INFO, "- save_dense: %i", m_settings_save.save_dense);
  LOG_F(INFO, "- save_guided: %i", m_settings_save.save_guided);
  LOG_F(INFO, "- save_imgs: %i", m_settings_save.save_imgs);
  LOG_F(INFO, "- save_normals: %i", m_settings_save.save_normals);
  LOG_F(INFO, "- save_sparse: %i", m_settings_save.save_sparse);
  LOG_F(INFO, "- save_thumb: %i", m_settings_save.save_thumb);

  //将与密集化相关的设置信息也输出到日志中
  m_densifier->printSettingsToLog();
}

//获取当前接收缓冲区中的帧数量（即队列深度）
uint32_t Densification::getQueueDepth() {
  return m_buffer_reco.size();
}
