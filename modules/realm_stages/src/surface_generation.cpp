

#include <realm_core/loguru.h>

#include <realm_stages/surface_generation.h>

using namespace realm;
using namespace stages;
using namespace realm::ortho;

SurfaceGeneration::SurfaceGeneration(const StageSettings::Ptr &settings, double rate)
: StageBase("surface_generation", (*settings)["path_output"].toString(), rate, (*settings)["queue_size"].toInt(), bool((*settings)["log_to_file"].toInt())),
  m_try_use_elevation((*settings)["try_use_elevation"].toInt() > 0),
  m_compute_all_frames((*settings)["compute_all_frames"].toInt() > 0),
  m_knn_max_iter((*settings)["knn_max_iter"].toInt()),
  m_is_projection_plane_offset_computed(false),
  m_projection_plane_offset(0.0),
  m_mode_surface_normals(static_cast<DigitalSurfaceModel::SurfaceNormalMode>((*settings)["mode_surface_normals"].toInt())),
  m_plane_reference(Plane{(cv::Mat_<double>(3, 1) << 0.0, 0.0, 0.0), (cv::Mat_<double>(3, 1) << 0.0, 0.0, 1.0)}),
  m_settings_save({(*settings)["save_elevation"].toInt() > 0,
                  (*settings)["save_normals"].toInt() > 0})
{
  //注册一个异步数据准备函数
  registerAsyncDataReadyFunctor([=]{ return !m_buffer.empty(); });
}

void SurfaceGeneration::addFrame(const Frame::Ptr &frame)
{
  // First update statistics about incoming frame rate
  //更新有关输入帧率的统计信息
  updateStatisticsIncoming();

  std::unique_lock<std::mutex> lock(m_mutex_buffer);
  //将帧添加到缓冲区的末尾
  m_buffer.push_back(frame);
  // Ringbuffer implementation
  //检查缓冲区的大小是否超过了指定的队列大小
  if (m_buffer.size() > m_queue_size)
  {
    updateStatisticsSkippedFrame();
    m_buffer.pop_front();
  }
  notify();
}

bool SurfaceGeneration::process()
{
  bool has_processed = false;
  if (!m_buffer.empty())
  {
    // Prepare timing
    long t;

    //从缓冲区中获取一个帧
    Frame::Ptr frame = getNewFrame();
    LOG_F(INFO, "Processing frame #%u...", frame->getFrameId());

    // Identify surface assumption for input frame and compute DSM
    t = getCurrentTimeMilliseconds();
    //用于存储计算出的数字表面模型
    DigitalSurfaceModel::Ptr dsm;
    //调用 computeSurfaceAssumption 函数来确定输入帧的表面假设
    SurfaceAssumption assumption = computeSurfaceAssumption(frame);
    //根据表面假设的不同，进入不同的分支 
    switch(assumption)
    {
      case SurfaceAssumption::PLANAR:
        LOG_F(INFO, "Surface assumption: PLANAR.");
        dsm = createPlanarSurface(frame);
        break;
      case SurfaceAssumption::ELEVATION:
        LOG_F(INFO, "Surface assumption: ELEVATION.");
        dsm = createElevationSurface(frame);
        break;
    }
    //从数字表面模型 dsm 中获取表面网格
    CvGridMap::Ptr surface = dsm->getSurfaceGrid();
    LOG_IF_F(INFO, m_verbose, "Timing [Compute DSM]: %lu ms", getCurrentTimeMilliseconds() - t);

    // Observed map should be empty at this point, but check before set
    t = getCurrentTimeMilliseconds();
    //检查帧中是否已经存在表面模型
    if (!frame->getSurfaceModel())
    //将计算得到的表面网格 surface 设置为帧的表面模型
      frame->setSurfaceModel(surface);
    else
      frame->getSurfaceModel()->add(*surface, REALM_OVERWRITE_ALL, true);
    //将计算得到的表面假设设置到帧中
    frame->setSurfaceAssumption(assumption);
    LOG_IF_F(INFO, m_verbose, "Timing [Container Add]: %lu ms", getCurrentTimeMilliseconds() - t);

    LOG_F(INFO, "Publishing frame for next stage...");

    // Publishes every iteration
    t = getCurrentTimeMilliseconds();
    publish(frame);//将经过处理的帧传递给下一个阶段进行处理
    LOG_IF_F(INFO, m_verbose, "Timing [Publish]: %lu ms", getCurrentTimeMilliseconds() - t);

    // Savings every iteration
    //保存表面模型的迭代结果，以及相关的统计信息
    t = getCurrentTimeMilliseconds();
    saveIter(*surface, frame->getFrameId());
    LOG_IF_F(INFO, m_verbose, "Timing [Saving]: %lu ms", getCurrentTimeMilliseconds() - t);

    has_processed = true;
  }
  return has_processed;
}

//在运行时更改阶段的参数
bool SurfaceGeneration::changeParam(const std::string& name, const std::string &val)
{
  std::unique_lock<std::mutex> lock(m_mutex_params);
  if (name == "try_use_elevation")
  {
    //字符串 "true" 或 "1" 转换为布尔值
    m_try_use_elevation = (val == "true" || val == "1");
    return true;
  }
  else if (name == "compute_all_frames")
  {
    m_compute_all_frames = (val == "true" || val == "1");
    return true;
  }

  return false;
}

void SurfaceGeneration::reset()
{
  //重置为0
  m_projection_plane_offset = 0.0;
  //设置为 false
  m_is_projection_plane_offset_computed = false;
}

//将处理过的帧对象传递给下一个阶段
void SurfaceGeneration::publish(const Frame::Ptr &frame)
{
  // First update statistics about outgoing frame rate
  updateStatisticsOutgoing();
  m_transport_frame(frame, "output/frame");
}

//将计算得到的地表模型和法线信息保存为图像文件
void SurfaceGeneration::saveIter(const CvGridMap &surface, uint32_t id)
{
  // Invalid points are marked with NaN
  if (!surface["elevation"].empty()) {
    cv::Mat valid = (surface["elevation"] == surface["elevation"]);

    if (m_settings_save.save_elevation)
      io::saveImageColorMap(surface["elevation"], valid, m_stage_path + "/elevation", "elevation", id,
                            io::ColormapType::ELEVATION);
    if (m_settings_save.save_normals && surface.exists("elevation_normal"))
      io::saveImageColorMap(surface["elevation_normal"], valid, m_stage_path + "/normals", "normal", id,
                            io::ColormapType::NORMALS);
  } else {
    LOG_F(WARNING, "Elevation surface was empty, skipping saveIter()!");
  }
}

//在阶段初始化时根据保存设置创建输出目录结构
void SurfaceGeneration::initStageCallback()
{
  // If we aren't saving any information, skip directory creation
  if (!(m_log_to_file || m_settings_save.save_required()))
  {
    return;
  }

  // Stage directory first
  if (!io::dirExists(m_stage_path))
  //创建阶段目录
    io::createDir(m_stage_path);

  // Then sub directories
  if (!io::dirExists(m_stage_path + "/elevation") && m_settings_save.save_elevation)
    io::createDir(m_stage_path + "/elevation");
  if (!io::dirExists(m_stage_path + "/normals") && m_settings_save.save_normals)
    io::createDir(m_stage_path + "/normals");
}

//打印日志
void SurfaceGeneration::printSettingsToLog()
{
  LOG_F(INFO, "### Stage process settings ###");
  LOG_F(INFO, "- try_use_elevation: %i", m_try_use_elevation);
  LOG_F(INFO, "- compute_all_frames: %i", m_compute_all_frames);
  LOG_F(INFO, "- mode_surface_normals: %i", static_cast<int>(m_mode_surface_normals));

  LOG_F(INFO, "### Stage save settings ###");
  LOG_F(INFO, "- save_elevation: %i", m_settings_save.save_elevation);
  LOG_F(INFO, "- save_normals: %i", m_settings_save.save_normals);

}
//获取输入缓冲区中待处理帧的数量（队列深度）
uint32_t SurfaceGeneration::getQueueDepth() {
  return m_buffer.size();
}

//从输入缓冲区中获取待处理的帧
Frame::Ptr SurfaceGeneration::getNewFrame()
{
  std::unique_lock<std::mutex> lock(m_mutex_buffer);
  Frame::Ptr frame = m_buffer.front();
  m_buffer.pop_front();
  updateStatisticsProcessedFrame();
  return (std::move(frame));
}

//根据帧的性质和参数，决定在处理过程中应该使用的表面假设
SurfaceAssumption SurfaceGeneration::computeSurfaceAssumption(const Frame::Ptr &frame)
{
  std::unique_lock<std::mutex> lock(m_mutex_params);
  if (m_try_use_elevation && frame->isKeyframe() && frame->hasAccuratePose())
  {
    LOG_F(INFO, "Frame is accurate and keyframe. Checking for dense information...");
    if (frame->getDepthmap())//帧是否具有深度图信息
      return SurfaceAssumption::ELEVATION;
  }
  return SurfaceAssumption::PLANAR;
}

//计算投影平面的偏移值
double SurfaceGeneration::computeProjectionPlaneOffset(const Frame::Ptr &frame)
{
  double offset = 0.0;

  // If scene depth is computed, there is definitely enough sparse points
  if (frame->isDepthComputed())//检查帧是否已经计算了场景的深度信息
  {
    //从稀疏点云中提取出 Z 坐标，并对这些坐标进行排序
    std::vector<double> z_coord;

    cv::Mat points = frame->getSparseCloud()->data();
    for (int i = 0; i < points.rows; ++i)
      z_coord.push_back(points.at<double>(i, 2));

    sort(z_coord.begin(), z_coord.end());
    //择排序后中间位置的 Z 坐标值作为投影平面的初始偏移值
    offset = z_coord[(z_coord.size() - 1) / 2];

    LOG_F(INFO, "Sparse cloud was utilized to compute an initial projection plane at elevation = %4.2f.", offset);

    // Only consider the plane offset computed if we had a sparse cloud to work with.
    m_is_projection_plane_offset_computed = true;
  }
  else
  {
    LOG_F(INFO, "No sparse cloud set in frame. Assuming the projection plane is at elevation = %4.2f.", offset);
  }

  return offset;
}

//创建数字表面模型
DigitalSurfaceModel::Ptr SurfaceGeneration::createPlanarSurface(const Frame::Ptr &frame)
{
  //检查是否已经计算了投影平面的偏移值,是否需要计算所有帧的情况
  if (!m_is_projection_plane_offset_computed || m_compute_all_frames)
  {
    //计算投影平面的偏移值
    m_projection_plane_offset = computeProjectionPlaneOffset(frame);
  }

  // Create planar surface in world frame
  //将投影平面的图像边界转换为在世界坐标系中的矩形区域 roi
  cv::Rect2d roi = frame->getCamera()->projectImageBoundsToPlaneRoi(m_plane_reference.pt, m_plane_reference.n);

  return std::make_shared<DigitalSurfaceModel>(roi, m_projection_plane_offset);
}

//创建高程表面模型
DigitalSurfaceModel::Ptr SurfaceGeneration::createElevationSurface(const Frame::Ptr &frame)
{
  // In case of elevated surface assumption there has to be a dense depthmap
  //获取帧的深度图像数据
  Depthmap::Ptr depthmap = frame->getDepthmap();

  // Create elevated 2.5D surface in world frame
  //据参考点和法向量计算投影平面的矩形区域 roi
  cv::Rect2d roi = frame->getCamera()->projectImageBoundsToPlaneRoi(m_plane_reference.pt, m_plane_reference.n);

  // We reproject the depthmap into a 3D point cloud first, before creating the surface model
  //将深度图像数据转换为三维点云数据
  cv::Mat img3d = stereo::reprojectDepthMap(depthmap->getCamera(), depthmap->data());

  // We want to organize the point cloud with row(i) = x, y, z. Therefore we have to reshape the matrix, which right
  // now is a 3 channel matrix with a point at every pixel. Reshaping the channel to 1 dimension results in a
  // 1x(cols*rows*3) matrix. But we want a new point in every row. Therefore the number of rows must be rows*cols.
  cv::Mat dense_cloud = img3d.reshape(1, img3d.rows*img3d.cols);

  return std::make_shared<DigitalSurfaceModel>(roi, dense_cloud, m_mode_surface_normals, m_knn_max_iter);
}