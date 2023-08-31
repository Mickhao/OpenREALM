

#include <realm_core/loguru.h>
#include <realm_core/tree_node.h>
#include <realm_stages/mosaicing.h>

#ifdef WITH_PCL
#include <realm_io/pcl_export.h>
#endif

/*将正射影像进一步处理，生成高程图(DEM)和几何校正的正射影像*/
using namespace realm;
using namespace stages;

//初始化 Mosaicing 阶段的各种设置和参数
Mosaicing::Mosaicing(const StageSettings::Ptr &stage_set, double rate)
    : StageBase("mosaicing", (*stage_set)["path_output"].toString(), rate, (*stage_set)["queue_size"].toInt(), bool((*stage_set)["log_to_file"].toInt())),
      m_utm_reference(nullptr),
      m_global_map(nullptr),
      //m_mesher(nullptr),
      m_gdal_writer(nullptr),
      m_publish_mesh_nth_iter(0),
      m_publish_mesh_every_nth_kf((*stage_set)["publish_mesh_every_nth_kf"].toInt()),
      m_do_publish_mesh_at_finish((*stage_set)["publish_mesh_at_finish"].toInt() > 0),
      m_downsample_publish_mesh((*stage_set)["downsample_publish_mesh"].toDouble()),
      m_use_surface_normals(true),
      m_th_elevation_min_nobs((*stage_set)["th_elevation_min_nobs"].toInt()),
      m_th_elevation_var((*stage_set)["th_elevation_variance"].toFloat()),
      m_settings_save({(*stage_set)["split_gtiff_channels"].toInt() > 0,
                      (*stage_set)["save_ortho_rgb_one"].toInt() > 0,
                      (*stage_set)["save_ortho_rgb_all"].toInt() > 0,
                      (*stage_set)["save_ortho_gtiff_one"].toInt() > 0,
                      (*stage_set)["save_ortho_gtiff_all"].toInt() > 0,
                      (*stage_set)["save_elevation_one"].toInt() > 0,
                      (*stage_set)["save_elevation_all"].toInt() > 0,
                      (*stage_set)["save_elevation_var_one"].toInt() > 0,
                      (*stage_set)["save_elevation_var_all"].toInt() > 0,
                      (*stage_set)["save_elevation_obs_angle_one"].toInt() > 0,
                      (*stage_set)["save_elevation_obs_angle_all"].toInt() > 0,
                       (*stage_set)["save_elevation_mesh_one"].toInt() > 0,
                       (*stage_set)["save_num_obs_one"].toInt() > 0,
                       (*stage_set)["save_num_obs_all"].toInt() > 0,
                       (*stage_set)["save_dense_ply"].toInt() > 0})
{
  //显示阶段的名称和设置信息
  std::cout << "Stage [" << m_stage_name << "]: Created Stage with Settings: " << std::endl;
  stage_set->print();

  if (m_settings_save.save_ortho_gtiff_all)
  {
    //建一个名为 mosaicing_gtiff_writer 的 io::GDALContinuousWriter 对象
    m_gdal_writer.reset(new io::GDALContinuousWriter("mosaicing_gtiff_writer", 100, true));
    //启动写入器
    m_gdal_writer->start();
  }
  //注册一个异步数据准备函数
  registerAsyncDataReadyFunctor([=]{ return !m_buffer.empty(); });
}

//析构函数
Mosaicing::~Mosaicing()
{
}

void Mosaicing::addFrame(const Frame::Ptr &frame)
{
  // First update statistics about incoming frame rate
  //更新有关传入帧速率的统计信息
  updateStatisticsIncoming();
  //检查传入帧是否具有表面模型（observed map）和正射影像（orthophoto）
  if (!frame->getSurfaceModel() || !frame->getOrthophoto())
  {
    LOG_F(INFO, "Input frame missing observed map. Dropping!");
    return;
  }
  //获取一个独占锁
  std::unique_lock<std::mutex> lock(m_mutex_buffer);
  //将传入的帧指针添加到缓冲区中
  m_buffer.push_back(frame);

  // Ringbuffer implementation for buffer with no pose
  //如果缓冲区的大小超过预设的队列大小
  if (m_buffer.size() > m_queue_size)
  {
    //从队列前面弹出一个帧，同时更新有关跳过帧的统计信息
    m_buffer.pop_front();
    updateStatisticsSkippedFrame();
  }
  notify();
}

bool Mosaicing::process()
{
  bool has_processed = false;
  if (!m_buffer.empty())
  {
    // Prepare timing
    long t;

    // Prepare output of incremental map update
    CvGridMap::Ptr map_update;

    //从缓冲区中获取一帧数据，其中包含表面模型和正射影像
    Frame::Ptr frame = getNewFrame();
    CvGridMap::Ptr surface_model = frame->getSurfaceModel();
    CvGridMap::Ptr orthophoto = frame->getOrthophoto();

    //创建一个 CvGridMap 对象，将表面模型和正射影像添加到其中
    CvGridMap::Ptr map = std::make_shared<CvGridMap>(orthophoto->roi(), orthophoto->resolution());
    map->add(*surface_model, REALM_OVERWRITE_ALL, false);
    map->add(*orthophoto, REALM_OVERWRITE_ALL, false);

    LOG_F(INFO, "Processing frame #%u...", frame->getFrameId());

    // Use surface normals only if setting was set to true AND actual data has normals
    //判断是否使用表面法线
    m_use_surface_normals = (m_use_surface_normals && map->exists("elevation_normal"));
    
    //如果为空,初始化 UTM 坐标参考和全局地图对象
    if (m_utm_reference == nullptr)
      m_utm_reference = std::make_shared<UTMPose>(frame->getGnssUtm());
    //检查 m_global_map 是否为空
    if (m_global_map == nullptr)
    {
      LOG_F(INFO, "Initializing global map...");
      //将 map 赋值给 m_global_map
      m_global_map = map;

      // Incremental update is equal to global map on initialization
      map_update = m_global_map;
    }
    else
    {
      LOG_F(INFO, "Adding new map data to global map...");
      //地图数据的融合
      t = getCurrentTimeMilliseconds();
      (*m_global_map).add(*map, REALM_OVERWRITE_ZERO, true);
      LOG_F(INFO, "Timing [Add New Map]: %lu ms", getCurrentTimeMilliseconds()-t);

      t = getCurrentTimeMilliseconds();
      CvGridMap::Overlap overlap = m_global_map->getOverlap(*map);
      LOG_F(INFO, "Timing [Compute Overlap]: %lu ms", getCurrentTimeMilliseconds()-t);

      if (overlap.first == nullptr && overlap.second == nullptr)
      {
        LOG_F(INFO, "No overlap detected. Add without blending...");
      }
      else
      {
        LOG_F(INFO, "Overlap detected. Add with blending...");

        t = getCurrentTimeMilliseconds();
        CvGridMap overlap_blended = blend(&overlap);
        (*m_global_map).add(overlap_blended, REALM_OVERWRITE_ALL, false);
        LOG_F(INFO, "Timing [Blending]: %lu ms", getCurrentTimeMilliseconds()-t);

        cv::Rect2d roi = overlap_blended.roi();
        LOG_F(INFO, "Overlap region: [%4.2f, %4.2f] [%4.2f x %4.2f]", roi.x, roi.y, roi.width, roi.height);
        LOG_F(INFO, "Overlap area: %6.2f", roi.area());
      }

      LOG_F(INFO, "Extracting updated map...");
      //创建一个更新的地图 map_update,包括 "color_rgb" 和 "elevation" 层的数据
      map_update = std::make_shared<CvGridMap>(m_global_map->getSubmap({"color_rgb", "elevation"}, overlap.first->roi()));
    }

    // Publishings every iteration
    //发布更新后的地图数据和当前帧数据
    LOG_F(INFO, "Publishing...");

    t = getCurrentTimeMilliseconds();
    publish(frame, m_global_map, map_update, frame->getTimestamp());
    LOG_F(INFO, "Timing [Publish]: %lu ms", getCurrentTimeMilliseconds()-t);


    // Savings every iteration
    //保存当前帧数据和更新后的地图数据
    t = getCurrentTimeMilliseconds();
    saveIter(frame->getFrameId(), map_update);
    LOG_F(INFO, "Timing [Saving]: %lu ms", getCurrentTimeMilliseconds()-t);

    // MVS export
    //m_frames.push_back(frame);

    has_processed = true;
  }
  return has_processed;
}

//地图数据的融合
CvGridMap Mosaicing::blend(CvGridMap::Overlap *overlap)
{
  // Overlap between global mosaic (ref) and new data (inp)
  CvGridMap ref = *overlap->first;
  CvGridMap src = *overlap->second;

  cv::Mat ref_not_elevated;
  cv::bitwise_not(ref["elevated"], ref_not_elevated);

  // There are aparently a number of issues with NaN comparisons breaking in various ways.  See:
  // https://github.com/opencv/opencv/issues/16465
  // To avoid these, use patchNaNs before using boolean comparisons
  //比较两个地图的 "elevation_angle" 层来确定融合的位置，生成mask
  cv::patchNaNs(ref["elevation_angle"],0);
  cv::Mat mask = (src["elevation_angle"] > ref["elevation_angle"]);

  //将重叠区域的数据从 src 地图中复制到 ref 地图中，并使用融合的遮罩进行掩码操作
  src["color_rgb"].copyTo(ref["color_rgb"], mask);
  src["elevation"].copyTo(ref["elevation"], mask);
  src["elevation_angle"].copyTo(ref["elevation_angle"], mask);
  cv::add(ref["num_observations"], cv::Mat::ones(ref.size().height, ref.size().width, CV_16UC1),
          ref["num_observations"], mask);

  return ref;
}

//保存地图数据的迭代结果
void Mosaicing::saveIter(uint32_t id, const CvGridMap::Ptr &map_update)
{
  // Check NaN
  //检查地图中是否存在 NaN 值
  //根据设置将不同层的数据保存为图像文件或者 GeoTIFF 文件
  cv::Mat valid = ((*m_global_map)["elevation"] == (*m_global_map)["elevation"]);

  if (m_settings_save.save_ortho_rgb_all)
    io::saveImage((*m_global_map)["color_rgb"], io::createFilename(m_stage_path + "/ortho/ortho_", id, ".png"));
  if (m_settings_save.save_elevation_all)
    io::saveImageColorMap((*m_global_map)["elevation"], valid, m_stage_path + "/elevation/color_map", "elevation", id, io::ColormapType::ELEVATION);
  if (m_settings_save.save_elevation_var_all)
    io::saveImageColorMap((*m_global_map)["elevation_var"], valid, m_stage_path + "/variance", "variance", id, io::ColormapType::ELEVATION);
  if (m_settings_save.save_elevation_obs_angle_all)
    io::saveImageColorMap((*m_global_map)["elevation_angle"], valid, m_stage_path + "/obs_angle", "angle", id, io::ColormapType::ELEVATION);
  if (m_settings_save.save_num_obs_all)
    io::saveImageColorMap((*m_global_map)["num_observations"], valid, m_stage_path + "/nobs", "nobs", id, io::ColormapType::NUM_OBS);
  if (m_settings_save.save_ortho_gtiff_all && m_gdal_writer != nullptr)
    m_gdal_writer->requestSaveGeoTIFF(std::make_shared<CvGridMap>(m_global_map->getSubmap({"color_rgb"})), m_utm_reference->zone, m_stage_path + "/ortho/ortho_iter.tif", true, m_settings_save.split_gtiff_channels);

    //io::saveGeoTIFF(*map_update, "color_rgb", _utm_reference->zone, io::createFilename(_stage_path + "/ortho/ortho_", id, ".tif"));
}

//将整个地图数据保存为图像、GeoTIFF 文件和点云等格式
void Mosaicing::saveAll()
{
  //检查是否已创建全局地图，并且该地图是否包含 "elevation" 层的数据
  if(!m_global_map || !m_global_map->exists("elevation"))
  {
    LOG_F(ERROR, "No global map created, skipping saveAll()");
    return;
  }

  // Check NaN
  //检查地图中的 NaN 值
  cv::Mat valid = ((*m_global_map)["elevation"] == (*m_global_map)["elevation"]);
  //根据设置保存不同层的数据
  // 2D map output
  if (m_settings_save.save_ortho_rgb_one)
    io::saveCvGridMapLayer(*m_global_map, m_utm_reference->zone, m_utm_reference->band, "color_rgb", m_stage_path + "/ortho/ortho.png");
  if (m_settings_save.save_elevation_one)
    io::saveImageColorMap((*m_global_map)["elevation"], valid, m_stage_path + "/elevation/color_map", "elevation", io::ColormapType::ELEVATION);
  if (m_settings_save.save_elevation_var_one)
    io::saveImageColorMap((*m_global_map)["elevation_var"], valid, m_stage_path + "/variance", "variance", io::ColormapType::ELEVATION);
  if (m_settings_save.save_elevation_obs_angle_one)
    io::saveImageColorMap((*m_global_map)["elevation_angle"], valid, m_stage_path + "/obs_angle", "angle", io::ColormapType::ELEVATION);
  if (m_settings_save.save_num_obs_one)
    io::saveImageColorMap((*m_global_map)["num_observations"], valid, m_stage_path + "/nobs", "nobs", io::ColormapType::ELEVATION);
  if (m_settings_save.save_num_obs_one)
    io::saveGeoTIFF(m_global_map->getSubmap({"num_observations"}), m_utm_reference->zone, m_stage_path + "/nobs/nobs.tif");
  if (m_settings_save.save_ortho_gtiff_one)
    io::saveGeoTIFF(m_global_map->getSubmap({"color_rgb"}), m_utm_reference->zone, m_stage_path + "/ortho/ortho.tif", true, m_settings_save.split_gtiff_channels);
  if (m_settings_save.save_elevation_one)
    io::saveGeoTIFF(m_global_map->getSubmap({"elevation"}), m_utm_reference->zone, m_stage_path + "/elevation/gtiff/elevation.tif");
  if (m_settings_save.save_elevation_obs_angle_one)
    io::saveGeoTIFF(m_global_map->getSubmap({"elevation_angle"}), m_utm_reference->zone, m_stage_path + "/obs_angle/angle.tif");

  //io::MvsExport::saveFrames(m_frames, m_stage_path + "/mvs");

  // 3D Point cloud output
  //将稠密点云数据保存为 PLY 文件,前提宏被定义
#if WITH_PCL
  if (m_settings_save.save_dense_ply)
  {
    if (m_global_map->exists("elevation_normal"))
      io::saveElevationPointsToPLY(*m_global_map, "elevation", "elevation_normal", "color_rgb", "valid", m_stage_path + "/elevation/ply", "elevation");
    else
      io::saveElevationPointsToPLY(*m_global_map, "elevation", "", "color_rgb", "valid", m_stage_path + "/elevation/ply", "elevation");
  }
#endif

  // 3D Mesh output
  if (m_settings_save.save_elevation_mesh_one)
  {
    //std::vector<cv::Point2i> vertex_ids = _mesher->buildMesh(*_global_map, "valid");
    //if (_global_map->exists("elevation_normal"))
    //  io::saveElevationMeshToPLY(*_global_map, vertex_ids, "elevation", "elevation_normal", "color_rgb", "valid", _stage_path + "/elevation/mesh", "elevation");
    //else
    //  io::saveElevationMeshToPLY(*_global_map, vertex_ids, "elevation", "", "color_rgb", "valid", _stage_path + "/elevation/mesh", "elevation");
  }
}

//在阶段需要重置时被调用
void Mosaicing::reset()
{
  LOG_F(INFO, "Reseted!");
}

//在阶段即将结束时被调用
void Mosaicing::finishCallback()
{
  // First polish results
  //执行后处理操作
  runPostProcessing();

  if (m_gdal_writer != nullptr)
  {
    m_gdal_writer->requestFinish();
    m_gdal_writer->join();
  }

  // Trigger savings
  saveAll();

  // Publish final mesh at the end
  //if (m_do_publish_mesh_at_finish)
  //  m_transport_mesh(createMeshFaces(m_global_map), "output/mesh");
}

void Mosaicing::runPostProcessing()
{

}

//从缓冲区中获取一个帧对象
Frame::Ptr Mosaicing::getNewFrame()
{
  std::unique_lock<std::mutex> lock(m_mutex_buffer);
  //从前面获取一个帧对象
  Frame::Ptr frame = m_buffer.front();
  //从缓冲区中移除该帧，更新已处理帧的统计信息
  m_buffer.pop_front();
  updateStatisticsProcessedFrame();
  return std::move(frame);
}

//在阶段初始化时被调用
void Mosaicing::initStageCallback()
{
  // If we aren't saving any information, skip directory creation
  //检查是否需要保存信息（日志或文件）
  if (!(m_log_to_file || m_settings_save.save_required()))
  {
    return;
  }

  // Stage directory first
  if (!io::dirExists(m_stage_path))
  //创建阶段目录 m_stage_path
    io::createDir(m_stage_path);

  // Then sub directories
  //根据不同的保存设置创建其他子目录
  if (!io::dirExists(m_stage_path + "/elevation") && m_settings_save.save_elevation())
    io::createDir(m_stage_path + "/elevation");
  if (!io::dirExists(m_stage_path + "/elevation/color_map") && m_settings_save.save_elevation_map())
    io::createDir(m_stage_path + "/elevation/color_map");
  if (!io::dirExists(m_stage_path + "/elevation/ply") && m_settings_save.save_dense_ply)
    io::createDir(m_stage_path + "/elevation/ply");
  if (!io::dirExists(m_stage_path + "/elevation/mesh") && m_settings_save.save_elevation_mesh_one)
    io::createDir(m_stage_path + "/elevation/mesh");
  if (!io::dirExists(m_stage_path + "/elevation/gtiff") && m_settings_save.save_elevation_map())
    io::createDir(m_stage_path + "/elevation/gtiff");

  if (!io::dirExists(m_stage_path + "/obs_angle") && m_settings_save.save_obs_angle())
    io::createDir(m_stage_path + "/obs_angle");
  if (!io::dirExists(m_stage_path + "/variance") && m_settings_save.save_variance())
    io::createDir(m_stage_path + "/variance");
  if (!io::dirExists(m_stage_path + "/ortho") && m_settings_save.save_ortho())
    io::createDir(m_stage_path + "/ortho");
  if (!io::dirExists(m_stage_path + "/nobs") && m_settings_save.save_nobs())
    io::createDir(m_stage_path + "/nobs");
  if (!io::dirExists(m_stage_path + "/mvs"))
    io::createDir(m_stage_path + "/mvs");
}

//将阶段的各种设置信息和缓冲区的队列深度输出到日志中
void Mosaicing::printSettingsToLog()
{
  LOG_F(INFO, "### Stage process settings ###");
  LOG_F(INFO, "- publish_mesh_nth_iter: %i", m_publish_mesh_nth_iter);
  LOG_F(INFO, "- publish_mesh_every_nth_kf: %i", m_publish_mesh_every_nth_kf);
  LOG_F(INFO, "- do_publish_mesh_at_finish: %i", m_do_publish_mesh_at_finish);
  LOG_F(INFO, "- downsample_publish_mesh: %4.2f", m_downsample_publish_mesh);
  LOG_F(INFO, "- use_surface_normals: %i", m_use_surface_normals);
  LOG_F(INFO, "- th_elevation_min_nobs: %i", m_th_elevation_min_nobs);
  LOG_F(INFO, "- th_elevation_var: %4.2f", m_th_elevation_var);

  LOG_F(INFO, "### Stage save settings ###");
  LOG_F(INFO, "- save_ortho_rgb_one: %i", m_settings_save.save_ortho_rgb_one);
  LOG_F(INFO, "- save_ortho_rgb_all: %i", m_settings_save.save_ortho_rgb_all);
  LOG_F(INFO, "- save_ortho_gtiff_one: %i", m_settings_save.save_ortho_gtiff_one);
  LOG_F(INFO, "- save_ortho_gtiff_all: %i", m_settings_save.save_ortho_gtiff_all);
  LOG_F(INFO, "- save_elevation_one: %i", m_settings_save.save_elevation_one);
  LOG_F(INFO, "- save_elevation_all: %i", m_settings_save.save_elevation_all);
  LOG_F(INFO, "- save_elevation_var_one: %i", m_settings_save.save_elevation_var_one);
  LOG_F(INFO, "- save_elevation_var_all: %i", m_settings_save.save_elevation_var_all);
  LOG_F(INFO, "- save_elevation_obs_angle_one: %i", m_settings_save.save_elevation_obs_angle_one);
  LOG_F(INFO, "- save_elevation_obs_angle_all: %i", m_settings_save.save_elevation_obs_angle_all);
  LOG_F(INFO, "- save_elevation_mesh_one: %i", m_settings_save.save_elevation_mesh_one);
  LOG_F(INFO, "- save_num_obs_one: %i", m_settings_save.save_num_obs_one);
  LOG_F(INFO, "- save_num_obs_all: %i", m_settings_save.save_num_obs_all);
  LOG_F(INFO, "- save_dense_ply: %i", m_settings_save.save_dense_ply);
}

//返回当前缓冲区中的队列深度
uint32_t Mosaicing::getQueueDepth() {
  return m_buffer.size();
}

//生成用于发布的三维网格的面（faces）
std::vector<Face> Mosaicing::createMeshFaces(const CvGridMap::Ptr &map)
{
  CvGridMap::Ptr mesh_sampled;
  //是否需要对网格进行下采样
  if (m_downsample_publish_mesh > 10e-6)
  {
    //进行下采样处理
    if (map && map->exists("elevation") && map->exists("color_rgb")) {

      // Downsampling was set by the user in settings
      LOG_F(INFO, "Downsampling mesh publish to %4.2f [m/gridcell]...", m_downsample_publish_mesh);
      mesh_sampled = std::make_shared<CvGridMap>(map->cloneSubmap({"elevation", "color_rgb"}));

      cv::Mat valid = ((*mesh_sampled)["elevation"] == (*mesh_sampled)["elevation"]);

      // TODO: Change resolution correction is not cool -> same in ortho rectification
      // Check ranges of input elevation, this is necessary to correct resizing interpolation errors
      double ele_min, ele_max;
      cv::Point2i min_loc, max_loc;
      cv::minMaxLoc((*mesh_sampled)["elevation"], &ele_min, &ele_max, &min_loc, &max_loc, valid);

      mesh_sampled->changeResolution(m_downsample_publish_mesh);

      // After resizing through bilinear interpolation there can occure bad elevation values at the border
      cv::Mat mask_low = ((*mesh_sampled)["elevation"] < ele_min);
      cv::Mat mask_high = ((*mesh_sampled)["elevation"] > ele_max);
      (*mesh_sampled)["elevation"].setTo(std::numeric_limits<float>::quiet_NaN(), mask_low);
      (*mesh_sampled)["elevation"].setTo(std::numeric_limits<float>::quiet_NaN(), mask_high);
    }
    else
    {
      LOG_F(WARNING, "Could not publish downsampled mesh, no global map existed.");
    }
  }
  else
  {
    LOG_F(INFO, "No downsampling of mesh publish...");
    // No downsampling was set
    //用传入的 map 作为 mesh_sampled
    mesh_sampled = map;
  }

  //std::vector<cv::Point2i> vertex_ids = _mesher->buildMesh(*mesh_sampled, "valid");
  //std::vector<Face> faces = cvtToMesh((*mesh_sampled), "elevation", "color_rgb", vertex_ids);
  //return faces;
  // Placeholder return, there are a few calls that try to use this
  return std::vector<Face>();
}

//发布处理过程中的数据，包括图像和网格
void Mosaicing::publish(const Frame::Ptr &frame, const CvGridMap::Ptr &map, const CvGridMap::Ptr &update, uint64_t timestamp)
{
  cv::Mat valid = ((*m_global_map)["elevation"] == (*m_global_map)["elevation"]);

  // First update statistics about outgoing frame rate
  //更新有关正在发布的帧速率的统计信息
  updateStatisticsOutgoing();

  //将彩色图像和高程图像的颜色映射（使用JET颜色映射）发布到指定路径
  m_transport_img((*m_global_map)["color_rgb"], "output/rgb");
  m_transport_img(analysis::convertToColorMapFromCVC1((*m_global_map)["elevation"],
                                                      valid,
                                                      cv::COLORMAP_JET), "output/elevation");
  //将完整地图以及更新后的部分地图（update）发布到不同的输出路径
  m_transport_cvgridmap(m_global_map->getSubmap({"color_rgb"}), m_utm_reference->zone, m_utm_reference->band, "output/full/ortho");
  m_transport_cvgridmap(update->getSubmap({"color_rgb"}), m_utm_reference->zone, m_utm_reference->band, "output/update/ortho");
  //_transport_cvgridmap(update->getSubmap({"elevation", "valid"}), _utm_reference->zone, _utm_reference->band, "output/update/elevation");

  //据 m_publish_mesh_every_nth_kf 和 m_publish_mesh_nth_iter 的值，决定是否发布网格数据
  if (m_publish_mesh_every_nth_kf > 0 && m_publish_mesh_every_nth_kf == m_publish_mesh_nth_iter)
  {
    //生成网格面数据
    std::vector<Face> faces = createMeshFaces(map);
    //通过多线程将网格面数据发布到指定路径
    std::thread t(m_transport_mesh, faces, "output/mesh");
    t.detach();
    m_publish_mesh_nth_iter = 0;
  }
  else if (m_publish_mesh_every_nth_kf > 0)
  {
    m_publish_mesh_nth_iter++;
  }
}