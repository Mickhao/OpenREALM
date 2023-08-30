

#include <realm_core/loguru.h>

#include <realm_stages/stage_base.h>

/*处理阶段的抽象基类
定义了一些纯虚函数来规范处理阶段的接口*/
using namespace realm;

//初始化
StageBase::StageBase(const std::string &name, const std::string &path, double rate, int queue_size, bool log_to_file)
: WorkerThreadBase("Stage [" + name + "]", static_cast<int64_t>(1/rate*1000.0), true),
  m_stage_name(name),
  m_stage_path(path),
  m_queue_size(queue_size),
  m_log_to_file(log_to_file),
  m_is_output_dir_initialized(false),
  m_t_statistics_period(10),
  m_counter_frames_in(0),
  m_counter_frames_out(0),
  m_timer_statistics_fps(new Timer(std::chrono::seconds(m_t_statistics_period), std::bind(&StageBase::evaluateStatistic, this)))
{
}

//打印一个警告消息，表示该阶段不支持参数的更改，并返回 false
bool StageBase::changeParam(const std::string &name, const std::string &val)
{
  LOG_F(WARNING, "Changing parameter not implemented for this stage!");
  return false;
}

void StageBase::initStagePath(const std::string &abs_path)
{
  // Set and create output directory
  //将输出路径设置为 abs_path 加上阶段名称（m_stage_name）
  m_stage_path = abs_path + "/" + m_stage_name;
  initStageCallback();
  m_is_output_dir_initialized = true;

  // Init logging if enabled
  //如果日志记录被启用
  if (m_log_to_file)
  {
    //将配置日志输出到阶段的日志文件（stage.log）
    loguru::add_file((m_stage_path + "/stage.log").c_str(), loguru::Append, loguru::Verbosity_MAX);
  }

  LOG_F(INFO, "Successfully initialized!");
  LOG_F(INFO, "Stage path set to: %s", m_stage_path.c_str());
  printSettingsToLog();
}

//获取阶段的统计信息
StageStatistics StageBase::getStageStatistics()
{
  std::unique_lock<std::mutex> lock(m_mutex_statistics);
  //将当前队列深度设置到统计信息中
  m_stage_statistics.queue_depth = getQueueDepth();
  //返回统计信息
  return m_stage_statistics;
}

void StageBase::registerAsyncDataReadyFunctor(const std::function<bool()> &func)
{
  // Return true if either the functor evaluates to true, or when a finish is requested.
  //在数据准备就绪的条件下，该函数对象将被调用
  m_data_ready_functor = ([=]{ return (func() || isFinishRequested()); });
}
//需要传输帧数据时会被调用
void StageBase::registerFrameTransport(const std::function<void(const Frame::Ptr&, const std::string&)> &func)
{
  m_transport_frame = func;
}
//需要传输位姿信息时会被调用
void StageBase::registerPoseTransport(const std::function<void(const cv::Mat &, uint8_t zone, char band, const std::string &)> &func)
{
  m_transport_pose = func;
}
//需要传输深度图像时会被调用
void StageBase::registerDepthMapTransport(const std::function<void(const cv::Mat&, const std::string&)> &func)
{
  m_transport_depth_map = func;
}
//需要传输点云数据时会被调用
void StageBase::registerPointCloudTransport(const std::function<void(const PointCloud::Ptr &, const std::string&)> &func)
{
  m_transport_pointcloud = func;
}
//需要传输图像数据时会被调用
void StageBase::registerImageTransport(const std::function<void(const cv::Mat&, const std::string&)> &func)
{
  m_transport_img = func;
}
//需要传输网格数据时会被调用
void StageBase::registerMeshTransport(const std::function<void(const std::vector<Face>&, const std::string&)> &func)
{
  m_transport_mesh = func;
}
//需要传输 CvGridMap 数据时会被调用
void StageBase::registerCvGridMapTransport(const std::function<void(const CvGridMap &, uint8_t zone, char band, const std::string&)> &func)
{
  m_transport_cvgridmap = func;
}
//设置统计信息更新的时间间隔
void StageBase::setStatisticsPeriod(uint32_t s)
{
  std::unique_lock<std::mutex> lock(m_mutex_statistics);
  m_t_statistics_period = s;
}
//在日志中记录当前的统计信息
void StageBase::logCurrentStatistics() const
{
  LOG_SCOPE_F(INFO, "Stage [%s] statistics", m_stage_name.c_str());
  LOG_F(INFO, "Total frames: %i ", m_stage_statistics.frames_total);
  LOG_F(INFO, "Processed frames: %i ", m_stage_statistics.frames_processed);
  LOG_F(INFO, "Bad frames: %i ", m_stage_statistics.frames_bad);
  LOG_F(INFO, "Dropped frames: %i ", m_stage_statistics.frames_dropped);
  LOG_F(INFO, "Fps in: %4.2f ", m_stage_statistics.fps_in);
  LOG_F(INFO, "Fps out: %4.2f ", m_stage_statistics.fps_out);
}
//接收到新帧时更新输入帧数的统计信息
void StageBase::updateStatisticsIncoming()
{
  std::unique_lock<std::mutex> lock(m_mutex_statistics);
  m_counter_frames_in++;
  m_stage_statistics.frames_total++;
}
//在跳过一帧时更新被丢弃帧的统计信息
void StageBase::updateStatisticsSkippedFrame()
{
  std::unique_lock<std::mutex> lock(m_mutex_statistics);
  m_stage_statistics.frames_dropped++;
}
//递增错误帧数的统计信息
void StageBase::updateStatisticsBadFrame()
{
  std::unique_lock<std::mutex> lock(m_mutex_statistics);
  m_stage_statistics.frames_bad++;
}
//递增已处理帧数的统计信息
void StageBase::updateStatisticsProcessedFrame()
{
  std::unique_lock<std::mutex> lock(m_mutex_statistics);
  m_stage_statistics.frames_processed++;
}

//更新关于传出帧的统计信息
void StageBase::updateStatisticsOutgoing()
{
    std::unique_lock<std::mutex> lock(m_mutex_statistics);
    m_counter_frames_out++;
    //更新为当前处理统计信息
    m_stage_statistics.process_statistics = getProcessingStatistics();

    uint32_t queue_depth = getQueueDepth();
    if (m_stage_statistics.queue_statistics.count == 0) {
      //当前队列深度
      m_stage_statistics.queue_statistics.min = queue_depth;
      m_stage_statistics.queue_statistics.max = queue_depth;
    } else {
      if (m_stage_statistics.queue_statistics.min > queue_depth) m_stage_statistics.queue_statistics.min = queue_depth;
      if (m_stage_statistics.queue_statistics.max > queue_depth) m_stage_statistics.queue_statistics.max = queue_depth;
    }
    m_stage_statistics.queue_statistics.count++;
    //计算队列深度的平均值，并更新
    m_stage_statistics.queue_statistics.avg =
        m_stage_statistics.queue_statistics.avg + ((double)queue_depth - m_stage_statistics.queue_statistics.avg)
                                                  / (double)m_stage_statistics.queue_statistics.count;
}

//在一段时间内评估传入和传出帧的帧率，并将统计信息记录下来
void StageBase::evaluateStatistic()
{
  std::unique_lock<std::mutex> lock(m_mutex_statistics);
  //计算传入帧的平均帧率 fps_in
  float fps_in = static_cast<float>(m_counter_frames_in) / m_t_statistics_period;
  //计算传出帧的平均帧率 fps_out
  float fps_out = static_cast<float>(m_counter_frames_out) / m_t_statistics_period;
  //重置为零
  m_counter_frames_in = 0;
  m_counter_frames_out = 0;
  //将当前的统计信息记录到日志中
  logCurrentStatistics();
  //将计算得到的帧率值存储到统计信息中
  m_stage_statistics.fps_in = fps_in;
  m_stage_statistics.fps_out = fps_out;
}