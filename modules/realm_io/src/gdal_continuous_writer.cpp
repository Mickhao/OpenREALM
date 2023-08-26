

#include <realm_io/gdal_continuous_writer.h>

using namespace realm;

io::GDALContinuousWriter::GDALContinuousWriter(const std::string &thread_name, int64_t sleep_time, bool verbose)
 : WorkerThreadBase(thread_name, sleep_time, verbose),
   m_queue_size(1)
{
}

void io::GDALContinuousWriter::requestSaveGeoTIFF(const CvGridMap::Ptr &map,
                                                  const uint8_t &zone,
                                                  const std::string &filename,
                                                  bool do_build_overview,
                                                  bool do_split_save,
                                                  io::GDALProfile gdal_profile)
{
  // Create new save job
  //创建一个新的保存任务对象 QueueElement，将要保存的信息封装在其中
  QueueElement::Ptr queue_element;
  queue_element.reset(new QueueElement{map, zone, filename, do_build_overview, do_split_save, gdal_profile});

  // Push it to the processing queue
  //将创建的保存任务对象添加到保存请求队列中。
  //在添加之前，会获取保存请求队列的互斥锁，确保线程安全
  m_mutex_save_requests.lock();
  m_save_requests.push_back(queue_element);
  if (m_save_requests.size() > m_queue_size)
  {
    m_save_requests.pop_front();
  }
  m_mutex_save_requests.unlock();
}

//用于处理保存请求队列中的任务
bool io::GDALContinuousWriter::process()
{
  //获取保存请求队列的互斥锁
  m_mutex_save_requests.lock();
  //如果保存请求队列不为空，表示有待处理的保存请求
  if (!m_save_requests.empty())
  {
    //从队列的末尾取出一个保存请求对象 QueueElement
    QueueElement::Ptr queue_element = m_save_requests.back();
    //从保存请求队列中弹出该保存请求对象
    m_save_requests.pop_back();
    //释放保存请求队列的互斥锁
    m_mutex_save_requests.unlock();

    //使用 saveGeoTIFF 函数将地理网格数据保存为 GeoTIFF 文件
    io::saveGeoTIFF(
      *queue_element->map,
      queue_element->zone,
      queue_element->filename,
      queue_element->do_build_overview,
      queue_element->do_split_save,
      queue_element->gdal_profile
    );

    return true;
  }
  //释放保存请求队列的互斥锁
  m_mutex_save_requests.unlock();
  return false;
}

void io::GDALContinuousWriter::reset()
{

}

void io::GDALContinuousWriter::finishCallback()
{
}