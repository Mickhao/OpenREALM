

#include <cstdio>

#include <realm_ortho/tile_cache.h>
#include <realm_io/cv_import.h>
#include <realm_io/cv_export.h>

/*管理和缓存地图瓦片数据*/
using namespace realm;

//初始化
TileCache::TileCache(const std::string &id, double sleep_time, const std::string &output_directory, bool verbose)
 : WorkerThreadBase("tile_cache_" + id, sleep_time, verbose),
   m_dir_toplevel(output_directory),
   m_has_init_directories(false),
   m_do_update(false)
{
  m_data_ready_functor = [=]{ return (m_do_update || isFinishRequested()); };
}

//析构函数，用于释放 TileCache 对象在内部分配的资源
TileCache::~TileCache()
{
  flushAll();
}

//设置输出文件夹的路径
void TileCache::setOutputFolder(const std::string &dir)
{
  std::lock_guard<std::mutex> lock(m_mutex_settings);
  //更新输出文件夹的路径
  m_dir_toplevel = dir;
}

bool TileCache::process()
{
  bool has_processed = false;

  //使用 try_lock 尝试获取 m_mutex_do_update 互斥锁
  if (m_mutex_do_update.try_lock())
  {
    long t;

    // Give update lock free as fast as possible, so we won't block other threads from adding data
    //将m_do_update 的值存储在局部变量 do_update
    bool do_update = m_do_update;
    m_do_update = false;
    //解锁
    m_mutex_do_update.unlock();

    if (do_update)//如果为true，说明之前标记为需要更新的数据现在可以被处理
    {
      int n_tiles_written = 0;

      t = getCurrentTimeMilliseconds();

      //遍历缓存的所有缩放级别
      for (auto &cached_elements_zoom : m_cache)
      {
        //获取当前缩放级别的预测区域
        cv::Rect2i roi_prediction = m_roi_prediction.at(cached_elements_zoom.first);
        //遍历缓存的所有列
        for (auto &cached_elements_column : cached_elements_zoom.second)
        {
          //遍历缓存的所有元素
          for (auto &cached_elements : cached_elements_column.second)
          {
            std::lock_guard<std::mutex> lock(cached_elements.second->mutex);
            cached_elements.second->tile->lock();

              //如果当前缓存元素对应的瓦片尚未被写入
            if (!cached_elements.second->was_written)
            {
              //增加已写入的瓦片数
              n_tiles_written++;
              //将瓦片数据写入磁盘
              write(cached_elements.second);
            }
              //检查当前缓存元素对应的瓦片数据是否已经在内存中
            if (isCached(cached_elements.second))
            {
              //获取坐标
              int tx = cached_elements.second->tile->x();
              int ty = cached_elements.second->tile->y();
              //检查瓦片的坐标是否在预测区域之外
              if (tx < roi_prediction.x || tx > roi_prediction.x + roi_prediction.width
                  || ty < roi_prediction.y || ty > roi_prediction.y + roi_prediction.height)
              {
                //将瓦片数据从内存中刷新到磁盘
                flush(cached_elements.second);
              }
            }
            cached_elements.second->tile->unlock();
          }
        }
      }

      LOG_IF_F(INFO, m_verbose, "Tiles written: %i", n_tiles_written);
      LOG_IF_F(INFO, m_verbose, "Timing [Cache Flush]: %lu ms", getCurrentTimeMilliseconds() - t);

      has_processed = true;
    }
  }
  return has_processed;
}

//清空缓存
void TileCache::reset()
{
  m_cache.clear();
}

//将瓦片数据添加到缓存中
void TileCache::add(int zoom_level, const std::vector<Tile::Ptr> &tiles, const cv::Rect2i &roi_idx)
{
  std::lock_guard<std::mutex> lock(m_mutex_cache);

  // Assuming all tiles are based on the same data, therefore have the same number of layers and layer names
  //获取要添加的瓦片的图层名称。这里假设所有的瓦片都基于相同的数据，因此它们具有相同的图层名称
  std::vector<std::string> layer_names = tiles[0]->data()->getAllLayerNames();

  std::vector<LayerMetaData> layer_meta;
  for (const auto &layer_name : layer_names)
  {
    // Saving the name and the type of the layer into the meta data
    //获取第一个瓦片的数据
    CvGridMap::Layer layer = tiles[0]->data()->getLayer(layer_name);
   //将图层的名称、数据类型和插值方式添加到 layer_meta 向量中
   layer_meta.emplace_back(LayerMetaData{layer_name, layer.data.type(), layer.interpolation});
  }

  //检查是否已经初始化目录结构
  if (!m_has_init_directories)
  {
    //创建图层的目录
    createDirectories(m_dir_toplevel + "/", layer_names, "");
    m_has_init_directories = true;
  }

  //使用 zoom_level 作为键，在缓存 m_cache 中查找相应缩放级别的数据
  auto it_zoom = m_cache.find(zoom_level);

  //获取当前的时间戳，以毫秒为单位
  long timestamp = getCurrentTimeMilliseconds();

  long t = getCurrentTimeMilliseconds();

  // Cache for this zoom level already exists
  //如果在缓存中找到了与当前缩放级别相对应的数据
  if (it_zoom != m_cache.end())
  {
    for (const auto &t : tiles)
    {
      // Here we find a tile grid for a specific zoom level and add the new tiles to it.
      // Important: Tiles that already exist will be overwritten!
      t->lock();
      //在当前缩放级别的缓存数据中查找与当前瓦片的列索引相对应的数据
      auto it_tile_x = it_zoom->second.find(t->x());
      if (it_tile_x == it_zoom->second.end())
      {
        // Zoom level exists, but tile column is
        //建目录以存储当前缩放级别和列索引对应的数据
        createDirectories(m_dir_toplevel + "/", layer_names, "/" + std::to_string(zoom_level) + "/" + std::to_string(t->x()));
        //将当前瓦片数据创建为 CacheElement 对象并添加到缓存中
        it_zoom->second[t->x()][t->y()].reset(new CacheElement{timestamp, layer_meta, t, false});
      }
      else
      {
        //在当前列索引的缓存数据中查找与当前瓦片行索引相对应的数据
        auto it_tile_xy = it_tile_x->second.find(t->y());
        if (it_tile_xy == it_tile_x->second.end())
        {
          // Zoom level and column was found, but tile did not yet exist
          //将当前瓦片数据创建为 CacheElement 对象并添加到缓存中
          it_tile_x->second[t->y()].reset(new CacheElement{timestamp, layer_meta, t, false});
        }
        else
        { //如果在当前列索引的缓存数据中找到了与当前瓦片行索引相对应的数据
          // Existing tile was found inside zoom level and column
          it_tile_xy->second->mutex.lock(); // note: mutex goes out of scope after this operation, no unlock needed.
          //将当前瓦片数据重新赋值为一个新的 CacheElement 对象
          it_tile_xy->second.reset(new CacheElement{timestamp, layer_meta, t, false});
        }
      }
      t->unlock();
    }
  }
  // Cache for this zoom level does not yet exist
  else
  {
    //创建目录以存储当前缩放级别对应的数据
    createDirectories(m_dir_toplevel + "/", layer_names, "/" + std::to_string(zoom_level));

    //用于存储新的缓存数据
    CacheElementGrid tile_grid;
    for (const auto &t : tiles)
    {
      // By assigning a new grid of tiles to the zoom level we overwrite all existing data. But in this case there was
      // no prior data found for the specific zoom level.
      t->lock();
      //在当前缩放级别的缓存数据中查找与当前瓦片的列索引相对应的数据
      auto it_tile_x = it_zoom->second.find(t->x());
      if (it_tile_x == it_zoom->second.end())
      //建目录以存储当前缩放级别和列索引对应的数据
        createDirectories(m_dir_toplevel + "/", layer_names, "/" + std::to_string(zoom_level) + "/" + std::to_string(t->x()));
        //将当前瓦片数据创建为 CacheElement 对象并添加到缓存中
      tile_grid[t->x()][t->y()].reset(new CacheElement{timestamp, layer_meta, t, false});
      t->unlock();
    }
    //将新创建的缓存数据（tile_grid）添加到缓存中
    m_cache[zoom_level] = tile_grid;
  }

  LOG_IF_F(INFO, m_verbose, "Timing [Cache Push]: %lu ms", getCurrentTimeMilliseconds() - t);

  //更新预测
  updatePrediction(zoom_level, roi_idx);

  std::lock_guard<std::mutex> lock1(m_mutex_do_update);
  m_do_update = true;//表示缓存已更新
  notify();
}

//从缓存中获取瓦片数据
Tile::Ptr TileCache::get(int tx, int ty, int zoom_level)
{
  //在缓存中查找与请求的缩放级别相对应的数据
  auto it_zoom = m_cache.find(zoom_level);
  if (it_zoom == m_cache.end())
  {
    return nullptr;
  }

  //在缓存的缩放级别数据中查找与请求的列索引相对应的数据
  auto it_tile_x = it_zoom->second.find(tx);
  if (it_tile_x == it_zoom->second.end())
  {
    return nullptr;
  }
  
  //在缓存的列索引数据中查找与请求的行索引相对应的数据
  auto it_tile_xy = it_tile_x->second.find(ty);
  if (it_tile_xy == it_tile_x->second.end())
  {
    return nullptr;
  }

  //对当前瓦片数据的互斥锁进行加锁
  std::lock_guard<std::mutex> lock(it_tile_xy->second->mutex);

  // Warning: We lock the tile now and return it to the calling thread locked. Therefore the responsibility to unlock
  // it is on the calling thread!
  //对瓦片数据本身加锁
  it_tile_xy->second->tile->lock();
  //如果瓦片数据未被缓存
  if (!isCached(it_tile_xy->second))
  {
    //加载瓦片数据
    load(it_tile_xy->second);
  }

  return it_tile_xy->second->tile;
}

//将所有瓦片数据写入磁盘并进行缓存清空
void TileCache::flushAll()
{
  //记录写入磁盘的瓦片数量
  int n_tiles_written = 0;
  //在日志中记录开始刷新所有瓦片数据的操作
  LOG_IF_F(INFO, m_verbose, "Flushing all tiles...");
  //记录当前时间
  long t = getCurrentTimeMilliseconds();

  //遍历缓存中的所有缩放级别
  for (auto &zoom_levels : m_cache)
  //历缓存中的所有列
    for (auto &cache_column : zoom_levels.second)
    //遍历缓存中的所有瓦片数据
      for (auto &cache_element : cache_column.second)
      {
        std::lock_guard<std::mutex> lock(cache_element.second->mutex);
        cache_element.second->tile->lock();
        if (!cache_element.second->was_written)//如果瓦片数据尚未被写入磁盘
        {
          //将瓦片数据写入磁盘
          write(cache_element.second);
          n_tiles_written++;
        }

        //释放瓦片数据的内存
        cache_element.second->tile->data() = nullptr;
        cache_element.second->tile->unlock();
      }

  LOG_IF_F(INFO, m_verbose, "Tiles written: %i", n_tiles_written);
  LOG_IF_F(INFO, m_verbose, "Timing [Flush All]: %lu ms", getCurrentTimeMilliseconds() - t);
}

//将所有瓦片数据从磁盘加载回内存并进行缓存
void TileCache::loadAll()
{
  //遍历缓存中的所有缩放级别
  for (auto &zoom_levels : m_cache)
  //遍历缓存中的所有列
    for (auto &cache_column : zoom_levels.second)
    //遍历缓存中的所有瓦片数据
      for (auto &cache_element : cache_column.second)
      {
        std::lock_guard<std::mutex> lock(cache_element.second->mutex);
        cache_element.second->tile->lock();
        if (!isCached(cache_element.second))
        //磁盘加载瓦片数据到内存中
          load(cache_element.second);
        cache_element.second->tile->unlock();
      }
}

void TileCache::load(const CacheElement::Ptr &element) const
{
  //遍历瓦片数据的每个图层的元数据
  for (const auto &meta : element->layer_meta)
  {
    //根据图层名称、缩放级别以及瓦片的行列号构建文件名
    std::string filename = m_dir_toplevel + "/"
                           + meta.name + "/"
                           + std::to_string(element->tile->zoom_level()) + "/"
                           + std::to_string(element->tile->x()) + "/"
                           + std::to_string(element->tile->y());

    int type = meta.type & CV_MAT_DEPTH_MASK;//从图层的元数据中获取数据类型

    //根据数据类型进行判断
    switch(type)
    {
      case CV_8U:
        filename += ".png";
        break;
      case CV_16U:
        filename += ".bin";
        break;
      case CV_32F:
        filename += ".bin";
        break;
      case CV_64F:
        filename += ".bin";
        break;
      default:
        throw(std::invalid_argument("Error reading tile: data type unknown!"));
    }

    //如果文件存在
    if (io::fileExists(filename))
    {
      //加载图像或二进制文件数据
      cv::Mat data = io::loadImage(filename);

      //使用加载的数据，将图层数据添加到瓦片数据中
      element->tile->data()->add(meta.name, data, meta.interpolation_flag);

      LOG_IF_F(INFO, m_verbose, "Read tile from disk: %s", filename.c_str());
    }
    else
    {
      LOG_IF_F(WARNING, m_verbose, "Failed reading tile from disk: %s", filename.c_str());
      throw(std::invalid_argument("Error loading tile."));
    }
  }
}

//将瓦片数据写入磁盘
void TileCache::write(const CacheElement::Ptr &element) const
{
  //遍历瓦片数据的每个图层的元数据
  for (const auto &meta : element->layer_meta)
  {
    //从瓦片数据中获取特定图层的数据
    cv::Mat data = element->tile->data()->get(meta.name);

    //根据图层名称、缩放级别以及瓦片的行列号构建文件名
    std::string filename = m_dir_toplevel + "/"
                           + meta.name + "/"
                           + std::to_string(element->tile->zoom_level()) + "/"
                           + std::to_string(element->tile->x()) + "/"
                           + std::to_string(element->tile->y());

    int type = data.type() & CV_MAT_DEPTH_MASK;

    //根据数据类型进行判断
    switch(type)
    {
      case CV_8U:
        filename += ".png";
        break;
      case CV_16U:
        filename += ".bin";
        break;
      case CV_32F:
        filename += ".bin";
        break;
      case CV_64F:
        filename += ".bin";
        break;
      default:
        throw(std::invalid_argument("Error writing tile: data type unknown!"));
    }

    //将数据写入磁盘
    io::saveImage(data, filename);

    element->was_written = true;
  }
}

//刷新
void TileCache::flush(const CacheElement::Ptr &element) const
{
  if (!element->was_written)
    write(element);//将数据写入磁盘

  for (const auto &meta : element->layer_meta)
  {
    element->tile->data()->remove(meta.name);//从瓦片的数据存储中移除特定图层的数据
  }

  LOG_IF_F(INFO, m_verbose, "Flushed tile (%i, %i, %i) [zoom, x, y]", element->tile->zoom_level(), element->tile->x(), element->tile->y());
}

//判断特定瓦片数据是否已经缓存
bool TileCache::isCached(const CacheElement::Ptr &element) const
{
  return !(element->tile->data()->empty());
}

size_t TileCache::estimateByteSize(const Tile::Ptr &tile) const
{
  tile->lock();
  //size_t bytes = tile->data().total() * tile->data().elemSize();
  tile->unlock();

  //return bytes;
  return 0;
}

//更新预测
void TileCache::updatePrediction(int zoom_level, const cv::Rect2i &roi_current)
{
  std::lock_guard<std::mutex> lock(m_mutex_roi_prev_request);
  std::lock_guard<std::mutex> lock1(m_mutex_roi_prediction);

  //查找m_roi_prev_request映射中是否存在给定缩放级别的键，来获取之前请求的区域
  auto it_roi_prev_request = m_roi_prev_request.find(zoom_level);
  if (it_roi_prev_request == m_roi_prev_request.end())//如果找不到之前请求的区域
  {
    // There was no previous request, so there can be no prediction which region of tiles might be needed in the next
    // processing step. Therefore set the current roi to be the prediction for the next request.
    m_roi_prediction[zoom_level] = roi_current;//当前的ROI设置为下一次请求的预测
  }
  else
  {
    // We have a previous roi that was requested, therefore we can extrapolate what the next request might look like
    // utilizing our current roi
    //根据当前ROI来预测下一次请求可能的区
    auto it_roi_prediction = m_roi_prediction.find(zoom_level);
    //根据当前请求和之前请求之间的差异来预测下一次请求可能的区域，从而更新预测的区域
    it_roi_prediction->second.x = roi_current.x + (roi_current.x - it_roi_prev_request->second.x);
    it_roi_prediction->second.y = roi_current.y + (roi_current.y - it_roi_prev_request->second.y);
    it_roi_prediction->second.width = roi_current.width + (roi_current.width - it_roi_prev_request->second.width);
    it_roi_prediction->second.height = roi_current.height + (roi_current.height - it_roi_prev_request->second.height);
  }

  //更新之前请求的ROI为当前ROI
  it_roi_prev_request->second = roi_current;
}

//在文件系统中创建目录
void TileCache::createDirectories(const std::string &toplevel, const std::vector<std::string> &layer_names, const std::string &tile_tree)
{
  //遍历layer_names中的每个名称
  for (const auto &layer_name : layer_names)
  {
    //在toplevel路径下创建一个与该名称对应的子目录
    io::createDir(toplevel + layer_name + tile_tree);
  }
}