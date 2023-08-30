

#include <realm_ortho/map_tiler.h>

/*生成地图瓦片并执行地图坐标转换*/
using namespace realm;

//设置是否启用详细输出
MapTiler::MapTiler(bool verbosity)
    : m_verbosity(verbosity),
      m_zoom_level_min(11),
      m_zoom_level_max(35),
      m_tile_size(256)
{
  //计算原点偏移量,用于将地理坐标转换为像素坐标
  m_origin_shift = 2 * M_PI * 6378137 / 2.0;

  // Setup lookup table for zoom level resolution
  //循环遍历，为不同缩放级别的 m_lookup_nrof_tiles_from_zoom 查找表赋值
  for (int i = 0; i < m_zoom_level_max; ++i)
  {
    m_lookup_nrof_tiles_from_zoom[i] = std::pow(2, i);
  }

  //计算不同缩放级别下的分辨率
  computeLookupResolutionFromZoom();
}

//获取给定缩放级别的分辨率
double MapTiler::getResolution(int zoom_level)
{
  //在查找表中查找给定缩放级别的分辨率
  auto it = m_lookup_resolution_from_zoom.find(zoom_level);
  //检查查找是否成功
  if (it != m_lookup_resolution_from_zoom.end())
    return it->second;
  else
    throw(std::invalid_argument("Error getting resolution for zoom level: Lookup table does not contain key!"));
}

//创建瓦片地图
std::map<int, MapTiler::TiledMap> MapTiler::createTiles(const CvGridMap::Ptr &map, int zoom_level_min, int zoom_level_max)
{
  // Set the region of interest before entering the loop, so that even though we transform the map itself the original
  // boundaries will matter
  //获取输入地图的感兴趣区域
  cv::Rect2d roi = map->roi();

  // Identify the zoom levels we work on
  //计算用于计算缩放级别的基准缩放级别
  int zoom_level_base = computeZoomForPixelSize(map->resolution());

  //检查是否指定了最小缩放级别和最大缩放级别
  if (zoom_level_min == -1)
  {
    zoom_level_min = zoom_level_base;
  }

  if (zoom_level_max == -1)
  {
    zoom_level_max = zoom_level_base;
  }

  //检查最小缩放级别是否小于最大缩放级别
  if (zoom_level_min > zoom_level_max)
    throw(std::invalid_argument("Error computing tiles: Minimum zoom level larger than maximum."));

  // Computation of tiles per zoom level
  //创建一个存储不同缩放级别瓦片地图的容器
  std::map<int, TiledMap> tiles_from_zoom;

  //循环遍历从最大缩放级别到最小缩放级别的范围，为每个缩放级别创建瓦片地图
  for (int zoom_level = zoom_level_max; zoom_level >= zoom_level_min; --zoom_level)
  {
    // 获取当前缩放级别的分辨率
    double zoom_resolution = getResolution(zoom_level);

    LOG_IF_F(INFO, m_verbosity, "Tileing map on zoom level %i, resolution = %4.4f", zoom_level, zoom_resolution);
    //使用当前缩放级别的分辨率来调整输入地图的分辨
    map->changeResolution(zoom_resolution);

    // Map is now in the right coordinate frame and has the correct resolution. It's time to start tileing it
    // Therefore first identify how many tiles we have to split our map into by computing the tile indices
    //计算覆盖感兴趣区域的瓦片索引范围
    cv::Rect2i tile_bounds_idx = computeTileBounds(roi, zoom_level);

    // With the tile indices we can compute the exact region of interest in the geographic frame in meters
    //使用瓦片索引范围计算在地理坐标系中感兴趣的瓦片区域
    cv::Rect2d tile_bounds_meters = computeTileBoundsMeters(tile_bounds_idx, zoom_level);

    // Because our map is not yet guaranteed to have exactly the size of the tile region, we have to perform padding to
    // to fit exactly the tile map boundaries
    //对输入地图进行扩展以适应瓦片地图边界
    map->extendToInclude(tile_bounds_meters);

    //使用嵌套循环遍历所有要创建的瓦片，并将其添加到 tiles 容器中
    std::vector<Tile::Ptr> tiles;
    for (int x = 0; x < tile_bounds_idx.width; ++x)
      // Note: Coordinate system of the tiles is up positive, while image is down positive. Therefore the inverse loop
      for (int y = tile_bounds_idx.height; y > 0; --y)
      {
        cv::Rect2i data_roi(x*256, y*256, 256, 256);
        Tile::Ptr tile_current = std::make_shared<Tile>(zoom_level, tile_bounds_idx.x + x, tile_bounds_idx.y + tile_bounds_idx.height - y, map->getSubmap(map->getAllLayerNames(), data_roi));
        tiles.push_back(tile_current);
      }

    //将每个缩放级别的瓦片地图添加到 tiles_from_zoom 容器中
    tiles_from_zoom[zoom_level] = TiledMap{tile_bounds_idx, tiles};
  }

  return tiles_from_zoom;
}

//计算不同缩放级别的分辨率
void MapTiler::computeLookupResolutionFromZoom(double latitude)
{
  for (int i = 0; i < m_zoom_level_max; ++i)
  {
    m_lookup_resolution_from_zoom[i] = computeZoomResolution(i, latitude);
  }
}

//将给定的纬度和经度转换为在指定缩放级别下的瓦片坐标
cv::Point2i MapTiler::computeTileFromLatLon(double lat, double lon, int zoom_level) const
{
  double lat_rad = lat * M_PI / 180.0;
  auto n = static_cast<double>(m_lookup_nrof_tiles_from_zoom.at(zoom_level));

  cv::Point2i pos;
  pos.x = static_cast<int>(std::floor((lon + 180.0) / 360.0 * n));
  pos.y = static_cast<int>(std::floor((1.0 - asinh(tan(lat_rad)) / M_PI) / 2.0 * n));
  return pos;
}

//将给定的像素坐标转换为在指定缩放级别下的地图坐标
cv::Point2d MapTiler::computeMetersFromPixels(int px, int py, int zoom_level)
{
  cv::Point2d meters;
  double resolution = m_lookup_resolution_from_zoom.at(zoom_level);
  meters.x = px * resolution - m_origin_shift;
  meters.y = py * resolution - m_origin_shift;
  return meters;
}

//将给定的地图坐标转换为在指定缩放级别下的像素坐标
cv::Point2i MapTiler::computePixelsFromMeters(double mx, double my, int zoom_level)
{
  cv::Point2i pixels;
  double resolution = m_lookup_resolution_from_zoom.at(zoom_level);
  pixels.x = (mx + m_origin_shift) / resolution;
  pixels.y = (my + m_origin_shift) / resolution;
  return pixels;
}

//将给定的像素坐标转换为在指定缩放级别下的瓦片坐标
cv::Point2i MapTiler::computeTileFromPixels(int px, int py, int zoom_level)
{
  cv::Point2i tile;
  tile.x = int(std::ceil(px / (double)(m_tile_size)) - 1);
  tile.y = int(std::ceil(py / (double)(m_tile_size)) - 1);
  return tile;
}

//将给定的地图坐标转换为在指定缩放级别下的瓦片坐标
cv::Point2i MapTiler::computeTileFromMeters(double mx, double my, int zoom_level)
{
  cv::Point2i pixels = computePixelsFromMeters(mx, my, zoom_level);
  return computeTileFromPixels(pixels.x, pixels.y, zoom_level);
}

//根据给定的地图区域（roi）和缩放级别，计算该区域在瓦片坐标系中的瓦片边界
cv::Rect2i MapTiler::computeTileBounds(const cv::Rect2d &roi, int zoom_level)
{
  cv::Point2i tile_idx_low = computeTileFromMeters(roi.x, roi.y, zoom_level);
  cv::Point2i tile_idx_high = computeTileFromMeters(roi.x + roi.width, roi.y + roi.height, zoom_level);
  // Note: +1 in both directions because tile origin sits in the lower left corner of the tile
  return cv::Rect2i(tile_idx_low.x, tile_idx_low.y, tile_idx_high.x - tile_idx_low.x + 1, tile_idx_high.y - tile_idx_low.y + 1);
}

//根据给定的瓦片坐标和缩放级别，计算该瓦片在地图坐标系中的地图边界
cv::Rect2d MapTiler::computeTileBoundsMeters(int tx, int ty, int zoom_level)
{
  cv::Point2d p_min = computeMetersFromPixels(tx * m_tile_size, ty * m_tile_size, zoom_level);
  cv::Point2d p_max = computeMetersFromPixels((tx + 1) * m_tile_size, (ty + 1) * m_tile_size, zoom_level);
  return cv::Rect2d(p_min.x, p_min.y, p_max.x - p_min.x, p_max.y - p_min.y);
}

//重载:根据给定的瓦片索引区域和缩放级别，计算该瓦片索引区域在地图坐标系中的地图边界
cv::Rect2d MapTiler::computeTileBoundsMeters(const cv::Rect2i &idx_roi, int zoom_level)
{
  cv::Rect2d tile_bounds_low = computeTileBoundsMeters(idx_roi.x, idx_roi.y, zoom_level);
  cv::Rect2d tile_bounds_high = computeTileBoundsMeters(idx_roi.x + idx_roi.width + 1, idx_roi.y + idx_roi.height + 1, zoom_level);
  return cv::Rect2d(tile_bounds_low.x, tile_bounds_low.y, tile_bounds_high.x - tile_bounds_low.x, tile_bounds_high.y - tile_bounds_low.y);
}

//计算给定瓦片坐标在指定缩放级别下的中心点的经纬度坐标（WGS84坐标系）
WGSPose MapTiler::computeLatLonForTile(int x, int y, int zoom_level) const
{
  auto n = static_cast<double>(m_lookup_nrof_tiles_from_zoom.at(zoom_level));
  double k = M_PI - 2.0 * M_PI * y / n;

  WGSPose wgs{};
  wgs.latitude = 180.0 / M_PI * atan(0.5 * (exp(k) - exp(-k)));
  wgs.longitude = x / n * 360.0 - 180;

  return wgs;
}

//根据给定的地面采样距离计算适合的缩放级别
//do_upscale 用于指定是否允许向上缩放
int MapTiler::computeZoomForPixelSize(double GSD, bool do_upscale) const
{
  for (int i = 0; i < m_zoom_level_max; ++i)
  {
    if (GSD >= m_lookup_resolution_from_zoom.at(i) + 10e-3)
    {
      if (do_upscale)
        return std::max(0, i);
      else
        return std::max(0, i - 1);
    }
  }
  return m_zoom_level_max - 1;
}

//计算在给定缩放级别和纬度下的地图分辨率
double MapTiler::computeZoomResolution(int zoom_level, double latitude) const
{
  if (fabs(latitude) < 10e-3)
    return (2 * M_PI * 6378137 / m_tile_size) / m_lookup_nrof_tiles_from_zoom.at(zoom_level);
  else
    return 156543.03 * cos(latitude*M_PI/180) / m_lookup_nrof_tiles_from_zoom.at(zoom_level);
}