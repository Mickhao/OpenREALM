

#include <iostream>
#include <algorithm>
#include <cmath>

#include <realm_core/loguru.h>
#include <realm_core/cv_grid_map.h>

/*CvGridMap 是一个用于表示网格地图的类
是一个多层的矩阵结构，每一层都有一个名称和一个类型
它可以存储不同类型的数据，如高程，正射影像，掩码等
还提供了一些函数来进行网格地图之间的拼接，裁剪，插值等*/
using namespace realm;

//初始化地图的分辨率为 1.0
CvGridMap::CvGridMap()
    : m_resolution(1.0)
{
}

//构造函数，通过给定的感兴趣区域（roi）和分辨率（resolution）来初始化地图
CvGridMap::CvGridMap(const cv::Rect2d &roi, double resolution)
    : m_resolution(resolution)
{
  setGeometry(roi, m_resolution);
}

//创建并返回当前地图的一个副本
CvGridMap CvGridMap::clone() const
{
  CvGridMap copy;
  copy.setGeometry(m_roi, m_resolution);
  for (const auto &layer : m_layers)
    copy.add(layer.name, layer.data.clone(), layer.interpolation);
  return std::move(copy);
}

//创建并返回当前地图的一个子地图副本，其中只包含指定的图层（layer_names）
CvGridMap CvGridMap::cloneSubmap(const std::vector<std::string> &layer_names)
{
  CvGridMap copy;
  copy.setGeometry(m_roi, m_resolution);
  for (const auto &layer_name : layer_names)
  {
    Layer layer = getLayer(layer_name);
    copy.add(layer.name, layer.data.clone(), layer.interpolation);
  }
  return copy;
}

/*将指定的图层数据添加到地图中
如果图层已经存在，则更新其数据
否则，将其添加到图层容器中
*/
void CvGridMap::add(const Layer &layer, bool is_data_empty)
{
  if (!is_data_empty && (layer.data.size().width != m_size.width || layer.data.size().height != m_size.height))
    throw(std::invalid_argument("Error: Adding Layer failed. Size of data does not match grid map size."));
  if (!isMatrixTypeValid(layer.data.type()))
    throw(std::invalid_argument("Error: Adding Layer failed. Matrix type is not supported."));

  // Add data if layer already exists, push to container if not
  if (!exists(layer.name))
    m_layers.push_back(layer);
  else
  {
    m_layers[findContainerIdx(layer.name)] = layer;
  }
}

//用于将指定名称的图层数据添加到地图中
void CvGridMap::add(const std::string &layer_name, const cv::Mat &layer_data, int interpolation)
{
  add(Layer{layer_name, layer_data, interpolation});
}

//用于将子地图数据添加到当前地图中
void CvGridMap::add(const CvGridMap &submap, int flag_overlap_handle, bool do_extend)
{
  //检查传入的子地图与当前地图的分辨率是否匹配
  if (fabs(resolution() - submap.resolution()) > std::numeric_limits<double>::epsilon())
    throw(std::invalid_argument("Error add submap: Resolution mismatch!"));

  // Check if extending the map is necessary(检查是否需要扩展当前地图)
  bool contains_sub_roi = containsRoi(submap.m_roi);

  // Create the theoretical ROI bounds to copy the submap data to
  cv::Rect2d copy_roi = submap.roi();

  // If extension is allowed, then extend the reference map first
  //如果 do_extend 为 true 且子地图不在当前地图的范围内，则会将当前地图扩展以包含子地图的数据
  if (do_extend && !contains_sub_roi)
  {
    extendToInclude(copy_roi);
  }
  // If extension is not allowed and the boundaries of the submap are outside the reference map, adjust the copy area
  //如果不允许扩展，并且子地图的范围超出了当前地图的边界，则会调整 copy_roi 以确保只复制重叠部分的数据
  else if(!do_extend && !contains_sub_roi)
  {
    // Get the overlapping region of interest of both rectangles
    copy_roi = (copy_roi & m_roi);

    // Check if there is an overlap at all
    if (copy_roi.area() < 10e-6)
    {
      LOG_F(WARNING, "Called 'add(Submap)' on submap outside the reference image without allowing to extend it.");
      return;
    }
  }
  // getting index of grid roi
  //获取子地图的重叠区域的索引范围 src_roi 和当前地图的索引范围 dst_roi
  cv::Rect2i src_roi(submap.atIndexROI(copy_roi));
  cv::Rect2i dst_roi(this->atIndexROI(copy_roi));

  // iterate through submap
  //循环遍历子地图中的每个图层 submap_layer
  for (const auto &submap_layer : submap.m_layers)
  {
    // Add of submap to this, if not existing
    //如果当前地图不存在名为 submap_layer.name 的图层，则会将一个空白的图层添加到当前地图中
    if (!exists(submap_layer.name))
      add(Layer{submap_layer.name, cv::Mat()}, true);

    // Now layers will exist, get it
    //获取当前地图中图层的索引 idx_layer
    uint32_t idx_layer = findContainerIdx(submap_layer.name);

    // But might be empty
    //如果当前地图中该图层的数据为空，则会根据数据类型创建一个与当前地图相同尺寸的空白数据矩阵
    if (m_layers[idx_layer].data.empty())
    {
      switch(m_layers[idx_layer].data.type())
      {
        case CV_32F:
          m_layers[idx_layer].data = cv::Mat(m_size, submap_layer.data.type(), std::numeric_limits<float>::quiet_NaN());
          break;
        case CV_64F:
          m_layers[idx_layer].data = cv::Mat(m_size, submap_layer.data.type(), std::numeric_limits<double>::quiet_NaN());
          break;
        default:
          m_layers[idx_layer].data = cv::Mat::zeros(m_size, submap_layer.data.type());
      }
    }

    // Get the data in the overlapping area of both mat
    //获取子地图和当前地图中重叠区域的数据矩阵，分别存储在 src_data_roi 和 dst_data_roi
    cv::Mat src_data_roi = submap_layer.data(src_roi);
    cv::Mat dst_data_roi = m_layers[idx_layer].data(dst_roi);

    // Final check for matrix size
    //检查矩阵的维度是否匹配，如果不匹配，则输出警告信息并跳过当前图层的处理
    if (src_data_roi.rows != dst_data_roi.rows || src_data_roi.cols != dst_data_roi.cols)
    {
      LOG_F(WARNING, "Overlap area could not be merged. Matrix dimensions mismatched!");
      continue;
    }

    // Now is finally the turn to calculate overlap result and copy it to src grid map
    //最终，使用指定的重叠处理方式 flag_overlap_handle 对重叠区域的数据进行合并，然后将合并后的数据复制回当前地图的相应图层中
    mergeMatrices(src_data_roi, dst_data_roi, flag_overlap_handle);
    dst_data_roi.copyTo(m_layers[idx_layer].data(dst_roi));
  }
}

//用于从当前地图中移除指定名称的图层
void CvGridMap::remove(const std::string &layer_name)
{
  auto it = m_layers.begin();
  while(it != m_layers.end())
  {
    if (it->name == layer_name)
    {
      m_layers.erase(it);
      break;
    }
    it++;
  }
}

//接受一个图层名称的向量，并将这些图层从当前地图中移除
void CvGridMap::remove(const std::vector<std::string> &layer_names)
{
  for (const auto &layer_name : layer_names)
    remove(layer_name);
}

//用于检查当前地图是否为空
bool CvGridMap::empty() const
{
  return (m_layers.size() == 0);
}

//用于检查指定名称的图层是否存在于当前地图中
bool CvGridMap::exists(const std::string &layer_name) const
{
  for (const auto &layer : m_layers)
  {
    if (layer.name == layer_name)
      return true;
  }
  return false;
}

//于检查当前地图的范围是否包含指定的矩形区域 roi
bool CvGridMap::containsRoi(const cv::Rect2d &roi) const
{
  return fabs((m_roi & roi).area() - roi.area()) < 10e-6;
}

//用于获取指定名称的图层的data
cv::Mat& CvGridMap::get(const std::string& layer_name)
{
  for (auto &layer : m_layers)
  {
    if (layer.name == layer_name)
      return layer.data;
  }
  throw std::out_of_range("No layer with name '" + layer_name + "' available.");
}

// 和上面函数类似，但不能通过这个引用修改图层的数据
const cv::Mat& CvGridMap::get(const std::string& layer_name) const
{
  for (const auto &layer : m_layers)
  {
    if (layer.name == layer_name)
      return layer.data;
  }
  throw std::out_of_range("No layer with name '" + layer_name + "' available.");
}

//用于获取指定名称的图层对象
CvGridMap::Layer CvGridMap::getLayer(const std::string& layer_name) const
{
  for (const auto& layer : m_layers)
    if (layer.name == layer_name)
      return layer;
  throw std::out_of_range("No layer with name '" + layer_name + "' available.");
}

//返回一个包含所有图层名称的字符串向量
std::vector<std::string> CvGridMap::getAllLayerNames() const
{
  std::vector<std::string> layer_names;
  for (const auto &layer : m_layers)
    layer_names.push_back(layer.name);
  return layer_names;
}

//用于获取当前地图的一个子地图，子地图包含指定名称的图层
CvGridMap CvGridMap::getSubmap(const std::vector<std::string> &layer_names) const
{
  CvGridMap submap;
  submap.setGeometry(m_roi, m_resolution);
  for (const auto &layer_name : layer_names)
  {
    Layer layer = getLayer(layer_name);
    submap.add(layer.name, layer.data, layer.interpolation);
  }
  return submap;
}

//用于获取指定矩形区域内的子地图，子地图包含指定的图层
CvGridMap CvGridMap::getSubmap(const std::vector<std::string> &layer_names, const cv::Rect2d &roi) const
{
  CvGridMap submap;
  submap.setGeometry(roi, m_resolution);
  //使用 getOverlap 函数获取当前地图与子地图之间的重叠区域
  CvGridMap::Overlap overlap = getOverlap(submap);

  //如果有重叠区域存在，它会返回重叠区域的当前地图的指定图层的子地图
  if (overlap.first != nullptr && overlap.second != nullptr)
    return overlap.first->getSubmap(layer_names);
  else
    throw(std::out_of_range("Error extracting submap: No overlap!"));
}

//
CvGridMap CvGridMap::getSubmap(const std::vector<std::string> &layer_names, const cv::Rect2i &roi) const
{
  CvGridMap submap;
  //通过 atPosition2d 函数将整数坐标转换为浮点坐标
  cv::Point2d pt = atPosition2d(roi.y, roi.x);
  submap.setGeometry(cv::Rect2d(pt.x, pt.y, (roi.width-1) * m_resolution, (roi.height - 1) * m_resolution), m_resolution);
  
  /*遍历指定的图层名称列表
  为子地图添加每个指定的图层
  只保留矩形区域内的数据
  */
  for (const auto &layer_name : layer_names)
  {
    Layer layer = getLayer(layer_name);
    submap.add(layer_name, layer.data(roi), layer.interpolation);
  }

  return submap;
}

//用于获取两个地图之间的重叠区域及其内容
CvGridMap::Overlap CvGridMap::getOverlap(const CvGridMap &other_map) const
{
  //通过对两个地图的边界框取交集来计算重叠的矩形区域
  cv::Rect2d overlap_roi = (m_roi & other_map.m_roi);

  // Check if overlap exists(检查重叠区域是否存在)
  //如果重叠区域的面积小于一个很小的阈值，就表示不存在重叠区域
  if (overlap_roi.area() < 10e-6)
    return Overlap{nullptr, nullptr};

  // getting matrix roi from both other and this map
  //使用 atIndexROI 函数将浮点坐标矩形区域转换为整数坐标矩形区域
  //获取两个地图在重叠区域内的矩形区域
  cv::Rect2i this_grid_roi(this->atIndexROI(overlap_roi));
  cv::Rect2i other_grid_roi(other_map.atIndexROI(overlap_roi));

  // Create this map as reference and initialize with layer names
  //map_ref作为参考地图
  auto map_ref = std::make_shared<CvGridMap>();
  map_ref->setGeometry(overlap_roi, m_resolution);
  /*遍历当前地图的图层
  为每个图层提取在重叠区域内的数据
  然后将该数据添加到参考地图的对应图层中
  */
  for (const auto &layer : m_layers)
  {
    cv::Mat overlap_data = layer.data(this_grid_roi).clone();
    map_ref->add(layer.name, overlap_data, layer.interpolation);
  }

  // Create other map as added map and initialize with other map layer names
  //map_added作为被添加地图
  auto map_added = std::make_shared<CvGridMap>();
  map_added->setGeometry(overlap_roi, m_resolution);
  for (const auto &layer : other_map.m_layers)
  {
    cv::Mat overlap_data = layer.data(other_grid_roi).clone();
    map_added->add(layer.name, overlap_data, layer.interpolation);
  }
  return std::make_pair(map_ref, map_added);
}

//返回一个 cv::Mat 的引用，允许你通过图层名称访问地图中的数据，并对其进行修改
cv::Mat& CvGridMap::operator[](const std::string& layer_name)
{
  return get(layer_name);
}

//返回一个 const cv::Mat 的引用，允许你通过图层名称访问地图中的数据，但不允许对其进行修改
const cv::Mat& CvGridMap::operator[](const std::string& layer_name) const
{
  return get(layer_name);
}

//用于设置网格地图的几何属性
void CvGridMap::setGeometry(const cv::Rect2d &roi, double resolution)
{
  if (roi.width < 10e-6 || roi.height < 10e-6)
    LOG_F(WARNING, "Grid dimension: %f x %f", roi.width, roi.height);
  if (m_resolution < 10e-6)
    throw(std::invalid_argument("Error: Resolution is zero!"));

  // Set basic members
  //更新地图的分辨率和几何属性
  m_resolution = resolution;
  fitGeometryToResolution(roi, m_roi, m_size);

  // Release all current data
  //释放所有当前图层的数据，使地图准备好接收新的数据
  for (auto &layer : m_layers)
    layer.data.release();
}

//于设置指定图层的插值方式
void CvGridMap::setLayerInterpolation(const std::string& layer_name, int interpolation)
{
  m_layers[findContainerIdx(layer_name)].interpolation = interpolation;
}

//用于扩展网格地图以包含给定的边界范围
void CvGridMap::extendToInclude(const cv::Rect2d &roi)
{
  if (roi.width <= 0.0 || roi.height <= 0.0)
    throw(std::invalid_argument("Error: Extending grid to include ROI failed. ROI dimensions zero!"));

  //计算出一个新的包围框 bounding_box，它包含了当前地图的边界和传入的 roi 的边界
  cv::Rect2d bounding_box;
  bounding_box.x = std::min({m_roi.x, roi.x});
  bounding_box.y = std::min({m_roi.y, roi.y});
  bounding_box.width = std::max({m_roi.x + m_roi.width, roi.x + roi.width}) - bounding_box.x;
  bounding_box.height = std::max({m_roi.y + m_roi.height, roi.y + roi.height}) - bounding_box.y;

  // The bounding box is the new ROI, next fit it to the given resolution and adjust the corresponding size of the grid
  //将新的包围框 bounding_box 适配到地图的分辨率，得到一个新的适合分辨率的边界范围 roi_set
  cv::Rect2d roi_set;
  fitGeometryToResolution(bounding_box, roi_set, m_size);

  // Add the grid growth to existing layer data
  //计算出在地图的四个方向上需要增加的大小，以便适配新的边界范围
  int size_x_right  = static_cast<int>(std::round((roi_set.x+roi_set.width - (m_roi.x + m_roi.width)) / m_resolution));
  int size_y_top    = static_cast<int>(std::round((roi_set.y+roi_set.height - (m_roi.y + m_roi.height)) / m_resolution));
  int size_x_left   = static_cast<int>(std::round((m_roi.x - roi_set.x) / m_resolution));
  int size_y_bottom = static_cast<int>(std::round((m_roi.y - roi_set.y) / m_resolution));

  if (size_x_left < 0) size_x_left = 0;
  if (size_y_bottom < 0) size_y_bottom = 0;
  if (size_x_right < 0) size_x_right = 0;
  if (size_y_top < 0) size_y_top = 0;
  
  //更新地图的边界范围 m_roi
  m_roi = roi_set;

  // afterwards add new size to existing layers
  /*遍历地图中的每个图层
  如果图层的数据不为空，则使用 cv::copyMakeBorder 函数在图层数据的上、下、左、右四个方向上添加指定大小的边界
  边界的填充值根据图层数据类型来确定
  */
  for (auto &layer : m_layers)
    if (!layer.data.empty())
    {
      switch(layer.data.type() & CV_MAT_DEPTH_MASK)
      {
        case CV_32F:
          cv::copyMakeBorder(layer.data, layer.data, size_y_top, size_y_bottom, size_x_left, size_x_right, cv::BORDER_CONSTANT, std::numeric_limits<float>::quiet_NaN());
          break;
        case CV_64F:
          cv::copyMakeBorder(layer.data, layer.data, size_y_top, size_y_bottom, size_x_left, size_x_right, cv::BORDER_CONSTANT, std::numeric_limits<double>::quiet_NaN());
          break;
        default:
          cv::copyMakeBorder(layer.data, layer.data, size_y_top, size_y_bottom, size_x_left, size_x_right, cv::BORDER_CONSTANT,0);
      }
    }
}

void CvGridMap::changeResolution(double resolution)
{
  //更新地图的分辨率为传入的值
  m_resolution = resolution;
  //使用新的分辨率来调整地图的几何范围和大小
  fitGeometryToResolution(m_roi, m_roi, m_size);
  /*遍历地图的每个图层
  如果图层的数据不为空且其尺寸与更新后的地图尺寸不匹配
  就使用 cv::resize 函数将图层数据调整到新的尺寸
  */
  for (auto &layer : m_layers)
    if (!layer.data.empty() && (layer.data.cols != m_size.width || layer.data.rows != m_size.height))
      cv::resize(layer.data, layer.data, m_size, layer.interpolation);
}

void CvGridMap::changeResolution(const cv::Size2i &size)
{
  // Compute the resizing depending on the final size of the matrix data. The -1 must be subtracted here, because the first
  // sample point is at (resolution/2, resolution/2) and the last at (width - resolution/2, height - resolution/2)
  //根据新尺寸计算出水平和垂直方向的缩放因子，以便将地图的尺寸调整到新的尺寸
  double x_factor = static_cast<double>(size.width-1) / (m_size.width - 1);
  double y_factor = static_cast<double>(size.height-1) / (m_size.height - 1);

  if (fabs(x_factor-y_factor) > 10e-6)
    throw(std::invalid_argument("Error changing resolution of CvGridMap: Desired size was not changed uniformly!"));
    //更新地图的分辨率
  m_resolution /= x_factor;
  //使用更新后的地图尺寸和分辨率来调整地图的几何范围和大小
  fitGeometryToResolution(m_roi, m_roi, m_size);
  /*遍历地图的每个图层
  如果图层的数据不为空且其尺寸与更新后的地图尺寸不匹配
  使用 cv::resize 函数将图层数据调整到新的尺寸
  */
  for (auto &layer : m_layers)
    if (!layer.data.empty() && (layer.data.cols != m_size.width || layer.data.rows != m_size.height))
      cv::resize(layer.data, layer.data, m_size, layer.interpolation);
}

//接受一个世界坐标位置 pos 作为参数，并返回与该位置最接近的地图矩阵索引
cv::Point2i CvGridMap::atIndex(const cv::Point2d &pos) const
{
  //将 pos 的 x 和 y 坐标分别四舍五入到与分辨率对齐的值，以获取最接近的矩阵索引
  double x_rounded = roundToResolution(pos.x, m_resolution);
  double y_rounded = roundToResolution(pos.y, m_resolution);

  double epsilon = m_resolution / 2;
  //检查是否所计算的索引在地图的有效范围内
  if (x_rounded < m_roi.x - epsilon || x_rounded > m_roi.x + m_roi.width + epsilon
      || y_rounded < m_roi.y - epsilon || y_rounded > m_roi.y + m_roi.height + epsilon)
    throw std::out_of_range("Requested world position is out of bounds.");
  
  //将计算得到的索引转换为 cv::Point2i 类型并返回
  cv::Point2i idx;
  idx.x = static_cast<int>(std::round((x_rounded - m_roi.x) / m_resolution));
  idx.y = static_cast<int>(std::round((m_roi.y + m_roi.height - y_rounded) / m_resolution));

  return idx;
}

//接受一个世界坐标的矩形范围 roi 作为参数，并返回该范围在地图矩阵中的索引矩形
cv::Rect2i CvGridMap::atIndexROI(const cv::Rect2d &roi) const
{
  // Get the four corners of the world coordinates in the matrix
  //使用 atIndex 函数将 roi 的左下角位置转换为地图矩阵索引
  cv::Point2i idx = atIndex(cv::Point2d(roi.x, roi.y+roi.height));
  //根据矩形的宽度和高度以及分辨率来计算矩阵索引矩形的大小
  int width = static_cast<int>(std::round(roi.width / m_resolution)) + 1;
  int height = static_cast<int>(std::round(roi.height / m_resolution)) + 1;

  return cv::Rect2i(idx.x, idx.y, width, height);
}

//接受矩阵的行索引 r 和列索引 c 作为参数，并返回与该矩阵索引对应的世界坐标位置
cv::Point2d CvGridMap::atPosition2d(uint32_t r, uint32_t c) const
{
  // check validity
  //确保给定的行和列索引在有效范围内
  assert(r < m_size.height && r >= 0);
  assert(c < m_size.width && c >= 0);
  assert(m_roi.width > 0 && m_roi.height > 0);
  assert(m_resolution > 0.0);

  //使用地图的几何范围、分辨率和索引值来计算世界坐标位置
  cv::Point2d pos;
  pos.x = m_roi.x + static_cast<double>(c) * m_resolution;
  pos.y = m_roi.y + m_roi.height - static_cast<double>(r) * m_resolution;  // ENU world frame
  return pos;
}

//返回在指定位置 (r, c) 和指定图层上的三维世界坐标点
cv::Point3d CvGridMap::atPosition3d(const int &r, const int &c, const std::string &layer_name) const
{
  //获取指定图层的数据矩阵
  cv::Mat layer_data = get(layer_name);

  // check validity
  //确保图层数据不为空，并且给定的行列索引在矩阵范围内
  if (layer_data.empty())
    throw(std::runtime_error("Error: Layer data empty! Requesting data failed."));
  if (r < 0 || r >= m_size.height || c < 0 || c >= m_size.width)
    throw(std::invalid_argument("Error: Requested position outside matrix boundaries!"));

  // create position and set data
  //使用地图的几何范围、分辨率和索引值来计算世界坐标位置
  cv::Point3d pos;
  pos.x = m_roi.x + static_cast<double>(c) * m_resolution;
  pos.y = m_roi.y + m_roi.height - static_cast<double>(r) * m_resolution;  // ENU world frame
  
  //根据图层数据的类型，获取指定位置的 z 分量
  if (layer_data.type() == CV_32F)
    pos.z = static_cast<double>(layer_data.at<float>(r, c));
  else if (layer_data.type() == CV_64F)
    pos.z = layer_data.at<double>(r, c);
  else
    throw(std::out_of_range("Error accessing 3d position in CvGridMap: z-coordinate data type not supported."));
  return pos;
}

//返回地图的分辨率
double CvGridMap::resolution() const
{
  return m_resolution;
}

//回地图的矩阵大小，即行数和列数
cv::Size2i CvGridMap::size() const
{
  return m_size;
}

//回地图的几何范围，即地图覆盖的世界坐标区域
cv::Rect2d CvGridMap::roi() const
{
  return m_roi;
}

//用于合并两个矩阵
void CvGridMap::mergeMatrices(const cv::Mat &from, cv::Mat &to, int flag_merge_handling)
{
  //根据标志参数的不同,进行不同的合并操作
  switch(flag_merge_handling)
  {
    case REALM_OVERWRITE_ALL:
      to = from;
      break;
      
      //根据不同数据类型的矩阵创建一个遮罩，然后根据遮罩条件进行部分值的覆盖
    case REALM_OVERWRITE_ZERO:
      cv::Mat mask;
      if (to.type() == CV_32F || to.type() == CV_64F)
        mask = (to != to) & (from == from);
      else
        mask = (to == 0) & (from > 0);
      from.copyTo(to, mask);
      break;
  }
}

//于验证给定的矩阵数据类型是否在支持的范围内
bool CvGridMap::isMatrixTypeValid(int type)
{
  switch(type & CV_MAT_DEPTH_MASK)
  {
    case CV_32F:
      return true;
    case CV_64F:
      return true;
    case CV_8U:
      return true;
    case CV_16U:
      return true;
    default:
      return false;
  }
}

//用于查找指定图层名称在图层容器中的索引
uint32_t CvGridMap::findContainerIdx(const std::string &layer_name)
{
  for (uint32_t i = 0; i < m_layers.size(); ++i)
    if (m_layers[i].name == layer_name)
      return i;
  throw(std::out_of_range("Error: Index for layer not found!"));
}

//用于将所需的矩形区域 roi_desired 调整为符合给定分辨率 m_resolution 的几何限制
void CvGridMap::fitGeometryToResolution(const cv::Rect2d &roi_desired, cv::Rect2d &roi_set, cv::Size2i &size_set)
{
  // We round the geometry of our region of interest to fit exactly into our resolution. So position x,y and dimensions
  // width,height are always multiples of the resolution
  roi_set.x = roundToResolution(roi_desired.x, m_resolution);
  roi_set.y = roundToResolution(roi_desired.y, m_resolution);
  roi_set.width = roundToResolution(roi_desired.width, m_resolution);
  roi_set.height = roundToResolution(roi_desired.height, m_resolution);

  // Please note, that the grid is always 1 element bigger than the width and height of the ROI. This is because the
  // samples in the world coordinate frame are in the center of the matrix elements. Therefore a resolution/2 is added
  // to all sides of the grid resulting in the additional + 1 as matrix size.
  size_set.width = static_cast<int>(std::round(roi_set.width / m_resolution)) + 1;
  size_set.height = static_cast<int>(std::round(roi_set.height / m_resolution)) + 1;
}

//用于将给定的值 value 调整为最接近的分辨率 resolution 的倍数
double CvGridMap::roundToResolution(double value, double resolution)
{
  double remainder = fmod(value, resolution);
  if (fabs(remainder) < 10e-6)
    return value;
  else
    return value + resolution - remainder;
}
