

#include <realm_ortho/dsm.h>

#include <opencv2/imgproc.hpp>

using namespace realm::ortho;
//处理数字地表模型

//初始化成员变量，创建了一个网格地图并添加高程信息
DigitalSurfaceModel::DigitalSurfaceModel(const cv::Rect2d &roi, double elevation)
: m_use_prior_normals(false),
  m_assumption(SurfaceAssumption::PLANAR),
  m_surface_normal_mode(SurfaceNormalMode::NONE)
{
  // Planar surface means an elevation of zero.(平面表面意味着高程为零)
  // Therefore based region of interest a grid(基于感兴趣区域创建一个网格地图，并用零填充)
  // map is created and filled with zeros.
  // Resolution is assumed to be 1.0m as default(默认情况下，分辨率假定为 1.0 米)
  m_surface = std::make_shared<CvGridMap>();
  m_surface->setGeometry(roi, 1.0);
  m_surface->add("elevation", cv::Mat::ones(m_surface->size(), CV_32FC1) * elevation);
}

//创建一个数字地表模型，它接受观测点云、表面法线模式和最大 KNN 迭代次数等参数
DigitalSurfaceModel::DigitalSurfaceModel(const cv::Rect2d &roi,
                                         const cv::Mat& points,
                                         SurfaceNormalMode mode,
                                         int knn_max_iter)
    : m_use_prior_normals(false),
      m_knn_max_iter(knn_max_iter),
      m_assumption(SurfaceAssumption::ELEVATION),
      m_surface_normal_mode(mode)
{
  // Elevation surface means, that prior information
  // about the surface is available. Here in the form
  // of a point cloud of observed points.
  // Strategy is now to:
  // 0) Filter input point cloud for outlier(过滤输入的点云以去除异常值)
  // 1) Create a Kd-tree of the observed point cloud(创建观测点云的 Kd 树)
  // 2) Estimate resolution of the point cloud(估计点云的分辨率)
  // 3) Create a grid map based on the previously computed resolution(基于先前计算的分辨率创建网格地图)

  // Check if prior normals were computed and can be used
  //检查是否计算了先前法线并可以使用
  if (points.cols >= 9)
    m_use_prior_normals = true;

  // 0) Filter input point cloud and create container
  //过滤输入点云并创建容器
  cv::Mat points_filtered = filterPointCloud(points);

  //初始化点云容器
  m_point_cloud.pts.clear();
  m_point_cloud.pts.resize((unsigned long)points_filtered.rows);
  for (int i = 0; i < points_filtered.rows; ++i)
  {
    m_point_cloud.pts[i].x = points_filtered.at<double>(i, 0);
    m_point_cloud.pts[i].y = points_filtered.at<double>(i, 1);
    m_point_cloud.pts[i].z = points_filtered.at<double>(i, 2);
  }

  // 1) Init Kd-tree(初始化 Kd 树)
  initKdTree(m_point_cloud);

  // 2) Estimate resolution based on the point cloud(基于点云估计分辨率)
  double GSD_estimated = computePointCloudGSD(m_point_cloud);

  // 3) Create grid map based on point cloud resolution and surface info
  //基于点云分辨率和表面信息创建网格地图
  m_surface = std::make_shared<CvGridMap>(roi, GSD_estimated);
  computeElevation(points_filtered);
}

cv::Mat DigitalSurfaceModel::filterPointCloud(const cv::Mat &points)
{
  assert(points.type() == CV_64F);
  assert(m_assumption == SurfaceAssumption::ELEVATION);

//  pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud(new pcl::PointCloud<pcl::PointXYZ>);
//  for (uint32_t i = 0; i < points.rows; ++i)
//  {
//    pcl::PointXYZ pt;
//
//    // Position informations are saved in 0,1,2
//    pt.x = (float) points.at<double>(i, 0);
//    pt.y = (float) points.at<double>(i, 1);
//    pt.z = (float) points.at<double>(i, 2);
//
//    point_cloud->points.push_back(pt);
//  }
//  point_cloud->width = (int)point_cloud->points.size();
//  point_cloud->height = 1;
//
//  pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
//  pcl::StatisticalOutlierRemoval<pcl::PointXYZ> outlier_removal;
//  outlier_removal.setInputCloud(point_cloud);
//  outlier_removal.setMeanK(6);
//  outlier_removal.setStddevMulThresh(1.0);
//  outlier_removal.filter(*point_cloud_filtered);
//
//  cv::Mat points_filtered;
//  points_filtered.reserve(point_cloud_filtered->points.size());
//
//  for (const auto &pt_pcl : point_cloud_filtered->points)
//  {
//     cv::Mat pt_cv(1, 9, CV_64F);
//     pt_cv.at<double>(0) = pt_pcl.x;
//     pt_cv.at<double>(1) = pt_pcl.y;
//     pt_cv.at<double>(2) = pt_pcl.z;
//     points_filtered.push_back(pt_cv);
//     pt_cv.at<double>(3) = pt_pcl.r/255.0;
//     pt_cv.at<double>(4) = pt_pcl.g/255.0;
//     pt_cv.at<double>(5) = pt_pcl.b/255.0;
//     pt_cv.at<double>(6) = pt_pcl.normal_x;
//     pt_cv.at<double>(7) = pt_pcl.normal_y;
//     pt_cv.at<double>(8) = pt_pcl.normal_z;
//  }

  return points;
}

//初始化 Kd 树
void DigitalSurfaceModel::initKdTree(const PointCloud<double> &point_cloud)
{
  assert(m_assumption == SurfaceAssumption::ELEVATION);

  // Build kd-tree for space hierarchy
  //建 PointCloudAdaptor_t 对象，将输入的点云数据包装为 KD-Tree 需要的数据适配器
  m_point_cloud_adaptor.reset(new PointCloudAdaptor_t(m_point_cloud));
  //创建 DigitalSurfaceModel::KdTree_t 对象,通过传递维度、数据适配器和一些参数来初始化
  m_kd_tree.reset(new DigitalSurfaceModel::KdTree_t(kDimensionKdTree, *m_point_cloud_adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(kMaxLeaf)));
  m_kd_tree->buildIndex();//调用 buildIndex 方法来构建 KD-Tree
}

//计算点云的地面采样距离的估计值
double DigitalSurfaceModel::computePointCloudGSD(const PointCloud<double> &point_cloud)
{
  // Prepare container(准备容器)
  size_t n = point_cloud.pts.size();
  //初始化容器 dists、tmp_indices 和 tmp_dists
  std::vector<double> dists;
  std::vector<size_t> tmp_indices(2);
  std::vector<double> tmp_dists(2);

  // Iterate through the point cloud and compute nearest neighbour distance
  //根据点云中的点数，计算迭代的步长 n_iter
  auto n_iter = static_cast<size_t>(0.01 * n);
  dists.reserve(n/n_iter+1);
  //遍历点云
  for (size_t i = 0; i < n; i+=n_iter)
  {
    //提取点的坐标
    PointCloud<double>::Point pt = point_cloud.pts[i];

    // Preparation of output container
    //清空容器
    tmp_indices.clear();
    tmp_dists.clear();

    // Initialization
    const double query_pt[3]{pt.x, pt.y, 0.0};

    //搜索查询点的最近邻点
    m_kd_tree->knnSearch(&query_pt[0], 2u, &tmp_indices[0], &tmp_dists[0]);

    // "closest point" distance is zero, because search point is also trained data
    // Therefore choose the one point that distance is above approx 0.0
    /*判断最近邻距离，如果第一个最近邻距离大于一个小的阈值（10e-5），将其添加到距离容器中。
    如果第一个最近邻距离很小，检查第二个最近邻距离，如果它大于阈值，也将其添加到距离容器中
    */
    if (tmp_dists[0] > 10e-5)
      dists.push_back(tmp_dists[0]);
    else if (tmp_dists[1] > 10e-5)
      dists.push_back(tmp_dists[1]);
  }
  // Compute and return average dist(计算和返回平均距离)
  return sqrt(accumulate(dists.begin(), dists.end(), 0.0)/dists.size());
}

//计算数字地面模型的高程和法线
void DigitalSurfaceModel::computeElevation(const cv::Mat &point_cloud)
{
  assert(m_assumption == SurfaceAssumption::ELEVATION);

  //获取地面模型的尺寸
  cv::Size2i size = m_surface->size();

  //创建elevation 图层和elevation_normal 图层
  // Essential layers / must have
  cv::Mat elevation = cv::Mat(size, CV_32FC1, std::numeric_limits<float>::quiet_NaN());

  // Optional computation according to flag
  cv::Mat elevation_normal(size, CV_32FC3, cv::Scalar(0.0, 0.0, 0.0));

  //遍历地面模型的每一个像素位置
  for (uint32_t r = 0; r < size.height; ++r)
    for (uint32_t c = 0; c < size.width; ++c)
    {
      //在当前位置 (r, c),提取在地面模型中对应的真实坐标 pt
      cv::Point2d pt = m_surface->atPosition2d(r, c);
      std::vector<double> query_pt{pt.x, pt.y, 0.0};

      // Prepare kd-search for nearest neighbors(准备进行 KD 树的最近邻搜索)
      //获取地面模型的分辨率 resolution
      double resolution = m_surface->resolution();
      //创建一个容器 indices_dists 用于存储最近邻点的索引和距离
      std::vector<std::pair<int, double>> indices_dists;

      // Process neighbor search, if no neighbors are found, extend search distance
      //循环执行最近邻搜索，搜索半径会逐步增加，直到找到至少三个最近邻点或者达到最大迭代次数
      for (int i = 0; i < m_knn_max_iter; ++i)
      {
        nanoflann::RadiusResultSet<double, int> result_set(static_cast<double>(i) * resolution, indices_dists);
        m_kd_tree->findNeighbors(result_set, &query_pt[0], nanoflann::SearchParams());

        // Process only if neighbours were found
        if (result_set.size() >= 3u)
        {
          //创建容器 distances，heights，points，和 normals_prior 来存储距离、高程、点的坐标和先验法线信息
          std::vector<double> distances;
          std::vector<double> heights;
          std::vector<PlaneFitter::Point> points;
          std::vector<PlaneFitter::Normal> normals_prior;
          //遍历最近邻点集合，将每个点的距离和高程信息添加到相应的容器中
          for (const auto &s : result_set.m_indices_dists)
          {
            distances.push_back(s.second);
            heights.push_back(point_cloud.at<double>(s.first, 2));
            points.emplace_back(PlaneFitter::Point{point_cloud.at<double>(s.first, 0),
                                                   point_cloud.at<double>(s.first, 1),
                                                   point_cloud.at<double>(s.first, 2)});
            if (point_cloud.cols >= 9)
              normals_prior.emplace_back(PlaneFitter::Normal{point_cloud.at<double>(s.first, 6),
                                                             point_cloud.at<double>(s.first, 7),
                                                             point_cloud.at<double>(s.first, 8)});
          }

          //估计当前位置的高程值
          elevation.at<float>(r, c) = interpolateHeight(heights, distances);

          //计算或插值当前位置的法线信息，并存储在 elevation_normal 图层中
          if ((m_surface_normal_mode == SurfaceNormalMode::NONE) && m_use_prior_normals)
            elevation_normal.at<cv::Vec3f>(r, c) = interpolateNormal(normals_prior, distances);
          else if (m_surface_normal_mode != SurfaceNormalMode::NONE)
            elevation_normal.at<cv::Vec3f>(r, c) = computeSurfaceNormal(points, distances);

          break;
        }
      }
    }

  //将计算得到的高程信息存储在 elevation 图层中
  m_surface->add("elevation", elevation);

  // If exact normal computation was chosen, remove high frequent noise from data
  if (m_surface_normal_mode == SurfaceNormalMode::RANDOM_NEIGHBOURS
      || m_surface_normal_mode == SurfaceNormalMode::FURTHEST_NEIGHBOURS)
      //进行中值滤波，以去除高频噪声
    cv::medianBlur(elevation_normal, elevation_normal, 5);

  // If normals were computed. set in surface map
  //如果法线信息已计算，将法线信息存储在 elevation_normal 图层中
  if (m_surface_normal_mode != SurfaceNormalMode::NONE || m_use_prior_normals)
    m_surface->add("elevation_normal", elevation_normal);
}

//获取数字地面模型的指针
realm::CvGridMap::Ptr DigitalSurfaceModel::getSurfaceGrid()
{
  return m_surface;
}

//进行插值计算得到一个高度值
float DigitalSurfaceModel::interpolateHeight(const std::vector<double> &heights, const std::vector<double> &dists)
{
  double numerator = 0.0;
  double denominator = 0.0;
  //循环遍历提供的高度和距离数组
  for (size_t i = 0u; i < heights.size(); ++i)
  {
    numerator += heights[i] / dists[i];
    denominator += 1.0 / dists[i];
  }
  return static_cast<float>(numerator / denominator);
}

//进行插值计算得到一个平均法线向量
cv::Vec3f DigitalSurfaceModel::interpolateNormal(const std::vector<PlaneFitter::Normal> &normals, const std::vector<double> &dists)
{
  //初始化了一个三维向量 numerator
  cv::Vec3f numerator(0.0f, 0.0f, 0.0f);
  double denominator = 0.0;
  //循环遍历提供的法线向量和距离数组
  for (size_t i = 0u; i < dists.size(); ++i)
  {
    numerator[0] += normals[i].x / dists[i];
    numerator[1] += normals[i].y / dists[i];
    numerator[2] += normals[i].z / dists[i];
    denominator += 1.0 / dists[i];
  }
  return cv::normalize(numerator / static_cast<float>(denominator));
}

//计算表面法线向量
cv::Vec3f DigitalSurfaceModel::computeSurfaceNormal(const std::vector<PlaneFitter::Point> &points, const std::vector<double> &dists)
{
  assert(m_surface_normal_mode != SurfaceNormalMode::NONE);

  // Creation of plane fitter
  PlaneFitter plane_fitter;

  // Vector to save point indices that are used for normal computation
  std::vector<size_t> indices_points;
  std::vector<PlaneFitter::Point> points_selected;
  
  //根据不同的 m_surface_normal_mode 值，选择不同的点集用于计算法线
  switch(m_surface_normal_mode)
  {
    case SurfaceNormalMode::NONE:
      // Should be unreachable code segment
      throw(std::runtime_error("Error computing surface normal: Function called despite mode set to 'NONE'"));

    case SurfaceNormalMode::RANDOM_NEIGHBOURS:
      // Select the first three points found(选择最近的三个点)
      indices_points = std::vector<size_t>{0, 1, 2};
      points_selected = std::vector<PlaneFitter::Point>{
          points[indices_points[0]],
          points[indices_points[1]],
          points[indices_points[2]]};
      break;

    case SurfaceNormalMode::FURTHEST_NEIGHBOURS:
      // Select furthest points found(选择距离最远的三个点)
      indices_points = getKmaxElementsIndices(dists, 3);
      points_selected = std::vector<PlaneFitter::Point>{
          points[indices_points[0]],
          points[indices_points[1]],
          points[indices_points[2]]};
      break;

    case SurfaceNormalMode::BEST_FIT:
      // All points used for best fit(使用所有点进行最佳拟合)
      points_selected = points;
      break;
  }

  // In case of 3 points no best fit is computed -> exact solution
  //估计一个平面，该平面代表了这些选定点所在的表面
  PlaneFitter::Plane plane = plane_fitter.estimate(points_selected);

  // Ensure surface points up
  //检查平面的法线方向是否朝上
  if (plane.n.z < 0)
    return cv::Vec3f((float)-plane.n.x, (float)-plane.n.y, (float)-plane.n.z);
  else
    return cv::Vec3f((float)plane.n.x, (float)plane.n.y, (float)plane.n.z);
}

//从给定的双精度浮点数向量中选择最大的 k 个元素的索引
std::vector<size_t> DigitalSurfaceModel::getKmaxElementsIndices(std::vector<double> vec, size_t k)
{
  size_t n = vec.size();
  //检查 k 是否等于向量 vec 的大小 n
  if (k == n)
  //返回一个固定的大小为3的索引向量 {0, 1, 2}
    return std::vector<size_t>{0, 1, 2};
  else if(k > n)
    throw(std::out_of_range("Error computing kMaxElementsIndices: Not enough input values."));

  std::vector<size_t> result; result.reserve(k);
  //对向量 vec 进行排序
  std::vector<size_t> indices_sorted = sort_indices(vec);

  //从向量 indices_sorted 中逆向选择最大的 k 个索引
  //将它们添加到结果向量 result 中
  for (size_t i = n; n - k < i; --i)
    result.push_back(indices_sorted[i-1]);

  return result;
}