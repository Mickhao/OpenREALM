

#include <realm_io/pcl_export.h>

#include <opencv2/imgproc.hpp>

#include <pcl/io/ply_io.h>
#include <pcl/conversions.h>
#include <pcl/point_types.h>

#include <realm_core/cv_grid_map.h>

namespace realm
{
namespace io
{

//将地图中的高程点数据以PLY格式保存到指定的目录和文件名中
void saveElevationPointsToPLY(const CvGridMap &map,
                              const std::string &ele_layer_name,
                              const std::string &normals_layer_name,
                              const std::string &color_layer_name,
                              const std::string &mask_layer_name,
                              const std::string &directory,
                              const std::string &name)
{
  //使用目录和文件名构建保存文件的完整路径
  std::string filename = (directory + "/" + name + ".ply");
  //将高程点数据保存为PLY文件
  saveElevationPoints(map, ele_layer_name, normals_layer_name, color_layer_name, mask_layer_name, filename, "ply");
}
//根据不同情况调用两个不同的保存函数
void saveElevationPoints(const CvGridMap &map,
                         const std::string &ele_layer_name,
                         const std::string &normals_layer_name,
                         const std::string &color_layer_name,
                         const std::string &mask_layer_name,
                         const std::string &filename,
                         const std::string &suffix)
{
  //确保高程图层存在
  assert(map.exists(ele_layer_name));
  //如果法线图层名称不为空，则确保法线图层存在
  assert(normals_layer_name.empty() ? map.exists(normals_layer_name) : true);
  //确保遮罩图层存在
  assert(map.exists(mask_layer_name));

  //根据是否有法线图层，选择调用不同的保存函数
  if (normals_layer_name.empty())
    saveElevationPointsRGB(map, ele_layer_name, color_layer_name, mask_layer_name, filename, suffix);
  else
    saveElevationPointsRGBNormal(map, ele_layer_name, normals_layer_name, color_layer_name, mask_layer_name, filename, suffix);
}

void saveElevationPointsRGB(const CvGridMap &map,
                            const std::string &ele_layer_name,
                            const std::string &color_layer_name,
                            const std::string &mask_layer_name,
                            const std::string &filename,
                            const std::string &suffix)
{
  //确保高程图层存在
  assert(map.exists(ele_layer_name));
  //确保颜色图层存在
  assert(map.exists(color_layer_name));
  //确保遮罩图层存在
  assert(map.exists(mask_layer_name));
  
  //获取地图的尺寸
  cv::Size2i size = map.size();
  //计算点的数量
  auto n = static_cast<uint32_t>(size.width*size.height);
  //获取高程图层的数据
  cv::Mat elevation = map[ele_layer_name];
  //创建遮罩图层，用于过滤无效的点
  cv::Mat mask = (elevation == elevation);
  //获取颜色图层的数据
  cv::Mat color = map[color_layer_name];
  //如果颜色图层的类型为CV_8UC4（BGRA），将其转换为CV_8UC3（BGR）
  if (color.type() == CV_8UC4)
    cv::cvtColor(color, color, cv::ColorConversionCodes::COLOR_BGRA2BGR);

  // Fill in the cloud data
  //构建点云数据
  pcl::PointCloud<pcl::PointXYZRGB> cloud;
  cloud.width    = n;
  cloud.height   = 1;
  cloud.is_dense = false;
  cloud.points.reserve(cloud.width * cloud.height);

  //遍历地图中的每个元素，创建点云数据
  for (int r = 0; r < size.height; ++r)
    for (int c = 0; c < size.width; ++c)
    {
      // Skip invalid grid elements
      //跳过无效的网格元素
      if (mask.at<uchar>(r, c) == 0)
        continue;
        //获取指定位置的3D坐标和颜色值，然后将它们添加到点云中
      cv::Point3d pt = map.atPosition3d(r, c, "elevation");
      cv::Vec3b bgr = color.at<cv::Vec3b>(r, c);

      pcl::PointXYZRGB pt_rgb;
      pt_rgb.x = static_cast<float>(pt.x);
      pt_rgb.y = static_cast<float>(pt.y);
      pt_rgb.z = static_cast<float>(pt.z);
      pt_rgb.r = bgr[2];
      pt_rgb.g = bgr[1];
      pt_rgb.b = bgr[0];

      cloud.push_back(pt_rgb);
    }
  cloud.resize(cloud.size());
  //如果文件格式后缀为"ply",将点云保存为二进制的PLY文件
  if (suffix == "ply")
    pcl::io::savePLYFileBinary(filename, cloud);
}

void saveElevationPointsRGBNormal(const CvGridMap &map,
                                  const std::string &ele_layer_name,
                                  const std::string &normals_layer_name,
                                  const std::string &color_layer_name,
                                  const std::string &mask_layer_name,
                                  const std::string &filename,
                                  const std::string &suffix)
{
  //确保各类图层存在
  assert(map.exists(ele_layer_name));
  assert(map.exists(normals_layer_name));
  assert(map.exists(color_layer_name));
  assert(map.exists(mask_layer_name));
  //获取地图的尺寸，计算点的数量
  cv::Size2i size = map.size();
  auto n = static_cast<uint32_t>(size.width*size.height);
  //获取高程、法线、颜色和遮罩图层的数据
  cv::Mat elevation = map[ele_layer_name];
  cv::Mat elevation_normals = map[normals_layer_name];
  cv::Mat mask = map[mask_layer_name];
  cv::Mat color = map[color_layer_name];
  if (color.type() == CV_8UC4)
    cv::cvtColor(color, color, cv::ColorConversionCodes::COLOR_BGRA2BGR);

  // Fill in the cloud data
  //构建点云数据
  pcl::PointCloud<pcl::PointXYZRGBNormal> cloud;
  cloud.width    = n;
  cloud.height   = 1;
  cloud.is_dense = false;
  cloud.points.reserve(cloud.width * cloud.height);

  //遍历地图中的每个元素，创建点云数据
  for (int r = 0; r < size.height; ++r)
    for (int c = 0; c < size.width; ++c)
    {
      // Skip invalid grid elements
      if (mask.at<uchar>(r, c) == 0)
        continue;

        //获取指定位置的3D坐标、法线和颜色值，然后将它们添加到点云中
      cv::Point3d pt = map.atPosition3d(r, c, "elevation");
      cv::Vec3f normal = elevation_normals.at<cv::Vec3f>(r, c);
      cv::Vec3b bgr = color.at<cv::Vec3b>(r, c);

      pcl::PointXYZRGBNormal pt_rgb;
      pt_rgb.x = static_cast<float>(pt.x);
      pt_rgb.y = static_cast<float>(pt.y);
      pt_rgb.z = static_cast<float>(pt.z);
      pt_rgb.r = bgr[2];
      pt_rgb.g = bgr[1];
      pt_rgb.b = bgr[0];
      pt_rgb.normal_x = normal[0];
      pt_rgb.normal_y = normal[1];
      pt_rgb.normal_z = normal[2];

      cloud.push_back(pt_rgb);
    }
    //调整点云的大小，确保没有冗余的点
  cloud.resize(cloud.size());

  //如果文件格式后缀为"ply",将点云保存为二进制的PLY文件
  if (suffix == "ply")
    pcl::io::savePLYFileBinary(filename, cloud);
}

void saveElevationMeshToPLY(const CvGridMap &map,
                            const std::vector<cv::Point2i> &vertices,
                            const std::string &ele_layer_name,
                            const std::string &normal_layer_name,
                            const std::string &color_layer_name,
                            const std::string &mask_layer_name,
                            const std::string &directory,
                            const std::string &name)
{
  //构建保存文件名
  std::string filename = (directory + "/" + name + ".ply");
  //将之前构建的文件路径传递给函数 saveElevationMeshToPLY ，用于保存网格数据为PLY文件
  saveElevationMeshToPLY(map, vertices, ele_layer_name, normal_layer_name, color_layer_name, mask_layer_name, filename);
}

void saveElevationMeshToPLY(const CvGridMap &map,
                            const std::vector<cv::Point2i> &vertices,
                            const std::string &ele_layer_name,
                            const std::string &normal_layer_name,
                            const std::string &color_layer_name,
                            const std::string &mask_layer_name,
                            const std::string &filename)
{
  //确保各类图层存在
  assert(map.exists(ele_layer_name));
  assert(map.exists(normal_layer_name));
  assert(map.exists(color_layer_name));
  assert(map.exists(mask_layer_name));
  //获取地图的尺寸，计算地图中点的总数
  cv::Size2i size = map.size();
  auto n = static_cast<uint32_t>(size.width*size.height);
  //获取高程、法线、颜色和遮罩数据
  cv::Mat elevation = map[ele_layer_name];
  cv::Mat elevation_normals = map[normal_layer_name];
  cv::Mat mask = map[mask_layer_name];
  cv::Mat color = map[color_layer_name];
  if (color.type() == CV_8UC4)
    cv::cvtColor(color, color, cv::ColorConversionCodes::COLOR_BGRA2BGR);

  // Fill in the cloud data
  //构建点云数据
  pcl::PointCloud<pcl::PointXYZRGBNormal> cloud;
  cloud.width    = n;
  cloud.height   = 1;
  cloud.is_dense = false;
  cloud.points.reserve(cloud.width * cloud.height);

  std::unordered_map<int, size_t> rc_to_idx;
  //遍历地图中的每个元素
  for (int r = 0; r < size.height; ++r)
    for (int c = 0; c < size.width; ++c)
    {
      // Skip invalid grid elements
      if (mask.at<uchar>(r, c) == 0)
        continue;

      cv::Point3d pt = map.atPosition3d(r, c, "elevation");
      cv::Vec3f normal = elevation_normals.at<cv::Vec3f>(r, c);
      cv::Vec3b bgr = color.at<cv::Vec3b>(r, c);

      pcl::PointXYZRGBNormal pt_rgb;
      pt_rgb.x = static_cast<float>(pt.x);
      pt_rgb.y = static_cast<float>(pt.y);
      pt_rgb.z = static_cast<float>(pt.z);
      pt_rgb.r = bgr[2];
      pt_rgb.g = bgr[1];
      pt_rgb.b = bgr[0];
      pt_rgb.normal_x = normal[0];
      pt_rgb.normal_y = normal[1];
      pt_rgb.normal_z = normal[2];

      cloud.push_back(pt_rgb);

      int hash = r*size.width+c;
      rc_to_idx[hash] = cloud.size()-1;
    }

  //用于追踪多边形的数量
  size_t count = 0;
  //创建存储多边形的容器 polygons
  std::vector<pcl::Vertices> polygons(vertices.size()/3);
  //循环遍历顶点坐标
  for (size_t i = 0; i < vertices.size(); i+=3, ++count)
    //循环遍历每个三角形的三个顶点
    for (uint8_t j = 0; j < 3; ++j)
    {
      //使用三角形中的顶点的行和列坐标来计算哈希值
      int hash = vertices[i+j].y*size.width + vertices[i+j].x;
      //将计算得到的哈希值传递给之前构建的 rc_to_idx 映射，以获取在点云中的索引
      //然后，它将该索引添加到当前多边形的顶点列表中，从而定义了该多边形的顶点序列
      polygons[count].vertices.push_back(rc_to_idx[hash]);
    }
  //创建 pcl::PolygonMesh 对象 mesh
  pcl::PolygonMesh mesh;
  //将点云数据转换为PCL格式
  pcl::toPCLPointCloud2(cloud, mesh.cloud);
  //将构建的多边形数据设置为 mesh 对象的多边形列表
  mesh.polygons = polygons;
  //将构建的多边形网格保存为PLY二进制文件
  pcl::io::savePLYFileBinary(filename, mesh);
}

}
}
