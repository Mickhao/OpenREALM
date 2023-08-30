

#include <realm_stages/conversions.h>
//将图像数据和网格数据转换为点云和三角面片的表示
namespace realm
{

//转换成点云表示
cv::Mat cvtToPointCloud(const cv::Mat &img3d, const cv::Mat &color, const cv::Mat &normals, const cv::Mat &mask)
{
  //检查深度图像是否为空
  if (img3d.empty())
    throw(std::invalid_argument("Error: Depth map empty. Conversion to point cloud failed."));
    //计算图像的总像素数量
  size_t n = (size_t)img3d.cols*img3d.rows;
  //将深度图像重塑为一个 n 行 1 列的矩阵，其中每个元素是一个像素的深度值
  cv::Mat points = img3d.reshape(1, n);

  if (!color.empty())
  {
    //将颜色图像重塑成与深度图像相同的形状
    cv::Mat color_reshaped = color.reshape(1, n);
    //将颜色图像数据的数据类型转换为双精度浮点型
    color_reshaped.convertTo(color_reshaped, CV_64FC1);

    if(color.channels() < 4)
    {
      //将颜色图像与深度图像连接在一起，形成一个点云矩阵
      cv::hconcat(points, color_reshaped, points);
    }
    else if (color.channels() == 4)
    {
      //取前三个通道（去掉 Alpha 通道），然后与深度图像连接在一起
      cv::hconcat(points, color_reshaped.colRange(0, 3), points);
    }
    else
    {
      throw(std::invalid_argument("Error converting depth map to point cloud: Color depth invalid."));
    }
  }

  if (!normals.empty())
  {
    //将法线图像重塑成与深度图像相同的形状
    cv::Mat normals_reshaped = normals.reshape(1, n);
    //将法线图像数据的数据类型转换为双精度浮点型
    normals_reshaped.convertTo(normals_reshaped, CV_64FC1);
    //将法线图像与之前的点云矩阵连接在一起
    cv::hconcat(points, normals_reshaped, points);
  }

  // Masking out undesired elements
  //将掩码图像重塑成与深度图像相同的形状
  cv::Mat mask_reshaped = mask.reshape(1, n);

  cv::Mat points_masked;
  points_masked.reserve(n);
  //将掩码为 255 的像素点对应的点云数据添加到一个新的点云矩阵中
  for (int r = 0; r < mask_reshaped.rows; ++r)
  {
    if (mask.at<uchar>(r) == 255)
      points_masked.push_back(points.row(r));
  }

  return cv::Mat(points_masked);

}

cv::Mat cvtToPointCloud(const CvGridMap &map,
                        const std::string &layer_elevation,
                        const std::string &layer_color,
                        const std::string &layer_normals,
                        const std::string &layer_mask)
{
  //确保图层存在
  assert(map.exists(layer_elevation));
  assert(!layer_color.empty() ? map.exists(layer_color) : true);
  assert(!layer_normals.empty() ? map.exists(layer_normals) : true);
  assert(!layer_mask.empty() ? map.exists(layer_mask) : true);
  //获取地图的尺寸，并计算图像中的总像素数量
  cv::Size2i size = map.size();
  size_t n = (size_t)size.width*size.height;
  //创建一个空的 cv::Mat 对象 points，准备用来存储点云数据
  cv::Mat points;
  points.reserve(n);

  // OPTIONAL
  //如果颜色图层存在，从地图中获取颜色图像数据
  cv::Mat color;
  if (map.exists(layer_color))
    color = map[layer_color];
    
    //如果法线图层存在，从地图中获取法线图像数据
  cv::Mat elevation_normal;
  if (map.exists(layer_normals))
    elevation_normal = map[layer_normals];
   
    //如果掩码图层存在，从地图中获取掩码图像数据
  cv::Mat mask;
  if (map.exists(layer_mask))
    mask = map[layer_mask];

  // Create one point per grid element
  //创建一个 img3d 图像，将高度图层中的数据填充到该图像中，生成一个 CV_64FC3 类型的图像
  cv::Mat img3d(size, CV_64FC3);
  for (int r = 0; r < size.height; ++r)
    for (int c = 0; c < size.width; ++c)
      img3d.at<cv::Point3d>(r, c) = map.atPosition3d(r, c, layer_elevation);
      
      //返回最终的点云矩阵
  return cvtToPointCloud(img3d, color, elevation_normal, mask);
}

std::vector<Face> cvtToMesh(const CvGridMap &map,
                            const std::string &layer_elevation,
                            const std::string &layer_color,
                            const std::vector<cv::Point2i> &vertex_ids)
{
  //确保图层不为空且存在于地图中
  assert(!layer_elevation.empty() && map.exists(layer_elevation));
  assert(!layer_color.empty() ? map.exists(layer_color) : true);

  // OPTIONAL
  //如果颜色图层不为空字符串，使用 map.exists 来确保颜色图层存在
  cv::Mat color;
  if (map.exists(layer_color))
  //从地图中获取颜色图像数据
    color = map[layer_color];

  // Create output vector of faces
  //创建一个空的输出向量 faces，用来存储生成的三角形面片
  size_t count = 0;
  std::vector<Face> faces(vertex_ids.size()/3);

//使用循环遍历传入的顶点标识向量 vertex_ids
  for (size_t i = 0; i < vertex_ids.size(); i+=3, ++count)
    for (size_t j = 0; j < 3; ++j)
    {
      //获取对应位置的三维坐标
      faces[count].vertices[j] = map.atPosition3d(vertex_ids[i+j].y, vertex_ids[i+j].x, layer_elevation);
      if (!color.empty())
        faces[count].color[j] = color.at<cv::Vec4b>(vertex_ids[i+j].y, vertex_ids[i+j].x);
      else
        faces[count].color[j] = cv::Vec4b(0, 0, 0, 255);
    }
  return faces;
}

} // namespace realm