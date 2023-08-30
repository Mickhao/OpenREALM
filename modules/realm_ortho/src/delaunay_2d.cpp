

#include <realm_ortho/delaunay_2d.h>

using namespace realm;
//使用 CGAL 库来构建二维的 Delaunay 三角剖分，以生成给定网格上的顶点

Delaunay2D::Delaunay2D()
{

}

//构建二维的 Delaunay 三角剖分函数
std::vector<cv::Point2i> Delaunay2D::buildMesh(const CvGridMap &grid, const std::string &mask)
{
  //CGAL boilerplate
  //定义一个 CGAL 内核
  typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
  //We define a vertex_base with info. The "info" (size_t) allow us to keep track of the original point index.
  //一个定制的顶点基类,在顶点上存储了额外的信息
  typedef CGAL::Triangulation_vertex_base_with_info_2<cv::Point2i, Kernel> VertexBase;
  //一个定制的面基类，用于在三角剖分中存储面的信息
  typedef CGAL::Constrained_triangulation_face_base_2<Kernel> FaceBase;
  //使用定制的顶点和面基类
  typedef CGAL::Triangulation_data_structure_2<VertexBase, FaceBase> TriangulationDataStruct;
  //使用上述定义的内核和数据结构
  typedef CGAL::Delaunay_triangulation_2<Kernel, TriangulationDataStruct> DelaunayTriangulation;
  //表示三角剖分中点的类型
  typedef DelaunayTriangulation::Point CGALPoint;

  cv::Mat valid;//存储有效性信息
  if (mask.empty())
  //将 valid 初始化为与网格大小相同的全白图像
    valid = cv::Mat::ones(grid.size(), CV_8UC1)*255;
  else
  //将 valid 初始化为根据 mask 从 grid 中提取的图像
    valid = grid[mask];

  std::vector< std::pair<CGALPoint, cv::Point2i>> pts;
  //预分配足够的空间来容纳点对，以避免不必要的内存重新分配
  pts.reserve(static_cast<size_t>(valid.rows * valid.cols));

  // Create and add vertices
  //嵌套循环遍历 valid 图像中的每个像素
  for (uint32_t r = 0; r < valid.rows; ++r)
    for (uint32_t c = 0; c < valid.cols; ++c)
      if (valid.at<uchar>(r, c) > 0)
      {
        //从 grid 中获取相应位置的二维坐标，并将其添加到 pts 中
        cv::Point2d pt = grid.atPosition2d(r, c);
        pts.push_back(std::make_pair(CGALPoint(pt.x, pt.y), cv::Point2i(c, r)));
      }

  // The DT is built
  //创建一个 Delaunay 三角剖分对象 dt
  DelaunayTriangulation dt(pts.begin(), pts.end());
  
  //声明一个 std::vector 来存储顶点 ID
  std::vector<cv::Point2i> vertex_ids;
  vertex_ids.reserve(dt.number_of_faces()*3);
  //遍历三角剖分中的面，并获取每个面的顶点 ID，将这些 ID 添加到 vertex_ids 中
  if (vertex_ids.capacity() > 0)
  {
    for (auto it = dt.faces_begin(); it != dt.faces_end(); ++it)
      for (int i = 0; i < 3; ++i)
        vertex_ids.push_back(it->vertex(i)->info());
  }

  return vertex_ids;
}