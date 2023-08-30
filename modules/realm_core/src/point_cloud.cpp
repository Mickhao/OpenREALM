#include <realm_core/point_cloud.h>

/*实现了一个点云类 PointCloud，用于管理点云数据*/
using namespace realm;

//创建一个空的点云对象
PointCloud::PointCloud()
{
}

//创建一个点云对象，并初始化点的标识符和数据
PointCloud::PointCloud(const std::vector<uint32_t> &point_ids, const cv::Mat &data)
 :  m_point_ids(point_ids),//点的标识符向量
    m_data(data) //点的数据矩阵 
{
  //检查数据矩阵的行数是否与点的标识符数量相匹配
  if (data.rows != m_point_ids.size())
    throw(std::invalid_argument("Error creating sparse cloud: Data - ID mismatch!"));
}

//判断点云是否为空
bool PointCloud::empty()
{
  if (m_data.empty() || m_data.rows == 0)
    return true;
  return false;
}

//返回点云数据矩阵的引用，允许外部修改数据
cv::Mat& PointCloud::data()
{
  return m_data;
}

//回点云中的点标识符向量
std::vector<uint32_t> PointCloud::getPointIds()
{
  return m_point_ids;
}

//返回点云中数据矩阵的行数
int PointCloud::size()
{
  return m_data.rows;
}