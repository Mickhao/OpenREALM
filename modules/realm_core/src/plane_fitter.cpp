

#include <eigen3/Eigen/Eigen>
#include <opencv2/core.hpp>

#include <realm_core/plane_fitter.h>

namespace realm
{

//估计平面
PlaneFitter::Plane PlaneFitter::estimate(const std::vector<PlaneFitter::Point> &points)
{
  size_t n = points.size();
  assert(n >= 3);

  if (n == 3)
  {
    // Exact solution with three points possible
    //调用 computeExactPlaneCentroid(points) 方法计算中心点
    PlaneFitter::Point centroid = computeExactPlaneCentroid(points);
    //调用 computeExactPlaneNormal(points) 方法计算法向量
    PlaneFitter::Normal vn = computeExactPlaneNormal(points);
    return PlaneFitter::Plane{centroid, vn};
  }
  else
  {
    //采用最佳拟合方法
    // Best fit / SVD decomposition with more than three points (over determined)
    //使用 Eigen 库创建一个矩阵 points_eigen，其中存储了输入点的坐标（x、y、z）
    Eigen::Matrix<Eigen::Vector3d::Scalar, Eigen::Dynamic, Eigen::Dynamic> points_eigen(3, n);

    for (size_t i = 0; i < n; ++i)
      points_eigen.col(i) << points[i].x, points[i].y, points[i].z;
      //计算所有点的坐标的平均值，以获得中心点坐标
    Eigen::Vector3d centroid(points_eigen.row(0).mean(), points_eigen.row(1).mean(), points_eigen.row(2).mean());
    //将点集的坐标减去中心点，以便进行奇异值分解
    points_eigen.row(0).array() -= centroid(0);
    points_eigen.row(1).array() -= centroid(1);
    points_eigen.row(2).array() -= centroid(2);
    //使用奇异值分解（SVD）进行计算，得到一个近似的法向量
    auto svd = points_eigen.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Vector3d vn = svd.matrixU().rightCols<1>();

    return PlaneFitter::Plane{centroid(0), centroid(1), centroid(2), vn(0), vn(1), vn(2)};
  }
}

//用于计算三个点确定的平面的法线向量
PlaneFitter::Normal PlaneFitter::computeExactPlaneNormal(const std::vector<PlaneFitter::Point> &points)
{
  assert(points.size() == 3);
  //计算v1
  PlaneFitter::Vector v1{
      (points[1].x - points[0].x),
      (points[1].y - points[0].y),
      (points[1].z - points[0].z)};
  //计算v2
  PlaneFitter::Vector v2{
      (points[2].x - points[0].x),
      (points[2].y - points[0].y),
      (points[2].z - points[0].z)};
  //计算v3
  PlaneFitter::Vector v3{
          v1.y*v2.z - v1.z*v2.y,
          v1.z*v2.x - v1.x*v2.z,
          v1.x*v2.y - v1.y*v2.x};
  //通过计算 v3 的长度来标准化法线向量
  double length = sqrt(v3.x*v3.x + v3.y*v3.y + v3.z*v3.z);

  return PlaneFitter::Normal{
     v3.x/length,
     v3.y/length,
     v3.z/length};
}

//用于计算三个点的中心点坐标
PlaneFitter::Point PlaneFitter::computeExactPlaneCentroid(const std::vector<PlaneFitter::Point> &points)
{
  assert(points.size() == 3);
  return PlaneFitter::Point{
      (points[0].x + points[1].x + points[2].x) / 3.0,
      (points[0].y + points[1].y + points[2].y) / 3.0,
      (points[0].z + points[1].z + points[2].z) / 3.0};
}

} // namespace realm

//实现了拟合平面的功能，可以根据不同数量的点来估计平面，并计算平面的法向量和中心点