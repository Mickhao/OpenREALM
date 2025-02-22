

#include <realm_vslam_base/dummy_referencer.h>

#include <realm_core/loguru.h>

/*虚拟的地理参考系统
用于处理相机坐标系与世界坐标系之间的变换关系*/
using namespace realm;

DummyReferencer::DummyReferencer(const cv::Mat &T_c2g)
  : m_transformation_c2g(T_c2g)
{
}

//初始化虚拟的地理参考系统
void DummyReferencer::init(const std::vector<Frame::Ptr> &frames)
{
  LOG_F(WARNING, "This is a dummy georeferenciation. You have to manually provide the transformation from camera to world frame. "
                 "Not call to 'init()' possible!");
}

//取从相机坐标系到世界坐标系的变换矩阵
cv::Mat DummyReferencer::getTransformation()
{
  std::unique_lock<std::mutex> lock(m_mutex_t_c2g);
  return m_transformation_c2g;
}

void DummyReferencer::update(const Frame::Ptr &frame)
{
}

bool DummyReferencer::isBuisy()
{
  return false;
}

bool DummyReferencer::isInitialized()
{
  return true;
}