

#include <realm_core/depthmap.h>

/*处理深度图像，包括存储图像数据、相机模型以及一些计算深度参数*/
using namespace realm;

Depthmap::Depthmap(const cv::Mat &data, const camera::Pinhole &cam)
 : m_data(data),//将传入的深度图像数据存储
 //使用传入的相机模型创建一个共享指针，并将其存储在 m_cam 成员变量中
   m_cam(std::make_shared<camera::Pinhole>(cam))

{
  //检查存储的深度图像数据的类型是否为 CV_32F
  if (m_data.type() != CV_32F)
    throw(std::invalid_argument("Error creating depth map: Matrix type not CV_32F"));
  //检查深度图像数据的尺寸是否与相机模型的尺寸匹配
  if (data.rows != m_cam->height() || data.cols != m_cam->width())
    throw(std::invalid_argument("Error creating depth map: Dimension mismatch! Camera size does not match data."));

  updateDepthParameters();
}

//用于获取与深度图像相关联的相机模型的共享指针
camera::Pinhole::ConstPtr Depthmap::getCamera() const
{
  return m_cam;
}

//用于获取深度图像中的最小深度值
double Depthmap::getMinDepth() const
{
  return m_min_depth;
}

//用于获取深度图像中的最大深度值
double Depthmap::getMaxDepth() const
{
  return m_max_depth;
}

//用于获取深度图像中的中值深度值
double Depthmap::getMedianDepth() const
{
  return m_med_depth;
}

cv::Mat& Depthmap::data()
{
  return m_data;
}

//用于计算深度图像的一些参数
void Depthmap::updateDepthParameters()
{
  // First grab min and max values
  cv::Mat mask = (m_data > 0);//创建一个布尔类型的掩码
  //使用掩码计算深度图像中的最小和最大深度值
  cv::minMaxLoc(m_data, &m_min_depth, &m_max_depth, nullptr, nullptr, mask);
  
  /*深度图像数据存储在一个名为 array 的向量中
  遍历深度图像的数据,并将数据插入到向量中
  */
  std::vector<float> array;
  if (m_data.isContinuous()) {
    array.assign((float*)m_data.data, (float*)m_data.data + m_data.total() * m_data.channels());
  } else {
    for (int i = 0; i < m_data.rows; ++i) {
      array.insert(array.end(), m_data.ptr<float>(i), m_data.ptr<float>(i) + m_data.cols * m_data.channels());
    }
  }

  // Invalid depth values are set to -1.0
  //创建一个名为 array_masked 的向量，其中只包含大于0的深度值
  std::vector<float> array_masked;
  std::copy_if(array.begin(), array.end(), back_inserter(array_masked),[](float n){ return  n > 0.0;});

  // Find median by sorting the array and get middle value
  // 对 array 向量进行排序，以便计算中值深度
  std::sort(array.begin(), array.end());
  //将排序后的 array 向量的中间值存储
  m_med_depth = array[array.size() / 2];
}