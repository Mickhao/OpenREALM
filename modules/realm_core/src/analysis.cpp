

#include <opencv2/imgproc/imgproc_c.h>

#include <realm_core/analysis.h>
#include <realm_core/loguru.h>

using namespace realm;

//使用了 analysis 中的 convertToColorMapFromCVC1 函数 ：将单通道浮点型图像转换为RGB彩色映射的图像
cv::Mat analysis::convertToColorMapFromCVC1(const cv::Mat &img, const cv::Mat &mask, cv::ColormapTypes flag)
{
  //断言输入图像的类型：32位浮点数、64位浮点数或16位无符号整数（三者其一）。确保输入图像是灰度图像
  assert(img.type() == CV_32FC1 || img.type() == CV_64FC1 || img.type() == CV_16UC1);

  //定义两个'cv::Mat'对象：'map_norm'、'map_colored' 用于存储图像数据
  cv::Mat map_norm;       //用于存储归一化后的图像数据
  cv::Mat map_colored;    //用于存储颜色映射后的图像数据

  // Normalization can be processed with mask or without
  /*使用'cv::normalize'函数对输入图像'img'进行归一化，然后存储在'map_norm',像素值范围为[0,255],
  用来匹配CV_8UC1d的范围。
  之后进行判断，如果'mask'为空，则不考虑掩膜来进行归一化；如果不为空，则需要考虑掩膜
  掩膜中非零区域的像素值会影响归一化
  */
  if (mask.empty())
    cv::normalize(img, map_norm, 0, 255, cv::NORM_MINMAX, CV_8UC1);
  else
    cv::normalize(img, map_norm, 0, 255, cv::NORM_MINMAX, CV_8UC1, mask);

  // Afterwards apply color map
  //使用'cv::applyColorMap'函数 ：将图像'map_norm'应用颜色映射，并将结果存储到'map_colored'中
  cv::applyColorMap(map_norm, map_colored, flag);

  // Set invalid pixels black
  //循环遍历'map_colored'。如果掩膜中对应的像素值为0，表示为无效像素，将其设置为黑色cv::Vec3b(0, 0, 0)
  for (int r = 0; r < map_colored.rows; ++r)
    for (int c = 0; c < map_colored.cols; ++c)
      if (mask.at<uchar>(r, c) == 0)
        map_colored.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 0, 0);

  return map_colored;  //最终返回图像并存储到'map_colored'
}

//使用了 analysis 中的 convertToColorMapFromCVC3 函数 ：将三通道浮点型图像转换为RGB彩色映射的图像
cv::Mat analysis::convertToColorMapFromCVC3(const cv::Mat &img, const cv::Mat &mask)
{
  //断言输入图像的类型：32位浮点数、64位浮点数或16位无符号整数（三者其一）。确保输入图像是灰度图像
  assert(img.type() == CV_32FC3 || img.type() == CV_64FC3 || img.type() == CV_16UC3);
  //定义两个对象'map_32fc3'、'map_8uc3'
  cv::Mat map_32fc3, map_8uc3;
  //使用'cv::cvtColor'函数：将输入的图像从XYZ色彩空间转换为BGR色彩空间，之后存储到'map_32fc3'
  cv::cvtColor(img, map_32fc3, CV_XYZ2BGR);
  /*对'map_32fc3'中的像素值乘以255，以便其将图像的像素值范围映射到0-255
  接着使用'onvertTo'函数将其转换为8位无符号整数型三通道图像，存储到'map_8uc3'*/
  map_32fc3 *= 255;
  map_32fc3.convertTo(map_8uc3, CV_8UC3);

  // Set invalid pixels black
  //循环遍历'map_colored'。如果掩膜中对应的像素值为0，表示为无效像素，将其设置为黑色cv::Vec3b(0, 0, 0)
  for (int r = 0; r < map_8uc3.rows; ++r)
    for (int c = 0; c < map_8uc3.cols; ++c)
      if (mask.at<uchar>(r, c) == 0)
        map_8uc3.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 0, 0);

  return map_8uc3;//最终返回图像并存储到'map_8uc3'
}

//图像处理