

#include <realm_io/cv_export.h>

#include <fstream>

/*对图像和数据的保存*/

namespace realm
{
namespace io
{
//用于保存立体图像对的函数
void saveStereoPair(const Frame::Ptr &frame_left, const Frame::Ptr &frame_right, const std::string &path)
{
  //获取左右相机帧的内部相机矩阵 K（相机内参数）和畸变系数 D
  // inner calib
  cv::Mat K_l = frame_left->getResizedCalibration();
  cv::Mat K_r = frame_right->getResizedCalibration();
  // distortion
  cv::Mat D_l = frame_left->getCamera()->distCoeffs();
  cv::Mat D_r = frame_right->getCamera()->distCoeffs();
  // exterior calib
  //获取左相机帧的世界坐标到相机坐标的外部相机矩阵 T_l_w2c
  //获取右相机帧的相机坐标到世界坐标的外部相机矩阵 T_r_c2w
  cv::Mat T_l_w2c = frame_left->getCamera()->Tw2c();
  cv::Mat T_r_c2w = frame_right->getCamera()->Tc2w();
  cv::Mat T_lr = T_l_w2c*T_r_c2w;
  // Now calculate transformation between the cameras
  // Formula: R = R_1^T * R_2
  //          t = R_1^T * t_2 - R_1^T*t_1
  //计算左相机坐标到右相机坐标的变换矩阵 T_lr
  //表示从左相机坐标系到右相机坐标系的变换
  cv::Mat R = T_lr.rowRange(0, 3).colRange(0, 3);
  cv::Mat t = T_lr.rowRange(0, 3).col(3);
  uint32_t width = frame_left->getResizedImageWidth();
  uint32_t height = frame_left->getResizedImageHeight();
  cv::Size img_size(width, height);
  // Rectify images by stereo parameters
  //计算校正后的双目摄像机参数 R1、R2、P1、P2 和视差矩阵 Q
  cv::Mat R1, R2;
  cv::Mat P1, P2;
  cv::Mat Q;
  cv::stereoRectify(K_l, D_l, K_r, D_r, img_size, R, t, R1, R2, P1, P2, Q);
  // create names
  //构造输出文件名
  char filename_img_left[1000];
  char filename_img_right[1000];
  char filename_intrinsic[1000];
  char filename_extrinsic[1000];
  sprintf(filename_img_left, (path + std::string("img_left_%06i.png")).c_str(), frame_right->getFrameId());
  sprintf(filename_img_right, (path + std::string("img_right_%06i.png")).c_str(), frame_right->getFrameId());
  sprintf(filename_intrinsic, (path + std::string("intrinsic_%06i.yml")).c_str(), frame_right->getFrameId());
  sprintf(filename_extrinsic, (path + std::string("extrinsic_%06i.yml")).c_str(), frame_right->getFrameId());
  // save image pair distorted
  //保存经过畸变校正的立体图像
  if (cv::imwrite(std::string(filename_img_left), frame_left->getResizedImageUndistorted())
      && cv::imwrite(std::string(filename_img_right), frame_right->getResizedImageUndistorted()))
  {
    std::cout << "Saved stereo images to:\n" << filename_img_left << "\n" << filename_img_right << std::endl;
  }

  // save 1) intrisics and 2) extrinsics to folder
  
  //使用 cv::FileStorage 保存内部相机参数和外部相机矩阵到 YAML 文件
  //分别保存在 filename_intrinsic 和 filename_extrinsic 中
  cv::FileStorage fs(std::string(filename_intrinsic), cv::FileStorage::WRITE);
  if(fs.isOpened())
  {
    fs << "M1" << K_l << "D1" << D_l << "M2" << K_r << "D2" << D_r;
    std::cout << "Saved stereo intrinsics to:\n" << filename_intrinsic << std::endl;
    fs.release();
  }
  else
  {
    std::cout << "Error: can not save the intrinsic parameters to file:\n" << filename_intrinsic << std::endl;
  }
  fs.open(std::string(filename_extrinsic), cv::FileStorage::WRITE);
  if(fs.isOpened())
  {
    fs << "R" << R << "T" << t << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
    std::cout << "Saved stereo extrinsics to:\n" << filename_extrinsic << std::endl;
    fs.release();
  }
  else
  {
    std::cout << "Error: can not save the extrinsic parameters to file:\n" << filename_extrinsic << std::endl;
  }
}

//保存图像到文件
void saveImage(const cv::Mat &img, const std::string &filename)
{
  //获取文件名的后缀，以确定文件格式
  std::string suffix = filename.substr(filename.size()-3, 3);
  //如果后缀是 "png" 或 "jpg"，则使用 OpenCV 的 cv::imwrite 函数将图像保存为 PNG 或 JPEG 格式
  if(suffix == "png" || suffix == "jpg")
    cv::imwrite(filename, img);
  //如果后缀是 "bin"，则调用 saveImageToBinary 函数将图像保存为二进制文件
  else if (suffix == "bin")
    saveImageToBinary(img, filename);
  else
    throw(std::invalid_argument("Error writing image: Unknown suffix"));
}

//将一个OpenCV图像（cv::Mat）保存为二进制文件
void saveImageToBinary(const cv::Mat &data, const std::string &filepath)
{
  //获取图像中每个元素的大小（以字节为单位）和类型
  int elem_size_in_bytes = (int)data.elemSize();
  //获取图像的类型（数据类型）
  int elem_type          = (int)data.type();
  //打开文件以进行写入二进制模式，使用给定的文件路径
  FILE* file = fopen((filepath).c_str(), "wb");
  //创建一个大小为4的整数数组，其中包含图像的列数、行数、每个元素的字节数和元素类型
  int size[4] = {data.cols, data.rows, elem_size_in_bytes, elem_type};
  //使用 fwrite 将上述整数数组写入文件
  fwrite(size, 4, sizeof(int), file);

  // Operating rowise, so even non-continuous matrices are properly written to binary
  //通过遍历图像的每一行，使用 data.ptr<void>(r) 获取当前行的指针，并使用 fwrite 将行数据写入文件
  for (int r = 0; r < data.rows; ++r)
    fwrite(data.ptr<void>(r), data.cols, elem_size_in_bytes, file);

  fclose(file);
}

//用于保存深度图像为图像文件
void saveDepthMap(const Depthmap::Ptr &depthmap, const std::string &filename, uint32_t id)
{
  //从深度图像中获取原始深度数据
  cv::Mat data = depthmap->data();
  //创建一个掩码（mask），用于标记有效深度值（大于0.0）的区域
  cv::Mat mask = (data > 0.0);

  //据深度数据的类型（CV_32FC1 或 CV_64FC1）进行归一化操作
  cv::Mat img_normalized;
  if (data.type() == CV_32FC1 || data.type() == CV_64FC1)
    cv::normalize(data, img_normalized, 0, 65535, cv::NormTypes::NORM_MINMAX, CV_16UC1, mask);
  else
    throw(std::invalid_argument("Error saving depth map: Mat type not supported!"));

    //创建文件名缓冲区，将图像的文件名格式化为指定的格式
  char buffer[1000];
  sprintf(buffer, filename.c_str(), id);
  
  //计算深度数据的最小值和最大值，并将这些信息写入与图像文件相同名称的元数据文件中
  double min_depth, max_depth;
  cv::minMaxLoc(data, &min_depth, &max_depth, nullptr, nullptr, mask);

  std::string full_filename = std::string(buffer);
  std::ofstream metafile(full_filename.substr(0, full_filename.size()-3) + "txt");
  metafile << "Scaling\nMin: " << min_depth << "\nMax: " << max_depth;

  //获取深度相机的姿态信息，并将其写入元数据文件
  cv::Mat pose = depthmap->getCamera()->pose();
  metafile << "\nPose:";
  metafile << "\n" << pose.at<double>(0, 0) << " " << pose.at<double>(0, 1) << " " << pose.at<double>(0, 2) << " " << pose.at<double>(0, 3);
  metafile << "\n" << pose.at<double>(1, 0) << " " << pose.at<double>(1, 1) << " " << pose.at<double>(1, 2) << " " << pose.at<double>(1, 3);
  metafile << "\n" << pose.at<double>(2, 0) << " " << pose.at<double>(2, 1) << " " << pose.at<double>(2, 2) << " " << pose.at<double>(2, 3);
  metafile.close();
  //使用OpenCV的 imwrite 函数将归一化后的深度图像保存为图像文件
  cv::imwrite(full_filename, img_normalized);
}

//用于将彩色图像应用颜色映射并保存为图像文件
void saveImageColorMap(const cv::Mat &img,
                       const cv::Mat &mask,
                       const std::string &directory,
                       const std::string &name,
                       ColormapType flag)
{
  std::string filename = (directory + "/" + name + ".png");
  saveImageColorMap(img, mask, filename, flag);
}

void saveImageColorMap(const cv::Mat &img,
                       const cv::Mat &mask,
                       const std::string &directory,
                       const std::string &name,
                       uint32_t frame_id,
                       ColormapType flag)
{
  std::string filename = io::createFilename(directory + "/" + name + "_", frame_id, ".png");
  saveImageColorMap(img, mask, filename, flag);
}

void saveImageColorMap(const cv::Mat &img,
                       float range_min,
                       float range_max,
                       const std::string &directory,
                       const std::string &name,
                       uint32_t frame_id,
                       ColormapType flag)
{
  std::string filename = io::createFilename(directory + "/" + name + "_", frame_id, ".png");
  cv::Mat mask;
  cv::inRange(img, range_min, range_max, mask);
  saveImageColorMap(img, mask, filename, flag);
}

void saveImageColorMap(const cv::Mat &img, const cv::Mat &mask, const std::string &filename, ColormapType flag)
{
  cv::Mat map_colored;
  switch(flag)
  {
    case ColormapType::DEPTH:
      map_colored = analysis::convertToColorMapFromCVC1(img, mask, cv::COLORMAP_JET);
      break;
    case ColormapType::ELEVATION:
      map_colored = analysis::convertToColorMapFromCVC1(img, mask, cv::COLORMAP_JET);
      break;
    case ColormapType::NORMALS:
      map_colored = analysis::convertToColorMapFromCVC3(img, mask);
      break;
  }
  cv::imwrite(filename, map_colored);
}

//用于保存 CvGridMap 的某个图层
void saveCvGridMapLayer(const CvGridMap &map, int zone, char band, const std::string &layer_name, const std::string &filename)
{
  cv::Mat data = map[layer_name];
  //将图层数据保存为图像文件
  saveImage(data, filename);
  //保存图层的元数据到 YAML 文件
  saveCvGridMapMeta(map, zone, band, filename.substr(0, filename.size()-3) + "yaml");
}

//用于保存 CvGridMap 的元数据到一个 YAML 文件
void saveCvGridMapMeta(const CvGridMap &map, int zone, char band, const std::string &filename)
{
  cv::FileStorage metafile(filename, cv::FileStorage::WRITE);
  //将 zone 和 band 作为键值对写入 YAML 文件，用于标识地理区域和波段信息
  metafile << "zone" << zone;
  metafile << "band" << band;
  //将 map.roi() 写入 YAML 文件
  metafile << "roi" << map.roi();
  //将 map.resolution() 写入 YAML 文件
  metafile << "resolution" << map.resolution();
  //释放文件对象
  metafile.release();
}

} // namespace io
} // namespace realm