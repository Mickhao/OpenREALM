

#include <realm_io/realm_export.h>

/*保存不同类型的数据到文件中*/

namespace realm
{
namespace io
{

void saveTimestamp(uint64_t timestamp,
                   uint32_t frame_id,
                   const std::string &filename)
{
// Open file and write data
  std::ofstream file;
  //以“追加”模式打开指定的文件
  file.open(filename.c_str(), std::ios::app);
  //这将frame_id和timestamp写入文件
  file << frame_id << " " << timestamp << std::endl;
}

void saveTrajectory(uint64_t timestamp,
                    const cv::Mat &pose,
                    const std::string &filepath)
{
  // Grab container data
  //将位姿矩阵的第四列提取出来,表示平移向量
  cv::Mat t = pose.col(3);
  //将位姿矩阵的前三行和前三列提取出来，表示旋转矩阵
  cv::Mat R = pose.rowRange(0, 3).colRange(0, 3);
  //将OpenCV的旋转矩阵数据复制到Eigen矩阵中
  Eigen::Matrix3d R_eigen;
  R_eigen << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
             R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
             R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);
  Eigen::Quaterniond q(R_eigen);

  // Open file and write data
  std::ofstream file;
  file.open(filepath.c_str(), std::ios::app);
  //将时间戳、平移向量和四元数写入文件中
  saveTrajectoryTUM(&file, timestamp, t.at<double>(0), t.at<double>(1), t.at<double>(2), q.x(), q.y(), q.z(), q.w());
  file.close();
}

void saveTrajectoryTUM(std::ofstream *file,
                       uint64_t timestamp,
                       double x,
                       double y,
                       double z,
                       double qx,
                       double qy,
                       double qz,
                       double qw)
{
  // TUM file format
  //设置了文件流的输出精度为10位小数
  (*file).precision(10);
  //将给定的数据格式化后写入到文件流中
  (*file) << timestamp << " " << x << " " << y << " " << z << " " << qx << " " << qy << " " << qz << " " << qw << std::endl;
}

void saveCvGridMap(const CvGridMap &map, const std::string &filepath)
{
  //从文件路径中提取后缀名
  std::string suffix = filepath.substr(filepath.size()-9, 9);
  //检查后缀名是否为".grid.bin"
  if(suffix != ".grid.bin")
    throw(std::invalid_argument("Error saving CvGridMap to binary. Suffix not supported!"));
    //以二进制写入模式打开指定的文件
  FILE* file = fopen((filepath).c_str(), "wb");

  // Writing the ROI
  //获取CvGridMap对象的ROI
  cv::Rect2d roi = map.roi();
  double roi_raw[4] = { roi.x, roi.y, roi.width, roi.height };
  fwrite(roi_raw, 4, sizeof(double), file);

  // Writing the resolution
  //获取地图的分辨率
  double resolution = map.resolution();
  fwrite(&resolution, 1, sizeof(double), file);

  //获取地图中所有图层的名称
  std::vector<std::string> layer_names = map.getAllLayerNames();

  // Writing nrof layers
  //获取图层数量并将其写入
  int nrof_layers = layer_names.size();
  fwrite(&nrof_layers, 1, sizeof(int), file);

  for (const auto &layer_name : layer_names)
  {
    //获取指定图层名称的图层数据
    CvGridMap::Layer layer = map.getLayer(layer_name);

    // Writing layer name
    //写入图层名称信息
    int length = layer_name.length();
    fwrite(&length, 1, sizeof(int), file);
    fwrite(layer_name.c_str(), length, sizeof(char), file);

    // Writing layer interpolation flag
    //写入图层插值标志
    int interpolation = layer.interpolation;
    fwrite(&interpolation, 1, sizeof(int), file);

    // Writing layer data
    //写入图层数据大小信息
    int elem_size_in_bytes = (int)layer.data.elemSize();
    int elem_type          = (int)layer.data.type();

    //构建表示数据大小和类型的数组，并写入文件
    int size[4] = {layer.data.cols, layer.data.rows, elem_size_in_bytes, elem_type};
    fwrite(size, 4, sizeof(int), file);

    // Operating rowise, so even non-continuous matrices are properly written to binary
    //逐行将图层数据写入文件
    for (int r = 0; r < layer.data.rows; ++r)
      fwrite(layer.data.ptr<void>(r), layer.data.cols, elem_size_in_bytes, file);
  }

  fclose(file);
}

} // namespace io
} // namespace realm