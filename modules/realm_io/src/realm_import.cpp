

#include <realm_io/realm_import.h>

#include <eigen3/Eigen/Eigen>

using namespace realm;

//从YAML文件中加载相机参数
camera::Pinhole io::loadCameraFromYaml(const std::string &filepath, double* fps)
{
  // Identify camera model
  //从指定的YAML文件中加载相机设置
  CameraSettings::Ptr settings = CameraSettingsFactory::load(filepath);

  // Load camera informations depending on model
  //判断相机模型是否为 pinhole 类型
  if ((*settings)["type"].toString() == "pinhole")
  {
    // Grab the fps, as it is note saved in the camera container
    if (fps != nullptr)
    //加载的帧率存储到 fps 指向的位置
      *fps = (*settings)["fps"].toDouble();

    // Create pinhole model
    //使用加载的参数创建 camera::Pinhole 对象,设置畸变参数
    camera::Pinhole cam((*settings)["fx"].toDouble(), (*settings)["fy"].toDouble(),
                        (*settings)["cx"].toDouble(), (*settings)["cy"].toDouble(),
              (uint32_t)(*settings)["width"].toInt(), (uint32_t)(*settings)["height"].toInt());
    cam.setDistortionMap((*settings)["k1"].toDouble(), (*settings)["k2"].toDouble(),
                         (*settings)["p1"].toDouble(), (*settings)["p2"].toDouble(), (*settings)["k3"].toDouble());
    return cam;
  }
  else
  {
      throw std::runtime_error("Unable to load camera settings from yaml file at: " + filepath);
  }

}

cv::Mat io::loadGeoreferenceFromYaml(const std::string &filepath)
{
  cv::Mat georeference;
  //打开指定路径的YAML文件进行读取操作
  cv::FileStorage fs(filepath, cv::FileStorage::READ);
  //从文件中读取键为 "transformation_w2g" 的值
  fs["transformation_w2g"] >> georeference;
  //释放文件
  fs.release();

  return georeference;
}

std::unordered_map<uint64_t, cv::Mat> io::loadTrajectoryFromTxtTUM(const std::string &directory,
                                                                   const std::string &filename)
{
  //将目录路径和文件名组合成完整的文件路径
  return io::loadTrajectoryFromTxtTUM(directory + "/" + filename);
};
//从文本文件加载TUM格式的轨迹数据
std::unordered_map<uint64_t, cv::Mat> io::loadTrajectoryFromTxtTUM(const std::string &filepath)
{
  // Prepare result
  //存储加载的轨迹数据
  std::unordered_map<uint64_t, cv::Mat> result;

  // Open file
  //打开要加载的文本文件并检查是否成功打开
  std::ifstream file(filepath);
  if (!file.is_open())
    throw(std::runtime_error("Error loading trajectory file from '" + filepath + "': Could not open file!"));

  // Iterating through every line
  //使用循环逐行读取文件内容并处理轨迹数据
  std::string str;
  while (std::getline(file, str))
  {
    // Tokenize input line
    std::vector<std::string> tokens = io::split(str.c_str(), ' ');
    if (tokens.size() < 7)//检查标记的数量是否小于7
      throw(std::runtime_error("Error loading trajectory file from '\" + (directory+filename) + \"': Not enough arguments in line!"));

    // Convert all tokens to values
    //将标记转换为相应的值
    uint64_t timestamp = std::stoul(tokens[0]);
    double x = std::stod(tokens[1]);
    double y = std::stod(tokens[2]);
    double z = std::stod(tokens[3]);
    double qx = std::stod(tokens[4]);
    double qy = std::stod(tokens[5]);
    double qz = std::stod(tokens[6]);
    double qw = std::stod(tokens[7]);

    // Convert Quaternions to Rotation matrix
    //使用四元数信息创建旋转矩阵
    Eigen::Quaterniond quat(qw, qx, qy, qz);
    Eigen::Matrix3d R_eigen = quat.toRotationMatrix();

    // Pose as 3x4 matrix
    //创建姿态矩阵
    cv::Mat pose = (cv::Mat_<double>(3, 4) << R_eigen(0, 0), R_eigen(0, 1), R_eigen(0, 2), x,
                                              R_eigen(1, 0), R_eigen(1, 1), R_eigen(1, 2), y,
                                              R_eigen(2, 0), R_eigen(2, 1), R_eigen(2, 2), z);
    result[timestamp] = pose;
  }
  return result;
};

cv::Mat io::loadSurfacePointsFromTxt(const std::string &filepath)
{
  // Prepare result
  //存储加载的表面点数据
  cv::Mat points;

  // Open file
  //打开要加载的文本文件并检查是否成功打开
  std::ifstream file(filepath);
  if (!file.is_open())
    throw(std::runtime_error("Error loading surface point file from '" + filepath + "': Could not open file!"));

  // Iterating through every line
  //使用循环逐行读取文件内容并处理表面点数据
  std::string str;
  while (std::getline(file, str))
  {
    // Tokenize input line
    std::vector<std::string> tokens = io::split(str.c_str(), ' ');
    if (tokens.size() < 2)
      throw(std::runtime_error("Error loading surface point file from '" + filepath + "': Not enough arguments in line!"));

    // Convert all tokens to values
    //将标记转换为坐标值
    double x = std::stod(tokens[0]);
    double y = std::stod(tokens[1]);
    double z = std::stod(tokens[2]);

    // Point as 1x3 mat
    //创建一个1x3的 cv::Mat
    cv::Mat pt = (cv::Mat_<double>(1, 3) << x, y, z);
    //单个点的 cv::Mat 添加到存储点数据的 cv::Mat 中
    points.push_back(pt);
  }
  return points;
}

//从二进制文件加载 CvGridMap 格式的地图数据
CvGridMap::Ptr io::loadCvGridMap(const std::string &filepath)
{
  //检查文件是否存在
  if (!io::fileExists(filepath))
    throw(std::invalid_argument("Error loading image: File does not exist!"));

    //获取文件路径的后缀
  std::string suffix = filepath.substr(filepath.size()-9, 9);

  //检查是否为 ".grid.bin"
  if(suffix != ".grid.bin")
    throw(std::invalid_argument("Error loading CvGridMap: Unknown suffix"));
    //打开要加载的二进制文件
  FILE* file = fopen(filepath.c_str(), "rb");

  size_t elements_read;

  double x, y, width, height;
  double resolution;
  //读取地图数据
  elements_read = fread(&x, sizeof(double), 1, file);
  elements_read = fread(&y, sizeof(double), 1, file);
  elements_read = fread(&width, sizeof(double), 1, file);
  elements_read = fread(&height, sizeof(double), 1, file);
  elements_read = fread(&resolution, sizeof(double), 1, file);

  int nrof_layers;

  elements_read = fread(&nrof_layers, sizeof(int), 1, file);

  //创建一个指向 CvGridMap 类型的智能指针，同时传递地图的矩形区域和分辨率
  auto map = std::make_shared<CvGridMap>(cv::Rect2d(x, y, width, height), resolution);
  
  //循环遍历每个图层，并读取图层的信息并添加到地图中
  for (int i = 0; i < nrof_layers; ++i)
  {
    int length;
    elements_read = fread(&length, sizeof(int), 1, file);

    char layer_name[length];
    elements_read = fread(&layer_name, sizeof(char), length, file);

    int interpolation;
    elements_read = fread(&interpolation, sizeof(int), 1, file);

    int header[4];
    elements_read = fread(header, sizeof(int), 4, file);

    if (elements_read != 4)
      throw(std::runtime_error("Error reading binary: Elements read do not match matrix dimension!"));

    int cols               = header[0];
    int rows               = header[1];
    int elem_size_in_bytes = header[2];
    int elem_type          = header[3];
    //存储图层数据
    cv::Mat data = cv::Mat::ones(rows, cols, elem_type);

    elements_read = fread(data.data, elem_size_in_bytes, (size_t)(cols * rows), file);

    if (elements_read != (size_t)(cols * rows))
      throw(std::runtime_error("Error reading binary: Elements read do not match matrix dimension!"));

    map->add(std::string(layer_name), data, interpolation);
  }
  //关闭已打开的文件
  fclose(file);

  return map;
}