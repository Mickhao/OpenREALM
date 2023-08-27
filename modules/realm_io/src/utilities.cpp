

#include <realm_core/loguru.h>

#include <realm_io/utilities.h>

using namespace realm;

bool io::fileExists(const std::string &filepath)
{
  //检查给定路径的文件是否存在
  return boost::filesystem::exists(filepath);
}


bool io::dirExists(const std::string &directory)
{
  //检查给定路径的目录是否存在
  return boost::filesystem::exists(directory);
}
//用于创建目录
void io::createDir(const std::string &directory)
{
  //检查目录是否已经存在
  if (io::dirExists(directory))
    return;
  //将传入的目录路径转换为 boost::filesystem::path 对象
  boost::filesystem::path dir(directory);
  try
  {
    //创建目录
    boost::filesystem::create_directory(dir);
  }
  catch (...)
  {
    LOG_F(WARNING, "Creating path failed: %s", directory.c_str());
  }
}
//用于删除文件或目录
bool io::removeFileOrDirectory(const std::string &path)
{
  boost::system::error_code error;
  //来删除指定路径下的文件或目录
  boost::filesystem::remove_all(path, error);

  if(error)
  {
    throw(std::runtime_error(error.message()));
  }
  return true;
}

//生成文件名
std::string io::createFilename(const std::string &prefix, uint32_t frame_id, const std::string &suffix)
{
  char filename[1000];
  //将前缀、帧ID和后缀组合成一个文件名字符串，并将其返回
  sprintf(filename, (prefix + "%06i" + suffix).c_str(), frame_id);
  return std::string(filename);
}
//获取系统的临时目录路径
std::string io::getTempDirectoryPath()
{
  boost::system::error_code error;
  //获取临时目录的路径，并将路径转换为字符串
  boost::filesystem::path path = boost::filesystem::temp_directory_path(error);

  if(error)
  {
    throw(std::runtime_error(error.message()));
  }
  return path.string();
}
//用于获取当前的日期和时间
std::string io::getDateTime()
{
  //获取当前时间的时间戳
  time_t     now = time(nullptr);
  //将时间戳转换为本地时间
  tm  tstruct = *localtime(&now);
  char tim[20];
  //将时间结构转换为指定格式的日期时间字符串
  strftime(tim, sizeof(tim), "%y-%m-%d_%H-%M-%S", &tstruct);
  return std::string(tim);
}
//从文件路径中提取帧ID
uint32_t io::extractFrameIdFromFilepath(const std::string &filepath)
{
  //将文件路径分割为不同的部分，然后从中提取文件名
  std::vector<std::string> tokens_path = io::split(filepath.c_str(), '/');
  std::vector<std::string> tokens_name = io::split(tokens_path.back().c_str(), '.');
  std::string filename = tokens_name[0];
  //将文件名的一部分转换为无符号长整型
  return static_cast<uint32_t>(std::stoul(filename.substr(filename.size()-4,filename.size())));
}
//将一个C字符串按照指定的字符分割成多个子字符串
std::vector<std::string> io::split(const char *str, char c)
{
  std::vector<std::string> result;

  do
  {
    //记录子字符串的起始位置
    const char *begin = str;

    //用于查找标记字符 c 或字符串结尾
    while(*str != c && *str)
      str++;

    result.emplace_back(std::string(begin, str));
  }
  while (0 != *str++);

  return result;
}

std::vector<std::string> io::getFileList(const std::string& dir, const std::string &suffix)
{
  std::vector<std::string> filenames;
  if (!dir.empty())//检查传入的目录路径是否为空
  {
    //将传入的目录路径转换为 boost::filesystem::path
    boost::filesystem::path apk_path(dir);
    //创建一个递归目录遍历迭代器的结束迭代器
    boost::filesystem::recursive_directory_iterator end;
    //遍历目录中的所有文件和子目录
    for (boost::filesystem::recursive_directory_iterator it(apk_path); it != end; ++it)
    {
      //获取当前迭代器指向的路径
      const boost::filesystem::path cp = (*it);
      //将路径转换为字符串
      const std::string &filepath = cp.string();
      //检查文件路径的后缀名是否与传入的后缀名相匹配
      if (suffix.empty() || filepath.substr(filepath.size() - suffix.size(), filepath.size()) == suffix)
        filenames.push_back(cp.string());
    }
  }
  //对文件路径列表进行排序
  std::sort(filenames.begin(), filenames.end());
  return filenames;
}
//计算根据给定航向角度（heading）计算相机的旋转矩阵
cv::Mat io::computeOrientationFromHeading(double heading)
{
  // Rotation to the world in camera frame
  //创建一个3x3的单位矩阵
  cv::Mat R_wc = cv::Mat::eye(3, 3, CV_64F);
  //将第二行第二列的元素设置为 -1
  R_wc.at<double>(1, 1) = -1;
  //将第三行第三列的元素设置为 -1
  R_wc.at<double>(2, 2) = -1;

  // Rotation around z considering uav heading
  //将航向角度转换为弧度
  double gamma = heading * M_PI / 180;
  //创建一个3x3的单位矩阵
  cv::Mat R_wc_z = cv::Mat::eye(3, 3, CV_64F);
  //设置 R_wc_z 矩阵的元素来表示绕 z 轴旋转
  R_wc_z.at<double>(0, 0) = cos(-gamma);
  R_wc_z.at<double>(0, 1) = -sin(-gamma);
  R_wc_z.at<double>(1, 0) = sin(-gamma);
  R_wc_z.at<double>(1, 1) = cos(-gamma);
  //绕 z 轴旋转的旋转矩阵和世界到相机坐标系的旋转矩阵相乘
  cv::Mat R = R_wc_z * R_wc;

  return R;
}