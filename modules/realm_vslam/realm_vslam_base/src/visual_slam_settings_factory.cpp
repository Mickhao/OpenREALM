

#include <realm_vslam_base/visual_slam_settings_factory.h>
#include <realm_io/utilities.h>

/*根据设置文件中指定的方法类型加载对应的视觉SLAM设置*/
using namespace realm;

VisualSlamSettings::Ptr VisualSlamSettingsFactory::load(const std::string &filepath, const std::string &directory)
{
  //从设置文件中提取 "type" 参数的值，指定要加载的视觉SLAM方法类型
  std::string method = VisualSlamSettings::sneakParameterFromFile<std::string>("type", filepath);
  if (method == "ORB_SLAM2")
    return loadOrbSlam2(filepath, directory);
  if (method == "ORB_SLAM3")
    return loadOrbSlam2(filepath, directory);
  if (method == "OPEN_VSLAM")
    return loadOpenVslam(filepath, directory);
  if (method == "SVO")
    return loadDefault<SvoSettings>(filepath, directory);
  if (method == "SVO2")
    return loadDefault<Svo2Settings>(filepath, directory);
  if (method == "DSO")
    return loadDefault<DsoSettings>(filepath, directory);
  if (method == "OV2SLAM")
    return loadDefault<Ov2SlamSettings>(filepath, directory);
  throw (std::invalid_argument("Error: Loading visual slam settings failed. Method '" + method + "' not recognized"));
}

//加载 ORB_SLAM2 的设置
VisualSlamSettings::Ptr VisualSlamSettingsFactory::loadOrbSlam2(const std::string &filepath, const std::string &directory)
{
  auto settings = std::make_shared<OrbSlamSettings>();
  settings->loadFromFile(filepath);

  // Check and correct paths
  settings->set("path_vocabulary", directory + settings->get("path_vocabulary").toString());
  return std::move(settings);
}

//加载 OPEN_VSLAM 的设置
VisualSlamSettings::Ptr VisualSlamSettingsFactory::loadOpenVslam(const std::string &filepath, const std::string &directory)
{
  auto settings = std::make_shared<OpenVslamSettings>();
  settings->loadFromFile(filepath);

  // Add full path to file based on current settings folder
  settings->set("path_vocabulary", directory + settings->get("path_vocabulary").toString());
  return std::move(settings);
}

template <typename T>
VisualSlamSettings::Ptr VisualSlamSettingsFactory::loadDefault(const std::string &filepath, const std::string &directory)
{
  auto settings = std::make_shared<T>();
  //从文件中加载设置
  settings->loadFromFile(filepath);
  return settings;
}