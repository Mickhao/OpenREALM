

#include <realm_densifier_base/densifier_settings_factory.h>

//根据文件中的配置加载不同的稠密化算法设置
using namespace realm;

DensifierSettings::Ptr DensifierSettingsFactory::load(const std::string &filepath, const std::string &directory)
{
  //从文件中读取名为"method"的参数
  std::string method = DensifierSettings::sneakParameterFromFile<std::string>("type", filepath);
  if (method == "DUMMY")//检查从文件中读取的算法类型是否为"DUMMY"
    return loadDefault<DensifierDummySettings>(filepath, directory);
  if (method == "PSL")//检查从文件中读取的算法类型是否为"PSL"
    return loadDefault<PlaneSweepSettings>(filepath, directory);
//if (method == "YOUR_IMPLEMENTATION")
//  return loadDefault<YOUR_IMPLEMENTATION_SETTINGS>(settings, directory, filename);
  throw (std::invalid_argument("Error: Loading densifier settings failed. Method '" + method + "' not recognized"));
}

//加载默认的算法设置
template <typename T>
DensifierSettings::Ptr DensifierSettingsFactory::loadDefault(const std::string &filepath, const std::string &directory)
{
  auto settings = std::make_shared<T>();
  settings->loadFromFile(filepath);
  return settings;
}