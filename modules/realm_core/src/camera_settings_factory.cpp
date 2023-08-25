

#include <realm_core/camera_settings_factory.h>

using namespace realm;
/*运用了'CameraSettingsFactory'类中的函数实现
函数从给定的'.yaml'文件中读取相机模型的类型，
如果模型类型为'pinhole'则调用'load<PinholeSettings>'来加载相机设置
如果未能识别，则抛出异常
*/
CameraSettings::Ptr CameraSettingsFactory::load(const std::string &filepath)
{
  // Identify camera model
  std::string model_type = CameraSettings::sneakParameterFromFile<std::string>("type", filepath);
  if (model_type == "pinhole")
    return load<PinholeSettings>(filepath);
  else
    throw(std::out_of_range("Error! Camera type '" + model_type + "' not recognized."));
}
/*函数从给定的'.yaml'文件中加载相机设置
并创建一个指向类型<T>的相机设置对象的共享指针'settings'
再从文件中加载设置
最后通过'std::move'将其相机设置对象传递出去
*/
template <typename T>
CameraSettings::Ptr CameraSettingsFactory::load(const std::string &filepath)
{
  // Read from settings file
  auto settings = std::make_shared<T>();
  settings->loadFromFile(filepath);
  return std::move(settings);
}

//相机设置工厂