

#include <realm_core/settings_base.h>

using namespace realm;

// PUBLIC
//为了管理一组设置参数

//重载:当你使用 settings[key] 时，它实际上调用了 get(key) 函数来获取参数的值并返回
SettingsBase::Variant SettingsBase::operator[](const std::string &key) const
{
  return get(key);
}

//用于从设置中获取给定键 key 对应的参数值
SettingsBase::Variant SettingsBase::get(const std::string &key) const
{
  auto it = m_parameters.find(key);
  if (it != m_parameters.end())
    return *it->second;
  else
    throw(std::out_of_range("Error: Parameter with name '" + key + "' could not be found in settings."));
}

//设置键为 key 的参数值为整数类型 val
void SettingsBase::set(const std::string &key, int val)
{
  m_parameters[key]->m_int_container.value = val;
}

//设置键为 key 的参数值为浮点数类型 val
void SettingsBase::set(const std::string &key, double val)
{
  m_parameters[key]->m_double_container.value = val;
}

//设置键为 key 的参数值为字符串类型 val
void SettingsBase::set(const std::string &key, const std::string &val)
{
  m_parameters[key]->m_string_container.value = val;
}

//设置键为 key 的参数值为 OpenCV 的矩阵（cv::Mat）类型 val
void SettingsBase::set(const std::string &key, const cv::Mat &val)
{
  m_parameters[key]->m_mat_container.value = val;
}

//用于从文件中加载参数设置
void SettingsBase::loadFromFile(const std::string &filepath)
{
  //打开指定路径的配置文件，创建 cv::FileStorage 对象 fs 以进行读取
  cv::FileStorage fs(filepath, cv::FileStorage::READ);
  //检查文件是否成功打开
  if (fs.isOpened())
  {
    //遍历每个参数对象 param
    for (auto &param : m_parameters)
    {
      //根据参数对象的类型,使用 fs 对应的键值将文件中的值读取到参数对象的容器中
      if (param.second->m_type == SettingsBase::Variant::VariantType::INT)    fs[param.first] >> param.second->m_int_container.value;
      if (param.second->m_type == SettingsBase::Variant::VariantType::DOUBLE) fs[param.first] >> param.second->m_double_container.value;
      if (param.second->m_type == SettingsBase::Variant::VariantType::STRING) fs[param.first] >> param.second->m_string_container.value;
      if (param.second->m_type == SettingsBase::Variant::VariantType::MATRIX) fs[param.first] >> param.second->m_mat_container.value;
    }
  }
  else
    throw std::out_of_range("Error. Could not load settings from file: " + filepath);
}

//用于检查给定键是否存在于参数集合中
bool SettingsBase::has(const std::string &key) const
{
  auto it = m_parameters.find(key);
  if (it != m_parameters.end())
    return true;
  else
    return false;
}

//用于将参数集合的内容打印到标准输出
void SettingsBase::print()
{
  std::cout.precision(6);
  for (auto &param : m_parameters)
  {
    if (param.second->m_type == SettingsBase::Variant::VariantType::INT)    std::cout << "\t<Param>[" << param.first << "]: " << param.second->toInt() << std::endl;
    if (param.second->m_type == SettingsBase::Variant::VariantType::DOUBLE) std::cout << "\t<Param>[" << param.first << "]: " << param.second->toDouble() << std::endl;
    if (param.second->m_type == SettingsBase::Variant::VariantType::STRING) std::cout << "\t<Param>[" << param.first << "]: " << param.second->toString() << std::endl;
    if (param.second->m_type == SettingsBase::Variant::VariantType::MATRIX) std::cout << "\t<Param>[" << param.first << "]: " << param.second->toMat() << std::endl;
  }
}

// PROTECTED

//接受一个键和一个整数类型的参数，并将参数的值添加到参数集合中
void SettingsBase::add(const std::string &key, const Parameter_t<int> &param)
{
  m_parameters[key].reset(new Variant(param));
}

//接受一个键和一个双精度浮点数类型的参数，并将参数的值添加到参数集合中
void SettingsBase::add(const std::string &key, const Parameter_t<double> &param)
{
  m_parameters[key].reset(new Variant(param));
}

//接受一个键和一个字符串类型的参数，并将参数的值添加到参数集合中
void SettingsBase::add(const std::string &key, const Parameter_t<std::string> &param)
{
  m_parameters[key].reset(new Variant(param));
}

//接受一个键和一个 OpenCV cv::Mat 类型的参数，并将参数的值添加到参数集合中
void SettingsBase::add(const std::string &key, const Parameter_t<cv::Mat> &param)
{
  m_parameters[key].reset(new Variant(param));
}