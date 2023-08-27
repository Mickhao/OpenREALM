

#include <realm_stages/stage_settings_factory.h>
#include <realm_stages/stage_settings.h>

using namespace realm;

//根据阶段类型和设置文件路径，加载相应阶段的设置
StageSettings::Ptr StageSettingsFactory::load(const std::string &stage_type_set,
                                              const std::string &filepath)
{
  std::string stage_type_read = StageSettings::sneakParameterFromFile<std::string>("type", filepath);
  if (stage_type_set != stage_type_read)
    throw(std::invalid_argument("Error: Could not load stage settings. Stage type mismatch!"));

  // Desired stage type and settings file match, start loading file
  //将读取的阶段类型与传入的 stage_type_set 进行比较，以确保它们匹配
  if (stage_type_set == "pose_estimation")
    return loadDefault<PoseEstimationSettings>(stage_type_set, filepath);
  if (stage_type_set == "densification")
    return loadDefault<DensificationSettings>(stage_type_set, filepath);
  if (stage_type_set == "surface_generation")
    return loadDefault<SurfaceGenerationSettings>(stage_type_set, filepath);
  if (stage_type_set == "ortho_rectification")
    return loadDefault<OrthoRectificationSettings>(stage_type_set, filepath);
  if (stage_type_set == "mosaicing")
    return loadDefault<MosaicingSettings>(stage_type_set, filepath);
  if (stage_type_set == "tileing")
    return loadDefault<TileingSettings>(stage_type_set, filepath);
  throw(std::invalid_argument("Error: Could not load stage settings. Did not recognize stage type: " + stage_type_set));
}

//根据特定阶段类型和设置文件路径，创建一个特定类型的设置对象，并从设置文件中加载设置
template <typename T>
StageSettings::Ptr StageSettingsFactory::loadDefault(const std::string &stage_type_set, const std::string &filepath)
{
  auto settings = std::make_shared<T>();
  //将设置文件的路径传递
  settings->loadFromFile(filepath);
  //返回前使用 std::move 来将对象从局部变量移动到返回值位置
  //以避免额外的复制开销
  return std::move(settings);
}