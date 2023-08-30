

#include <realm_densifier_base/densifier_factory.h>

/*据传入的设置来创建相应的稠密化算法实例*/
using namespace realm;

DensifierIF::Ptr densifier::DensifierFactory::create(const DensifierSettings::Ptr &settings)
{
  //检查设置中的稠密化算法类型是否为"DUMMY"
  if ((*settings)["type"].toString() == "DUMMY")
    return std::make_shared<densifier::Dummy>(settings);
  //判断是否定义了USE_CUDA宏
#ifdef USE_CUDA
  //检查设置中的稠密化算法类型是否为"PSL"
  if ((*settings)["type"].toString() == "PSL")
    return std::make_shared<densifier::PlaneSweep>(settings);
#endif
  throw std::invalid_argument("Error: Densifier framework '" + (*settings)["type"].toString() + "' not found");
}