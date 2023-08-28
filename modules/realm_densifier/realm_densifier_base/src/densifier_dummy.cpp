

#include <realm_densifier_base/densifier_dummy.h>

using namespace realm;

densifier::Dummy::Dummy(const DensifierSettings::Ptr &settings)
: m_resizing((*settings)["resizing"].toDouble())
{

}

//稠密化
Depthmap::Ptr densifier::Dummy::densify(const std::deque<Frame::Ptr> &frames, uint8_t ref_idx)
{
  return nullptr;
}

//获取输入帧数量
uint8_t densifier::Dummy::getNrofInputFrames()
{
  return 0;
}

//获取调整因子
double densifier::Dummy::getResizeFactor()
{
  return m_resizing;
}

//打印设置到日志的函数
void densifier::Dummy::printSettingsToLog()
{
  LOG_F(INFO, "### Dummy settings ###");
  LOG_F(INFO, "- no settings");
}