

#include <realm_core/timer.h>

/*用于计时器功能的封装类
提供了一些函数来启动或停止计时器
以及获取计时器当前或总共经过的时间*/
using namespace realm;

//创建一个定时器对象
Timer::Timer(const std::chrono::milliseconds &period, const std::function<void()> &func)
    : m_period(period),//传入的时间间隔
      m_func(func), //传入的函数
      m_in_flight(true) //表示定时器处于活动状态
{
  // Lambda implementation for threading
  //启动一个新线程
  m_thread = std::thread([this] 
  {
    while (m_in_flight)
    {
      //等待指定的时间间隔
      this->interruptable_wait_for(m_period);
      if (m_in_flight) //如果为真,则调用成员函数 
      {
        m_func();
      }
    }
  });
}

Timer::~Timer()
{
  {
    //创建一个 std::lock_guard 对象 l，使用互斥锁 m_stop_mtx 对象进行上锁
    std::lock_guard<std::mutex> l(m_stop_mtx);
      m_in_flight = false;//表示定时器不再处于活动状态
  }
  m_cond.notify_one();//发送一个通知，以唤醒可能在等待中的线程
  m_thread.join();//等待定时器所在的线程结束
}

//获取当前时间的秒数，精确到整秒
long Timer::getCurrentTimeSeconds()
{
  using namespace std::chrono;
  seconds s = duration_cast<seconds>(system_clock::now().time_since_epoch());
  return s.count();
}

//获取当前时间的毫秒数，精确到毫秒
long Timer::getCurrentTimeMilliseconds()
{
  using namespace std::chrono;
  milliseconds ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
  return ms.count();
}

//获取当前时间的微秒数，精确到微秒
long Timer::getCurrentTimeMicroseconds()
{
  using namespace std::chrono;
  microseconds ms = duration_cast<microseconds>(system_clock::now().time_since_epoch());
  return ms.count();
}

//获取当前时间的纳秒数，精确到纳秒
long Timer::getCurrentTimeNanoseconds()
{
  using namespace std::chrono;
  nanoseconds ns = duration_cast<nanoseconds>(system_clock::now().time_since_epoch());
  return ns.count();
}