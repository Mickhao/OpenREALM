

#define LOGURU_IMPLEMENTATION 1
#include <realm_core/loguru.h>

#include <functional>

#include <realm_core/worker_thread_base.h>

/*实现一个基本的工作线程类 WorkerThreadBase
用于管理工作线程的启动、停止、重置等*/
using namespace realm;

//构造函数接受线程名称、睡眠时间和是否启用详细输出作为参数，并设置了各种线程控制标志和函数对象
WorkerThreadBase::WorkerThreadBase(const std::string &thread_name, int64_t sleep_time, bool verbose)
: m_thread_name(thread_name),
  m_sleep_time(sleep_time),
  m_finish_requested(false),
  m_reset_requested(false),
  m_stop_requested(false),
  m_is_stopped(false),
  m_verbose(verbose),
  m_data_ready_functor([=]{ return isFinishRequested(); })
{
  //如果睡眠时间为0，抛出异常
  if (m_sleep_time == 0)
    throw(std::runtime_error("Error: Worker thread was created with 0s sleep time."));
}

//启动工作线程
void WorkerThreadBase::start()
{
  startCallback();
  m_thread = std::thread(std::bind(&WorkerThreadBase::run, this));
}

//等待工作线程结束
void WorkerThreadBase::join()
{
  //如果线程可以被加入，它将会等待线程执行完毕
  if (m_thread.joinable())
    m_thread.join();
}

void WorkerThreadBase::run()
{
  // To have better readability in the log file we set the thread name
  //调用 loguru::set_thread_name() 来设置线程的名称
  loguru::set_thread_name(m_thread_name.c_str());

  //使用 LOG_IF_F 宏记录线程开始循环的日志信息，如果 m_verbose 设置为 true，则输出线程名称和信息
  LOG_IF_F(INFO, m_verbose, "Thread '%s' starting loop...", m_thread_name.c_str());
  bool is_first_run = true;


  while (!isFinishRequested())
  {
    //创建一个 std::unique_lock<std::mutex> 对象 lock，并锁定了互斥锁 m_mutex_processing
    std::unique_lock<std::mutex> lock(m_mutex_processing);
    if (!is_first_run)
    //在 isFinishRequested() 返回 true 之前，或者超时之前，线程都会等待在这里
      m_condition_processing.wait_for(lock, std::chrono::milliseconds(m_sleep_time), m_data_ready_functor);
    else
    //如果是第一次运行，直接将 is_first_run 设为 false，表示不再是第一次运行循环
      is_first_run = false;

    // Handle stops and finish
    if (isStopRequested())
    {
      //记录一条日志信息:指明线程已经被停止。
      LOG_IF_F(INFO, m_verbose, "Thread '%s' stopped!", m_thread_name.c_str());
      //检查线程是否处于停止状态同时还要确保没有收到终止请求
      while (isStopped() && !isFinishRequested())
      {
        //检查是否有重置请求
        if (isResetRequested())
        {
          reset();
          LOG_IF_F(INFO, m_verbose, "Thread '%s' reset!", m_thread_name.c_str());
        }
        //线程会休眠一段时间,时间长度为 m_sleep_time 毫秒
        std::this_thread::sleep_for(std::chrono::milliseconds(m_sleep_time));
      }
      //记录一条日志信息:表示线程已经恢复到正常运行状态
      LOG_IF_F(INFO, m_verbose, "Thread '%s' resumed to loop!", m_thread_name.c_str());
    }

    // Check if reset was requested and execute if necessary
    if (isResetRequested())
    {
      reset();
      LOG_IF_F(INFO, m_verbose, "Thread '%s' reset!", m_thread_name.c_str());
    }

    // Calls to derived classes implementation of process()
    //调用 getCurrentTimeMilliseconds() 获取当前的时间戳
    long t = getCurrentTimeMilliseconds();
    //如果 process() 返回 true，表示处理逻辑执行成功，进入下面的步骤
    if (process())
    {
      //记录一条日志，显示当前处理的总时间
      LOG_IF_F(INFO,
               m_verbose,
               "Timing [Total]: %lu ms",
               getCurrentTimeMilliseconds() - t);

      // Only Update statistics for processing if we did work
      //获取处理时间，并进行统计信息的更新
      std::unique_lock<std::mutex> stat_lock(m_mutex_statistics);
      long timing = getCurrentTimeMilliseconds() - t;
      //更新最小、最大、平均处理时间
      if (m_process_statistics.count == 0) {
        m_process_statistics.min = (double)timing;
        m_process_statistics.max = (double)timing;
      } else {
        if (m_process_statistics.min > (double)timing) m_process_statistics.min = (double)timing;
        if (m_process_statistics.max < (double)timing) m_process_statistics.max = (double)timing;
      }
      //增加统计信息的处理次数，并计算新的平均处理时间
      m_process_statistics.count++;
      m_process_statistics.avg = m_process_statistics.avg + ((double)timing - m_process_statistics.avg) / (double)m_process_statistics.count;
    }

  }
  LOG_IF_F(INFO, m_verbose, "Thread '%s' finished!", m_thread_name.c_str());
}

//用于将线程从停止状态恢复为运行状态
void WorkerThreadBase::resume()
{
  std::unique_lock<std::mutex> lock(m_mutex_is_stopped);
  if (m_is_stopped)
    m_is_stopped = false;
}

//用于请求线程停止
void WorkerThreadBase::requestStop()
{
  std::unique_lock<std::mutex> lock(m_mutex_stop_requested);
  m_stop_requested = true;
  LOG_IF_F(INFO, m_verbose, "Thread '%s' received stop request...", m_thread_name.c_str());
}

//用于请求线程重置
void WorkerThreadBase::requestReset()
{
  std::unique_lock<std::mutex> lock(m_mutex_reset_requested);
  m_reset_requested = true;
  LOG_IF_F(INFO, m_verbose, "Thread '%s' received reset request...", m_thread_name.c_str());
}

//用于请求线程结束
void WorkerThreadBase::requestFinish()
{
  std::unique_lock<std::mutex> lock(m_mutex_finish_requested);
  m_finish_requested = true;
  LOG_IF_F(INFO, m_verbose, "Thread '%s' received finish request...", m_thread_name.c_str());
  finishCallback();
}

//用于检查是否有停止请求
bool WorkerThreadBase::isStopRequested()
{
  std::unique_lock<std::mutex> lock(m_mutex_stop_requested);
  std::unique_lock<std::mutex> lock1(m_mutex_is_stopped);
  if (m_stop_requested && !m_is_stopped)
  {
    m_is_stopped = true;
  }
  return m_stop_requested;
}

//用于检查是否有重置请求
bool WorkerThreadBase::isResetRequested()
{
  std::unique_lock<std::mutex> lock(m_mutex_reset_requested);
  return m_reset_requested;
}

//用于检查是否有结束请求
bool WorkerThreadBase::isFinishRequested()
{
  std::unique_lock<std::mutex> lock(m_mutex_finish_requested);
  return m_finish_requested;
}

//用于检查线程是否已经停止
bool WorkerThreadBase::isStopped()
{
  std::unique_lock<std::mutex> lock(m_mutex_is_stopped);
  if (m_is_stopped)
    m_stop_requested = false;
  return m_is_stopped;
}

//用于通知正在等待的线程，以便它们从等待状态中唤醒
void WorkerThreadBase::notify()
{
  m_condition_processing.notify_one();
}

//用于获取线程的处理统计信息，包括最小、最大、平均处理时间等
Statistics WorkerThreadBase::getProcessingStatistics() {
  std::unique_lock<std::mutex> lock(m_mutex_statistics);
  return m_process_statistics;
}

//用于获取当前的时间（以毫秒为单位），并返回该时间的毫秒表示
long WorkerThreadBase::getCurrentTimeMilliseconds()
{
  using namespace std::chrono;
  milliseconds ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
  return ms.count();
}