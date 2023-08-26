#include <realm_io/mvs_export.h>
#include <realm_core/loguru.h>

#include <unordered_map>

using namespace realm::io;

void MvsExport::saveFrames(const std::vector<Frame::Ptr> &frames, const std::string &directory)
{
  Frame::Ptr first_frame = frames.front();
  //获取第一帧的UTM坐标信息
  UTMPose utm = first_frame->getGnssUtm();
  //创建MVS接口对象MvsInterface，并初始化平台信息
  MvsInterface interface{};

  MvsInterface::Platform platform;
  platform.name    = "OpenREALM";
 
  //创建相机（Camera）信息
  MvsInterface::Platform::Camera camera;
  camera.name      = first_frame->getCameraId();
  camera.width     = first_frame->getCamera()->width();
  camera.height    = first_frame->getCamera()->height();
  camera.K         = first_frame->getCamera()->K();
  camera.R         = MvsInterface::Mat33d(1, 0, 0, 0, 1, 0, 0, 0, 1);
  camera.C         = MvsInterface::Pos3d(0, 0, 0);
  //将相机添加到平台的相机列表中
  platform.cameras.push_back(camera);
  
  //创建无序映射 sparse_cloud_map
  std::unordered_map<uint32_t, MvsInterface::Vertex> sparse_cloud_map;
  //创建帧指针的向量 frames_selected
  std::vector<Frame::Ptr> frames_selected;
  
  //遍历输入的帧对象的向量 frames，并将每个帧对象添加到 frames_selected 向量中
  for (int i = 0; i < frames.size(); i++)
    frames_selected.push_back(frames[i]);

  for (int i = 0; i < frames_selected.size(); ++i)
  {
    //获取 frames_selected 向量中的帧对象 f
    Frame::Ptr f = frames_selected.at(i);
    //获取帧 f 的相机位姿 t
    cv::Mat t = f->getCamera()->t();
    
    //
    MvsInterface::Platform::Pose pose;
    //从帧（f）对象获取相机的旋转矩阵（R）。这个旋转矩阵描述了相机的姿态（方向）
    pose.R         = f->getCamera()->R();
    pose.C         = MvsInterface::Pos3d(t.at<double>(0) - utm.easting, t.at<double>(1) - utm.northing, t.at<double>(2));
    //将姿态对象添加到 MVS 接口中的平台对象（platform.poses）中，以保存相机在场景中的姿态信息
    platform.poses.push_back(pose);

    //构建图像文件名
    std::string filename = directory + "/image_" + std::to_string(i) + ".png";
    //输出文件名
    std::cout << "filename: " << filename << std::endl;
    //保存图像
    cv::imwrite(filename, f->getImageUndistorted());

    //创建 MvsInterface::Image 对象
    MvsInterface::Image img;
    img.name       = filename;
    img.cameraID   = 0;
    img.platformID = 0;
    img.poseID     = i;
    //将构建的 MvsInterface::Image 对象添加到 interface.images 中
    interface.images.push_back(img);

    //创建 MvsInterface::Vertex::View 对象
    MvsInterface::Vertex::View view;
    view.imageID    = i;
    view.confidence = 0.;

    //f->getSparseCloud() 返回稀疏点云对象的指针
    cv::Mat points = f->getSparseCloud()->data();
    //返回点云中每个点的唯一标识 ID（通常是无符号整数）的向量
    std::vector<uint32_t> point_ids = f->getSparseCloud()->getPointIds();

    //遍历点云中的每个点
    for (int j = 0; j < points.rows; ++j)
    {
      //获取当前点的唯一标识 ID
      uint32_t id = point_ids[j];
      //在稀疏点云映射中查找是否已存在具有相同 ID 的点
      auto it = sparse_cloud_map.find(id);
      //如果找到具有相同 ID 的点
      if (it != sparse_cloud_map.end())
      {
        //将当前视图 view 添加到已存在的点的视图列表中
        it->second.views.push_back(view);
      }
      else
      {
        //创建一个新的 MvsInterface::Vertex 对象 vertex
        MvsInterface::Vertex vertex;
        //从稀疏点云的 points 矩阵中获取当前点的坐标，并将坐标转换为UTM坐标系下的坐标
        vertex.X.x = static_cast<float>(points.at<double>(j, 0) - utm.easting);
        vertex.X.y = static_cast<float>(points.at<double>(j, 1) - utm.northing);
        vertex.X.z = static_cast<float>(points.at<double>(j, 2));
        //将当前视图 view 添加到新点的视图列表中
        vertex.views.push_back(view);
        //将新点的ID和关联信息添加到 sparse_cloud_map 中
        sparse_cloud_map[id] = vertex;
      }
    }
  }

  //遍历存储稀疏点云数据的 sparse_cloud_map
  for (const auto& it : sparse_cloud_map)
  //将每个点的数据（包括位置和视图信息）添加到 interface 中的 vertices 列表中
    interface.vertices.push_back(it.second);
  //将前面定义的相机平台信息（包括相机和姿态信息）添加到 interface 中的 platforms 列表中
  interface.platforms.push_back(platform);

  //将收集到的所有信息保存为一个MVS数据文件
  MvsArchive::SerializeSave(interface, directory + "/data.mvs");
}