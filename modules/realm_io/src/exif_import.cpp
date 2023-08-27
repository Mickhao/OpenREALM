

#include <realm_io/exif_import.h>
#include <realm_core/timer.h>

#include <realm_core/loguru.h>

namespace realm
{

io::Exiv2FrameReader::Exiv2FrameReader(const FrameTags &tags)
: m_frame_tags(tags)
{
}

std::map<std::string, bool> io::Exiv2FrameReader::probeImage(const std::string &filepath)
{
  //创建一个空的 std::map 容器，用于存储标签的存在情况
  std::map<std::string, bool> tag_existence;
  //将 m_frame_tags.camera_id 作为键
  //false 作为值插入到 tag_existence 容器中
  //其他亦是
  tag_existence[m_frame_tags.camera_id] = false;
  tag_existence[m_frame_tags.timestamp] = false;
  tag_existence[m_frame_tags.latitude] = false;
  tag_existence[m_frame_tags.latituderef] = false;
  tag_existence[m_frame_tags.longitude] = false;
  tag_existence[m_frame_tags.longituderef] = false;
  tag_existence[m_frame_tags.altitude] = false;
  tag_existence[m_frame_tags.heading] = false;

//使用 Exiv2 图像工厂打开指定的图像文件
  Exiv2ImagePointer exif_img = Exiv2::ImageFactory::open(filepath);
  if (exif_img.get())//检查是否成功打开图像文件
  {
    //读取图像文件的元数据
    exif_img->readMetadata();
    //获取图像的 Exif 数据对象的引用
    Exiv2::ExifData &exif_data = exif_img->exifData();
    //获取图像的 XMP 数据对象的引用
    Exiv2::XmpData &xmp_data = exif_img->xmpData();

    //对每个标签进行探测（probeTag）操作，以判断标签是否存在,将结果存储在 tag_existence 容器中
    tag_existence[m_frame_tags.camera_id] = probeTag(m_frame_tags.camera_id, exif_data, xmp_data);
    tag_existence[m_frame_tags.timestamp] = probeTag(m_frame_tags.timestamp, exif_data, xmp_data);
    tag_existence[m_frame_tags.latitude] = probeTag(m_frame_tags.latitude, exif_data, xmp_data);
    tag_existence[m_frame_tags.latituderef] = probeTag(m_frame_tags.latituderef, exif_data, xmp_data);
    tag_existence[m_frame_tags.longitude] = probeTag(m_frame_tags.longitude, exif_data, xmp_data);
    tag_existence[m_frame_tags.longituderef] = probeTag(m_frame_tags.longituderef, exif_data, xmp_data);
    tag_existence[m_frame_tags.altitude] = probeTag(m_frame_tags.altitude, exif_data, xmp_data);
    tag_existence[m_frame_tags.heading] = probeTag(m_frame_tags.heading, exif_data, xmp_data);
  }

  return tag_existence;
}

Frame::Ptr io::Exiv2FrameReader::loadFrameFromExiv2(const std::string &camera_id, const camera::Pinhole::Ptr &cam, const std::string &filepath)
{
  //打开指定的图像文件，得到一个 Exiv2 图像对象指针 exif_img
  Exiv2ImagePointer exif_img = Exiv2::ImageFactory::open(filepath);
  if (exif_img.get())//检查是否成功打开图像文件
  {
    // Read exif and xmp metadata
    //读取图像文件的元数据，包括 Exif 数据和 XMP 数据
    exif_img->readMetadata();
    Exiv2::ExifData &exif_data = exif_img->exifData();
    Exiv2::XmpData &xmp_data = exif_img->xmpData();

    // Read image data
    //取指定路径的图像文件，加载为一个彩色图像的 CV Mat 对象
    cv::Mat img = cv::imread(filepath, cv::IMREAD_COLOR);

    /*========== ESSENTIAL KEYS ==========*/
    //从图像文件路径中提取帧 ID
    uint32_t frame_id = io::extractFrameIdFromFilepath(filepath);
    //定义一个空字符串，用于存储相机 ID
    std::string camera_id_set;
    if (camera_id.empty())//检查相机 ID 的存在情况
    {
      if (!readMetaTagCameraId(exif_data, xmp_data, &camera_id_set))
        camera_id_set = "unknown_id";
    }

    WGSPose wgs{0};
    //从图像的元数据中读取经纬度、海拔和航向信息
    if (!readMetaTagLatitude(exif_data, xmp_data, &wgs.latitude))
      wgs.latitude = 0.0;

    if (!readMetaTagLongitude(exif_data, xmp_data, &wgs.longitude))
      wgs.longitude = 0.0;

    if (!readMetaTagAltitude(exif_data, xmp_data, &wgs.altitude))
      wgs.altitude = 0.0;

    if (!readMetaTagHeading(exif_data, xmp_data, &wgs.heading))
      wgs.heading = 0.0;
      //将读取到的经纬度信息 wgs 转换为 UTM 坐标
    UTMPose utm = gis::convertToUTM(wgs);

    //LOG_F(INFO, "WGS: %f %f", wgs.latitude, wgs.longitude);
    //LOG_F(INFO, "UTM: %d %c %f %f", utm.zone, utm.band, utm.easting, utm.northing);

    /*========== OPTIONAL KEYS ==========*/
    uint64_t timestamp_val;//存储时间戳的值
    //从 Exif 数据和 XMP 数据中读取时间戳信息。如果无法读取时间戳信息，将使用当前时间的毫秒级时间戳作为默认值
    if (!readMetaTagTimestamp(exif_data, xmp_data, &timestamp_val))
      timestamp_val = Timer::getCurrentTimeMilliseconds();
        //使用读取到的信息，创建一个 Frame 对象
    return std::make_shared<Frame>(camera_id, frame_id, timestamp_val, img, utm, cam, computeOrientationFromHeading(utm.heading));
  }
  return nullptr;
}

bool io::Exiv2FrameReader::readMetaTagCameraId(Exiv2::ExifData &exif_data, Exiv2::XmpData &xmp_data, std::string* camera_id)
{
  //检查是否应该在 XMP 数据中查找相机 ID 信息
  if (isXmpTag(m_frame_tags.camera_id))
  {
    //检查 XMP 数据是否包含指定的 XMP 标签
    if (xmp_data.findKey(Exiv2::XmpKey(m_frame_tags.camera_id)) != xmp_data.end())
    {
      //从 XMP 数据中提取相机 ID 信息并将其存储在 camera_id 指向的字符串对象中
      *camera_id = xmp_data[m_frame_tags.camera_id].toString();
      return true;
    }
  }
  else
  {
    //检查 Exif 数据是否包含指定的 Exif 标签
    if (exif_data.findKey(Exiv2::ExifKey(m_frame_tags.camera_id)) != exif_data.end())
    {
      //从 Exif 数据中提取相机 ID 信息并将其存储在 camera_id 指向的字符串对象中
      *camera_id = exif_data[m_frame_tags.camera_id].toString();
      return true;
    }
  }
  return false;
}

bool io::Exiv2FrameReader::readMetaTagTimestamp(Exiv2::ExifData &exif_data, Exiv2::XmpData &xmp_data, uint64_t* timestamp)
{
  //检查是否应该在 XMP 数据中查找时间戳信息
  if (isXmpTag(m_frame_tags.timestamp))
  {
    //检查 XMP 数据是否包含指定的 XMP 标
    if (xmp_data.findKey(Exiv2::XmpKey(m_frame_tags.timestamp)) != xmp_data.end())
    {
      //从 XMP 数据中提取时间戳信息并将其转换为 uint64_t 类型，然后将其存储在 timestamp 指向的变量中
      *timestamp = std::stoul(xmp_data[m_frame_tags.timestamp].toString());
      return true;
    }
  }
  else
  {
    //检查 Exif 数据是否包含指定的 Exif 标签
    if (exif_data.findKey(Exiv2::ExifKey(m_frame_tags.timestamp)) != exif_data.end())
    {
      //从 Exif 数据中提取时间戳信息并将其转换为 uint64_t 类型，然后将其存储在 timestamp 指向的变量中
      *timestamp = std::stoul(exif_data[m_frame_tags.timestamp].toString());
      return true;
    }
  }
  return false;
}

bool io::Exiv2FrameReader::readMetaTagLatitude(Exiv2::ExifData &exif_data, Exiv2::XmpData &xmp_data, double* latitude)
{
  //存储纬度的度分秒（DMS）
  double latitude_dms[3];
  if (isXmpTag(m_frame_tags.latitude))//检查是否应该在 XMP 数据中查找纬度信息
  {
    //检查 XMP 数据是否同时包含指定的纬度标签和纬度参考标签
    if (xmp_data.findKey(Exiv2::XmpKey(m_frame_tags.latitude)) != xmp_data.end() &&
        xmp_data.findKey(Exiv2::XmpKey(m_frame_tags.latituderef)) != xmp_data.end())
    {
      //从 XMP 数据中分别提取度、分和秒的值，并将这些值存储在 latitude_dms 数组中
      latitude_dms[0] = xmp_data[m_frame_tags.latitude].toFloat(0);
      latitude_dms[1] = xmp_data[m_frame_tags.latitude].toFloat(1);
      latitude_dms[2] = xmp_data[m_frame_tags.latitude].toFloat(2);
      //将度分秒的表示转换为十进制
      *latitude = cvtAngleDegMinSecToDecimal(latitude_dms);

      //检查纬度参考标签的值
      if (xmp_data[m_frame_tags.latituderef].toString(0) == "S")
      {
	      *latitude *= -1.0;
      }
     
      return true;
    }
  }
  else
  {
    //检查 Exif 数据是否同时包含指定的纬度标签和纬度参考标签.其余操作与上面相同
    if (exif_data.findKey(Exiv2::ExifKey(m_frame_tags.latitude)) != exif_data.end() &&
        exif_data.findKey(Exiv2::ExifKey(m_frame_tags.latituderef)) != exif_data.end())
    {
      latitude_dms[0] = exif_data[m_frame_tags.latitude].toFloat(0);
      latitude_dms[1] = exif_data[m_frame_tags.latitude].toFloat(1);
      latitude_dms[2] = exif_data[m_frame_tags.latitude].toFloat(2);
      *latitude = cvtAngleDegMinSecToDecimal(latitude_dms);

      if (exif_data[m_frame_tags.latituderef].toString(0) == "S")
      {
	      *latitude *= -1.0;
      }
      return true;
    }
  }
  return false;
}

bool io::Exiv2FrameReader::readMetaTagLongitude(Exiv2::ExifData &exif_data, Exiv2::XmpData &xmp_data, double* longitude)
{
  //存储经度的度分秒（DMS）,其余和上面相同
  double longitude_dms[3];
  if (isXmpTag(m_frame_tags.longitude))
  {
    if (xmp_data.findKey(Exiv2::XmpKey(m_frame_tags.longitude)) != xmp_data.end() &&
        xmp_data.findKey(Exiv2::XmpKey(m_frame_tags.longituderef)) != xmp_data.end())
    {
      longitude_dms[0] = xmp_data[m_frame_tags.longitude].toFloat(0);
      longitude_dms[1] = xmp_data[m_frame_tags.longitude].toFloat(1);
      longitude_dms[2] = xmp_data[m_frame_tags.longitude].toFloat(2);
      *longitude = cvtAngleDegMinSecToDecimal(longitude_dms);

      if (xmp_data[m_frame_tags.longituderef].toString(0) == "W")
      {
	      *longitude *= -1.0;
      }
      return true;
    }
  }
  else
  {
    if (exif_data.findKey(Exiv2::ExifKey(m_frame_tags.longitude)) != exif_data.end() &&
        exif_data.findKey(Exiv2::ExifKey(m_frame_tags.longituderef)) != exif_data.end())
    {
      longitude_dms[0] = exif_data[m_frame_tags.longitude].toFloat(0);
      longitude_dms[1] = exif_data[m_frame_tags.longitude].toFloat(1);
      longitude_dms[2] = exif_data[m_frame_tags.longitude].toFloat(2);
      *longitude = cvtAngleDegMinSecToDecimal(longitude_dms);

      if (exif_data[m_frame_tags.longituderef].toString(0) == "W")
      {
	      *longitude *= -1.0;
      }
      return true;
    }
  }
  return false;
}

bool io::Exiv2FrameReader::readMetaTagAltitude(Exiv2::ExifData &exif_data, Exiv2::XmpData &xmp_data, double* altitude)
{
  //检查是否应该在 XMP 数据中查找海拔高度信息
  if (isXmpTag(m_frame_tags.altitude))
  {
    //检查 XMP 数据是否包含指定的海拔高度标签
    if (xmp_data.findKey(Exiv2::XmpKey(m_frame_tags.altitude)) != xmp_data.end())
    {
      //从 XMP 数据中读取海拔高度值，并将其转换为 double 类型，并将结果存储在 altitude
      *altitude = static_cast<double>(xmp_data[m_frame_tags.altitude].toFloat());
      return true;
    }
  }
  else
  {
    //检查 Exif 数据是否包含指定的海拔高度标签
    if (exif_data.findKey(Exiv2::ExifKey(m_frame_tags.altitude)) != exif_data.end())
    {
      *altitude = static_cast<double>(exif_data[m_frame_tags.altitude].toFloat());
      return true;
    }
  }
  return false;
}

bool io::Exiv2FrameReader::readMetaTagHeading(Exiv2::ExifData &exif_data, Exiv2::XmpData &xmp_data, double* heading)
{
  //检查是否应该在 XMP 数据中查找航向角信息
  if (isXmpTag(m_frame_tags.heading))
  {
    //检查 XMP 数据是否包含指定的航向角标签
    if (xmp_data.findKey(Exiv2::XmpKey(m_frame_tags.heading)) != xmp_data.end())
    {
      //从 XMP 数据中读取航向角值，并将其转换为 double 类型，并将结果存储在 heading
      *heading = static_cast<double>(xmp_data[m_frame_tags.heading].toFloat());
      return true;
    }
  }
  else
  {
    //检查 Exif 数据是否包含指定的航向角标签
    if (exif_data.findKey(Exiv2::ExifKey(m_frame_tags.heading)) != exif_data.end())
    {
      *heading = static_cast<double>(exif_data[m_frame_tags.heading].toFloat());
      return true;
    }
  }
  return false;
}

double io::Exiv2FrameReader::cvtAngleDegMinSecToDecimal(const double* angle)
{
  //从输入数组中获取度、分和秒的值
  double angle_deg = angle[0];
  //将分和秒的值转换为度的分数形式
  double angle_min = angle[1]/60;
  double angle_sec = angle[2]/3600;
  //将度、分钟和秒钟的十进制值相加，得到最终的十进制表示的角度值
  return angle_deg + angle_min + angle_sec;
}

bool io::Exiv2FrameReader::isXmpTag(const std::string &tag)
{
  //将输入标签字符串按照 . 进行分割结果存储在 tokens
  std::vector<std::string> tokens = io::split(tag.c_str(), '.');
  //代码检查分割后的结果中的第一个元素是否为 "Xmp"
  return tokens[0] == "Xmp";
}

bool io::Exiv2FrameReader::probeTag(const std::string &tag, Exiv2::ExifData &exif_data, Exiv2::XmpData &xmp_data)
{
  if (isXmpTag(tag))//断给定的标签是否为 XMP 格式的标签
  {
    //检查是否可以在 XMP 数据中找到相应的标签键
    if (xmp_data.findKey(Exiv2::XmpKey(tag)) != xmp_data.end())
    {
      return true;
    }
  }
  else
  {
    //查是否可以在 Exif 数据中找到相应的标签键
    if (exif_data.findKey(Exiv2::ExifKey(tag)) != exif_data.end())
    {
      return true;
    }
  }
  return false;
}

}
