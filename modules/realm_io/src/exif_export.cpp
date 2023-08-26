

#include <realm_io/exif_export.h>

namespace realm
{
namespace io
{

//用于将帧保存为带有 EXIF 信息的 JPEG 图像
void saveExifImage(const Frame::Ptr &frame,
                   const std::string &directory,
                   const std::string &name,
                   uint32_t id,
                   bool use_resized)
{
  //生成一个保存图像的文件名 filename，通过将目录、名称和标识符组合而成
  std::string filename = io::createFilename(directory + "/" + name + "_", id, ".jpg");
  //根据 use_resized 的值，选择使用调整大小后的图像还是原始图像
  //并调用 saveExifImage 函数将图像保存到文件中
  if (use_resized)
    saveExifImage(frame->getTimestamp(),
                  frame->getResizedImageRaw(),
                  frame->getResizedCamera(),
                  frame->getGnssUtm(),
                  frame->getCameraId(),
                  frame->getFrameId(),
                  filename);
  else
    saveExifImage(frame->getTimestamp(),
                  frame->getImageRaw(),
                  frame->getCamera(),
                  frame->getGnssUtm(),
                  frame->getCameraId(),
                  frame->getFrameId(),
                  filename);
}

//用于将图像保存为带有 EXIF 信息的 JPEG 文件
void saveExifImage(uint64_t timestamp,
                   const cv::Mat& img,
                   const camera::Pinhole::ConstPtr &cam,
                   const UTMPose &utm_ref,
                   const std::string &camera_id,
                   uint32_t image_id,
                   const std::string &filename)
{
  //创建一个 ExifMetaTag 类型的 meta 对象，用于存储 EXIF 和 XMP 数据
  ExifMetaTag meta;

  // General Tags
  //填充一些常规的 EXIF 标签，如图像处理软件、相机型号、拍摄时间等
  meta.exif_data["Exif.Image.ProcessingSoftware"] = "REALM";
  meta.exif_data["Exif.Image.Model"]              = camera_id;
  meta.exif_data["Exif.Photo.DateTimeOriginal"]   = getDateTime();
  meta.exif_data["Exif.Image.ImageNumber"]        = image_id;            // Long

  // GNSS info conversion
  //将 UTM 坐标转换为 WGS84 坐标，并生成对应的 EXIF 标签
  std::vector<std::string> gnss_info = createGNSSExifTag(gis::convertToWGS84(utm_ref));
  meta.exif_data["Exif.GPSInfo.GPSVersionID"]      = gnss_info[0];            // Byte
  meta.exif_data["Exif.GPSInfo.GPSLatitudeRef"]    = gnss_info[3];            // Ascii
  meta.exif_data["Exif.GPSInfo.GPSLatitude"]       = gnss_info[4];            // Rational
  meta.exif_data["Exif.GPSInfo.GPSLongitudeRef"]   = gnss_info[5];            // Ascii
  meta.exif_data["Exif.GPSInfo.GPSLongitude"]      = gnss_info[6];            // Rational
  meta.exif_data["Exif.GPSInfo.GPSAltitudeRef"]    = gnss_info[1];            // Byte
  meta.exif_data["Exif.GPSInfo.GPSAltitude"]       = gnss_info[2];            // Rational

  // Camera calibration tags
  //提取相机的校准参数，如 cx、cy、fx、fy，并写入 XMP 数据
  meta.xmp_data["Xmp.exif.Timestamp"] = timestamp;
  meta.xmp_data["Xmp.exif.cx"] = cam->cx();
  meta.xmp_data["Xmp.exif.cy"] = cam->cy();
  meta.xmp_data["Xmp.exif.fx"] = cam->fx();
  meta.xmp_data["Xmp.exif.fy"] = cam->fy();

  // Writing image data from opencv mat is not straightforward, see http://dev.exiv2.org/boards/3/topics/2795 for info
  // Basic idea: encode img data before passing to exif creation
  //使用 OpenCV 的 imencode 函数将图像编码为 JPEG 格式，并将编码后的数据存储在缓冲区中
  std::vector<uchar> buffer;
  cv::imencode(".jpg", img, buffer);
  //使用 Exiv2 库创建一个图像对象 exiv2_file，将编码后的图像数据传递给它
  std::unique_ptr<Exiv2::Image> exiv2_file = Exiv2::ImageFactory::open(&buffer[0], buffer.size());

  if (exiv2_file != nullptr)
  {
    //将之前创建的 EXIF 和 XMP 数据设置到图像对象中
    exiv2_file->setExifData(meta.exif_data);
    exiv2_file->setXmpData(meta.xmp_data);
    //使用 writeMetadata 方法将 EXIF 数据写入图像对象
    exiv2_file->writeMetadata();

    // Workaround for writing exif tags to image. File must be created before opening with exif API   
    FILE * file1 = ::fopen(filename.c_str(),"w");
    if (file1 != nullptr)
      ::fclose(file1);
    else
      throw(std::runtime_error("Error creating exif file: Opening '" + filename + "' failed!"));

    //建一个图像文件，并将图像对象中的数据写入文件
    Exiv2::FileIo file(filename);
    if (!file.open())
    {
      file.write(exiv2_file->io());
      file.close();
    }
    else
      throw(std::runtime_error("Error creating exif file: Opening '" + filename + "' failed!"));
  }
  else
    throw(std::runtime_error("Error creating exif image data: Opening failed!"));
}

//用于根据给定的 WGS84 坐标生成相应的 GPS 信息的 EXIF 标签
std::vector<std::string> createGNSSExifTag(const WGSPose &wgs)
{
  //创建一个包含 7 个字符串元素的向量 vect，用于存储不同的 GPS 信息字段
  std::vector<std::string> vect(7);
  // GNSS Version ID
  //将 GNSS 版本信息设置为 "2 2 2 2"，并存储在 vect[0] 中
  vect[0] = "2 2 2 2";

  // Altitude
  //根据海拔高度的正负情况，将海拔信息设置为 "0"（海拔高于海平面）或 "1"（海拔低于海平面）
  //并将绝对值的整数部分存储在 vect[2] 中
  if (wgs.altitude >= 0.0 )
    vect[1] = "0";      // Above Sea Level
  else
    vect[1] = "1";
  vect[2] =  std::to_string((int)floor(fabs(wgs.altitude))) + "/1";

  // Latitude
  //根据纬度的正负情况，将纬度信息设置为 "N"（北纬）或 "S"（南纬）
  //并将纬度的度分秒表示形式存储在 vect[4] 中
  if (wgs.latitude >= 0.0 )
    vect[3] = "N";  // Above Equator
  else
    vect[3] = "S";
  vect[4] = cvtAngleDecimalToDegMinSec(wgs.latitude);

  // Longitude
  //根据经度的正负情况，将经度信息设置为 "E"（东经）或 "W"（西经）
  //并将经度的度分秒表示形式存储在 vect[6] 中
  if (wgs.longitude >= 0.0 )
    vect[5] = "E";     // East of green meridian
  else
    vect[5] = "W";
  vect[6] = cvtAngleDecimalToDegMinSec(wgs.longitude);
  return vect;
}

//用于将以十进制表示的角度转换为度分秒的形式
std::string cvtAngleDecimalToDegMinSec(double angle)
{
  
  std::stringstream result;
  //计算角度的绝对值 angle_abs
  double angle_abs = fabs(angle);
  //将角度的整数部分作为度数部分，并将其存储在 angle_deg 中
  auto angle_deg = static_cast<int>(floor(angle_abs));
  //计算角度小数部分的分钟表示，将其存储在 angle_rem 中
  double angle_rem = (angle_abs - angle_deg)*60;
  //将其转换为整数部分存储在 angle_min 中
  auto angle_min = static_cast<int>(floor(angle_rem));
  //计算角度小数部分的秒表示，将其存储在 angle_sec 中
  auto angle_sec = static_cast<int>(floor((angle_rem - angle_min)*6000));
  //使用字符串流 (std::stringstream) 构建一个字符串，以 "度/分/秒" 的格式表示角度
  result << angle_deg << "/1 " << angle_min << "/1 " << angle_sec << "/100";
  return result.str();
}

} // namespace io
} // namespace realm