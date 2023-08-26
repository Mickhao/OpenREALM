

#include <opencv2/imgproc.hpp>

#include <realm_core/loguru.h>
#include <realm_core/timer.h>
#include <realm_io/gis_export.h>

using namespace realm;

//
void io::saveGeoTIFF(const CvGridMap &map,
                     const uint8_t &zone,
                     const std::string &filename,
                     bool do_build_overview,
                     bool do_split_save,
                     GDALProfile gdal_profile)
{
  //获取当前时间，作为开始保存的时间戳
  long t = Timer::getCurrentTimeMilliseconds();

  //获取地图中的所有图层名称
  std::vector<std::string> layer_names = map.getAllLayerNames();
  //检查图层数量是否大于 1
  if (layer_names.size() > 1)
    throw(std::invalid_argument("Error: Exporting Gtiff from CvGridMap is currently supported for single layer objects only."));
    //获取地图的图像数据，取第一个图层的数据
  cv::Mat img = map[map.getAllLayerNames()[0]];
  //将图像从 BGR 或 BGRA 颜色通道格式转换为 RGB 或 RGBA 格式
  cv::Mat img_converted;
  if (img_converted.channels() == 3)
    cv::cvtColor(img, img_converted, cv::ColorConversionCodes::COLOR_BGR2RGB);
  else if (img.channels() == 4)
    cv::cvtColor(img, img_converted, cv::ColorConversionCodes::COLOR_BGRA2RGBA);
  else
    img_converted = img;
    //调用 io::computeGDALDatasetMeta 函数计算地图的 GDAL 数据集元数据
  GDALDatasetMeta* meta = io::computeGDALDatasetMeta(map, zone);

  //如果不需要分割保存或图像只有一个通道，则直接调用 io::saveGeoTIFFtoFile 函数保存 GeoTIFF 文件，并打印保存信息
  if (!do_split_save || img_converted.channels() == 1)
  {
    io::saveGeoTIFFtoFile(img_converted, *meta, filename, do_build_overview, gdal_profile);

    LOG_F(INFO, "GeoTIFF saved, t = [%4.2f s], location: %s", (Timer::getCurrentTimeMilliseconds()-t)/1000.0, filename.c_str());
  }
  else
  {
    //如果需要分割保存且图像通道数大于 1，则对每个通道进行分别保存
    cv::Mat img_bands[img_converted.channels()];
    cv::split(img_converted, img_bands);

    //根据通道索引，修改文件名，以 "_r"、"_g"、"_b" 或 "_a" 作为后缀插入到文件名前缀中
    for (int i = 0; i < img_converted.channels(); ++i)
    {
      std::string filename_split = filename;
      switch(i)
      {
        case 0:
          filename_split.insert(filename_split.size()-4, "_r");
          break;
        case 1:
          filename_split.insert(filename_split.size()-4, "_g");
          break;
        case 2:
          filename_split.insert(filename_split.size()-4, "_b");
          break;
        case 3:
          filename_split.insert(filename_split.size()-4, "_a");
          break;
        default:
          throw(std::invalid_argument("Error: Exporting GeoTIFF split is only supported up to 4 channels."));
      }
      //调用 io::saveGeoTIFFtoFile 函数分别保存每个通道的 GeoTIFF 文件，并打印保存信息
      io::saveGeoTIFFtoFile(img_converted, *meta, filename_split, do_build_overview, gdal_profile);

      LOG_F(INFO, "GeoTIFF saved, t = [%4.2f s], location: %s", (Timer::getCurrentTimeMilliseconds()-t)/1000.0, filename_split.c_str());

      t = Timer::getCurrentTimeMilliseconds();
    }
  }

  delete meta;
}

void io::saveGeoTIFFtoFile(const cv::Mat &data,
                           const GDALDatasetMeta &meta,
                           const std::string &filename,
                           bool do_build_overviews,
                           GDALProfile gdal_profile)
{
  //定义 GeoTIFF 的文件格式为 "GTiff"，初始化选项指针为 nullptr
  const char *format = "GTiff";
  char **options = nullptr;

  //创建 GDAL 驱动和数据集对象
  GDALDriver* driver;
  GDALDataset* dataset_mem;
  GDALDatasetH dataset_tif;

  //调用 GDALAllRegister() 函数来注册所有的 GDAL 驱动，确保只被调用一次
  GDALAllRegister(); // -> This should be called only once according to the docs, move it in the future

  // First a memory drive is created and the dataset loaded. This is done to provide the functionality of adding
  // internal overviews before translating the data to tif format. This is particularly used for Cloud Optimized GeoTIFFS,
  // see also: https://geoexamples.com/other/2019/02/08/cog-tutorial.html
  // 调用 generateMemoryDataset 函数，创建一个内存数据集，将图像数据加载进去，并提供添加内部缩略图的功能
  dataset_mem = generateMemoryDataset(data, meta);
  
  //如果 do_build_overviews 为真，即需要构建缩略图
  //将使用一组指定的缩略图分辨率大小构建缩略图
  if (do_build_overviews)
  {
    int overview_list[10] = { 2, 4, 8, 16, 32, 64, 128, 256, 1024, 2048 };
    dataset_mem->BuildOverviews("NEAREST", 10, overview_list, 0, nullptr, GDALDummyProgress, nullptr);
  }

  // The previously created dataset in memory is now finally translated to .tif format. All prior information is copied
  // and additional options added.
  driver = GetGDALDriverManager()->GetDriverByName(format);
  options = getExportOptionsGeoTIFF(gdal_profile);

  // Check if multi channel image, therefore RGB/BGR
  //如果图像通道数为 3 或 4，即为 RGB 或 RGBA 图像，将 PHOTOMETRIC 选项设置为 "RGB"
  if (data.channels() == 3 || data.channels() == 4)
    options = CSLSetNameValue( options, "PHOTOMETRIC", "RGB" );
    
    //使用 GDALCreateCopy 函数创建一个与内存数据集一样的 .tif 格式的文件数据集
  dataset_tif = GDALCreateCopy(driver, filename.c_str(), dataset_mem, 0, options, NULL, NULL);

  GDALClose((GDALDatasetH) dataset_mem);
  GDALClose(dataset_tif);
}

//
io::GDALDatasetMeta* io::computeGDALDatasetMeta(const CvGridMap &map, uint8_t zone)
{
  //存储图像的元数据
  auto meta = new GDALDatasetMeta();
  //获取地理感知信息
  cv::Rect2d roi = map.roi();
  double GSD = map.resolution();
  //获取图像的第一个图层,并分析其数据类型，以确定要保存的图像的数据类型
  cv::Mat img = map[map.getAllLayerNames()[0]];

  //根据图像数据类型设置 GDALDatasetMeta 对象的 datatype 字段
  //对应于 GDAL 中的数据类型
  switch(img.type() & CV_MAT_DEPTH_MASK)
  {
    case CV_8U:
      meta->datatype = GDT_Byte;
      break;
    case CV_16U:
      meta->datatype = GDT_Int16;
      break;
    case CV_32F:
      meta->datatype = GDT_Float32;
      break;
    case CV_64F:
      meta->datatype = GDT_Float64;
      break;
    default:
      throw(std::invalid_argument("Error saving GTiff: Image format not recognized!"));
  }

  // Creating geo informations for GDAL and OGR
  //设置地理信息
  //包括投影带号、左上角 x 坐标、地面分辨率、旋转参数和左上角 y 坐标
  meta->zone = zone;
  meta->geoinfo[0] = roi.x;
  meta->geoinfo[1] = GSD;
  meta->geoinfo[2] = 0.0;
  meta->geoinfo[3] = roi.y + roi.height;
  meta->geoinfo[4] = 0.0;
  meta->geoinfo[5] = -GSD;

  return meta;
}

GDALDataset* io::generateMemoryDataset(const cv::Mat &data, const io::GDALDatasetMeta &meta)
{
  //使用获取到的驱动程序创建一个空的内存数据集（dataset）
  GDALDriver* driver = nullptr;
  GDALDataset* dataset = nullptr;
  //初始化 OGRSpatialReference 对象（oSRS）
  //用于处理地理空间参考信息（投影和坐标系）
  OGRSpatialReference oSRS;
  gis::initAxisMappingStrategy(&oSRS);
  
  char **options = nullptr;
  //设置数据集的地理变换参数，包括地理坐标系中的原点坐标、像素宽度和高度等
  driver = GetGDALDriverManager()->GetDriverByName("MEM");
  dataset = driver->Create("", data.cols, data.rows, data.channels(), meta.datatype, options);

  char *pszSRS_WKT = nullptr;
  double geoinfo[6] = {meta.geoinfo[0], meta.geoinfo[1], meta.geoinfo[2], meta.geoinfo[3], meta.geoinfo[4], meta.geoinfo[5]};
  //设置数据集的投影信息，包括投影带号和使用的地理坐标系（WGS84）
  dataset->SetGeoTransform(geoinfo);
  oSRS.SetUTM(meta.zone, TRUE);
  oSRS.SetWellKnownGeogCS("WGS84");
  //将投影信息导出为 WKT（Well-Known Text）格式
  //并将其设置为数据集的投影信息
  oSRS.exportToWkt(&pszSRS_WKT);
  dataset->SetProjection(pszSRS_WKT);
  CPLFree(pszSRS_WKT);
  
  //拆分图像的通道，存储到 img_bands 数组中
  cv::Mat img_bands[data.channels()];
  cv::split(data, img_bands);
  //遍历图像的各个通道，分别将每个通道的数据写入对应的 GDALRasterBand 中
  for (int i = 1; i <= data.channels(); i++)
  {
    GDALRasterBand *band = dataset->GetRasterBand(i);
    CPLErr error_code = band->RasterIO(GF_Write, 0, 0, data.cols, data.rows, img_bands[i - 1].data, data.cols, data.rows, meta.datatype, 0, 0);

    if (error_code != CE_None)
      throw(std::runtime_error("Error saving GeoTIFF: Unhandled error code."));
      //设置每个 GDALRasterBand 中的 NaN 值，确保正确处理缺失数据
    setGDALBandNan(band, data);
  }
  return dataset;
}

char** io::getExportOptionsGeoTIFF(GDALProfile gdal_profile)
{
  //初始化一个空的选项数组 options
  char** options = nullptr;
  /*根据给定的 gdal_profile 枚举值，切换到不同的情况:
    如果 gdal_profile 为 GDALProfile::COG（云优化的GeoTIFF配置）
    则设置一系列选项，包括像素的排列方式、块大小、光度模型、压缩方式等
    如果 gdal_profile 不是已知的值，抛出异常，表示未知的 GDAL 导出配置
  */
  switch(gdal_profile)
  {
    case GDALProfile::COG:
      options = CSLSetNameValue( options, "INTERLEAVE", "PIXEL" );
      options = CSLSetNameValue( options, "TILED", "YES" );
      options = CSLSetNameValue( options, "BLOCKXSIZE", "256" );
      options = CSLSetNameValue( options, "BLOCKYSIZE", "256" );
      options = CSLSetNameValue( options, "PHOTOMETRIC", "MINISBLACK");
      options = CSLSetNameValue( options, "BIGTIFF", "IF_SAFER");
      options = CSLSetNameValue( options, "COPY_SRC_OVERVIEWS", "YES" );
      options = CSLSetNameValue( options, "COMPRESS", "LZW" );
      break;
    default:
      throw(std::invalid_argument("Error: Unknown GDAL export profile."));
  }
  return options;
}

//根据图像的数据类型设置 GDAL 栅格波段的无效值（NaN值）
void io::setGDALBandNan(GDALRasterBand *band, const cv::Mat &data)
{
  //如果图像的数据类型是 CV_8UC1、CV_8UC2、CV_8UC3 或 CV_8UC4（8位无符号整数通道图像），则将栅格波段的无效值设置为 0
  if (data.type() == CV_8UC1 || data.type() == CV_8UC2 || data.type() == CV_8UC3 || data.type() == CV_8UC4)
    band->SetNoDataValue(0);
  //如果图像的数据类型是 CV_16UC1（16位无符号整数通道图像），则将栅格波段的无效值设置为 0
  else if (data.type() == CV_16UC1)
    band->SetNoDataValue(0);
  //如果图像的数据类型是 CV_32F（32位浮点数通道图像），则将栅格波段的无效值设置为浮点数的 NaN 值
  else if (data.type() == CV_32F)
    band->SetNoDataValue(std::numeric_limits<float>::quiet_NaN());
  //如果图像的数据类型是 CV_64F（64位浮点数通道图像），则将栅格波段的无效值设置为双精度浮点数的 NaN 值
  else if (data.type() == CV_64F)
    band->SetNoDataValue(std::numeric_limits<double>::quiet_NaN());
}