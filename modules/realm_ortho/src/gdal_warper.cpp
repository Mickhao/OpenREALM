

#include <realm_ortho/gdal_warper.h>

#include <opencv2/highgui.hpp>

/*将栅格数据进行投影变换*/
using namespace realm;

//初始化
gis::GdalWarper::GdalWarper()
 : m_epsg_target(0),
   m_nrof_threads(-1)
{
  //注册GDAL驱动
  GDALAllRegister();

  // Warping is done in RAM and no data needs to be saved to the disk for now
  //初始化用于将栅格数据保存在内存中的驱动
  m_driver = GetGDALDriverManager()->GetDriverByName("MEM");
}

//设置目标EPSG
void gis::GdalWarper::setTargetEPSG(int epsg_code)
{
  m_epsg_target = epsg_code;
}

//设置线程数量
void gis::GdalWarper::setNrofThreads(int nrof_threads)
{
  m_nrof_threads = nrof_threads;
}

//执行实际的投影变换
CvGridMap::Ptr gis::GdalWarper::warpRaster(const CvGridMap &map, uint8_t zone)
{
  //=======================================//
  //
  //      Step 1: Check validity
  //
  //=======================================//

  //检查输入数据和设置，确保进行地图扭曲（warping）的准备工作和参数都是有效的
  if (m_epsg_target == 0)
    throw(std::runtime_error("Error warping map: Target EPSG was not set!"));

    //获取地图对象中所有图层的名称
  std::vector<std::string> layer_names = map.getAllLayerNames();
  //检查图层数量是否大于1
  if (layer_names.size() > 1)
    throw(std::invalid_argument("Error warping map: There is more than one layer in the map. This is currently not supported."));

  //=======================================//
  //
  //      Step 2: Prepare datasets
  //
  //=======================================//

  // Extract data and metadata
  //地图中获取第一个图层的数据
  cv::Mat data = map[layer_names[0]];
  //计算数据集的元数据
  io::GDALDatasetMeta* meta = io::computeGDALDatasetMeta(map, zone);

  // Convert the source map into a GDAL dataset
  //将数据和元数据转换为GDAL的数据集
  GDALDataset* dataset_mem_src;
  dataset_mem_src = io::generateMemoryDataset(data, *meta);

  // Get source coordinate system.
  //取源数据集的投影参考（坐标系）信息
  const char *gdal_proj_src = GDALGetProjectionRef(dataset_mem_src);
  //返回源数据集的投影参考的字符串表示
  CPLAssert(gdal_proj_src != NULL && strlen(gdal_proj_src) > 0);

  // Set target coordinate system
  char *gdal_proj_dst = nullptr;
  OGRSpatialReference oSRS;
  gis::initAxisMappingStrategy(&oSRS);

  //将目标EPSG代码指定的坐标系导入到oSRS对象中
  oSRS.importFromEPSG(m_epsg_target);
  //将目标坐标系转换为Well-Known Text（WKT）格式
  oSRS.exportToWkt(&gdal_proj_dst);
  //使用CPLAssert宏检查目标坐标系的WKT字符串是否成功导出
  CPLAssert(gdal_proj_dst != NULL && strlen(gdal_proj_dst) > 0);

  // Create a transformer that maps from source pixel/line coordinates
  // to destination georeferenced coordinates (not destination
  // pixel line).  We do that by omitting the destination dataset
  // handle (setting it to NULL).
  //创建一个用于地图投影变换的变换器（projector）
  void *projector = GDALCreateGenImgProjTransformer(dataset_mem_src, gdal_proj_src, NULL, gdal_proj_dst, FALSE, 0, 1);
  //使用CPLAssert宏来检查变换器是否成功创建
  CPLAssert( projector != NULL );

  // Get approximate output georeferenced bounds and resolution for file.
  //获取扭曲后数据集的大致地理信息
  double geoinfo_target[6];
  int warped_cols = 0, warped_rows = 0;
  CPLErr eErr = GDALSuggestedWarpOutput(dataset_mem_src, GDALGenImgProjTransform, projector, geoinfo_target, &warped_cols , &warped_rows  );
  //检查获取扭曲后数据集地理信息的操作是否成功
  CPLAssert( eErr == CE_None );
  //销毁之前创建的变换器，释放相关资源
  GDALDestroyGenImgProjTransformer(projector);

  // Create the output object.
  //通过GDAL驱动程序创建目标数据集
  GDALDataset* dataset_mem_dst = m_driver->Create("", warped_cols , warped_rows , data.channels(), meta->datatype, nullptr );
  //检查目标数据集是否成功创建
  CPLAssert( hDstDS != NULL );

  // Write out the projection definition.
  //设置目标数据集的投影信息
  GDALSetProjection(dataset_mem_dst, gdal_proj_dst);
  //设置目标数据集的地理转换信息
  GDALSetGeoTransform(dataset_mem_dst, geoinfo_target);

  //释放之前分配的目标坐标系字符串的内存
  CPLFree(gdal_proj_dst);

  //=======================================//
  //
  //      Step 3: Prepare warping
  //
  //=======================================//

  double no_data_value;
  /*根据源数据的类型（单精度浮点或双精度浮点）
  将no_data_value设置为相应的NaN值，以指示无效数据
  如果源数据不是浮点类型，则将no_data_value设置为0.0
  */
  switch(data.type() & CV_MAT_DEPTH_MASK)
  {
    case CV_32F:
      no_data_value = std::numeric_limits<float>::quiet_NaN();
      break;
    case CV_64F:
      no_data_value = std::numeric_limits<double>::quiet_NaN();
      break;
    default:
      no_data_value = 0.0;
  }

/*根据图层的插值方法
选择GDAL库中提供的相应的重采样算法
*/
  GDALResampleAlg resample_alg;
  switch(map.getLayer(layer_names[0]).interpolation)
  {
    case cv::INTER_NEAREST:
      resample_alg = GRA_NearestNeighbour;
      break;
    case cv::INTER_LINEAR:
      resample_alg = GRA_Bilinear;
      break;
    case cv::INTER_CUBIC:
      resample_alg = GRA_Cubic;
      break;
    default:
      resample_alg = GRA_Bilinear;
  }

  char** warper_system_options = nullptr;
  //初始化输出数据集,设置为NO_DATA
  warper_system_options = CSLSetNameValue(warper_system_options, "INIT_DEST", "NO_DATA");
  //如果 m_nrof_threads 小于等于0，则将其设置为 ALL_CPUS
  //否则，将其设置为 m_nrof_threads
  if (m_nrof_threads <= 0)
    warper_system_options = CSLSetNameValue(warper_system_options, "NUM_THREADS", "ALL_CPUS");
  else
    warper_system_options = CSLSetNameValue(warper_system_options, "NUM_THREADS", std::to_string(m_nrof_threads).c_str());

  // Setup warp options.
  //创建一个 GDALWarpOptions 结构，并对其成员进行配置
  GDALWarpOptions *warper_options = GDALCreateWarpOptions();
  warper_options->eResampleAlg = resample_alg;
  warper_options->papszWarpOptions = warper_system_options;
  warper_options->padfSrcNoDataReal = new double(no_data_value);
  warper_options->padfDstNoDataReal = new double(no_data_value);
  warper_options->hSrcDS = dataset_mem_src;
  warper_options->hDstDS = dataset_mem_dst;
  warper_options->nBandCount = 0;
  warper_options->nSrcAlphaBand = (data.channels() == 4 ? data.channels() : 0);
  warper_options->nDstAlphaBand = (data.channels() == 4 ? data.channels() : 0);

  // Establish reprojection transformer.
  //创建一个执行投影变换的变换器
  warper_options->pTransformerArg = GDALCreateGenImgProjTransformer(
                                       dataset_mem_src,
                                       GDALGetProjectionRef(dataset_mem_src),
                                       dataset_mem_dst,
                                       GDALGetProjectionRef(dataset_mem_dst),
                                       FALSE, 0.0, 1 );

  //设置变换函数为 GDALGenImgProjTransform
  warper_options->pfnTransformer = GDALGenImgProjTransform;

  //=======================================//
  //
  //      Step 4: Warping
  //
  //=======================================//
  //图像重采样

  //初始化
  GDALWarpOperation warping;
  warping.Initialize(warper_options);
  //用 ChunkAndWarpImage 函数来执行重采样操作，将目标数据集的图像数据映射到重采样后的图像中
  warping.ChunkAndWarpImage(0, 0, GDALGetRasterXSize(dataset_mem_dst), GDALGetRasterYSize(dataset_mem_dst));

  //获取重采样后的图像数据
  int raster_cols = dataset_mem_dst->GetRasterXSize();
  int raster_rows = dataset_mem_dst->GetRasterYSize();
  int raster_channels = dataset_mem_dst->GetRasterCount();

  int single_channel_type;
  //根据原始数据类型确定单通道数据类型
  switch(data.type() & CV_MAT_DEPTH_MASK)
  {
    case CV_8U:
      single_channel_type = CV_8UC1;
      break;
    case CV_16U:
      single_channel_type = CV_16UC1;
      break;
    case CV_32F:
      single_channel_type = CV_32FC1;
      break;
    case CV_64F:
      single_channel_type = CV_64FC1;
      break;
  }

  //循环遍历每个波段
  std::vector<cv::Mat> warped_data_split;
  for(int i = 1; i <= raster_channels; ++i)
  {
    // Save the channel in var not in the vector of Mat
    //读取目标数据集的每个波段的像素值到 cv::Mat bckVar 中
    cv::Mat bckVar(raster_rows, raster_cols, single_channel_type);

    GDALRasterBand *band = dataset_mem_dst->GetRasterBand(i);
    band->SetNoDataValue(no_data_value);

    eErr = band->RasterIO(GF_Read, 0, 0, raster_cols, raster_rows, bckVar.data, raster_cols, raster_rows, band->GetRasterDataType(), 0, 0);
    CPLAssert( eErr == CE_None );

    fixGdalNoData(bckVar);

    warped_data_split.push_back(bckVar);
  }

  //合并为一个多通道的warped_data
  cv::Mat warped_data;
  cv::merge(warped_data_split, warped_data);

  //释放资源
  GDALDestroyGenImgProjTransformer(warper_options->pTransformerArg );
  GDALDestroyWarpOptions(warper_options );
  delete meta;

  //=======================================//
  //
  //      Step 5: Compute output
  //
  //=======================================//

  double warped_geoinfo[6];
  //获取重采样后图像数据的地理变换参数
  dataset_mem_dst->GetGeoTransform(warped_geoinfo);

  //计算重采样后图像的分辨率
  double warped_resolution = warped_geoinfo[1];

  //使用重采样后图像的地理变换参数和行数、列数
  //计算出重采样后图像的区域信息 warped_roi
  cv::Rect2d warped_roi;
  warped_roi.x = warped_geoinfo[0];
  warped_roi.y = warped_geoinfo[3] - warped_data.rows * warped_resolution;
  warped_roi.width = warped_data.cols * warped_resolution - warped_resolution;
  warped_roi.height = warped_data.rows * warped_resolution - warped_resolution;
  
  //使用计算出的 warped_roi 和 warped_resolution 构造一个 CvGridMap 对象 output
  auto output = std::make_shared<CvGridMap>(warped_roi, warped_resolution);
  //将经过重采样的图像数据 warped_data 添加到 output 中，同时传递图像的插值方式
  output->add(layer_names[0], warped_data, map.getLayer(layer_names[0]).interpolation);
  //关闭目标数据集 dataset_mem_dst 和源数据集 dataset_mem_src
  GDALClose(dataset_mem_dst);
  GDALClose(dataset_mem_src);

  return output;
}

/*CvGridMap::Ptr gis::GdalWarper::warpImage(const CvGridMap &map, uint8_t zone)
{
  //=======================================//
  //
  //      Step 1: Check validity
  //
  //=======================================//

  if (_epsg_target == 0)
    throw(std::runtime_error("Error warping map: Target EPSG was not set!"));

  //=======================================//
  //
  //      Step 2: Prepare datasets
  //
  //=======================================//

  // Get source coordinate system
  OGRSpatialReference src_SRS;
  src_SRS.SetUTM(zone, TRUE);
  src_SRS.SetWellKnownGeogCS("WGS84");

  // Set target coordinate system
  OGRSpatialReference dst_SRS;
  dst_SRS.importFromEPSG(_epsg_target);

  // Create transformator
  OGRCoordinateTransformation *tf = OGRCreateCoordinateTransformation(&src_SRS, &dst_SRS);

  cv::Rect2d roi = map.roi();

  double x[4] = { roi.x, roi.x,              roi.x + roi.width, roi.x + roi.width };
  double y[4] = { roi.y, roi.y + roi.height, roi.y,             roi.y + roi.height };

  bool success = tf->Transform(4, x, y);

  if (success)
  {
    double w = std::min({x[0], x[1], x[2], x[3]});
    double e = std::max({x[0], x[1], x[2], x[3]});
    double s = std::min({y[0], y[1], y[2], y[3]});
    double n = std::max({y[0], y[1], y[2], y[3]});
    cv::Rect2d roi_warped(w, s, e-w, n-s);

    cv::Mat src_points = (cv::Mat_<float>(4, 2) <<
                                                -roi.width/2, -roi.height/2,
        -roi.width/2,  roi.height/2,
        roi.width/2, -roi.height/2,
        roi.width/2,  roi.height/2);

    double resolution = map.resolution() * roi_warped.width/roi.width;
    double tx = roi_warped.x + roi_warped.width/2;
    double ty = roi_warped.y + roi_warped.height/2;

    cv::Mat dst_points(4, 2, CV_32F);
    for (int i = 0; i < 4; ++i)
    {
      dst_points.at<float>(i, 0) = (float)(x[i] - tx);
      dst_points.at<float>(i, 1) = (float)(y[i] - ty);
    }

    cv::Mat H = cv::getPerspectiveTransform(src_points, dst_points);

    CvGridMap map_clone = map.clone();
    auto map_warped = std::make_shared<CvGridMap>(roi_warped, resolution);

    for (const auto &layer_name : map.getAllLayerNames())
    {
      map_clone.changeResolution(resolution);
      const CvGridMap::Layer &layer = map_clone.getLayer(layer_name);

      cv::Mat data_warped;
      cv::warpPerspective(layer.data, data_warped, H, map_warped->size());

      map_warped->add(layer.name, data_warped, layer.interpolation);
    }

    return map_warped;
  }
  return nullptr;
}*/

void gis::GdalWarper::warpPoints()
{
  /*OGRSpatialReference s_SRS;
    const char* s_WKT = dataset_mem_src->GetProjectionRef();
    s_SRS.importFromWkt(const_cast<char **>(&s_WKT));
    OGRCoordinateTransformation *coordinate_transformation;

    double x = blub[0], y = blub[3];
    coordinate_transformation = OGRCreateCoordinateTransformation(&oSRS, &s_SRS);
    coordinate_transformation->Transform(1, &x, &y);*/
}

//修复GDAL数据集中无效值
void gis::GdalWarper::fixGdalNoData(cv::Mat &data)
{
  //输入数据的类型是否是 CV_32F
  if (data.type() == CV_32F)
  {
    //创建一个掩码，将接近零的值标记为真
    cv::Mat mask;
    cv::inRange(data, -std::numeric_limits<float>::epsilon(), std::numeric_limits<float>::epsilon(), mask);
    //将掩码标记为真的像素值设置为 std::numeric_limits<float>::quiet_NaN()
    data.setTo(std::numeric_limits<float>::quiet_NaN(), mask);
  }
  else if (data.type() == CV_64F)
  {
    cv::Mat mask;
    cv::inRange(data, -std::numeric_limits<double>::epsilon(), std::numeric_limits<double>::epsilon(), mask);
    data.setTo(std::numeric_limits<double>::quiet_NaN(), mask);
  }
}