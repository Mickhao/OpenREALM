

#include <realm_core/conversions.h>

/*使用GDAL库来进行坐标转换*/
using namespace realm;

/*'convertToUTM'函数先根据输入的经度计算所属的UTM带号和带号字母。
然后创建两个'GRSpatialReference'对象，分别用于WGS84和UTM的坐标系
对 ogr_utm 进行设置，指定UTM带号和北半球/南半球。
创建 OGRCoordinateTransformation 对象，用于执行坐标转换。
对输入的经纬度进行转换，将经度赋值给 x，纬度赋值给 y，然后调用 Transform 方法进行坐标转换。
最后，根据转换结果创建并返回一个 UTMPose 对象。
*/
UTMPose gis::convertToUTM(const WGSPose &wgs)
{
  // TODO: Check if utm conversions are valid everywhere (not limited to utm32)
  // Compute zone（计算带号）zone:表示UTM的经度带号
  double lon_tmp = (wgs.longitude+180)-int((wgs.longitude+180)/360)*360-180;
  auto zone = static_cast<int>(1 + (wgs.longitude+180.0)/6.0);
  if(wgs.latitude >= 56.0 && wgs.latitude < 64.0 && lon_tmp >= 3.0 && lon_tmp < 12.0)
    zone = 32;
  if(wgs.latitude >= 72.0 && wgs.latitude < 84.0)
  {
    if(      lon_tmp >= 0.0  && lon_tmp <  9.0 ) zone = 31;
    else if( lon_tmp >= 9.0  && lon_tmp < 21.0 ) zone = 33;
    else if( lon_tmp >= 21.0 && lon_tmp < 33.0 ) zone = 35;
    else if( lon_tmp >= 33.0 && lon_tmp < 42.0 ) zone = 37;
  }

  // Compute band（计算带字母） band：表示MGRS的纬度带字母
  char band = UTMBand(wgs.latitude, wgs.longitude);

  int is_northern = (wgs.latitude < 0.0 ? 0 : 1);
  //根据经纬度的正负来判断所在的地理位置是北半球还是南半球,0南1北

  OGRSpatialReference ogr_wgs;
  gis::initAxisMappingStrategy(&ogr_wgs);
  
  ogr_wgs.SetWellKnownGeogCS("WGS84");

  OGRSpatialReference ogr_utm;
  gis::initAxisMappingStrategy(&ogr_utm);

  ogr_utm.SetWellKnownGeogCS("WGS84");
  ogr_utm.SetUTM(zone, is_northern);

  OGRCoordinateTransformation* coord_trans = OGRCreateCoordinateTransformation(&ogr_wgs, &ogr_utm);

  double x = wgs.longitude;
  double y = wgs.latitude;
  bool result = coord_trans->Transform(1, &x, &y);

  // Underlying library is C malloc, have to clean up the resulting pointer to avoid a leak
  delete coord_trans;  //完成后释放，避免内存泄漏

  if (!result) {
    throw(std::runtime_error("Error converting utm coordinates to wgs84: Transformation failed"));
  }

  return UTMPose(x, y, wgs.altitude, wgs.heading, (uint8_t)zone, band);
}

/*创建两个 OGRSpatialReference 对象，分别用于UTM和WGS84的坐标系。
对 ogr_utm 进行设置，指定UTM带号和北半球/南半球。
对 ogr_wgs 进行设置，指定WGS84坐标系。
创建 OGRCoordinateTransformation 对象，用于执行坐标转换。
对输入的UTM坐标进行转换，将东移坐标赋值给 x，北移坐标赋值给 y，然后调用 Transform 方法进行坐标转换。
最后，根据转换结果创建并返回一个 WGSPose 对象。
*/
WGSPose gis::convertToWGS84(const UTMPose &utm)
{
  OGRSpatialReference ogr_utm;
  gis::initAxisMappingStrategy(&ogr_utm);

  ogr_utm.SetWellKnownGeogCS("WGS84");
  ogr_utm.SetUTM(utm.zone, TRUE);

  OGRSpatialReference ogr_wgs;
  gis::initAxisMappingStrategy(&ogr_wgs);
  ogr_wgs.SetWellKnownGeogCS("WGS84");

  OGRCoordinateTransformation* coord_trans = OGRCreateCoordinateTransformation(&ogr_utm, &ogr_wgs);

  double x = utm.easting;
  double y = utm.northing;
  bool result = coord_trans->Transform(1, &x, &y);

  // Underlying library is C malloc, have to clean up the resulting pointer to avoid a leak
  delete coord_trans;

  if (!result) {
    throw(std::runtime_error("Error converting utm coordinates to wgs84: Transformation failed"));
  }

  return WGSPose{y, x, utm.altitude, utm.heading};
}

/*定义了一个名为 initAxisMappingStrategy 的函数
该函数的作用是根据 GDAL 版本的不同来初始化坐标轴映射策略。
如果GDAL版本号大于等于3，就将坐标轴映射策略设为传统的 GIS 顺序
即：经度在前，纬度在后
*/
void gis::initAxisMappingStrategy(OGRSpatialReference *oSRS)
{
#if GDAL_VERSION_MAJOR >= 3
  {
    oSRS->SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
  }
#endif
}
