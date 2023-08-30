/**
* This file is part of OpenREALM.
*
* Copyright (C) 2020 Alexander Kern <laxnpander at gmail dot com> (Braunschweig University of Technology)
* For more information see <https://github.com/laxnpander/OpenREALM>
*
* OpenREALM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* OpenREALM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with OpenREALM. If not, see <http://www.gnu.org/licenses/>.
*/

#include <realm_io/cv_import.h>
#include <realm_io/utilities.h>

/*从文件加载图像数据*/
using namespace realm;

//用于从文件中加载图像数据
cv::Mat io::loadImage(const std::string &filepath)
{
  //检查文件是否存在
  if (!io::fileExists(filepath))
    throw(std::invalid_argument("Error loading image: File does not exist!"));
    //提取文件路径的后缀部分，以确定图像文件的格式。这里假设文件后缀为3个字符
  std::string suffix = filepath.substr(filepath.size()-3, 3);

  //如果后缀是 "png" 或 "jpg"，则使用 OpenCV 的 cv::imread 函数以未修改的形式读取图像文件
  if(suffix == "png" || suffix == "jpg")
    return cv::imread(filepath, cv::IMREAD_UNCHANGED);
  //如果后缀是 "bin"，则调用 loadImageFromBinary 函数以二进制格式加载图像数据
  else if (suffix == "bin")
    return loadImageFromBinary(filepath);
  else
    throw(std::invalid_argument("Error writing image: Unknown suffix"));
}

//用于从二进制文件中加载图像数据
cv::Mat io::loadImageFromBinary(const std::string &filepath)
{
  //使用 fopen 函数以二进制读取模式打开文件
  FILE* file = fopen(filepath.c_str(), "rb");
  //从文件中读取一个包含4个 int 型元素的头部信息，存储在 header 数组中
  int header[4];

  size_t elements_read;
  elements_read = fread(header, sizeof(int), 4, file);
  //检查从文件中读取的元素数量是否为4，如果不是则抛出异常 
  if (elements_read != 4)
    throw(std::runtime_error("Error reading binary: Elements read do not match matrix dimension!"));
    //根据头部信息提取图像的列数、行数、元素字节大小和数据类型
  int cols               = header[0];
  int rows               = header[1];
  int elem_size_in_bytes = header[2];
  int elem_type          = header[3];

  //创建一个大小为(rows, cols)、数据类型为elem_type的cv::Mat矩阵，表示图像数据
  cv::Mat data = cv::Mat::ones(rows, cols, elem_type);
  //使用 fread 函数从文件中读取图像数据，将数据写入到矩阵的内存中
  elements_read = fread(data.data, elem_size_in_bytes, (size_t)(cols * rows), file);
  //检查从文件中读取的元素数量是否与预期的元素数量相匹配
  if (elements_read != (size_t)(cols * rows))
    throw(std::runtime_error("Error reading binary: Elements read do not match matrix dimension!"));
    //关闭文件
  fclose(file);

  return data;
}