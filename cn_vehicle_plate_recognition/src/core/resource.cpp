/* \file resource.cpp
 * Includes some constants
 */

#include "../../include/core/resource.h"

/* \namespace pr
 * Namespace where all C++ Plate Recognition functionality resides
 */
namespace pr
{

const char Resources::chars[] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z' };
const int Resources::numCharacters = 34;

const char Resources::sp_chars[] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z' };
const int Resources::numSPCharacters = 30;


const std::string Resources::cn_chars[] = {"京", "冀", "吉", "宁", "川", "晋", "桂", "沪", "津", "浙", "湘", "琼", "皖", "粤", "苏", "豫", "贵", "赣", "辽", "鄂", "闽", "陕", "鲁", "黑"};
const int Resources::numCNCharacters = 24;

} /* end for namespace pr */
