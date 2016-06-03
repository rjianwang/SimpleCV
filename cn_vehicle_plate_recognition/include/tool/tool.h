/* \file tool.h
 * Define some utilites
 */

#include <stdlib.h>
#include <vector>
#include <string>

#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>

#include "../stdafx.h"

/* \namespace pr
 * Namespace where all C++ Plate Recognition functionality resides
 */
namespace pr
{

class Char;

// get items in the directory "filepath"
std::vector<std::string> getFiles(const std::string filepath);
// An implementation of qsort
void qsort(std::vector<Char> &input, int low, int high);

} /* end for namespace pr */
