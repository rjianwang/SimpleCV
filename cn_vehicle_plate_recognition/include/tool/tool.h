
#include <stdlib.h>
#include <vector>
#include <string>

#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>

#include "../stdafx.h"

class CharSegment;

class Util
{
public:
    static std::vector<std::string> getFiles(const std::string filepath);

public:
    static void qsort(std::vector<CharSegment> &input, int low, int high);
};
