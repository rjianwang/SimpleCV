#include "StdAfx.h"

#include "stdlib.h"
#include <vector>
#include <string>

#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>

#include "OCR.h"

class Util
{
public:
    static std::vector<std::string> getFiles(const std::string filepath);

public:
    static int compare(const CharSegment *a, const CharSegment *b)
    {
        return (a->pos.x - b->pos.x);
    }
};
