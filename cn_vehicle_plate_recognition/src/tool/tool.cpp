/* \file tool.h
 * Define some utilities
 */

#include <iostream>
#include <string>
#include "../../include/core/ocr.h"
#include "../../include/core/char.h"
#include "../../include/tool/tool.h"

/* \namespace pr
 * Namespace where all C++ Plate Recognition functionality residies
 */
namespace pr
{

// Get items in the directory "filepath"
std::vector<std::string> getFiles(const std::string filepath)
{
    std::vector<std::string> files;
    DIR *dir;
    struct dirent *file;

    if (!(dir = opendir(filepath.c_str())))
    {
        std::cout << "Invalid file path: " 
            << filepath
            << std::endl;
    }

    while ((file = readdir(dir)) != NULL)
    {
        struct stat sb;
        stat(file->d_name, &sb);
        if (strcmp(".", file->d_name) == 0 || strcmp("..", file->d_name) == 0)
            continue;
        std::string filename = file->d_name;  
        files.push_back(filename);
    }

    closedir(dir);

    return files;
}

// An implementation of qsort
void qsort(std::vector<Char> &input, int low, int high)
{
    int i = low, j = high;
    Char temp = input[low];

    while (i < j)
    {
        while (i < j && temp.position.x <= input[j].position.x) j--;
        if (i < j)
        {
            input[i] = input[j];
            i++;
        }
        while (i < j && temp.position.x > input[i].position.x) i++;
        if (i < j)
        {
            input[j] = input[i];
            j--;
        }
    }
    input[i] = temp;

    if (low < i) qsort(input, low, i - 1);
    if (i < high) qsort(input, i + 1, high);
}

} /* end for namespace pr */
