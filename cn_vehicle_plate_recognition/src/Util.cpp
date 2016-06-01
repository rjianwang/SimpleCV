#include "Util.h"

std::vector<std::string> Util::getFiles(const std::string filepath)
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

void Util::qsort(std::vector<CharSegment> &input, int low, int high)
{
    int i = low, j = high;
    CharSegment temp = input[low];

    while (i < j)
    {
        while (i < j && temp.pos.x <= input[j].pos.x) j--;
        if (i < j)
        {
            input[i] = input[j];
            i++;
        }
        while (i < j && temp.pos.x > input[i].pos.x) i++;
        if (i < j)
        {
            input[j] = input[i];
            j--;
        }
    }
    input[i] = temp;

    if (low < i) qsort(input, low, i - 1);
    if (i < high) qsort(input, j + 1, high);
}
