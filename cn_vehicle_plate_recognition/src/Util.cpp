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
