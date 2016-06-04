#include <string>
#include <unistd.h>
#include "../include/stdafx.h"
#include "../include/tool/tool.h"

using namespace pr;

int main(int argc, char* argv[])
{
    std::string filepath = "../data/plates/";
    std::vector<std::string> files = getFiles(filepath);
    for (int i = 0; i < files.size(); i++)
    {
        std::string cmd = "./bin/plate_recognition -detect ";
        cmd = cmd + filepath + files[i];
        system(cmd.c_str());
        sleep(1);
    }
    return 0;
}
