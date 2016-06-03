/* \file resource.h
 * Includes some constant
 */

#pragma once

#include <string>
#include <iostream>

/* \namespace pr
 * Namespace where all C++ Plate Recognition functionality resides
 */
namespace pr
{

/* \class Resources
 * Definition of some constant
 */
class Resources
{
public:
    const static char chars[];      // digits & letters
    const static int numCharacters; // number of digits & digits

    const static char sp_chars[];
    const static int numSPCharacters;

    const static std::string cn_chars[]; // Chinese characters
    const static int numCNCharacters;    // number of Chinese characters
}; /* end for class Resources */

} /* end for namespace pr */
