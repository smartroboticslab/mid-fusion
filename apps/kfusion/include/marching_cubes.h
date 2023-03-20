/* Copyright (c) 2011-2013 Gerhard Reitmayr, TU Graz
 * Copyright (c) 2013 Jan Jachnik
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */
//
// This file is added to the project by hh1013
//
#ifndef MARCHING_CUBES_H
#define MARCHING_CUBES_H

#include <commons.h>
#include <volume.hpp>
#include <string>
#include <stdint.h>

struct triangle
{
    float3 vertex[3];
};

struct coloured_vertex{
    float3 pt;
    uchar3 col;

    coloured_vertex(float3 _pt, uchar3 _col)
    {
        pt = _pt;
        col = _col;
    }

    coloured_vertex(){}
};

struct coloured_triangle
{
    coloured_vertex vertex[3];
};



uint8_t getCubeIndex(int x, int y, int z, const Volume & vol);
float3 LinearInterpolate(uint3 a, uint3 b, const Volume & vol);
coloured_vertex LinearInterpolateColor(uint3 a, uint3 b, const Volume & vol, const RGBVolume & pho);
float3 calcPt(int edge, int x, int y, int z, const Volume & vol);

void marchingCubes(const Volume vol, std::vector<Triangle>& triangles, 
                   std::string filename);
void marchingCubesColor(const Volume vol, const RGBVolume pho, std::string filename);

#endif
