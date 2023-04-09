#include <iostream>
// #include <bits/stdc++.h>
using namespace std;

int xnor_popcount(int a, int b)
{
    return __builtin_popcount(~(a^b));
}

/*
* Estimate the number of SIMD operations assuming DWH alignment
*/
int estimate_SIMD_ops(int width, int height, int in_depth, int out_depth, int tile_width, int tile_height, int tile_depth) {
    return ceil(width/tile_width)*ceil(tile_width/2)*ceil(height/tile_height)*ceil(tile_height/2)*
           ceil(out_depth/tile_depth)*ceil(tile_depth/64)*9*in_depth;
}

/* Todo */
float* zero_pad_fp();
int* zero_pad_bn();
