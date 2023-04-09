#include <stdint.h>
#include <immintrin.h>

typedef struct {
    unsigned int b0: 1;
    unsigned int b1: 1;
    unsigned int b2: 1;
    unsigned int b3: 1;
    unsigned int b4: 1;
    unsigned int b5: 1;
    unsigned int b6: 1;
    unsigned int b7: 1;
    unsigned int b8: 1;
    unsigned int b9: 1;
    unsigned int b10: 1;
    unsigned int b11: 1;
    unsigned int b12: 1;
    unsigned int b13: 1;
    unsigned int b14: 1;
    unsigned int b15: 1;
    unsigned int b16: 1;
    unsigned int b17: 1;
    unsigned int b18: 1;
    unsigned int b19: 1;
    unsigned int b20: 1;
    unsigned int b21: 1;
    unsigned int b22: 1;
    unsigned int b23: 1;
    unsigned int b24: 1;
    unsigned int b25: 1;
    unsigned int b26: 1;
    unsigned int b27: 1;
    unsigned int b28: 1;
    unsigned int b29: 1;
    unsigned int b30: 1;
    unsigned int b31: 1;
    unsigned int b32: 1;
    unsigned int b33: 1;
    unsigned int b34: 1;
    unsigned int b35: 1;
    unsigned int b36: 1;
    unsigned int b37: 1;
    unsigned int b38: 1;
    unsigned int b39: 1;
    unsigned int b40: 1;
    unsigned int b41: 1;
    unsigned int b42: 1;
    unsigned int b43: 1;
    unsigned int b44: 1;
    unsigned int b45: 1;
    unsigned int b46: 1;
    unsigned int b47: 1;
    unsigned int b48: 1;
    unsigned int b49: 1;
    unsigned int b50: 1;
    unsigned int b51: 1;
    unsigned int b52: 1;
    unsigned int b53: 1;
    unsigned int b54: 1;
    unsigned int b55: 1;
    unsigned int b56: 1;
    unsigned int b57: 1;
    unsigned int b58: 1;
    unsigned int b59: 1;
    unsigned int b60: 1;
    unsigned int b61: 1;
    unsigned int b62: 1;
    unsigned int b63: 1;
    unsigned int b64: 1;
} bit64_t;

typedef union{
    bit64_t b;
    uint64_t u;
} bit64_u;


// used in SSE
typedef union{
    __m128i m;
    int64_t i[2];
} m128_u;

// used in AVX256
typedef union{
    __m256i m;
    int64_t i[4];
} m256_u;
           
// used in AVX512
typedef union{
    __m256i m;
    int64_t i[8];
} m512_u;

void bitpack_binarization_input(float* inputs, bit64_u* outputs, ) {

}

void bitpack_binarization_weight(float* B, bit64_u* Bb, int k, int n) {
    float *p;
    int i, j;
    bit64_u bit64;
    for (j = 0; j < k; j+=1)
    {
        for (i = 0; i < n; i+=64)
        {
        p = &B[i*k+j];
        // fuse bit-packing into binarization
        bit64.b.b0  = p[0]   <=0.0f;
        bit64.b.b1  = p[1*k] <=0.0f;
        bit64.b.b2  = p[2*k] <=0.0f;
        bit64.b.b3  = p[3*k] <=0.0f;
        bit64.b.b4  = p[4*k] <=0.0f;
        bit64.b.b5  = p[5*k] <=0.0f;
        bit64.b.b6  = p[6*k] <=0.0f;
        bit64.b.b7  = p[7*k] <=0.0f;
        bit64.b.b8  = p[8*k] <=0.0f;
        bit64.b.b9  = p[9*k] <=0.0f;
        bit64.b.b10 = p[10*k]<=0.0f;
        bit64.b.b11 = p[11*k]<=0.0f;
        bit64.b.b12 = p[12*k]<=0.0f;
        bit64.b.b13 = p[13*k]<=0.0f;
        bit64.b.b14 = p[14*k]<=0.0f;
        bit64.b.b15 = p[15*k]<=0.0f;
        bit64.b.b16 = p[16*k]<=0.0f;
        bit64.b.b17 = p[17*k]<=0.0f;
        bit64.b.b18 = p[18*k]<=0.0f;
        bit64.b.b19 = p[19*k]<=0.0f;
        bit64.b.b20 = p[20*k]<=0.0f;
        bit64.b.b21 = p[21*k]<=0.0f;
        bit64.b.b22 = p[22*k]<=0.0f;
        bit64.b.b23 = p[23*k]<=0.0f;
        bit64.b.b24 = p[24*k]<=0.0f;
        bit64.b.b25 = p[25*k]<=0.0f;
        bit64.b.b26 = p[26*k]<=0.0f;
        bit64.b.b27 = p[27*k]<=0.0f;
        bit64.b.b28 = p[28*k]<=0.0f;
        bit64.b.b29 = p[29*k]<=0.0f;
        bit64.b.b30 = p[30*k]<=0.0f;
        bit64.b.b31 = p[31*k]<=0.0f;
        bit64.b.b32 = p[32*k]<=0.0f;
        bit64.b.b33 = p[33*k]<=0.0f;
        bit64.b.b34 = p[34*k]<=0.0f;
        bit64.b.b35 = p[35*k]<=0.0f;
        bit64.b.b36 = p[36*k]<=0.0f;
        bit64.b.b37 = p[37*k]<=0.0f;
        bit64.b.b38 = p[38*k]<=0.0f;
        bit64.b.b39 = p[39*k]<=0.0f;
        bit64.b.b40 = p[40*k]<=0.0f;
        bit64.b.b41 = p[41*k]<=0.0f;
        bit64.b.b42 = p[42*k]<=0.0f;
        bit64.b.b43 = p[43*k]<=0.0f;
        bit64.b.b44 = p[44*k]<=0.0f;
        bit64.b.b45 = p[45*k]<=0.0f;
        bit64.b.b46 = p[46*k]<=0.0f;
        bit64.b.b47 = p[47*k]<=0.0f;
        bit64.b.b48 = p[48*k]<=0.0f;
        bit64.b.b49 = p[49*k]<=0.0f;
        bit64.b.b50 = p[50*k]<=0.0f;
        bit64.b.b51 = p[51*k]<=0.0f;
        bit64.b.b52 = p[52*k]<=0.0f;
        bit64.b.b53 = p[53*k]<=0.0f;
        bit64.b.b54 = p[54*k]<=0.0f;
        bit64.b.b55 = p[55*k]<=0.0f;
        bit64.b.b56 = p[56*k]<=0.0f;
        bit64.b.b57 = p[57*k]<=0.0f;
        bit64.b.b58 = p[58*k]<=0.0f;
        bit64.b.b59 = p[59*k]<=0.0f;
        bit64.b.b58 = p[60*k]<=0.0f;
        bit64.b.b59 = p[61*k]<=0.0f;
        bit64.b.b62 = p[62*k]<=0.0f;
        bit64.b.b63 = p[63*k]<=0.0f;
        // do transposition implicitly
        Bb[(j*n+i)>>6].u = bit64.u;
    } }
}