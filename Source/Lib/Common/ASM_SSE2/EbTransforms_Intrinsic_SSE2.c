/*
* Copyright(c) 2019 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

#include "EbDefinitions.h"
#include "EbIntrinMacros16bit_SSE2.h"
#include <emmintrin.h>

/*****************************
* Defines
*****************************/

#define MACRO_TRANS_2MAC_NO_SAVE(XMM_1, XMM_2, XMM_3, XMM_4, XMM_OFFSET, OFFSET1, OFFSET2, SHIFT)\
    XMM_3 = _mm_load_si128((__m128i *)(TransformAsmConst + OFFSET1));\
    XMM_4 = _mm_load_si128((__m128i *)(TransformAsmConst + OFFSET2));\
    XMM_3 = _mm_madd_epi16(XMM_3, XMM_1);\
    XMM_4 = _mm_madd_epi16(XMM_4, XMM_2);\
    XMM_3 = _mm_srai_epi32(_mm_add_epi32(XMM_4, _mm_add_epi32(XMM_3, XMM_OFFSET)), SHIFT);\
    XMM_3 = _mm_packs_epi32(XMM_3, XMM_3);

#define MACRO_TRANS_2MAC(XMM_1, XMM_2, XMM_3, XMM_4, XMM_OFFSET, OFFSET1, OFFSET2, SHIFT, OFFSET3)\
    MACRO_TRANS_2MAC_NO_SAVE(XMM_1, XMM_2, XMM_3, XMM_4, XMM_OFFSET, OFFSET1, OFFSET2, SHIFT)\
    _mm_storel_epi64((__m128i *)(transform_coefficients+OFFSET3), XMM_3);

#define TRANS8x8_OFFSET_83_36    0
#define TRANS8x8_OFFSET_36_N83  (8 + TRANS8x8_OFFSET_83_36)
#define TRANS8x8_OFFSET_89_75   (8 + TRANS8x8_OFFSET_36_N83)
#define TRANS8x8_OFFSET_50_18   (8 + TRANS8x8_OFFSET_89_75)
#define TRANS8x8_OFFSET_75_N18  (8 + TRANS8x8_OFFSET_50_18)
#define TRANS8x8_OFFSET_N89_N50 (8 + TRANS8x8_OFFSET_75_N18)
#define TRANS8x8_OFFSET_50_N89  (8 + TRANS8x8_OFFSET_N89_N50)
#define TRANS8x8_OFFSET_18_75   (8 + TRANS8x8_OFFSET_50_N89)
#define TRANS8x8_OFFSET_18_N50  (8 + TRANS8x8_OFFSET_18_75)
#define TRANS8x8_OFFSET_75_N89  (8 + TRANS8x8_OFFSET_18_N50)
#define TRANS8x8_OFFSET_256     (8 + TRANS8x8_OFFSET_75_N89)
#define TRANS8x8_OFFSET_64_64   (8 + TRANS8x8_OFFSET_256)
#define TRANS8x8_OFFSET_N18_N50 (8 + TRANS8x8_OFFSET_64_64)
#define TRANS8x8_OFFSET_N75_N89 (8 + TRANS8x8_OFFSET_N18_N50)
#define TRANS8x8_OFFSET_N36_N83 (8 + TRANS8x8_OFFSET_N75_N89)
#define TRANS8x8_OFFSET_N83_N36 (8 + TRANS8x8_OFFSET_N36_N83)
#define TRANS8x8_OFFSET_36_83   (8 + TRANS8x8_OFFSET_N83_N36)
#define TRANS8x8_OFFSET_50_89   (8 + TRANS8x8_OFFSET_36_83)
#define TRANS8x8_OFFSET_18_N75  (8 + TRANS8x8_OFFSET_50_89)
#define TRANS8x8_OFFSET_N64_64  (8 + TRANS8x8_OFFSET_18_N75)
#define TRANS8x8_OFFSET_64_N64  (8 + TRANS8x8_OFFSET_N64_64)
#define TRANS8x8_OFFSET_N75_N18 (8 + TRANS8x8_OFFSET_64_N64)
#define TRANS8x8_OFFSET_89_N50  (8 + TRANS8x8_OFFSET_N75_N18)
#define TRANS8x8_OFFSET_83_N36  (8 + TRANS8x8_OFFSET_89_N50)
#define TRANS8x8_OFFSET_N36_83  (8 + TRANS8x8_OFFSET_83_N36)
#define TRANS8x8_OFFSET_N83_36  (8 + TRANS8x8_OFFSET_N36_83)
#define TRANS8x8_OFFSET_89_N75  (8 + TRANS8x8_OFFSET_N83_36)
#define TRANS8x8_OFFSET_50_N18  (8 + TRANS8x8_OFFSET_89_N75)

#define MACRO_CALC_EVEN_ODD(XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7, XMM8)\
    even0 = _mm_add_epi16(XMM1, XMM8);\
    even1 = _mm_add_epi16(XMM2, XMM7);\
    even2 = _mm_add_epi16(XMM3, XMM6);\
    even3 = _mm_add_epi16(XMM4, XMM5);\
    odd0 = _mm_sub_epi16(XMM1, XMM8);\
    odd1 = _mm_sub_epi16(XMM2, XMM7);\
    odd2 = _mm_sub_epi16(XMM3, XMM6);\
    odd3 = _mm_sub_epi16(XMM4, XMM5);

#define MACRO_TRANS_4MAC_NO_SAVE(XMM1, XMM2, XMM3, XMM4, XMM_RET, XMM_OFFSET, MEM, OFFSET1, OFFSET2, SHIFT)\
    XMM_RET = _mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(_mm_add_epi32(_mm_madd_epi16(XMM1, _mm_load_si128((__m128i*)(MEM+OFFSET1))),\
                                                                         _mm_madd_epi16(XMM3, _mm_load_si128((__m128i*)(MEM+OFFSET2)))), XMM_OFFSET), SHIFT),\
                              _mm_srai_epi32(_mm_add_epi32(_mm_add_epi32(_mm_madd_epi16(XMM2, _mm_load_si128((__m128i*)(MEM+OFFSET1))),\
                                                                         _mm_madd_epi16(XMM4, _mm_load_si128((__m128i*)(MEM+OFFSET2)))), XMM_OFFSET), SHIFT));

#define MACRO_TRANS_8MAC(XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7, XMM8, XMM_OFST, MEM, OFST1, OFST2, OFST3, OFST4, SHIFT, INSTR, DST, OFST5)\
    sum1 = _mm_add_epi32(_mm_madd_epi16(XMM1, _mm_loadu_si128((__m128i *)(MEM + OFST1))), _mm_madd_epi16(XMM2, _mm_loadu_si128((__m128i *)(MEM + OFST2))));\
    sum2 = _mm_add_epi32(_mm_madd_epi16(XMM3, _mm_loadu_si128((__m128i *)(MEM + OFST3))), _mm_madd_epi16(XMM4, _mm_loadu_si128((__m128i *)(MEM + OFST4))));\
    sum1 = _mm_srai_epi32(_mm_add_epi32(XMM_OFST, _mm_add_epi32(sum1, sum2)), SHIFT);\
    sum3 = _mm_add_epi32(_mm_madd_epi16(XMM5, _mm_loadu_si128((__m128i *)(MEM + OFST1))), _mm_madd_epi16(XMM6, _mm_loadu_si128((__m128i *)(MEM + OFST2))));\
    sum4 = _mm_add_epi32(_mm_madd_epi16(XMM7, _mm_loadu_si128((__m128i *)(MEM + OFST3))), _mm_madd_epi16(XMM8, _mm_loadu_si128((__m128i *)(MEM + OFST4))));\
    sum3 = _mm_srai_epi32(_mm_add_epi32(XMM_OFST, _mm_add_epi32(sum3, sum4)), SHIFT);\
    sum = _mm_packs_epi32(sum1, sum3);\
    INSTR((__m128i *)(DST + OFST5), sum);

#define MACRO_TRANS_8MAC_PF_N2(XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7, XMM8, XMM_OFST, MEM, OFST1, OFST2, OFST3, OFST4, SHIFT, INSTR, DST, OFST5)\
    sum1 = _mm_add_epi32(_mm_madd_epi16(XMM1, _mm_loadu_si128((__m128i *)(MEM + OFST1))), _mm_madd_epi16(XMM2, _mm_loadu_si128((__m128i *)(MEM + OFST2))));\
    sum2 = _mm_add_epi32(_mm_madd_epi16(XMM3, _mm_loadu_si128((__m128i *)(MEM + OFST3))), _mm_madd_epi16(XMM4, _mm_loadu_si128((__m128i *)(MEM + OFST4))));\
    sum1 = _mm_srai_epi32(_mm_add_epi32(XMM_OFST, _mm_add_epi32(sum1, sum2)), SHIFT);\
    /*sum3 = _mm_add_epi32(_mm_madd_epi16(XMM5, _mm_loadu_si128((__m128i *)(MEM + OFST1))), _mm_madd_epi16(XMM6, _mm_loadu_si128((__m128i *)(MEM + OFST2))));*/\
    /*sum4 = _mm_add_epi32(_mm_madd_epi16(XMM7, _mm_loadu_si128((__m128i *)(MEM + OFST3))), _mm_madd_epi16(XMM8, _mm_loadu_si128((__m128i *)(MEM + OFST4))));*/\
    /*sum3 = _mm_srai_epi32(_mm_add_epi32(XMM_OFST, _mm_add_epi32(sum3, sum4)), SHIFT);*/\
    /*sum = _mm_packs_epi32(sum1, sum3);*/\
    sum = _mm_packs_epi32(sum1, sum1);\
    INSTR((__m128i *)(DST + OFST5), sum);
#define MACRO_TRANS_8MAC_PF_N4(XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7, XMM8, XMM_OFST, MEM, OFST1, OFST2, OFST3, OFST4, SHIFT, INSTR, DST, OFST5)\
    sum1 = _mm_add_epi32(_mm_madd_epi16(XMM1, _mm_loadu_si128((__m128i *)(MEM + OFST1))), _mm_madd_epi16(XMM2, _mm_loadu_si128((__m128i *)(MEM + OFST2))));\
    sum2 = _mm_add_epi32(_mm_madd_epi16(XMM3, _mm_loadu_si128((__m128i *)(MEM + OFST3))), _mm_madd_epi16(XMM4, _mm_loadu_si128((__m128i *)(MEM + OFST4))));\
    sum1 = _mm_srai_epi32(_mm_add_epi32(XMM_OFST, _mm_add_epi32(sum1, sum2)), SHIFT);\
    /*sum3 = _mm_add_epi32(_mm_madd_epi16(XMM5, _mm_loadu_si128((__m128i *)(MEM + OFST1))), _mm_madd_epi16(XMM6, _mm_loadu_si128((__m128i *)(MEM + OFST2))));*/\
    /*sum4 = _mm_add_epi32(_mm_madd_epi16(XMM7, _mm_loadu_si128((__m128i *)(MEM + OFST3))), _mm_madd_epi16(XMM8, _mm_loadu_si128((__m128i *)(MEM + OFST4))));*/\
    /*sum3 = _mm_srai_epi32(_mm_add_epi32(XMM_OFST, _mm_add_epi32(sum3, sum4)), SHIFT);*/\
    /*sum = _mm_packs_epi32(sum1, sum3);*/\
    sum = _mm_packs_epi32(sum1, sum1);\
    INSTR((__m128i *)(DST + OFST5), sum);

void PfreqTranspose32Type1_SSE2(
    int16_t *src,
    uint32_t  src_stride,
    int16_t *dst,
    uint32_t  dst_stride)
{
    uint32_t i, j;
    for (i = 0; i < 2; i++)
    {
        for (j = 0; j < 2; j++)
        {
            __m128i a0, a1, a2, a3, a4, a5, a6, a7;
            __m128i b0, b1, b2, b3, b4, b5, b6, b7;

            a0 = _mm_loadu_si128((const __m128i *)(src + (8 * i + 0)*src_stride + 8 * j));
            a1 = _mm_loadu_si128((const __m128i *)(src + (8 * i + 1)*src_stride + 8 * j));
            a2 = _mm_loadu_si128((const __m128i *)(src + (8 * i + 2)*src_stride + 8 * j));
            a3 = _mm_loadu_si128((const __m128i *)(src + (8 * i + 3)*src_stride + 8 * j));
            a4 = _mm_loadu_si128((const __m128i *)(src + (8 * i + 4)*src_stride + 8 * j));
            a5 = _mm_loadu_si128((const __m128i *)(src + (8 * i + 5)*src_stride + 8 * j));
            a6 = _mm_loadu_si128((const __m128i *)(src + (8 * i + 6)*src_stride + 8 * j));
            a7 = _mm_loadu_si128((const __m128i *)(src + (8 * i + 7)*src_stride + 8 * j));

            b0 = _mm_unpacklo_epi16(a0, a4);
            b1 = _mm_unpacklo_epi16(a1, a5);
            b2 = _mm_unpacklo_epi16(a2, a6);
            b3 = _mm_unpacklo_epi16(a3, a7);
            b4 = _mm_unpackhi_epi16(a0, a4);
            b5 = _mm_unpackhi_epi16(a1, a5);
            b6 = _mm_unpackhi_epi16(a2, a6);
            b7 = _mm_unpackhi_epi16(a3, a7);

            a0 = _mm_unpacklo_epi16(b0, b2);
            a1 = _mm_unpacklo_epi16(b1, b3);
            a2 = _mm_unpackhi_epi16(b0, b2);
            a3 = _mm_unpackhi_epi16(b1, b3);
            a4 = _mm_unpacklo_epi16(b4, b6);
            a5 = _mm_unpacklo_epi16(b5, b7);
            a6 = _mm_unpackhi_epi16(b4, b6);
            a7 = _mm_unpackhi_epi16(b5, b7);

            b0 = _mm_unpacklo_epi16(a0, a1);
            b1 = _mm_unpackhi_epi16(a0, a1);
            b2 = _mm_unpacklo_epi16(a2, a3);
            b3 = _mm_unpackhi_epi16(a2, a3);
            b4 = _mm_unpacklo_epi16(a4, a5);
            b5 = _mm_unpackhi_epi16(a4, a5);
            b6 = _mm_unpacklo_epi16(a6, a7);
            b7 = _mm_unpackhi_epi16(a6, a7);

            _mm_storeu_si128((__m128i *)(dst + (8 * j + 0)*dst_stride + 8 * i), b0);
            _mm_storeu_si128((__m128i *)(dst + (8 * j + 1)*dst_stride + 8 * i), b1);
            _mm_storeu_si128((__m128i *)(dst + (8 * j + 2)*dst_stride + 8 * i), b2);
            _mm_storeu_si128((__m128i *)(dst + (8 * j + 3)*dst_stride + 8 * i), b3);
            _mm_storeu_si128((__m128i *)(dst + (8 * j + 4)*dst_stride + 8 * i), b4);
            _mm_storeu_si128((__m128i *)(dst + (8 * j + 5)*dst_stride + 8 * i), b5);
            _mm_storeu_si128((__m128i *)(dst + (8 * j + 6)*dst_stride + 8 * i), b6);
            _mm_storeu_si128((__m128i *)(dst + (8 * j + 7)*dst_stride + 8 * i), b7);
        }
    }
}
