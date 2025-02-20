#ifndef MATRIX_TESTS_ISA_LOAD_STORE_H_
#define MATRIX_TESTS_ISA_LOAD_STORE_H_

#include "riscv_matrix.h"
#include "riscv_vector.h"
#include "utils.h"

const int8_t ls_i8_src[8 * 8] = {
    -104, -10,  -90,  3,   94,   -2,  -118, 29,   -85, 38,  -33, 9,    71,
    78,   -99,  60,   -24, -74,  105, 113,  -109, 6,   -90, -11, -121, 11,
    36,   43,   -119, 2,   -112, 93,  75,   -81,  -33, 93,  76,  -37,  -22,
    -48,  -111, 3,    -20, -95,  1,   35,   118,  61,  -17, 44,  -67,  32,
    26,   57,   99,   125, -111, -20, 47,   18,   58,  -91, 101, -55};

const uint8_t ls_u8_src[8 * 8] = {
    207, 5,   111, 236, 252, 228, 169, 233, 196, 212, 163, 92,  6,
    194, 210, 0,   242, 18,  253, 241, 81,  128, 47,  76,  41,  77,
    215, 243, 208, 103, 191, 63,  149, 116, 231, 147, 170, 118, 71,
    138, 225, 82,  183, 157, 253, 118, 235, 167, 164, 10,  14,  234,
    1,   185, 117, 168, 203, 15,  218, 155, 24,  226, 213, 42};

const int16_t ls_i16_src[8 * 8] = {
    -23997, 17495,  -28277, -12330, 32211,  -29618, 32313,  6838,
    11880,  -12599, 26269,  16085,  9099,   12733,  -3749,  24365,
    9992,   18387,  3481,   -24222, 31931,  -15238, 31074,  -15059,
    -10926, 31700,  20625,  12431,  -27996, 1202,   -17818, -17851,
    12653,  27145,  -28709, 2286,   -3347,  -31529, 23882,  -9706,
    27081,  -18867, 19059,  15439,  -21660, -185,   -17332, 28000,
    -28951, 1149,   -19049, 3786,   14214,  -1826,  -30139, -3204,
    18679,  -3943,  -7297,  235,    4739,   -13147, 31919,  -7128};

const uint16_t ls_u16_src[8 * 8] = {
    47635, 32078, 30437, 7700,  26787, 38769, 29389, 27496, 12741, 59672, 13672,
    41327, 1826,  50319, 40691, 9741,  54293, 61231, 31082, 62098, 22621, 18484,
    1702,  26018, 64471, 62370, 22739, 10504, 8138,  21625, 58047, 51833, 48798,
    17443, 50133, 31376, 25471, 44740, 12180, 47240, 8082,  19719, 36103, 34577,
    43767, 5283,  41783, 17760, 52160, 46022, 44706, 10810, 38435, 38969, 32268,
    48021, 48445, 39969, 42184, 18168, 58035, 5370,  24788, 3211};

const int32_t ls_i32_src[8 * 4] = {
    -1989395023, -371395999,  -292694948,  -715124779, -1585026718, 1453185758,
    928105862,   1593085348,  -1020204260, 1818520508, 124686649,   1130429991,
    -136089165,  -1848214585, 438467929,   -569797084, 1064847958,  -550645351,
    -2078563007, -1463696107, -104057128,  -810714651, 2077413965,  669938450,
    -352468570,  -458989738,  -1327085764, 1306514821, -855276358,  -296410604,
    -706236095,  18622971};

const uint32_t ls_u32_src[8 * 4] = {
    304998111,  3717196571, 3997307068, 2196761274, 4077833894, 1123632122,
    482662781,  3793536030, 3539736721, 3337514271, 1294703199, 180555887,
    2055538577, 2911995115, 2112284567, 2889961585, 563278589,  517327472,
    1061103040, 3309178420, 3184042195, 3132004753, 181300616,  1147395575,
    967238162,  591859874,  1287156978, 1108163567, 608611587,  2098800955,
    4179820169, 3222669076};

const int64_t ls_i64_src[8 * 2] = {
    3470500552315383091, -7157888958374847902, 7417905495295585108,
    5202680171157528021, -3514936454023114372, 5441021249183388904,
    5987232322858384342, -1005246747348027534, 4487668338078190916,
    6755468198055268843, -2217304082079286145, 667046590290870190,
    1892091047668225348, -6330731506964637536, 4906681605308898160,
    -4973348527113500138};

const uint64_t ls_u64_src[8 * 2] = {
    3124289805900100564ull,  4529684755180831705ull,  5699014898879032867ull,
    16070993490469498570ull, 9189157692509778032ull,  8719509931675924159ull,
    14843607066879663471ull, 13066048337374181024ull, 17163888103613926985ull,
    3136496091786589846ull,  2089865526332178830ull,  2592717781086028892ull,
    9708187782959804916ull,  14020885944247640632ull, 3488625519223632894ull,
    2736891790469435972ull};

const fp16_t ls_f16_src[8 * 8] = {
    -79.3,  26.98,  -8.516, -61.25, 35.2,   -19.08, 79.4,   63.78,
    98.7,   -25.08, -81.7,  -50.16, -1.693, -60.3,  -74.9,  33.7,
    -41.03, 67.25,  85.1,   61.94,  -48.88, -30.42, 1.392,  -52.62,
    71.94,  58.9,   68.06,  -1.808, 90.44,  61.72,  6.324,  16.17,
    33.25,  84.56,  -88.7,  -62.25, -35.12, 55.,    0.9956, 58.7,
    -4.574, 4.426,  -36.7,  -5.,    2.719,  -78.56, 40.25,  -78.5,
    78.44,  -87.06, 3.79,   -11.47, 17.22,  -77.8,  60.1,   -38.4,
    97.2,   -89.6,  -23.44, -49.06, 20.64,  -54.56, -45.94, 4.516};

const fp32_t ls_f32_src[8 * 4] = {
    32.07479,   85.11509,  63.82503,   -34.532795, 89.74828,   2.2306035,
    -84.539314, -67.7678,  -38.8788,   43.48091,   7.0056033,  -74.30909,
    85.49286,   21.704903, 58.53529,   -78.33169,  16.63223,   -68.76382,
    92.94964,   26.658998, 41.838226,  -68.62157,  -23.913134, -68.00235,
    -6.499663,  -58.17478, 0.13445568, 77.884834,  82.459045,  0.11977486,
    86.96803,   94.97192};

const fp64_t ls_f64_src[8 * 2] = {
    -38.73212158, 8.40068948,   -63.09381612, 52.63072007,
    40.37105744,  89.67115255,  70.98073472,  -72.24281866,
    59.04989384,  -73.13543867, -62.78058286, -93.13538432,
    68.52618759,  8.6628601,    42.43194053,  -22.04015233};

static void test_mla_m_msa_m_i8() {
  enum { M = 8, K = 8 };
  const size_t stride = K * sizeof(int8_t);
  SET_MBA0_I8();
  msettilem(M);
  msettilek(K);
  mint8_t ms = mla_m(ls_i8_src, stride);
  msa_m(ms, i8_buffer, stride);
  EXCEPT_I8_ARRAY_EQ(ls_i8_src, i8_buffer, M * K, "MLA & MSA I8 ");
}

static void test_mla_m_msa_m_u8() {
  enum { M = 8, K = 8 };
  const size_t stride = K * sizeof(uint8_t);
  SET_MBA0_I8();
  msettilem(M);
  msettilek(K);
  muint8_t ms = mla_m(ls_u8_src, stride);
  msa_m(ms, u8_buffer, stride);
  EXCEPT_U8_ARRAY_EQ(ls_u8_src, u8_buffer, M * K, "MLA & MSA U8 ");
}

static void test_mla_m_msa_m_i16() {
  enum { M = 8, K = 8 };
  const size_t stride = K * sizeof(int16_t);
  SET_MBA0_I16();
  msettilem(M);
  msettilek(K);
  mint16_t ms = mla_m(ls_i16_src, stride);
  msa_m(ms, i16_buffer, stride);
  EXCEPT_I16_ARRAY_EQ(ls_i16_src, i16_buffer, M * K, "MLA & MSA I16");
}

static void test_mla_m_msa_m_u16() {
  enum { M = 8, K = 8 };
  const size_t stride = K * sizeof(uint16_t);
  SET_MBA0_I16();
  msettilem(M);
  msettilek(K);
  muint16_t ms = mla_m(ls_u16_src, stride);
  msa_m(ms, u16_buffer, stride);
  EXCEPT_U16_ARRAY_EQ(ls_u16_src, u16_buffer, M * K, "MLA & MSA U16");
}

static void test_mla_m_msa_m_f16() {
  enum { M = 8, K = 8 };
  const size_t stride = K * sizeof(fp16_t);
  SET_MBA0_F16();
  msettilem(M);
  msettilek(K);
  mfloat16_t ms = mla_m(ls_f16_src, stride);
  msa_m(ms, f16_buffer, stride);
  EXCEPT_F16_ARRAY_EQ(ls_f16_src, f16_buffer, M * K, "MLA & MSA F16");
}

static void test_mla_m_msa_m_i32() {
  enum { M = 8, K = 4 };
  const size_t stride = K * sizeof(int32_t);
  SET_MBA0_I32();
  msettilem(M);
  msettilek(K);
  mint32_t ms = mla_m(ls_i32_src, stride);
  msa_m(ms, i32_buffer, stride);
  EXCEPT_I32_ARRAY_EQ(ls_i32_src, i32_buffer, M * K, "MLA & MSA I32");
}

static void test_mla_m_msa_m_u32() {
  enum { M = 8, K = 4 };
  const size_t stride = K * sizeof(uint32_t);
  SET_MBA0_I32();
  msettilem(M);
  msettilek(K);
  muint32_t ms = mla_m(ls_u32_src, stride);
  msa_m(ms, u32_buffer, stride);
  EXCEPT_U32_ARRAY_EQ(ls_u32_src, u32_buffer, M * K, "MLA & MSA U32");
}

static void test_mla_m_msa_m_f32() {
  enum { M = 8, K = 4 };
  const size_t stride = K * sizeof(fp32_t);
  SET_MBA0_F32();
  msettilem(M);
  msettilek(K);
  mfloat32_t ms = mla_m(ls_f32_src, stride);
  msa_m(ms, f32_buffer, stride);
  EXCEPT_F32_ARRAY_EQ(ls_f32_src, f32_buffer, M * K, "MLA & MSA F32");
}

static void test_mla_m_msa_m_i64() {
  const size_t M = 8;
  const size_t K = 2;
  const size_t stride = K * sizeof(int64_t);
  SET_MBA0_I64();
  msettilem(M);
  msettilek(K);
  mint64_t ms = mla_m(ls_i64_src, stride);
  msa_m(ms, i64_buffer, stride);
  EXCEPT_I64_ARRAY_EQ(ls_i64_src, i64_buffer, M * K, "MLA & MSA I64");
}

static void test_mla_m_msa_m_u64() {
  const size_t M = 8;
  const size_t K = 2;
  const size_t stride = K * sizeof(uint64_t);
  SET_MBA0_I64();
  msettilem(M);
  msettilek(K);
  muint64_t ms = mla_m(ls_u64_src, stride);
  msa_m(ms, u64_buffer, stride);
  EXCEPT_U64_ARRAY_EQ(ls_u64_src, u64_buffer, M * K, "MLA & MSA U64");
}

static void test_mla_m_msa_m_f64() {
  const size_t M = 8;
  const size_t K = 2;
  const size_t stride = K * sizeof(fp64_t);
  SET_MBA0_F64();
  msettilem(M);
  msettilek(K);
  mfloat64_t ms = mla_m(ls_f64_src, stride);
  msa_m(ms, f64_buffer, stride);
  EXCEPT_F64_ARRAY_EQ(ls_f64_src, f64_buffer, M * K, "MLA & MSA F64");
}

static void test_mlb_m_msb_m_i8() {
  const size_t K = 8;
  const size_t N = 8;
  const size_t stride = N * sizeof(int8_t);
  SET_MBA0_I8();
  msettilek(K);
  msettilen(N);
  mint8_t ms = mlb_m(ls_i8_src, stride);
  msb_m(ms, i8_buffer, stride);
  EXCEPT_I8_ARRAY_EQ(ls_i8_src, i8_buffer, K * N, "MLB & MSB I8 ");
}

static void test_mlb_m_msb_m_u8() {
  const size_t K = 8;
  const size_t N = 8;
  const size_t stride = N * sizeof(uint8_t);
  SET_MBA0_I8();
  msettilek(K);
  msettilen(N);
  muint8_t ms = mlb_m(ls_u8_src, stride);
  msb_m(ms, u8_buffer, stride);
  EXCEPT_U8_ARRAY_EQ(ls_u8_src, u8_buffer, K * N, "MLB & MSB U8 ");
}

static void test_mlb_m_msb_m_i16() {
  const size_t K = 8;
  const size_t N = 8;
  const size_t stride = N * sizeof(int16_t);
  SET_MBA0_I16();
  msettilek(K);
  msettilen(N);
  mint16_t ms = mlb_m(ls_i16_src, stride);
  msb_m(ms, i16_buffer, stride);
  EXCEPT_I16_ARRAY_EQ(ls_i16_src, i16_buffer, K * N, "MLB & MSB I16");
}

static void test_mlb_m_msb_m_u16() {
  const size_t K = 8;
  const size_t N = 8;
  const size_t stride = N * sizeof(uint16_t);
  SET_MBA0_I16();
  msettilek(K);
  msettilen(N);
  muint16_t ms = mlb_m(ls_u16_src, stride);
  msb_m(ms, u16_buffer, stride);
  EXCEPT_U16_ARRAY_EQ(ls_u16_src, u16_buffer, K * N, "MLB & MSB U16");
}

static void test_mlb_m_msb_m_f16() {
  const size_t K = 8;
  const size_t N = 8;
  const size_t stride = N * sizeof(fp16_t);
  SET_MBA0_F16();
  msettilek(K);
  msettilen(N);
  mfloat16_t ms = mlb_m(ls_f16_src, stride);
  msb_m(ms, f16_buffer, stride);
  EXCEPT_F16_ARRAY_EQ(ls_f16_src, f16_buffer, K * N, "MLB & MSB F16");
}

static void test_mlb_m_msb_m_i32() {
  const size_t K = 4;
  const size_t N = 4;
  const size_t stride = N * sizeof(int32_t);
  SET_MBA0_I32();
  msettilek(K);
  msettilen(N);
  mint32_t ms = mlb_m(ls_i32_src, stride);
  msb_m(ms, i32_buffer, stride);
  EXCEPT_I32_ARRAY_EQ(ls_i32_src, i32_buffer, K * N, "MLB & MSB I32");
}

static void test_mlb_m_msb_m_u32() {
  const size_t K = 4;
  const size_t N = 4;
  const size_t stride = N * sizeof(uint32_t);
  SET_MBA0_I32();
  msettilek(K);
  msettilen(N);
  muint32_t ms = mlb_m(ls_u32_src, stride);
  msb_m(ms, u32_buffer, stride);
  EXCEPT_U32_ARRAY_EQ(ls_u32_src, u32_buffer, K * N, "MLB & MSB U32");
}

static void test_mlb_m_msb_m_f32() {
  const size_t K = 4;
  const size_t N = 4;
  const size_t stride = N * sizeof(fp32_t);
  SET_MBA0_F32();
  msettilek(K);
  msettilen(N);
  mfloat32_t ms = mlb_m(ls_f32_src, stride);
  msb_m(ms, f32_buffer, stride);
  EXCEPT_F32_ARRAY_EQ(ls_f32_src, f32_buffer, K * N, "MLB & MSB F32");
}

static void test_mlb_m_msb_m_i64() {
  const size_t K = 2;
  const size_t N = 2;
  const size_t stride = N * sizeof(int64_t);
  SET_MBA0_I64();
  msettilek(K);
  msettilen(N);
  mint64_t ms = mlb_m(ls_i64_src, stride);
  msb_m(ms, i64_buffer, stride);
  EXCEPT_I64_ARRAY_EQ(ls_i64_src, i64_buffer, K * N, "MLB & MSB I64");
}

static void test_mlb_m_msb_m_u64() {
  const size_t K = 2;
  const size_t N = 2;
  const size_t stride = N * sizeof(uint64_t);
  SET_MBA0_I64();
  msettilek(K);
  msettilen(N);
  muint64_t ms = mlb_m(ls_u64_src, stride);
  msb_m(ms, u64_buffer, stride);
  EXCEPT_U64_ARRAY_EQ(ls_u64_src, u64_buffer, K * N, "MLB & MSB U64");
}

static void test_mlb_m_msb_m_f64() {
  const size_t K = 2;
  const size_t N = 2;
  const size_t stride = N * sizeof(fp64_t);
  SET_MBA0_F64();
  msettilek(K);
  msettilen(N);
  mfloat64_t ms = mlb_m(ls_f64_src, stride);
  msb_m(ms, f64_buffer, stride);
  EXCEPT_F64_ARRAY_EQ(ls_f64_src, f64_buffer, K * N, "MLB & MSB F64");
}

static void test_mlc_m_msc_m_i8() {
  const size_t M = 8;
  const size_t N = 16;
  const size_t stride = N * sizeof(int8_t);
  SET_MBA0_I8();
  msettilem(M);
  msettilen(N);
  mint8_t ms = mlc_m(ls_i8_src, stride);
  msc_m(ms, i8_buffer, stride);
  EXCEPT_I8_ARRAY_EQ(ls_i8_src, i8_buffer, M * N, "MLC & MSC I8 ");
}

static void test_mlc_m_msc_m_u8() {
  const size_t M = 8;
  const size_t N = 16;
  const size_t stride = N * sizeof(uint8_t);
  SET_MBA0_I8();
  msettilem(M);
  msettilen(N);
  muint8_t ms = mlc_m(ls_u8_src, stride);
  msc_m(ms, u8_buffer, stride);
  EXCEPT_U8_ARRAY_EQ(ls_u8_src, u8_buffer, M * N, "MLC & MSC U8 ");
}

static void test_mlc_m_msc_m_i16() {
  enum { M = 8, N = 8 };
  const size_t stride = N * sizeof(int16_t);
  SET_MBA0_I16();
  msettilem(M);
  msettilen(N);
  mint16_t ms = mlc_m(ls_i16_src, stride);
  msc_m(ms, i16_buffer, stride);
  EXCEPT_I16_ARRAY_EQ(ls_i16_src, i16_buffer, M * N, "MLC & MSC I16");
}

static void test_mlc_m_msc_m_u16() {
  enum { M = 8, N = 8 };
  const size_t stride = N * sizeof(uint16_t);
  SET_MBA0_I16();
  msettilem(M);
  msettilen(N);
  muint16_t ms = mlc_m(ls_u16_src, stride);
  msc_m(ms, u16_buffer, stride);
  EXCEPT_U16_ARRAY_EQ(ls_u16_src, u16_buffer, M * N, "MLC & MSC U16");
}

static void test_mlc_m_msc_m_f16() {
  enum { M = 8, N = 8 };
  const size_t stride = N * sizeof(fp16_t);
  SET_MBA0_F16();
  msettilem(M);
  msettilen(N);
  mfloat16_t ms = mlc_m(ls_f16_src, stride);
  msc_m(ms, f16_buffer, stride);
  EXCEPT_F16_ARRAY_EQ(ls_f16_src, f16_buffer, M * N, "MLC & MSC F16");
}

static void test_mlc_m_msc_m_i32() {
  const size_t M = 8;
  const size_t N = 4;
  const size_t stride = N * sizeof(int32_t);
  SET_MBA0_I32();
  msettilem(M);
  msettilen(N);
  mint32_t ms = mlc_m(ls_i32_src, stride);
  msc_m(ms, i32_buffer, stride);
  EXCEPT_I32_ARRAY_EQ(ls_i32_src, i32_buffer, M * N, "MLC & MSC I32");
}

static void test_mlc_m_msc_m_u32() {
  const size_t M = 8;
  const size_t N = 4;
  const size_t stride = N * sizeof(uint32_t);
  SET_MBA0_I32();
  msettilem(M);
  msettilen(N);
  muint32_t ms = mlc_m(ls_u32_src, stride);
  msc_m(ms, u32_buffer, stride);
  EXCEPT_U32_ARRAY_EQ(ls_u32_src, u32_buffer, M * N, "MLC & MSC U32");
}

static void test_mlc_m_msc_m_f32() {
  const size_t M = 8;
  const size_t N = 4;
  const size_t stride = N * sizeof(fp32_t);
  SET_MBA0_F32();
  msettilem(M);
  msettilen(N);
  mfloat32_t ms = mlc_m(ls_f32_src, stride);
  msc_m(ms, f32_buffer, stride);
  EXCEPT_F32_ARRAY_EQ(ls_f32_src, f32_buffer, M * N, "MLC & MSC F32");
}

static void test_mlc_m_msc_m_i64() {
  const size_t M = 8;
  const size_t N = 2;
  const size_t stride = N * sizeof(int64_t);
  SET_MBA0_I64();
  msettilem(M);
  msettilen(N);
  mint64_t ms = mlc_m(ls_i64_src, stride);
  msc_m(ms, i64_buffer, stride);
  EXCEPT_I64_ARRAY_EQ(ls_i64_src, i64_buffer, M * N, "MLC & MSC I64");
}

static void test_mlc_m_msc_m_u64() {
  const size_t M = 8;
  const size_t N = 2;
  const size_t stride = N * sizeof(uint64_t);
  SET_MBA0_I64();
  msettilem(M);
  msettilen(N);
  muint64_t ms = mlc_m(ls_u64_src, stride);
  msc_m(ms, u64_buffer, stride);
  EXCEPT_U64_ARRAY_EQ(ls_u64_src, u64_buffer, M * N, "MLC & MSC U64");
}

static void test_mlc_m_msc_m_f64() {
  const size_t M = 8;
  const size_t N = 2;
  const size_t stride = N * sizeof(fp64_t);
  SET_MBA0_F64();
  msettilem(M);
  msettilen(N);
  mfloat64_t ms = mlc_m(ls_f64_src, stride);
  msc_m(ms, f64_buffer, stride);
  EXCEPT_F64_ARRAY_EQ(ls_f64_src, f64_buffer, M * N, "MLC & MSC F64");
}

static void test_mltr_m_mstr_m_i8() {
  SET_MBA0_I8();
  mint8_t ms = mltr_m(ls_i8_src, 8 * sizeof(int8_t));
  mstr_m(ms, i8_buffer, 8 * sizeof(int8_t));
  EXCEPT_I8_ARRAY_EQ(ls_i8_src, i8_buffer, 8 * 8, "MLTR & MSTR I8 ");
}

static void test_mltr_m_mstr_m_u8() {
  SET_MBA0_I8();
  muint8_t ms = mltr_m(ls_u8_src, 8 * sizeof(uint8_t));
  mstr_m(ms, u8_buffer, 8 * sizeof(uint8_t));
  EXCEPT_U8_ARRAY_EQ(ls_u8_src, u8_buffer, 8 * 8, "MLTR & MSTR U8 ");
}

static void test_mltr_m_mstr_m_i16() {
  SET_MBA0_I16();
  mint16_t ms = mltr_m(ls_i16_src, 8 * sizeof(int16_t));
  mstr_m(ms, i16_buffer, 8 * sizeof(int16_t));
  EXCEPT_I16_ARRAY_EQ(ls_i16_src, i16_buffer, 8 * 8, "MLTR & MSTR I16");
}

static void test_mltr_m_mstr_m_u16() {
  SET_MBA0_I16();
  muint16_t ms = mltr_m(ls_u16_src, 8 * sizeof(uint16_t));
  mstr_m(ms, u16_buffer, 8 * sizeof(uint16_t));
  EXCEPT_U16_ARRAY_EQ(ls_u16_src, u16_buffer, 8 * 8, "MLTR & MSTR U16");
}

static void test_mltr_m_mstr_m_f16() {
  SET_MBA0_F16();
  mfloat16_t ms = mltr_m(ls_f16_src, 8 * sizeof(fp16_t));
  mstr_m(ms, f16_buffer, 8 * sizeof(fp16_t));
  EXCEPT_F16_ARRAY_EQ(ls_f16_src, f16_buffer, 8 * 8, "MLTR & MSTR F16");
}

static void test_mltr_m_mstr_m_i32() {
  SET_MBA0_I32();
  mint32_t ms = mltr_m(ls_i32_src, 4 * sizeof(int32_t));
  mstr_m(ms, i32_buffer, 4 * sizeof(int32_t));
  EXCEPT_I32_ARRAY_EQ(ls_i32_src, i32_buffer, 8 * 4, "MLTR & MSTR I32");
}

static void test_mltr_m_mstr_m_u32() {
  SET_MBA0_I32();
  muint32_t ms = mltr_m(ls_u32_src, 4 * sizeof(uint32_t));
  mstr_m(ms, u32_buffer, 4 * sizeof(uint32_t));
  EXCEPT_U32_ARRAY_EQ(ls_u32_src, u32_buffer, 8 * 4, "MLTR & MSTR U32");
}

static void test_mltr_m_mstr_m_f32() {
  SET_MBA0_F32();
  mfloat32_t ms = mltr_m(ls_f32_src, 4 * sizeof(fp32_t));
  mstr_m(ms, f32_buffer, 4 * sizeof(fp32_t));
  EXCEPT_F32_ARRAY_EQ(ls_f32_src, f32_buffer, 8 * 4, "MLTR & MSTR F32");
}

static void test_mltr_m_mstr_m_i64() {
  SET_MBA0_I64();
  mint64_t ms = mltr_m(ls_i64_src, 2 * sizeof(int64_t));
  mstr_m(ms, i64_buffer, 2 * sizeof(int64_t));
  EXCEPT_I64_ARRAY_EQ(ls_i64_src, i64_buffer, 8 * 2, "MLTR & MSTR I64");
}

static void test_mltr_m_mstr_m_u64() {
  SET_MBA0_I64();
  muint64_t ms = mltr_m(ls_u64_src, 2 * sizeof(uint64_t));
  mstr_m(ms, u64_buffer, 2 * sizeof(uint64_t));
  EXCEPT_U64_ARRAY_EQ(ls_u64_src, u64_buffer, 8 * 2, "MLTR & MSTR U64");
}

static void test_mltr_m_mstr_m_f64() {
  SET_MBA0_F64();
  mfloat64_t ms = mltr_m(ls_f64_src, 2 * sizeof(fp64_t));
  mstr_m(ms, f64_buffer, 2 * sizeof(fp64_t));
  EXCEPT_F64_ARRAY_EQ(ls_f64_src, f64_buffer, 8 * 2, "MLTR & MSTR F64");
}

static void test_mlacc_m_msacc_m_i8() {
  SET_MBA0_I8();
  mint8_t ms = mlacc_m(ls_i8_src, 16 * sizeof(int8_t));
  msacc_m(ms, i8_buffer, 16 * sizeof(int8_t));
  EXCEPT_I8_ARRAY_EQ(ls_i8_src, i8_buffer, 8 * 16, "MLACC & MSACC I8 ");
}

static void test_mlacc_m_msacc_m_u8() {
  SET_MBA0_I8();
  muint8_t ms = mlacc_m(ls_u8_src, 16 * sizeof(uint8_t));
  msacc_m(ms, u8_buffer, 16 * sizeof(uint8_t));
  EXCEPT_U8_ARRAY_EQ(ls_u8_src, u8_buffer, 8 * 16, "MLACC & MSACC U8 ");
}

static void test_mlacc_m_msacc_m_i16() {
  SET_MBA0_I16();
  mint16_t ms = mlacc_m(ls_i16_src, 8 * sizeof(int16_t));
  msacc_m(ms, i16_buffer, 8 * sizeof(int16_t));
  EXCEPT_I16_ARRAY_EQ(ls_i16_src, i16_buffer, 8 * 8, "MLACC & MSACC I16");
}

static void test_mlacc_m_msacc_m_u16() {
  SET_MBA0_I16();
  muint16_t ms = mlacc_m(ls_u16_src, 8 * sizeof(uint16_t));
  msacc_m(ms, u16_buffer, 8 * sizeof(uint16_t));
  EXCEPT_U16_ARRAY_EQ(ls_u16_src, u16_buffer, 8 * 8, "MLACC & MSACC U16");
}

static void test_mlacc_m_msacc_m_f16() {
  SET_MBA0_F16();
  mfloat16_t ms = mlacc_m(ls_f16_src, 8 * sizeof(fp16_t));
  msacc_m(ms, f16_buffer, 8 * sizeof(fp16_t));
  EXCEPT_F16_ARRAY_EQ(ls_f16_src, f16_buffer, 8 * 8, "MLACC & MSACC F16");
}

static void test_mlacc_m_msacc_m_i32() {
  SET_MBA0_I32();
  mint32_t ms = mlacc_m(ls_i32_src, 4 * sizeof(int32_t));
  msacc_m(ms, i32_buffer, 4 * sizeof(int32_t));
  EXCEPT_I32_ARRAY_EQ(ls_i32_src, i32_buffer, 8 * 4, "MLACC & MSACC I32");
}

static void test_mlacc_m_msacc_m_u32() {
  SET_MBA0_I32();
  muint32_t ms = mlacc_m(ls_u32_src, 4 * sizeof(uint32_t));
  msacc_m(ms, u32_buffer, 4 * sizeof(uint32_t));
  EXCEPT_U32_ARRAY_EQ(ls_u32_src, u32_buffer, 8 * 4, "MLACC & MSACC U32");
}

static void test_mlacc_m_msacc_m_f32() {
  SET_MBA0_F32();
  mfloat32_t ms = mlacc_m(ls_f32_src, 4 * sizeof(fp32_t));
  msacc_m(ms, f32_buffer, 4 * sizeof(fp32_t));
  EXCEPT_F32_ARRAY_EQ(ls_f32_src, f32_buffer, 8 * 4, "MLACC & MSACC F32");
}

static void test_mlacc_m_msacc_m_i64() {
  SET_MBA0_I64();
  mint64_t ms = mlacc_m(ls_i64_src, 2 * sizeof(int64_t));
  msacc_m(ms, i64_buffer, 2 * sizeof(int64_t));
  EXCEPT_I64_ARRAY_EQ(ls_i64_src, i64_buffer, 8 * 2, "MLACC & MSACC I64");
}

static void test_mlacc_m_msacc_m_u64() {
  SET_MBA0_I64();
  muint64_t ms = mlacc_m(ls_u64_src, 2 * sizeof(uint64_t));
  msacc_m(ms, u64_buffer, 2 * sizeof(uint64_t));
  EXCEPT_U64_ARRAY_EQ(ls_u64_src, u64_buffer, 8 * 2, "MLACC & MSACC U64");
}

static void test_mlacc_m_msacc_m_f64() {
  SET_MBA0_F64();
  mfloat64_t ms = mlacc_m(ls_f64_src, 2 * sizeof(fp64_t));
  msacc_m(ms, f64_buffer, 2 * sizeof(fp64_t));
  EXCEPT_F64_ARRAY_EQ(ls_f64_src, f64_buffer, 8 * 2, "MLACC & MSACC F64");
}

static void test_mlat_m_msat_m_i8() {
  const int8_t s[8 * 8] = {
      54,   2,   -114, -116, 75,  -61,  78,   36,   -95, 59,  -89, -94, 101,
      -69,  -59, 9,    -106, -87, 2,    -48,  -105, -58, 23,  -95, 83,  119,
      -125, 78,  93,   14,   -33, 75,   94,   104,  54,  -35, -84, -8,  38,
      -5,   -43, 122,  -32,  -63, -126, 21,   30,   -58, 102, 79,  -80, 53,
      117,  -19, 114,  -18,  -42, -128, -107, 9,    116, 123, 65,  -50};
  const int8_t ts[8 * 8] = {
      54,  -95, -106, 83,   94,  -43, 102,  -42, 2,    59,  -87,  119,  104,
      122, 79,  -128, -114, -89, 2,   -125, 54,  -32,  -80, -107, -116, -94,
      -48, 78,  -35,  -63,  53,  9,   75,   101, -105, 93,  -84,  -126, 117,
      116, -61, -69,  -58,  14,  -8,  21,   -19, 123,  78,  -59,  23,   -33,
      38,  30,  114,  65,   36,  9,   -95,  75,  -5,   -58, -18,  -50};
  enum { M = 8, K = 8 };
  SET_MBA0_I8();
  msettilem(M);
  msettilek(K);
  mint8_t ms1 = mlat_m(s, K * sizeof(int8_t));
  msa_m(ms1, i8_buffer, M * sizeof(int8_t));
  EXCEPT_I8_ARRAY_EQ(ts, i8_buffer, M * K, "MLAT & MSA I8 ");
  mint8_t ms2 = mla_m(s, K * sizeof(int8_t));
  msat_m(ms2, i8_buffer, M * sizeof(int8_t));
  EXCEPT_I8_ARRAY_EQ(ts, i8_buffer, M * K, "MLA & MSAT I8 ");
}

static void test_mlat_m_msat_m_u8() {
  const uint8_t s[8 * 8] = {
      39,  131, 201, 123, 103, 7,   188, 152, 204, 254, 186, 9,   41,
      99,  254, 36,  112, 122, 145, 135, 38,  14,  39,  7,   167, 213,
      20,  186, 139, 50,  46,  206, 27,  172, 13,  154, 66,  129, 218,
      28,  65,  74,  150, 252, 26,  251, 252, 159, 197, 20,  161, 74,
      139, 225, 81,  85,  209, 125, 229, 204, 103, 79,  169, 38};
  const uint8_t ts[8 * 8] = {
      39,  204, 112, 167, 27,  65,  197, 209, 131, 254, 122, 213, 172,
      74,  20,  125, 201, 186, 145, 20,  13,  150, 161, 229, 123, 9,
      135, 186, 154, 252, 74,  204, 103, 41,  38,  139, 66,  26,  139,
      103, 7,   99,  14,  50,  129, 251, 225, 79,  188, 254, 39,  46,
      218, 252, 81,  169, 152, 36,  7,   206, 28,  159, 85,  38};
  enum { M = 8, K = 8 };
  SET_MBA0_I8();
  msettilem(M);
  msettilek(K);
  muint8_t ms1 = mlat_m(s, K * sizeof(uint8_t));
  msa_m(ms1, u8_buffer, M * sizeof(uint8_t));
  EXCEPT_U8_ARRAY_EQ(ts, u8_buffer, M * K, "MLAT & MSA U8 ");
  muint8_t ms2 = mla_m(s, K * sizeof(uint8_t));
  msat_m(ms2, u8_buffer, M * sizeof(uint8_t));
  EXCEPT_U8_ARRAY_EQ(ts, u8_buffer, M * K, "MLA & MSAT U8 ");
}

static void test_mlat_m_msat_m_i16() {
  const int16_t s[8 * 8] = {
      31586,  -31453, 13394,  7264,   -2415,  -26334, -32247, 5493,
      -32039, -10813, 20283,  -16845, 18468,  -10819, 11470,  2429,
      -22600, -26219, -12585, 8591,   -6186,  3707,   27292,  22706,
      -4737,  -27315, -26135, -8543,  -18826, 17644,  15644,  31638,
      3380,   13313,  -21615, 29286,  11236,  28259,  26800,  5731,
      -13216, 19120,  31078,  -31874, -25234, -21606, 9870,   19232,
      154,    20107,  9422,   4395,   -32717, -13853, 18732,  2025,
      3389,   -6269,  -6455,  14456,  -20358, 353,    3776,   -25439};
  const int16_t ts[8 * 8] = {
      31586,  -32039, -22600, -4737,  3380,   -13216, 154,    3389,
      -31453, -10813, -26219, -27315, 13313,  19120,  20107,  -6269,
      13394,  20283,  -12585, -26135, -21615, 31078,  9422,   -6455,
      7264,   -16845, 8591,   -8543,  29286,  -31874, 4395,   14456,
      -2415,  18468,  -6186,  -18826, 11236,  -25234, -32717, -20358,
      -26334, -10819, 3707,   17644,  28259,  -21606, -13853, 353,
      -32247, 11470,  27292,  15644,  26800,  9870,   18732,  3776,
      5493,   2429,   22706,  31638,  5731,   19232,  2025,   -25439};
  enum { M = 8, K = 8 };
  SET_MBA0_I16();
  msettilem(M);
  msettilek(K);
  mint16_t ms1 = mlat_m(s, K * sizeof(int16_t));
  msa_m(ms1, i16_buffer, M * sizeof(int16_t));
  EXCEPT_I16_ARRAY_EQ(ts, i16_buffer, M * K, "MLAT & MSA I16");
  mint16_t ms2 = mla_m(s, K * sizeof(int16_t));
  msat_m(ms2, i16_buffer, M * sizeof(int16_t));
  EXCEPT_I16_ARRAY_EQ(ts, i16_buffer, M * K, "MLA & MSAT I16");
}

static void test_mlat_m_msat_m_u16() {
  const uint16_t s[8 * 8] = {
      12385, 10883, 996,   15552, 56666, 61279, 59590, 33005, 17475, 57383,
      58774, 65103, 9794,  1640,  13169, 20351, 24897, 12188, 29655, 24967,
      34548, 18028, 37163, 36305, 18176, 49305, 15765, 2280,  8512,  42145,
      48177, 49424, 42921, 41871, 42491, 15157, 33870, 42000, 44669, 18555,
      7777,  44180, 11892, 58859, 45124, 38025, 60127, 63629, 52425, 64544,
      33472, 60190, 65510, 30660, 4526,  45075, 62684, 22568, 16189, 29895,
      29236, 22794, 9002,  51794};
  const uint16_t ts[8 * 8] = {
      12385, 17475, 24897, 18176, 42921, 7777,  52425, 62684, 10883, 57383,
      12188, 49305, 41871, 44180, 64544, 22568, 996,   58774, 29655, 15765,
      42491, 11892, 33472, 16189, 15552, 65103, 24967, 2280,  15157, 58859,
      60190, 29895, 56666, 9794,  34548, 8512,  33870, 45124, 65510, 29236,
      61279, 1640,  18028, 42145, 42000, 38025, 30660, 22794, 59590, 13169,
      37163, 48177, 44669, 60127, 4526,  9002,  33005, 20351, 36305, 49424,
      18555, 63629, 45075, 51794};
  enum { M = 8, K = 8 };
  SET_MBA0_I16();
  msettilem(M);
  msettilek(K);
  muint16_t ms1 = mlat_m(s, K * sizeof(uint16_t));
  msa_m(ms1, u16_buffer, M * sizeof(uint16_t));
  EXCEPT_U16_ARRAY_EQ(ts, u16_buffer, M * K, "MLAT & MSA U16");
  muint16_t ms2 = mla_m(s, K * sizeof(uint16_t));
  msat_m(ms2, u16_buffer, M * sizeof(uint16_t));
  EXCEPT_U16_ARRAY_EQ(ts, u16_buffer, M * K, "MLA & MSAT U16");
}

static void test_mlat_m_msat_m_f16() {
  const fp16_t s[8 * 8] = {
      -4.668,  -6.375,  -2.062, -1.217, 1.101,  -3.342, 8.1,    -8.41,
      -7.62,   3.635,   -9.81,  -7.977, -1.759, -1.041, 1.65,   2.084,
      -9.58,   2.127,   9.44,   -2.994, -2.176, 7.727,  -1.753, -7.484,
      -4.62,   -0.7017, -9.94,  -8.92,  8.38,   -4.03,  -6.02,  -1.999,
      -3.547,  2.604,   4.92,   9.44,   5.996,  -5.402, 4.363,  8.52,
      -0.9136, 6.46,    1.201,  -6.605, -7.33,  -2.4,   0.4043, 1.771,
      -6.14,   7.43,    6.03,   3.322,  -4.863, 1.169,  5.766,  -4.26,
      -5.72,   8.234,   -9.016, -3.746, 3.814,  3.838,  -7.246, 2.035};
  const fp16_t ts[8 * 8] = {
      -4.668, -7.62,  -9.58,  -4.62,   -3.547, -0.9136, -6.14,  -5.72,
      -6.375, 3.635,  2.127,  -0.7017, 2.604,  6.46,    7.43,   8.234,
      -2.062, -9.81,  9.44,   -9.94,   4.92,   1.201,   6.03,   -9.016,
      -1.217, -7.977, -2.994, -8.92,   9.44,   -6.605,  3.322,  -3.746,
      1.101,  -1.759, -2.176, 8.38,    5.996,  -7.33,   -4.863, 3.814,
      -3.342, -1.041, 7.727,  -4.03,   -5.402, -2.4,    1.169,  3.838,
      8.1,    1.65,   -1.753, -6.02,   4.363,  0.4043,  5.766,  -7.246,
      -8.41,  2.084,  -7.484, -1.999,  8.52,   1.771,   -4.26,  2.035};
  enum { M = 8, K = 8 };
  SET_MBA0_F16();
  msettilem(M);
  msettilek(K);
  mfloat16_t ms1 = mlat_m(s, K * sizeof(fp16_t));
  msa_m(ms1, f16_buffer, M * sizeof(fp16_t));
  EXCEPT_F16_ARRAY_EQ(ts, f16_buffer, M * K, "MLAT & MSA F16");
  mfloat16_t ms2 = mla_m(s, K * sizeof(fp16_t));
  msat_m(ms2, f16_buffer, M * sizeof(fp16_t));
  EXCEPT_F16_ARRAY_EQ(ts, f16_buffer, M * K, "MLA & MSAT F16");
}

static void test_mlat_m_msat_m_i32() {
  const int32_t s[4 * 4] = {1176523711,  735229723,  -288426488,  -154942308,
                            901321450,   -935720867, -2045334519, 1733858699,
                            -677136741,  2079519650, -162916509,  -1832517247,
                            -2043041587, 1179468014, 263897191,   -1749280769};
  const int32_t ts[4 * 4] = {1176523711, 901321450,   -677136741,  -2043041587,
                             735229723,  -935720867,  2079519650,  1179468014,
                             -288426488, -2045334519, -162916509,  263897191,
                             -154942308, 1733858699,  -1832517247, -1749280769};
  enum { M = 4, K = 4 };
  SET_MBA0_I32();
  msettilem(M);
  msettilek(K);
  mint32_t ms1 = mlat_m(s, K * sizeof(int32_t));
  msa_m(ms1, i32_buffer, M * sizeof(int32_t));
  EXCEPT_I32_ARRAY_EQ(ts, i32_buffer, M * K, "MLAT & MSA I32");
  mint32_t ms2 = mla_m(s, K * sizeof(int32_t));
  msat_m(ms2, i32_buffer, M * sizeof(int32_t));
  EXCEPT_I32_ARRAY_EQ(ts, i32_buffer, M * K, "MLA & MSAT I32");
}

static void test_mlat_m_msat_m_u32() {
  const uint32_t s[4 * 4] = {3412453616, 3105338827, 1378087635, 3882013307,
                             2547263509, 540713997,  2119105730, 4185212428,
                             3981656098, 3402392960, 918264035,  114060537,
                             3795774855, 4087743283, 1477339864, 1103756575};
  const uint32_t ts[4 * 4] = {3412453616, 2547263509, 3981656098, 3795774855,
                              3105338827, 540713997,  3402392960, 4087743283,
                              1378087635, 2119105730, 918264035,  1477339864,
                              3882013307, 4185212428, 114060537,  1103756575};
  enum { M = 4, K = 4 };
  SET_MBA0_I32();
  msettilem(M);
  msettilek(K);
  muint32_t ms1 = mlat_m(s, K * sizeof(uint32_t));
  msa_m(ms1, u32_buffer, M * sizeof(uint32_t));
  EXCEPT_U32_ARRAY_EQ(ts, u32_buffer, M * K, "MLAT & MSA U32");
  muint32_t ms2 = mla_m(s, K * sizeof(uint32_t));
  msat_m(ms2, u32_buffer, M * sizeof(uint32_t));
  EXCEPT_U32_ARRAY_EQ(ts, u32_buffer, M * K, "MLA & MSAT U32");
}

static void test_mlat_m_msat_m_f32() {
  const fp32_t s[4 * 4] = {-65.38936, -24.07832,  -88.26531, 26.302822,
                           -77.28905, -75.71926,  27.796894, 86.04906,
                           81.37337,  -58.170525, -52.42903, -96.008934,
                           72.43898,  44.179466,  -26.26543, 85.663315};
  const fp32_t ts[4 * 4] = {-65.38936, -77.28905, 81.37337,   72.43898,
                            -24.07832, -75.71926, -58.170525, 44.179466,
                            -88.26531, 27.796894, -52.42903,  -26.26543,
                            26.302822, 86.04906,  -96.008934, 85.663315};
  enum { M = 4, K = 4 };
  SET_MBA0_F32();
  msettilem(M);
  msettilek(K);
  mfloat32_t ms1 = mlat_m(s, K * sizeof(fp32_t));
  msa_m(ms1, f32_buffer, M * sizeof(fp32_t));
  EXCEPT_F32_ARRAY_EQ(ts, f32_buffer, M * K, "MLAT & MSA F32");
  mfloat32_t ms2 = mla_m(s, K * sizeof(fp32_t));
  msat_m(ms2, f32_buffer, M * sizeof(fp32_t));
  EXCEPT_F32_ARRAY_EQ(ts, f32_buffer, M * K, "MLA & MSAT F32");
}

static void test_mlat_m_msat_m_i64() {
  const int64_t s[2 * 2] = {8195051659744675313, 2389382246251008106,
                            -5866374732065456732, -3945797158342723027};
  const int64_t ts[2 * 2] = {8195051659744675313, -5866374732065456732,
                             2389382246251008106, -3945797158342723027};
  enum { M = 2, K = 2 };
  SET_MBA0_I64();
  msettilem(M);
  msettilek(K);
  mint64_t ms1 = mlat_m(s, K * sizeof(int64_t));
  msa_m(ms1, i64_buffer, M * sizeof(int64_t));
  EXCEPT_I64_ARRAY_EQ(ts, i64_buffer, M * K, "MLAT & MSA I64");
  mint64_t ms2 = mla_m(s, K * sizeof(int64_t));
  msat_m(ms2, i64_buffer, M * sizeof(int64_t));
  EXCEPT_I64_ARRAY_EQ(ts, i64_buffer, M * K, "MLA & MSAT I64");
}

static void test_mlat_m_msat_m_u64() {
  const uint64_t s[2 * 2] = {9562317294127927450ull, 1025333062614921458ull,
                             11343385185794464382ull, 14629545821165543537ull};
  const uint64_t ts[2 * 2] = {9562317294127927450ull, 11343385185794464382ull,
                              1025333062614921458ull, 14629545821165543537ull};
  enum { M = 2, K = 2 };
  SET_MBA0_I64();
  msettilem(M);
  msettilek(K);
  muint64_t ms1 = mlat_m(s, K * sizeof(uint64_t));
  msa_m(ms1, u64_buffer, M * sizeof(uint64_t));
  EXCEPT_U64_ARRAY_EQ(ts, u64_buffer, M * K, "MLAT & MSA U64");
  muint64_t ms2 = mla_m(s, K * sizeof(uint64_t));
  msat_m(ms2, u64_buffer, M * sizeof(uint64_t));
  EXCEPT_U64_ARRAY_EQ(ts, u64_buffer, M * K, "MLA & MSAT U64");
}

static void test_mlat_m_msat_m_f64() {
  const fp64_t s[2 * 2] = {64.00968902, -69.34802143, 55.86185758, 78.15189359};
  const fp64_t ts[2 * 2] = {64.00968902, 55.86185758, -69.34802143,
                            78.15189359};
  enum { M = 2, K = 2 };
  SET_MBA0_F64();
  msettilem(M);
  msettilek(K);
  mfloat64_t ms1 = mlat_m(s, K * sizeof(fp64_t));
  msa_m(ms1, f64_buffer, M * sizeof(fp64_t));
  EXCEPT_F64_ARRAY_EQ(ts, f64_buffer, M * K, "MLAT & MSA F64");
  mfloat64_t ms2 = mla_m(s, K * sizeof(fp64_t));
  msat_m(ms2, f64_buffer, M * sizeof(fp64_t));
  EXCEPT_F64_ARRAY_EQ(ts, f64_buffer, M * K, "MLA & MSAT F64");
}

static void test_mlbt_m_msbt_m_i8() {
  const int8_t s[8 * 8] = {
      54,   2,   -114, -116, 75,  -61,  78,   36,   -95, 59,  -89, -94, 101,
      -69,  -59, 9,    -106, -87, 2,    -48,  -105, -58, 23,  -95, 83,  119,
      -125, 78,  93,   14,   -33, 75,   94,   104,  54,  -35, -84, -8,  38,
      -5,   -43, 122,  -32,  -63, -126, 21,   30,   -58, 102, 79,  -80, 53,
      117,  -19, 114,  -18,  -42, -128, -107, 9,    116, 123, 65,  -50};
  const int8_t ts[8 * 8] = {
      54,  -95, -106, 83,   94,  -43, 102,  -42, 2,    59,  -87,  119,  104,
      122, 79,  -128, -114, -89, 2,   -125, 54,  -32,  -80, -107, -116, -94,
      -48, 78,  -35,  -63,  53,  9,   75,   101, -105, 93,  -84,  -126, 117,
      116, -61, -69,  -58,  14,  -8,  21,   -19, 123,  78,  -59,  23,   -33,
      38,  30,  114,  65,   36,  9,   -95,  75,  -5,   -58, -18,  -50};
  const size_t K = 8;
  const size_t N = 8;
  SET_MBA0_I8();
  msettilek(K);
  msettilen(N);
  mint8_t ms1 = mlbt_m(s, N * sizeof(int8_t));
  msb_m(ms1, i8_buffer, K * sizeof(int8_t));
  EXCEPT_I8_ARRAY_EQ(ts, i8_buffer, K * N, "MLBT & MSB I8 ");
  mint8_t ms2 = mlb_m(s, N * sizeof(int8_t));
  msbt_m(ms2, i8_buffer, K * sizeof(int8_t));
  EXCEPT_I8_ARRAY_EQ(ts, i8_buffer, K * N, "MLB & MSBT I8 ");
}

static void test_mlbt_m_msbt_m_u8() {
  const uint8_t s[8 * 8] = {
      39,  131, 201, 123, 103, 7,   188, 152, 204, 254, 186, 9,   41,
      99,  254, 36,  112, 122, 145, 135, 38,  14,  39,  7,   167, 213,
      20,  186, 139, 50,  46,  206, 27,  172, 13,  154, 66,  129, 218,
      28,  65,  74,  150, 252, 26,  251, 252, 159, 197, 20,  161, 74,
      139, 225, 81,  85,  209, 125, 229, 204, 103, 79,  169, 38};
  const uint8_t ts[8 * 8] = {
      39,  204, 112, 167, 27,  65,  197, 209, 131, 254, 122, 213, 172,
      74,  20,  125, 201, 186, 145, 20,  13,  150, 161, 229, 123, 9,
      135, 186, 154, 252, 74,  204, 103, 41,  38,  139, 66,  26,  139,
      103, 7,   99,  14,  50,  129, 251, 225, 79,  188, 254, 39,  46,
      218, 252, 81,  169, 152, 36,  7,   206, 28,  159, 85,  38};
  const size_t K = 8;
  const size_t N = 8;
  SET_MBA0_I8();
  msettilek(K);
  msettilen(N);
  muint8_t ms1 = mlbt_m(s, N * sizeof(uint8_t));
  msb_m(ms1, u8_buffer, K * sizeof(uint8_t));
  EXCEPT_U8_ARRAY_EQ(ts, u8_buffer, K * N, "MLBT & MSB U8 ");
  muint8_t ms2 = mlb_m(s, N * sizeof(uint8_t));
  msbt_m(ms2, u8_buffer, K * sizeof(uint8_t));
  EXCEPT_U8_ARRAY_EQ(ts, u8_buffer, K * N, "MLB & MSBT U8 ");
}

static void test_mlbt_m_msbt_m_i16() {
  const int16_t s[8 * 8] = {
      31586,  -31453, 13394,  7264,   -2415,  -26334, -32247, 5493,
      -32039, -10813, 20283,  -16845, 18468,  -10819, 11470,  2429,
      -22600, -26219, -12585, 8591,   -6186,  3707,   27292,  22706,
      -4737,  -27315, -26135, -8543,  -18826, 17644,  15644,  31638,
      3380,   13313,  -21615, 29286,  11236,  28259,  26800,  5731,
      -13216, 19120,  31078,  -31874, -25234, -21606, 9870,   19232,
      154,    20107,  9422,   4395,   -32717, -13853, 18732,  2025,
      3389,   -6269,  -6455,  14456,  -20358, 353,    3776,   -25439};
  const int16_t ts[8 * 8] = {
      31586,  -32039, -22600, -4737,  3380,   -13216, 154,    3389,
      -31453, -10813, -26219, -27315, 13313,  19120,  20107,  -6269,
      13394,  20283,  -12585, -26135, -21615, 31078,  9422,   -6455,
      7264,   -16845, 8591,   -8543,  29286,  -31874, 4395,   14456,
      -2415,  18468,  -6186,  -18826, 11236,  -25234, -32717, -20358,
      -26334, -10819, 3707,   17644,  28259,  -21606, -13853, 353,
      -32247, 11470,  27292,  15644,  26800,  9870,   18732,  3776,
      5493,   2429,   22706,  31638,  5731,   19232,  2025,   -25439};
  const size_t K = 8;
  const size_t N = 8;
  SET_MBA0_I16();
  msettilek(K);
  msettilen(N);
  mint16_t ms1 = mlbt_m(s, N * sizeof(int16_t));
  msb_m(ms1, i16_buffer, K * sizeof(int16_t));
  EXCEPT_I16_ARRAY_EQ(ts, i16_buffer, K * N, "MLBT & MSB I16");
  mint16_t ms2 = mlb_m(s, N * sizeof(int16_t));
  msbt_m(ms2, i16_buffer, K * sizeof(int16_t));
  EXCEPT_I16_ARRAY_EQ(ts, i16_buffer, K * N, "MLB & MSBT I16");
}

static void test_mlbt_m_msbt_m_u16() {
  const uint16_t s[8 * 8] = {
      12385, 10883, 996,   15552, 56666, 61279, 59590, 33005, 17475, 57383,
      58774, 65103, 9794,  1640,  13169, 20351, 24897, 12188, 29655, 24967,
      34548, 18028, 37163, 36305, 18176, 49305, 15765, 2280,  8512,  42145,
      48177, 49424, 42921, 41871, 42491, 15157, 33870, 42000, 44669, 18555,
      7777,  44180, 11892, 58859, 45124, 38025, 60127, 63629, 52425, 64544,
      33472, 60190, 65510, 30660, 4526,  45075, 62684, 22568, 16189, 29895,
      29236, 22794, 9002,  51794};
  const uint16_t ts[8 * 8] = {
      12385, 17475, 24897, 18176, 42921, 7777,  52425, 62684, 10883, 57383,
      12188, 49305, 41871, 44180, 64544, 22568, 996,   58774, 29655, 15765,
      42491, 11892, 33472, 16189, 15552, 65103, 24967, 2280,  15157, 58859,
      60190, 29895, 56666, 9794,  34548, 8512,  33870, 45124, 65510, 29236,
      61279, 1640,  18028, 42145, 42000, 38025, 30660, 22794, 59590, 13169,
      37163, 48177, 44669, 60127, 4526,  9002,  33005, 20351, 36305, 49424,
      18555, 63629, 45075, 51794};
  const size_t K = 8;
  const size_t N = 8;
  SET_MBA0_I16();
  msettilek(K);
  msettilen(N);
  muint16_t ms1 = mlbt_m(s, N * sizeof(uint16_t));
  msb_m(ms1, u16_buffer, K * sizeof(uint16_t));
  EXCEPT_U16_ARRAY_EQ(ts, u16_buffer, K * N, "MLBT & MSB U16");
  muint16_t ms2 = mlb_m(s, N * sizeof(uint16_t));
  msbt_m(ms2, u16_buffer, K * sizeof(uint16_t));
  EXCEPT_U16_ARRAY_EQ(ts, u16_buffer, K * N, "MLB & MSBT U16");
}

static void test_mlbt_m_msbt_m_f16() {
  const fp16_t s[8 * 8] = {
      -4.668,  -6.375,  -2.062, -1.217, 1.101,  -3.342, 8.1,    -8.41,
      -7.62,   3.635,   -9.81,  -7.977, -1.759, -1.041, 1.65,   2.084,
      -9.58,   2.127,   9.44,   -2.994, -2.176, 7.727,  -1.753, -7.484,
      -4.62,   -0.7017, -9.94,  -8.92,  8.38,   -4.03,  -6.02,  -1.999,
      -3.547,  2.604,   4.92,   9.44,   5.996,  -5.402, 4.363,  8.52,
      -0.9136, 6.46,    1.201,  -6.605, -7.33,  -2.4,   0.4043, 1.771,
      -6.14,   7.43,    6.03,   3.322,  -4.863, 1.169,  5.766,  -4.26,
      -5.72,   8.234,   -9.016, -3.746, 3.814,  3.838,  -7.246, 2.035};
  const fp16_t ts[8 * 8] = {
      -4.668, -7.62,  -9.58,  -4.62,   -3.547, -0.9136, -6.14,  -5.72,
      -6.375, 3.635,  2.127,  -0.7017, 2.604,  6.46,    7.43,   8.234,
      -2.062, -9.81,  9.44,   -9.94,   4.92,   1.201,   6.03,   -9.016,
      -1.217, -7.977, -2.994, -8.92,   9.44,   -6.605,  3.322,  -3.746,
      1.101,  -1.759, -2.176, 8.38,    5.996,  -7.33,   -4.863, 3.814,
      -3.342, -1.041, 7.727,  -4.03,   -5.402, -2.4,    1.169,  3.838,
      8.1,    1.65,   -1.753, -6.02,   4.363,  0.4043,  5.766,  -7.246,
      -8.41,  2.084,  -7.484, -1.999,  8.52,   1.771,   -4.26,  2.035};
  const size_t K = 8;
  const size_t N = 8;
  SET_MBA0_F16();
  msettilek(K);
  msettilen(N);
  mfloat16_t ms1 = mlbt_m(s, N * sizeof(fp16_t));
  msb_m(ms1, f16_buffer, K * sizeof(fp16_t));
  EXCEPT_F16_ARRAY_EQ(ts, f16_buffer, K * N, "MLBT & MSB F16");
  mfloat16_t ms2 = mlb_m(s, N * sizeof(fp16_t));
  msbt_m(ms2, f16_buffer, K * sizeof(fp16_t));
  EXCEPT_F16_ARRAY_EQ(ts, f16_buffer, K * N, "MLB & MSBT F16");
}

static void test_mlbt_m_msbt_m_i32() {
  const int32_t s[4 * 4] = {1176523711,  735229723,  -288426488,  -154942308,
                            901321450,   -935720867, -2045334519, 1733858699,
                            -677136741,  2079519650, -162916509,  -1832517247,
                            -2043041587, 1179468014, 263897191,   -1749280769};
  const int32_t ts[4 * 4] = {1176523711, 901321450,   -677136741,  -2043041587,
                             735229723,  -935720867,  2079519650,  1179468014,
                             -288426488, -2045334519, -162916509,  263897191,
                             -154942308, 1733858699,  -1832517247, -1749280769};
  const size_t K = 4;
  const size_t N = 4;
  SET_MBA0_I32();
  msettilek(K);
  msettilen(N);
  mint32_t ms1 = mlbt_m(s, N * sizeof(int32_t));
  msb_m(ms1, i32_buffer, K * sizeof(int32_t));
  EXCEPT_I32_ARRAY_EQ(ts, i32_buffer, K * N, "MLBT & MSB I32");
  mint32_t ms2 = mlb_m(s, N * sizeof(int32_t));
  msbt_m(ms2, i32_buffer, K * sizeof(int32_t));
  EXCEPT_I32_ARRAY_EQ(ts, i32_buffer, K * N, "MLB & MSBT I32");
}

static void test_mlbt_m_msbt_m_u32() {
  const uint32_t s[4 * 4] = {3412453616, 3105338827, 1378087635, 3882013307,
                             2547263509, 540713997,  2119105730, 4185212428,
                             3981656098, 3402392960, 918264035,  114060537,
                             3795774855, 4087743283, 1477339864, 1103756575};
  const uint32_t ts[4 * 4] = {3412453616, 2547263509, 3981656098, 3795774855,
                              3105338827, 540713997,  3402392960, 4087743283,
                              1378087635, 2119105730, 918264035,  1477339864,
                              3882013307, 4185212428, 114060537,  1103756575};
  const size_t K = 4;
  const size_t N = 4;
  SET_MBA0_I32();
  msettilek(K);
  msettilen(N);
  muint32_t ms1 = mlbt_m(s, N * sizeof(uint32_t));
  msb_m(ms1, u32_buffer, K * sizeof(uint32_t));
  EXCEPT_U32_ARRAY_EQ(ts, u32_buffer, K * N, "MLBT & MSB U32");
  muint32_t ms2 = mlb_m(s, N * sizeof(uint32_t));
  msbt_m(ms2, u32_buffer, K * sizeof(uint32_t));
  EXCEPT_U32_ARRAY_EQ(ts, u32_buffer, K * N, "MLB & MSBT U32");
}

static void test_mlbt_m_msbt_m_f32() {
  const fp32_t s[4 * 4] = {-65.38936, -24.07832,  -88.26531, 26.302822,
                           -77.28905, -75.71926,  27.796894, 86.04906,
                           81.37337,  -58.170525, -52.42903, -96.008934,
                           72.43898,  44.179466,  -26.26543, 85.663315};
  const fp32_t ts[4 * 4] = {-65.38936, -77.28905, 81.37337,   72.43898,
                            -24.07832, -75.71926, -58.170525, 44.179466,
                            -88.26531, 27.796894, -52.42903,  -26.26543,
                            26.302822, 86.04906,  -96.008934, 85.663315};
  const size_t K = 4;
  const size_t N = 4;
  SET_MBA0_F32();
  msettilek(K);
  msettilen(N);
  mfloat32_t ms1 = mlbt_m(s, N * sizeof(fp32_t));
  msb_m(ms1, f32_buffer, K * sizeof(fp32_t));
  EXCEPT_F32_ARRAY_EQ(ts, f32_buffer, K * N, "MLBT & MSB F32");
  mfloat32_t ms2 = mlb_m(s, N * sizeof(fp32_t));
  msbt_m(ms2, f32_buffer, K * sizeof(fp32_t));
  EXCEPT_F32_ARRAY_EQ(ts, f32_buffer, K * N, "MLB & MSBT F32");
}

static void test_mlbt_m_msbt_m_i64() {
  const int64_t s[2 * 2] = {8195051659744675313, 2389382246251008106,
                            -5866374732065456732, -3945797158342723027};
  const int64_t ts[2 * 2] = {8195051659744675313, -5866374732065456732,
                             2389382246251008106, -3945797158342723027};
  const size_t K = 2;
  const size_t N = 2;
  SET_MBA0_I64();
  msettilek(K);
  msettilen(N);
  mint64_t ms1 = mlbt_m(s, N * sizeof(int64_t));
  msb_m(ms1, i64_buffer, K * sizeof(int64_t));
  EXCEPT_I64_ARRAY_EQ(ts, i64_buffer, K * N, "MLBT & MSB I64");
  mint64_t ms2 = mlb_m(s, N * sizeof(int64_t));
  msbt_m(ms2, i64_buffer, K * sizeof(int64_t));
  EXCEPT_I64_ARRAY_EQ(ts, i64_buffer, K * N, "MLB & MSBT I64");
}

static void test_mlbt_m_msbt_m_u64() {
  const uint64_t s[2 * 2] = {9562317294127927450ull, 1025333062614921458ull,
                             11343385185794464382ull, 14629545821165543537ull};
  const uint64_t ts[2 * 2] = {9562317294127927450ull, 11343385185794464382ull,
                              1025333062614921458ull, 14629545821165543537ull};
  const size_t K = 2;
  const size_t N = 2;
  SET_MBA0_I64();
  msettilek(K);
  msettilen(N);
  muint64_t ms1 = mlbt_m(s, N * sizeof(uint64_t));
  msb_m(ms1, u64_buffer, K * sizeof(uint64_t));
  EXCEPT_U64_ARRAY_EQ(ts, u64_buffer, K * N, "MLBT & MSB U64");
  muint64_t ms2 = mlb_m(s, N * sizeof(uint64_t));
  msbt_m(ms2, u64_buffer, K * sizeof(uint64_t));
  EXCEPT_U64_ARRAY_EQ(ts, u64_buffer, K * N, "MLB & MSBT U64");
}

static void test_mlbt_m_msbt_m_f64() {
  const fp64_t s[2 * 2] = {64.00968902, -69.34802143, 55.86185758, 78.15189359};
  const fp64_t ts[2 * 2] = {64.00968902, 55.86185758, -69.34802143,
                            78.15189359};
  const size_t K = 2;
  const size_t N = 2;
  SET_MBA0_F64();
  msettilek(K);
  msettilen(N);
  mfloat64_t ms1 = mlbt_m(s, N * sizeof(fp64_t));
  msb_m(ms1, f64_buffer, K * sizeof(fp64_t));
  EXCEPT_F64_ARRAY_EQ(ts, f64_buffer, K * N, "MLBT & MSB F64");
  mfloat64_t ms2 = mlb_m(s, N * sizeof(fp64_t));
  msbt_m(ms2, f64_buffer, K * sizeof(fp64_t));
  EXCEPT_F64_ARRAY_EQ(ts, f64_buffer, K * N, "MLB & MSBT F64");
}

static void test_mlct_m_msct_m_i8() {
  const int8_t s[8 * 8] = {
      54,   2,   -114, -116, 75,  -61,  78,   36,   -95, 59,  -89, -94, 101,
      -69,  -59, 9,    -106, -87, 2,    -48,  -105, -58, 23,  -95, 83,  119,
      -125, 78,  93,   14,   -33, 75,   94,   104,  54,  -35, -84, -8,  38,
      -5,   -43, 122,  -32,  -63, -126, 21,   30,   -58, 102, 79,  -80, 53,
      117,  -19, 114,  -18,  -42, -128, -107, 9,    116, 123, 65,  -50};
  const int8_t ts[8 * 8] = {
      54,  -95, -106, 83,   94,  -43, 102,  -42, 2,    59,  -87,  119,  104,
      122, 79,  -128, -114, -89, 2,   -125, 54,  -32,  -80, -107, -116, -94,
      -48, 78,  -35,  -63,  53,  9,   75,   101, -105, 93,  -84,  -126, 117,
      116, -61, -69,  -58,  14,  -8,  21,   -19, 123,  78,  -59,  23,   -33,
      38,  30,  114,  65,   36,  9,   -95,  75,  -5,   -58, -18,  -50};
  enum { M = 8, N = 8 };
  SET_MBA0_I8();
  msettilem(M);
  msettilen(N);
  mint8_t ms1 = mlct_m(s, N * sizeof(int8_t));
  msc_m(ms1, i8_buffer, M * sizeof(int8_t));
  EXCEPT_I8_ARRAY_EQ(ts, i8_buffer, M * N, "MLCT & MSC I8 ");
  mint8_t ms2 = mlc_m(s, N * sizeof(int8_t));
  msct_m(ms2, i8_buffer, M * sizeof(int8_t));
  EXCEPT_I8_ARRAY_EQ(ts, i8_buffer, M * N, "MLC & MSCT I8 ");
}

static void test_mlct_m_msct_m_u8() {
  const uint8_t s[8 * 8] = {
      39,  131, 201, 123, 103, 7,   188, 152, 204, 254, 186, 9,   41,
      99,  254, 36,  112, 122, 145, 135, 38,  14,  39,  7,   167, 213,
      20,  186, 139, 50,  46,  206, 27,  172, 13,  154, 66,  129, 218,
      28,  65,  74,  150, 252, 26,  251, 252, 159, 197, 20,  161, 74,
      139, 225, 81,  85,  209, 125, 229, 204, 103, 79,  169, 38};
  const uint8_t ts[8 * 8] = {
      39,  204, 112, 167, 27,  65,  197, 209, 131, 254, 122, 213, 172,
      74,  20,  125, 201, 186, 145, 20,  13,  150, 161, 229, 123, 9,
      135, 186, 154, 252, 74,  204, 103, 41,  38,  139, 66,  26,  139,
      103, 7,   99,  14,  50,  129, 251, 225, 79,  188, 254, 39,  46,
      218, 252, 81,  169, 152, 36,  7,   206, 28,  159, 85,  38};
  enum { M = 8, N = 8 };
  SET_MBA0_I8();
  msettilem(M);
  msettilen(N);
  muint8_t ms1 = mlct_m(s, N * sizeof(uint8_t));
  msc_m(ms1, u8_buffer, M * sizeof(uint8_t));
  EXCEPT_U8_ARRAY_EQ(ts, u8_buffer, M * N, "MLCT & MSC U8 ");
  muint8_t ms2 = mlc_m(s, N * sizeof(uint8_t));
  msct_m(ms2, u8_buffer, M * sizeof(uint8_t));
  EXCEPT_U8_ARRAY_EQ(ts, u8_buffer, M * N, "MLC & MSCT U8 ");
}

static void test_mlct_m_msct_m_i16() {
  const int16_t s[8 * 8] = {
      31586,  -31453, 13394,  7264,   -2415,  -26334, -32247, 5493,
      -32039, -10813, 20283,  -16845, 18468,  -10819, 11470,  2429,
      -22600, -26219, -12585, 8591,   -6186,  3707,   27292,  22706,
      -4737,  -27315, -26135, -8543,  -18826, 17644,  15644,  31638,
      3380,   13313,  -21615, 29286,  11236,  28259,  26800,  5731,
      -13216, 19120,  31078,  -31874, -25234, -21606, 9870,   19232,
      154,    20107,  9422,   4395,   -32717, -13853, 18732,  2025,
      3389,   -6269,  -6455,  14456,  -20358, 353,    3776,   -25439};
  const int16_t ts[8 * 8] = {
      31586,  -32039, -22600, -4737,  3380,   -13216, 154,    3389,
      -31453, -10813, -26219, -27315, 13313,  19120,  20107,  -6269,
      13394,  20283,  -12585, -26135, -21615, 31078,  9422,   -6455,
      7264,   -16845, 8591,   -8543,  29286,  -31874, 4395,   14456,
      -2415,  18468,  -6186,  -18826, 11236,  -25234, -32717, -20358,
      -26334, -10819, 3707,   17644,  28259,  -21606, -13853, 353,
      -32247, 11470,  27292,  15644,  26800,  9870,   18732,  3776,
      5493,   2429,   22706,  31638,  5731,   19232,  2025,   -25439};
  enum { M = 8, N = 8 };
  SET_MBA0_I16();
  msettilem(M);
  msettilen(N);
  mint16_t ms1 = mlct_m(s, N * sizeof(int16_t));
  msc_m(ms1, i16_buffer, M * sizeof(int16_t));
  EXCEPT_I16_ARRAY_EQ(ts, i16_buffer, M * N, "MLCT & MSC I16");
  mint16_t ms2 = mlc_m(s, N * sizeof(int16_t));
  msct_m(ms2, i16_buffer, M * sizeof(int16_t));
  EXCEPT_I16_ARRAY_EQ(ts, i16_buffer, M * N, "MLC & MSCT I16");
}

static void test_mlct_m_msct_m_u16() {
  const uint16_t s[8 * 8] = {
      12385, 10883, 996,   15552, 56666, 61279, 59590, 33005, 17475, 57383,
      58774, 65103, 9794,  1640,  13169, 20351, 24897, 12188, 29655, 24967,
      34548, 18028, 37163, 36305, 18176, 49305, 15765, 2280,  8512,  42145,
      48177, 49424, 42921, 41871, 42491, 15157, 33870, 42000, 44669, 18555,
      7777,  44180, 11892, 58859, 45124, 38025, 60127, 63629, 52425, 64544,
      33472, 60190, 65510, 30660, 4526,  45075, 62684, 22568, 16189, 29895,
      29236, 22794, 9002,  51794};
  const uint16_t ts[8 * 8] = {
      12385, 17475, 24897, 18176, 42921, 7777,  52425, 62684, 10883, 57383,
      12188, 49305, 41871, 44180, 64544, 22568, 996,   58774, 29655, 15765,
      42491, 11892, 33472, 16189, 15552, 65103, 24967, 2280,  15157, 58859,
      60190, 29895, 56666, 9794,  34548, 8512,  33870, 45124, 65510, 29236,
      61279, 1640,  18028, 42145, 42000, 38025, 30660, 22794, 59590, 13169,
      37163, 48177, 44669, 60127, 4526,  9002,  33005, 20351, 36305, 49424,
      18555, 63629, 45075, 51794};
  enum { M = 8, N = 8 };
  SET_MBA0_I16();
  msettilem(M);
  msettilen(N);
  muint16_t ms1 = mlct_m(s, N * sizeof(uint16_t));
  msc_m(ms1, u16_buffer, M * sizeof(uint16_t));
  EXCEPT_U16_ARRAY_EQ(ts, u16_buffer, M * N, "MLCT & MSC U16");
  muint16_t ms2 = mlc_m(s, N * sizeof(uint16_t));
  msct_m(ms2, u16_buffer, M * sizeof(uint16_t));
  EXCEPT_U16_ARRAY_EQ(ts, u16_buffer, M * N, "MLC & MSCT U16");
}

static void test_mlct_m_msct_m_f16() {
  const fp16_t s[8 * 8] = {
      -4.668,  -6.375,  -2.062, -1.217, 1.101,  -3.342, 8.1,    -8.41,
      -7.62,   3.635,   -9.81,  -7.977, -1.759, -1.041, 1.65,   2.084,
      -9.58,   2.127,   9.44,   -2.994, -2.176, 7.727,  -1.753, -7.484,
      -4.62,   -0.7017, -9.94,  -8.92,  8.38,   -4.03,  -6.02,  -1.999,
      -3.547,  2.604,   4.92,   9.44,   5.996,  -5.402, 4.363,  8.52,
      -0.9136, 6.46,    1.201,  -6.605, -7.33,  -2.4,   0.4043, 1.771,
      -6.14,   7.43,    6.03,   3.322,  -4.863, 1.169,  5.766,  -4.26,
      -5.72,   8.234,   -9.016, -3.746, 3.814,  3.838,  -7.246, 2.035};
  const fp16_t ts[8 * 8] = {
      -4.668, -7.62,  -9.58,  -4.62,   -3.547, -0.9136, -6.14,  -5.72,
      -6.375, 3.635,  2.127,  -0.7017, 2.604,  6.46,    7.43,   8.234,
      -2.062, -9.81,  9.44,   -9.94,   4.92,   1.201,   6.03,   -9.016,
      -1.217, -7.977, -2.994, -8.92,   9.44,   -6.605,  3.322,  -3.746,
      1.101,  -1.759, -2.176, 8.38,    5.996,  -7.33,   -4.863, 3.814,
      -3.342, -1.041, 7.727,  -4.03,   -5.402, -2.4,    1.169,  3.838,
      8.1,    1.65,   -1.753, -6.02,   4.363,  0.4043,  5.766,  -7.246,
      -8.41,  2.084,  -7.484, -1.999,  8.52,   1.771,   -4.26,  2.035};
  enum { M = 8, N = 8 };
  SET_MBA0_F16();
  msettilem(M);
  msettilen(N);
  mfloat16_t ms1 = mlct_m(s, N * sizeof(fp16_t));
  msc_m(ms1, f16_buffer, M * sizeof(fp16_t));
  EXCEPT_F16_ARRAY_EQ(ts, f16_buffer, M * N, "MLCT & MSC F16");
  mfloat16_t ms2 = mlc_m(s, N * sizeof(fp16_t));
  msct_m(ms2, f16_buffer, M * sizeof(fp16_t));
  EXCEPT_F16_ARRAY_EQ(ts, f16_buffer, M * N, "MLC & MSCT F16");
}

static void test_mlct_m_msct_m_i32() {
  const int32_t s[4 * 4] = {1176523711,  735229723,  -288426488,  -154942308,
                            901321450,   -935720867, -2045334519, 1733858699,
                            -677136741,  2079519650, -162916509,  -1832517247,
                            -2043041587, 1179468014, 263897191,   -1749280769};
  const int32_t ts[4 * 4] = {1176523711, 901321450,   -677136741,  -2043041587,
                             735229723,  -935720867,  2079519650,  1179468014,
                             -288426488, -2045334519, -162916509,  263897191,
                             -154942308, 1733858699,  -1832517247, -1749280769};
  enum { M = 4, N = 4 };
  SET_MBA0_I32();
  msettilem(M);
  msettilen(N);
  mint32_t ms1 = mlct_m(s, N * sizeof(int32_t));
  msc_m(ms1, i32_buffer, M * sizeof(int32_t));
  EXCEPT_I32_ARRAY_EQ(ts, i32_buffer, M * N, "MLCT & MSC I32");
  mint32_t ms2 = mlc_m(s, N * sizeof(int32_t));
  msct_m(ms2, i32_buffer, M * sizeof(int32_t));
  EXCEPT_I32_ARRAY_EQ(ts, i32_buffer, M * N, "MLC & MSCT I32");
}

static void test_mlct_m_msct_m_u32() {
  const uint32_t s[4 * 4] = {3412453616, 3105338827, 1378087635, 3882013307,
                             2547263509, 540713997,  2119105730, 4185212428,
                             3981656098, 3402392960, 918264035,  114060537,
                             3795774855, 4087743283, 1477339864, 1103756575};
  const uint32_t ts[4 * 4] = {3412453616, 2547263509, 3981656098, 3795774855,
                              3105338827, 540713997,  3402392960, 4087743283,
                              1378087635, 2119105730, 918264035,  1477339864,
                              3882013307, 4185212428, 114060537,  1103756575};
  enum { M = 4, N = 4 };
  SET_MBA0_I32();
  msettilem(M);
  msettilen(N);
  muint32_t ms1 = mlct_m(s, N * sizeof(uint32_t));
  msc_m(ms1, u32_buffer, M * sizeof(uint32_t));
  EXCEPT_U32_ARRAY_EQ(ts, u32_buffer, M * N, "MLCT & MSC U32");
  muint32_t ms2 = mlc_m(s, N * sizeof(uint32_t));
  msct_m(ms2, u32_buffer, M * sizeof(uint32_t));
  EXCEPT_U32_ARRAY_EQ(ts, u32_buffer, M * N, "MLC & MSCT U32");
}

static void test_mlct_m_msct_m_f32() {
  const fp32_t s[4 * 4] = {-65.38936, -24.07832,  -88.26531, 26.302822,
                           -77.28905, -75.71926,  27.796894, 86.04906,
                           81.37337,  -58.170525, -52.42903, -96.008934,
                           72.43898,  44.179466,  -26.26543, 85.663315};
  const fp32_t ts[4 * 4] = {-65.38936, -77.28905, 81.37337,   72.43898,
                            -24.07832, -75.71926, -58.170525, 44.179466,
                            -88.26531, 27.796894, -52.42903,  -26.26543,
                            26.302822, 86.04906,  -96.008934, 85.663315};
  enum { M = 4, N = 4 };
  SET_MBA0_F32();
  msettilem(M);
  msettilen(N);
  mfloat32_t ms1 = mlct_m(s, N * sizeof(fp32_t));
  msc_m(ms1, f32_buffer, M * sizeof(fp32_t));
  EXCEPT_F32_ARRAY_EQ(ts, f32_buffer, M * N, "MLCT & MSC F32");
  mfloat32_t ms2 = mlc_m(s, N * sizeof(fp32_t));
  msct_m(ms2, f32_buffer, M * sizeof(fp32_t));
  EXCEPT_F32_ARRAY_EQ(ts, f32_buffer, M * N, "MLC & MSCT F32");
}

static void test_mlct_m_msct_m_i64() {
  const int64_t s[2 * 2] = {8195051659744675313, 2389382246251008106,
                            -5866374732065456732, -3945797158342723027};
  const int64_t ts[2 * 2] = {8195051659744675313, -5866374732065456732,
                             2389382246251008106, -3945797158342723027};
  enum { M = 2, N = 2 };
  SET_MBA0_I64();
  msettilem(M);
  msettilen(N);
  mint64_t ms1 = mlct_m(s, N * sizeof(int64_t));
  msc_m(ms1, i64_buffer, M * sizeof(int64_t));
  EXCEPT_I64_ARRAY_EQ(ts, i64_buffer, M * N, "MLCT & MSC I64");
  mint64_t ms2 = mlc_m(s, N * sizeof(int64_t));
  msct_m(ms2, i64_buffer, M * sizeof(int64_t));
  EXCEPT_I64_ARRAY_EQ(ts, i64_buffer, M * N, "MLC & MSCT I64");
}

static void test_mlct_m_msct_m_u64() {
  const uint64_t s[2 * 2] = {9562317294127927450ull, 1025333062614921458ull,
                             11343385185794464382ull, 14629545821165543537ull};
  const uint64_t ts[2 * 2] = {9562317294127927450ull, 11343385185794464382ull,
                              1025333062614921458ull, 14629545821165543537ull};
  enum { M = 2, N = 2 };
  SET_MBA0_I64();
  msettilem(M);
  msettilen(N);
  muint64_t ms1 = mlct_m(s, N * sizeof(uint64_t));
  msc_m(ms1, u64_buffer, M * sizeof(uint64_t));
  EXCEPT_U64_ARRAY_EQ(ts, u64_buffer, M * N, "MLCT & MSC U64");
  muint64_t ms2 = mlc_m(s, N * sizeof(uint64_t));
  msct_m(ms2, u64_buffer, M * sizeof(uint64_t));
  EXCEPT_U64_ARRAY_EQ(ts, u64_buffer, M * N, "MLC & MSCT U64");
}

static void test_mlct_m_msct_m_f64() {
  const fp64_t s[2 * 2] = {64.00968902, -69.34802143, 55.86185758, 78.15189359};
  const fp64_t ts[2 * 2] = {64.00968902, 55.86185758, -69.34802143,
                            78.15189359};
  enum { M = 2, N = 2 };
  SET_MBA0_F64();
  msettilem(M);
  msettilen(N);
  mfloat64_t ms1 = mlct_m(s, N * sizeof(fp64_t));
  msc_m(ms1, f64_buffer, M * sizeof(fp64_t));
  EXCEPT_F64_ARRAY_EQ(ts, f64_buffer, M * N, "MLCT & MSC F64");
  mfloat64_t ms2 = mlc_m(s, N * sizeof(fp64_t));
  msct_m(ms2, f64_buffer, M * sizeof(fp64_t));
  EXCEPT_F64_ARRAY_EQ(ts, f64_buffer, M * N, "MLC & MSCT F64");
}

static void test_mta_m_i8() {
  enum { M = 8, K = 8 };
  const size_t stride = K * sizeof(int8_t);
  const int8_t src[M * K] = {
      77,  -68, -26, 71,   -85, -69, 112,  5,    53,   -69, 83,  77,   69,
      -59, -6,  118, -97,  25,  84,  -111, 119,  -63,  3,   -30, -112, 55,
      87,  19,  37,  -123, 116, 58,  -126, 69,   -41,  -99, 10,  93,   101,
      126, 10,  84,  91,   -18, -47, -82,  -116, -86,  -68, 125, -96,  39,
      119, -62, 4,   24,   -42, 36,  6,    -32,  -120, 98,  35,  -62};
  const int8_t ans[M * K] = {
      77,   53,   -97, -112, -126, 10,  -68, -42, -68, -69, 25, 55,  69,
      84,   125,  36,  -26,  83,   84,  87,  -41, 91,  -96, 6,  71,  77,
      -111, 19,   -99, -18,  39,   -32, -85, 69,  119, 37,  10, -47, 119,
      -120, -69,  -59, -63,  -123, 93,  -82, -62, 98,  112, -6, 3,   116,
      101,  -116, 4,   35,   5,    118, -30, 58,  126, -86, 24, -62};
  SET_MBA0_I8();
  msettilem(M);
  msettilek(K);
  mint8_t ms = mla_m(src, stride);
  mint8_t md = mta_m(ms);
  msa_m(md, i8_buffer, stride);
  EXCEPT_I8_ARRAY_EQ(ans, i8_buffer, M * K, "MTA_M I8");
}

static void test_mta_m_i16() {
  enum { M = 8, K = 8 };
  const size_t stride = K * sizeof(int16_t);
  const int16_t src[M * K] = {
      30929, 31992,  -2344,  19075,  -21099, -31491, 25567,  18606,
      15671, -29372, 15063,  5918,   -28908, 5831,   16439,  -22996,
      3983,  -17084, 4239,   31552,  -29243, 15558,  21084,  -17660,
      26409, -5278,  27318,  -31836, 12439,  19643,  16126,  25816,
      5204,  -10070, 23900,  24804,  4844,   10858,  -18868, 6917,
      -6906, 16397,  -11842, 18979,  15989,  14717,  -5948,  -3461,
      2724,  -10162, 30617,  24872,  31029,  -99,    -27003, -23555,
      13896, -6126,  2187,   4225,   -28379, 4882,   11637,  -29368};
  const int16_t ans[M * K] = {
      30929,  15671,  3983,   26409,  5204,   -6906,  2724,   13896,
      31992,  -29372, -17084, -5278,  -10070, 16397,  -10162, -6126,
      -2344,  15063,  4239,   27318,  23900,  -11842, 30617,  2187,
      19075,  5918,   31552,  -31836, 24804,  18979,  24872,  4225,
      -21099, -28908, -29243, 12439,  4844,   15989,  31029,  -28379,
      -31491, 5831,   15558,  19643,  10858,  14717,  -99,    4882,
      25567,  16439,  21084,  16126,  -18868, -5948,  -27003, 11637,
      18606,  -22996, -17660, 25816,  6917,   -3461,  -23555, -29368};
  SET_MBA0_I16();
  msettilem(M);
  msettilek(K);
  mint16_t ms = mla_m(src, stride);
  mint16_t md = mta_m(ms);
  msa_m(md, i16_buffer, stride);
  EXCEPT_I16_ARRAY_EQ(ans, i16_buffer, M * K, "MTA_M I16");
}

static void test_mta_m_i32() {
  enum { M = 4, K = 4 };
  const size_t stride = K * sizeof(int32_t);
  const int32_t src[M * K] = {
      -1681124598, -1292122543, -2028898099, -2045352292,
      -1888223806, -619176567,  2129772106,  294845680,
      -1344465868, -1642749564, 1092118040,  736267622,
      -912961762,  -655803685,  -1442065192, 467457903};
  const int32_t ans[M * K] = {
      -1681124598, -1888223806, -1344465868, -912961762,
      -1292122543, -619176567,  -1642749564, -655803685,
      -2028898099, 2129772106,  1092118040,  -1442065192,
      -2045352292, 294845680,   736267622,   467457903};
  SET_MBA0_I32();
  msettilem(M);
  msettilek(K);
  mint32_t ms = mla_m(src, stride);
  mint32_t md = mta_m(ms);
  msa_m(md, i32_buffer, stride);
  EXCEPT_I32_ARRAY_EQ(ans, i32_buffer, M * K, "MTA_M I32");
}

static void test_mta_m_i64() {
  enum { M = 2, K = 2 };
  const size_t stride = K * sizeof(int64_t);
  const int64_t src[M * K] = {-125363293662549438, 7432419646672721022,
                              -2936165396028250924, -3423249875061222608};
  const int64_t ans[M * K] = {-125363293662549438, -2936165396028250924,
                              7432419646672721022, -3423249875061222608};
  SET_MBA0_I64();
  msettilem(M);
  msettilek(K);
  mint64_t ms = mla_m(src, stride);
  mint64_t md = mta_m(ms);
  msa_m(md, i64_buffer, stride);
  EXCEPT_I64_ARRAY_EQ(ans, i64_buffer, M * K, "MTA_M I64");
}

static void test_mta_m_u8() {
  enum { M = 8, K = 8 };
  const size_t stride = K * sizeof(uint8_t);
  const uint8_t src[M * K] = {
      95,  250, 228, 97,  253, 236, 51,  182, 179, 21,  94,  127, 114,
      5,   231, 109, 121, 8,   212, 28,  177, 82,  40,  252, 192, 79,
      3,   221, 157, 195, 107, 24,  106, 202, 78,  144, 127, 57,  86,
      236, 59,  191, 23,  242, 32,  87,  93,  156, 91,  140, 164, 177,
      151, 179, 245, 204, 63,  212, 235, 46,  248, 86,  224, 71};
  const uint8_t ans[M * K] = {
      95,  179, 121, 192, 106, 59,  91,  63,  250, 21,  8,   79, 202,
      191, 140, 212, 228, 94,  212, 3,   78,  23,  164, 235, 97, 127,
      28,  221, 144, 242, 177, 46,  253, 114, 177, 157, 127, 32, 151,
      248, 236, 5,   82,  195, 57,  87,  179, 86,  51,  231, 40, 107,
      86,  93,  245, 224, 182, 109, 252, 24,  236, 156, 204, 71};
  SET_MBA0_I8();
  msettilem(M);
  msettilek(K);
  muint8_t ms = mla_m(src, stride);
  muint8_t md = mta_m(ms);
  msa_m(md, u8_buffer, stride);
  EXCEPT_U8_ARRAY_EQ(ans, u8_buffer, M * K, "MTA_M U8");
}

static void test_mta_m_u16() {
  enum { M = 8, K = 8 };
  const size_t stride = K * sizeof(uint16_t);
  const uint16_t src[M * K] = {
      25570, 41452, 3929,  17553, 39826, 5858,  23422, 25449, 2671,  52676,
      35333, 21801, 51814, 56172, 40104, 14848, 56365, 42600, 26097, 38242,
      58392, 5342,  39285, 36714, 52224, 32844, 59183, 8869,  59262, 75,
      18977, 65231, 53459, 30958, 25173, 23693, 61305, 19295, 16173, 11282,
      64095, 33860, 52416, 20816, 8220,  51630, 27705, 48754, 24673, 46715,
      9824,  28027, 24889, 27056, 11529, 11787, 6899,  5502,  14756, 35893,
      31189, 16010, 47528, 34047};
  const uint16_t ans[M * K] = {
      25570, 2671,  56365, 52224, 53459, 64095, 24673, 6899,  41452, 52676,
      42600, 32844, 30958, 33860, 46715, 5502,  3929,  35333, 26097, 59183,
      25173, 52416, 9824,  14756, 17553, 21801, 38242, 8869,  23693, 20816,
      28027, 35893, 39826, 51814, 58392, 59262, 61305, 8220,  24889, 31189,
      5858,  56172, 5342,  75,    19295, 51630, 27056, 16010, 23422, 40104,
      39285, 18977, 16173, 27705, 11529, 47528, 25449, 14848, 36714, 65231,
      11282, 48754, 11787, 34047};
  SET_MBA0_I16();
  msettilem(M);
  msettilek(K);
  muint16_t ms = mla_m(src, stride);
  muint16_t md = mta_m(ms);
  msa_m(md, u16_buffer, stride);
  EXCEPT_U16_ARRAY_EQ(ans, u16_buffer, M * K, "MTA_M U16");
}

static void test_mta_m_u32() {
  enum { M = 4, K = 4 };
  const size_t stride = K * sizeof(uint32_t);
  const uint32_t src[M * K] = {1853700213, 1224691227, 579172120,  2544000723,
                               3705217052, 3847582878, 2047817136, 2873319869,
                               944198550,  2495765483, 825898774,  920734754,
                               2888440314, 3623959478, 3650400147, 2853937346};
  const uint32_t ans[M * K] = {1853700213, 3705217052, 944198550,  2888440314,
                               1224691227, 3847582878, 2495765483, 3623959478,
                               579172120,  2047817136, 825898774,  3650400147,
                               2544000723, 2873319869, 920734754,  2853937346};
  SET_MBA0_I32();
  msettilem(M);
  msettilek(K);
  muint32_t ms = mla_m(src, stride);
  muint32_t md = mta_m(ms);
  msa_m(md, u32_buffer, stride);
  EXCEPT_U32_ARRAY_EQ(ans, u32_buffer, M * K, "MTA_M U32");
}

static void test_mta_m_u64() {
  enum { M = 2, K = 2 };
  const size_t stride = K * sizeof(uint64_t);
  const uint64_t src[M * K] = {1021280658148866281, 1524863831652402790,
                               1028769707722912970, 1169303502346608508};
  const uint64_t ans[M * K] = {1021280658148866281, 1028769707722912970,
                               1524863831652402790, 1169303502346608508};
  SET_MBA0_I64();
  msettilem(M);
  msettilek(K);
  muint64_t ms = mla_m(src, stride);
  muint64_t md = mta_m(ms);
  msa_m(md, u64_buffer, stride);
  EXCEPT_U64_ARRAY_EQ(ans, u64_buffer, M * K, "MTA_M U64");
}

static void test_mta_m_f16() {
  enum { M = 8, K = 8 };
  const size_t stride = K * sizeof(fp16_t);
  const fp16_t src[M * K] = {
      8.586,  -7.586, -9.125, -5.676,   -0.9844, 2.707,  1.951,  3.246,
      -1.897, 6.254,  -8.48,  2.453,    -1.325,  1.384,  -7.79,  1.528,
      -8.21,  6.438,  -9.484, 9.72,     1.674,   -9.35,  -1.623, 6.336,
      -4.633, 4.223,  3.107,  2.379,    5.547,   -1.083, 5.652,  -7.16,
      4.906,  0.6665, -2.396, -6.883,   6.082,   -7.652, 6.363,  -0.02127,
      7.344,  0.6743, -5.246, -0.773,   1.587,   -7.555, -1.096, 1.293,
      0.439,  -4.9,   -2.473, 4.9,      8.586,   4.92,   9.68,   -4.81,
      -5.55,  -9.805, -6.926, -0.06555, 1.454,   6.69,   -5.918, -0.6934};
  const fp16_t ans[M * K] = {
      8.586,   -1.897, -8.21,  -4.633, 4.906,    7.344,  0.439,  -5.55,
      -7.586,  6.254,  6.438,  4.223,  0.6665,   0.6743, -4.9,   -9.805,
      -9.125,  -8.48,  -9.484, 3.107,  -2.396,   -5.246, -2.473, -6.926,
      -5.676,  2.453,  9.72,   2.379,  -6.883,   -0.773, 4.9,    -0.06555,
      -0.9844, -1.325, 1.674,  5.547,  6.082,    1.587,  8.586,  1.454,
      2.707,   1.384,  -9.35,  -1.083, -7.652,   -7.555, 4.92,   6.69,
      1.951,   -7.79,  -1.623, 5.652,  6.363,    -1.096, 9.68,   -5.918,
      3.246,   1.528,  6.336,  -7.16,  -0.02127, 1.293,  -4.81,  -0.6934};
  SET_MBA0_F16();
  msettilem(M);
  msettilek(K);
  mfloat16_t ms = mla_m(src, stride);
  mfloat16_t md = mta_m(ms);
  msa_m(md, f16_buffer, stride);
  EXCEPT_F16_ARRAY_EQ(ans, f16_buffer, M * K, "MTA_M F16");
}

static void test_mta_m_f32() {
  enum { M = 4, K = 4 };
  const size_t stride = K * sizeof(fp32_t);
  const fp32_t src[M * K] = {-5.317063,  2.6362946, 3.2381415,  6.414423,
                             -2.2291253, 3.17508,   -2.7422917, -9.390402,
                             -6.324778,  4.798573,  0.21453322, 2.723796,
                             0.8812181,  9.901346,  9.690255,   3.6316826};
  const fp32_t ans[M * K] = {-5.317063, -2.2291253, -6.324778,  0.8812181,
                             2.6362946, 3.17508,    4.798573,   9.901346,
                             3.2381415, -2.7422917, 0.21453322, 9.690255,
                             6.414423,  -9.390402,  2.723796,   3.6316826};
  SET_MBA0_F32();
  msettilem(M);
  msettilek(K);
  mfloat32_t ms = mla_m(src, stride);
  mfloat32_t md = mta_m(ms);
  msa_m(md, f32_buffer, stride);
  EXCEPT_F32_ARRAY_EQ(ans, f32_buffer, M * K, "MTA_M F32");
}

static void test_mta_m_f64() {
  enum { M = 2, K = 2 };
  const size_t stride = K * sizeof(fp64_t);
  const fp64_t src[M * K] = {-4.97872986, 4.66672172, 0.18207431, -8.41076301};
  const fp64_t ans[M * K] = {-4.97872986, 0.18207431, 4.66672172, -8.41076301};
  SET_MBA0_F64();
  msettilem(M);
  msettilek(K);
  mfloat64_t ms = mla_m(src, stride);
  mfloat64_t md = mta_m(ms);
  msa_m(md, f64_buffer, stride);
  EXCEPT_F64_ARRAY_EQ(ans, f64_buffer, M * K, "MTA_M F64");
}

static void test_mta_m() {
  test_mta_m_i8();
  test_mta_m_i16();
  test_mta_m_i32();
  test_mta_m_i64();
  test_mta_m_u8();
  test_mta_m_u16();
  test_mta_m_u32();
  test_mta_m_u64();
  test_mta_m_f16();
  test_mta_m_f32();
  test_mta_m_f64();
}

static void test_mtb_m_i8() {
  enum { N = 8, K = 8 };
  const size_t stride = N * sizeof(int8_t);
  const int8_t src[K * N] = {
      77,  -68, -26, 71,   -85, -69, 112,  5,    53,   -69, 83,  77,   69,
      -59, -6,  118, -97,  25,  84,  -111, 119,  -63,  3,   -30, -112, 55,
      87,  19,  37,  -123, 116, 58,  -126, 69,   -41,  -99, 10,  93,   101,
      126, 10,  84,  91,   -18, -47, -82,  -116, -86,  -68, 125, -96,  39,
      119, -62, 4,   24,   -42, 36,  6,    -32,  -120, 98,  35,  -62};
  const int8_t ans[K * N] = {
      77,   53,   -97, -112, -126, 10,  -68, -42, -68, -69, 25, 55,  69,
      84,   125,  36,  -26,  83,   84,  87,  -41, 91,  -96, 6,  71,  77,
      -111, 19,   -99, -18,  39,   -32, -85, 69,  119, 37,  10, -47, 119,
      -120, -69,  -59, -63,  -123, 93,  -82, -62, 98,  112, -6, 3,   116,
      101,  -116, 4,   35,   5,    118, -30, 58,  126, -86, 24, -62};
  SET_MBA0_I8();
  msettilen(N);
  msettilek(K);
  mint8_t ms = mlb_m(src, stride);
  mint8_t md = mtb_m(ms);
  msb_m(md, i8_buffer, stride);
  EXCEPT_I8_ARRAY_EQ(ans, i8_buffer, K * N, "MTB_M I8");
}

static void test_mtb_m_i16() {
  enum { N = 8, K = 8 };
  const size_t stride = N * sizeof(int16_t);
  const int16_t src[K * N] = {
      30929, 31992,  -2344,  19075,  -21099, -31491, 25567,  18606,
      15671, -29372, 15063,  5918,   -28908, 5831,   16439,  -22996,
      3983,  -17084, 4239,   31552,  -29243, 15558,  21084,  -17660,
      26409, -5278,  27318,  -31836, 12439,  19643,  16126,  25816,
      5204,  -10070, 23900,  24804,  4844,   10858,  -18868, 6917,
      -6906, 16397,  -11842, 18979,  15989,  14717,  -5948,  -3461,
      2724,  -10162, 30617,  24872,  31029,  -99,    -27003, -23555,
      13896, -6126,  2187,   4225,   -28379, 4882,   11637,  -29368};
  const int16_t ans[K * N] = {
      30929,  15671,  3983,   26409,  5204,   -6906,  2724,   13896,
      31992,  -29372, -17084, -5278,  -10070, 16397,  -10162, -6126,
      -2344,  15063,  4239,   27318,  23900,  -11842, 30617,  2187,
      19075,  5918,   31552,  -31836, 24804,  18979,  24872,  4225,
      -21099, -28908, -29243, 12439,  4844,   15989,  31029,  -28379,
      -31491, 5831,   15558,  19643,  10858,  14717,  -99,    4882,
      25567,  16439,  21084,  16126,  -18868, -5948,  -27003, 11637,
      18606,  -22996, -17660, 25816,  6917,   -3461,  -23555, -29368};
  SET_MBA0_I16();
  msettilen(N);
  msettilek(K);
  mint16_t ms = mlb_m(src, stride);
  mint16_t md = mtb_m(ms);
  msb_m(md, i16_buffer, stride);
  EXCEPT_I16_ARRAY_EQ(ans, i16_buffer, K * N, "MTB_M I16");
}

static void test_mtb_m_i32() {
  enum { N = 4, K = 4 };
  const size_t stride = N * sizeof(int32_t);
  const int32_t src[K * N] = {
      -1681124598, -1292122543, -2028898099, -2045352292,
      -1888223806, -619176567,  2129772106,  294845680,
      -1344465868, -1642749564, 1092118040,  736267622,
      -912961762,  -655803685,  -1442065192, 467457903};
  const int32_t ans[K * N] = {
      -1681124598, -1888223806, -1344465868, -912961762,
      -1292122543, -619176567,  -1642749564, -655803685,
      -2028898099, 2129772106,  1092118040,  -1442065192,
      -2045352292, 294845680,   736267622,   467457903};
  SET_MBA0_I32();
  msettilen(N);
  msettilek(K);
  mint32_t ms = mlb_m(src, stride);
  mint32_t md = mtb_m(ms);
  msb_m(md, i32_buffer, stride);
  EXCEPT_I32_ARRAY_EQ(ans, i32_buffer, K * N, "MTB_M I32");
}

static void test_mtb_m_i64() {
  enum { N = 2, K = 2 };
  const size_t stride = N * sizeof(int64_t);
  const int64_t src[K * N] = {-125363293662549438, 7432419646672721022,
                              -2936165396028250924, -3423249875061222608};
  const int64_t ans[K * N] = {-125363293662549438, -2936165396028250924,
                              7432419646672721022, -3423249875061222608};
  SET_MBA0_I64();
  msettilen(N);
  msettilek(K);
  mint64_t ms = mlb_m(src, stride);
  mint64_t md = mtb_m(ms);
  msb_m(md, i64_buffer, stride);
  EXCEPT_I64_ARRAY_EQ(ans, i64_buffer, K * N, "MTB_M I64");
}

static void test_mtb_m_u8() {
  enum { N = 8, K = 8 };
  const size_t stride = N * sizeof(uint8_t);
  const uint8_t src[K * N] = {
      95,  250, 228, 97,  253, 236, 51,  182, 179, 21,  94,  127, 114,
      5,   231, 109, 121, 8,   212, 28,  177, 82,  40,  252, 192, 79,
      3,   221, 157, 195, 107, 24,  106, 202, 78,  144, 127, 57,  86,
      236, 59,  191, 23,  242, 32,  87,  93,  156, 91,  140, 164, 177,
      151, 179, 245, 204, 63,  212, 235, 46,  248, 86,  224, 71};
  const uint8_t ans[K * N] = {
      95,  179, 121, 192, 106, 59,  91,  63,  250, 21,  8,   79, 202,
      191, 140, 212, 228, 94,  212, 3,   78,  23,  164, 235, 97, 127,
      28,  221, 144, 242, 177, 46,  253, 114, 177, 157, 127, 32, 151,
      248, 236, 5,   82,  195, 57,  87,  179, 86,  51,  231, 40, 107,
      86,  93,  245, 224, 182, 109, 252, 24,  236, 156, 204, 71};
  SET_MBA0_I8();
  msettilen(N);
  msettilek(K);
  muint8_t ms = mlb_m(src, stride);
  muint8_t md = mtb_m(ms);
  msb_m(md, u8_buffer, stride);
  EXCEPT_U8_ARRAY_EQ(ans, u8_buffer, K * N, "MTB_M U8");
}

static void test_mtb_m_u16() {
  enum { N = 8, K = 8 };
  const size_t stride = N * sizeof(uint16_t);
  const uint16_t src[K * N] = {
      25570, 41452, 3929,  17553, 39826, 5858,  23422, 25449, 2671,  52676,
      35333, 21801, 51814, 56172, 40104, 14848, 56365, 42600, 26097, 38242,
      58392, 5342,  39285, 36714, 52224, 32844, 59183, 8869,  59262, 75,
      18977, 65231, 53459, 30958, 25173, 23693, 61305, 19295, 16173, 11282,
      64095, 33860, 52416, 20816, 8220,  51630, 27705, 48754, 24673, 46715,
      9824,  28027, 24889, 27056, 11529, 11787, 6899,  5502,  14756, 35893,
      31189, 16010, 47528, 34047};
  const uint16_t ans[K * N] = {
      25570, 2671,  56365, 52224, 53459, 64095, 24673, 6899,  41452, 52676,
      42600, 32844, 30958, 33860, 46715, 5502,  3929,  35333, 26097, 59183,
      25173, 52416, 9824,  14756, 17553, 21801, 38242, 8869,  23693, 20816,
      28027, 35893, 39826, 51814, 58392, 59262, 61305, 8220,  24889, 31189,
      5858,  56172, 5342,  75,    19295, 51630, 27056, 16010, 23422, 40104,
      39285, 18977, 16173, 27705, 11529, 47528, 25449, 14848, 36714, 65231,
      11282, 48754, 11787, 34047};
  SET_MBA0_I16();
  msettilen(N);
  msettilek(K);
  muint16_t ms = mlb_m(src, stride);
  muint16_t md = mtb_m(ms);
  msb_m(md, u16_buffer, stride);
  EXCEPT_U16_ARRAY_EQ(ans, u16_buffer, K * N, "MTB_M U16");
}

static void test_mtb_m_u32() {
  enum { N = 4, K = 4 };
  const size_t stride = N * sizeof(uint32_t);
  const uint32_t src[K * N] = {1853700213, 1224691227, 579172120,  2544000723,
                               3705217052, 3847582878, 2047817136, 2873319869,
                               944198550,  2495765483, 825898774,  920734754,
                               2888440314, 3623959478, 3650400147, 2853937346};
  const uint32_t ans[K * N] = {1853700213, 3705217052, 944198550,  2888440314,
                               1224691227, 3847582878, 2495765483, 3623959478,
                               579172120,  2047817136, 825898774,  3650400147,
                               2544000723, 2873319869, 920734754,  2853937346};
  SET_MBA0_I32();
  msettilen(N);
  msettilek(K);
  muint32_t ms = mlb_m(src, stride);
  muint32_t md = mtb_m(ms);
  msb_m(md, u32_buffer, stride);
  EXCEPT_U32_ARRAY_EQ(ans, u32_buffer, K * N, "MTB_M U32");
}

static void test_mtb_m_u64() {
  enum { N = 2, K = 2 };
  const size_t stride = N * sizeof(uint64_t);
  const uint64_t src[K * N] = {1021280658148866281, 1524863831652402790,
                               1028769707722912970, 1169303502346608508};
  const uint64_t ans[K * N] = {1021280658148866281, 1028769707722912970,
                               1524863831652402790, 1169303502346608508};
  SET_MBA0_I64();
  msettilen(N);
  msettilek(K);
  muint64_t ms = mlb_m(src, stride);
  muint64_t md = mtb_m(ms);
  msb_m(md, u64_buffer, stride);
  EXCEPT_U64_ARRAY_EQ(ans, u64_buffer, K * N, "MTB_M U64");
}

static void test_mtb_m_f16() {
  enum { N = 8, K = 8 };
  const size_t stride = N * sizeof(fp16_t);
  const fp16_t src[K * N] = {
      8.586,  -7.586, -9.125, -5.676,   -0.9844, 2.707,  1.951,  3.246,
      -1.897, 6.254,  -8.48,  2.453,    -1.325,  1.384,  -7.79,  1.528,
      -8.21,  6.438,  -9.484, 9.72,     1.674,   -9.35,  -1.623, 6.336,
      -4.633, 4.223,  3.107,  2.379,    5.547,   -1.083, 5.652,  -7.16,
      4.906,  0.6665, -2.396, -6.883,   6.082,   -7.652, 6.363,  -0.02127,
      7.344,  0.6743, -5.246, -0.773,   1.587,   -7.555, -1.096, 1.293,
      0.439,  -4.9,   -2.473, 4.9,      8.586,   4.92,   9.68,   -4.81,
      -5.55,  -9.805, -6.926, -0.06555, 1.454,   6.69,   -5.918, -0.6934};
  const fp16_t ans[K * N] = {
      8.586,   -1.897, -8.21,  -4.633, 4.906,    7.344,  0.439,  -5.55,
      -7.586,  6.254,  6.438,  4.223,  0.6665,   0.6743, -4.9,   -9.805,
      -9.125,  -8.48,  -9.484, 3.107,  -2.396,   -5.246, -2.473, -6.926,
      -5.676,  2.453,  9.72,   2.379,  -6.883,   -0.773, 4.9,    -0.06555,
      -0.9844, -1.325, 1.674,  5.547,  6.082,    1.587,  8.586,  1.454,
      2.707,   1.384,  -9.35,  -1.083, -7.652,   -7.555, 4.92,   6.69,
      1.951,   -7.79,  -1.623, 5.652,  6.363,    -1.096, 9.68,   -5.918,
      3.246,   1.528,  6.336,  -7.16,  -0.02127, 1.293,  -4.81,  -0.6934};
  SET_MBA0_F16();
  msettilen(N);
  msettilek(K);
  mfloat16_t ms = mlb_m(src, stride);
  mfloat16_t md = mtb_m(ms);
  msb_m(md, f16_buffer, stride);
  EXCEPT_F16_ARRAY_EQ(ans, f16_buffer, K * N, "MTB_M F16");
}

static void test_mtb_m_f32() {
  enum { N = 4, K = 4 };
  const size_t stride = N * sizeof(fp32_t);
  const fp32_t src[K * N] = {-5.317063,  2.6362946, 3.2381415,  6.414423,
                             -2.2291253, 3.17508,   -2.7422917, -9.390402,
                             -6.324778,  4.798573,  0.21453322, 2.723796,
                             0.8812181,  9.901346,  9.690255,   3.6316826};
  const fp32_t ans[K * N] = {-5.317063, -2.2291253, -6.324778,  0.8812181,
                             2.6362946, 3.17508,    4.798573,   9.901346,
                             3.2381415, -2.7422917, 0.21453322, 9.690255,
                             6.414423,  -9.390402,  2.723796,   3.6316826};
  SET_MBA0_F32();
  msettilen(N);
  msettilek(K);
  mfloat32_t ms = mlb_m(src, stride);
  mfloat32_t md = mtb_m(ms);
  msb_m(md, f32_buffer, stride);
  EXCEPT_F32_ARRAY_EQ(ans, f32_buffer, K * N, "MTB_M F32");
}

static void test_mtb_m_f64() {
  enum { N = 2, K = 2 };
  const size_t stride = N * sizeof(fp64_t);
  const fp64_t src[K * N] = {-4.97872986, 4.66672172, 0.18207431, -8.41076301};
  const fp64_t ans[K * N] = {-4.97872986, 0.18207431, 4.66672172, -8.41076301};
  SET_MBA0_F64();
  msettilen(N);
  msettilek(K);
  mfloat64_t ms = mlb_m(src, stride);
  mfloat64_t md = mtb_m(ms);
  msb_m(md, f64_buffer, stride);
  EXCEPT_F64_ARRAY_EQ(ans, f64_buffer, K * N, "MTB_M F64");
}

static void test_mtb_m() {
  test_mtb_m_i8();
  test_mtb_m_i16();
  test_mtb_m_i32();
  test_mtb_m_i64();
  test_mtb_m_u8();
  test_mtb_m_u16();
  test_mtb_m_u32();
  test_mtb_m_u64();
  test_mtb_m_f16();
  test_mtb_m_f32();
  test_mtb_m_f64();
}

static void test_mtc_m_i8() {
  enum { M = 8, N = 8 };
  const size_t stride = N * sizeof(int8_t);
  const int8_t src[M * N] = {
      77,  -68, -26, 71,   -85, -69, 112,  5,    53,   -69, 83,  77,   69,
      -59, -6,  118, -97,  25,  84,  -111, 119,  -63,  3,   -30, -112, 55,
      87,  19,  37,  -123, 116, 58,  -126, 69,   -41,  -99, 10,  93,   101,
      126, 10,  84,  91,   -18, -47, -82,  -116, -86,  -68, 125, -96,  39,
      119, -62, 4,   24,   -42, 36,  6,    -32,  -120, 98,  35,  -62};
  const int8_t ans[M * N] = {
      77,   53,   -97, -112, -126, 10,  -68, -42, -68, -69, 25, 55,  69,
      84,   125,  36,  -26,  83,   84,  87,  -41, 91,  -96, 6,  71,  77,
      -111, 19,   -99, -18,  39,   -32, -85, 69,  119, 37,  10, -47, 119,
      -120, -69,  -59, -63,  -123, 93,  -82, -62, 98,  112, -6, 3,   116,
      101,  -116, 4,   35,   5,    118, -30, 58,  126, -86, 24, -62};
  SET_MBA0_I8();
  msettilem(M);
  msettilen(N);
  mint8_t ms = mlc_m(src, stride);
  mint8_t md = mtc_m(ms);
  msc_m(md, i8_buffer, stride);
  EXCEPT_I8_ARRAY_EQ(ans, i8_buffer, M * N, "MTC_M I8");
}

static void test_mtc_m_i16() {
  enum { M = 8, N = 8 };
  const size_t stride = N * sizeof(int16_t);
  const int16_t src[M * N] = {
      30929, 31992,  -2344,  19075,  -21099, -31491, 25567,  18606,
      15671, -29372, 15063,  5918,   -28908, 5831,   16439,  -22996,
      3983,  -17084, 4239,   31552,  -29243, 15558,  21084,  -17660,
      26409, -5278,  27318,  -31836, 12439,  19643,  16126,  25816,
      5204,  -10070, 23900,  24804,  4844,   10858,  -18868, 6917,
      -6906, 16397,  -11842, 18979,  15989,  14717,  -5948,  -3461,
      2724,  -10162, 30617,  24872,  31029,  -99,    -27003, -23555,
      13896, -6126,  2187,   4225,   -28379, 4882,   11637,  -29368};
  const int16_t ans[M * N] = {
      30929,  15671,  3983,   26409,  5204,   -6906,  2724,   13896,
      31992,  -29372, -17084, -5278,  -10070, 16397,  -10162, -6126,
      -2344,  15063,  4239,   27318,  23900,  -11842, 30617,  2187,
      19075,  5918,   31552,  -31836, 24804,  18979,  24872,  4225,
      -21099, -28908, -29243, 12439,  4844,   15989,  31029,  -28379,
      -31491, 5831,   15558,  19643,  10858,  14717,  -99,    4882,
      25567,  16439,  21084,  16126,  -18868, -5948,  -27003, 11637,
      18606,  -22996, -17660, 25816,  6917,   -3461,  -23555, -29368};
  SET_MBA0_I16();
  msettilem(M);
  msettilen(N);
  mint16_t ms = mlc_m(src, stride);
  mint16_t md = mtc_m(ms);
  msc_m(md, i16_buffer, stride);
  EXCEPT_I16_ARRAY_EQ(ans, i16_buffer, M * N, "MTC_M I16");
}

static void test_mtc_m_i32() {
  enum { M = 4, N = 4 };
  const size_t stride = N * sizeof(int32_t);
  const int32_t src[M * N] = {
      -1681124598, -1292122543, -2028898099, -2045352292,
      -1888223806, -619176567,  2129772106,  294845680,
      -1344465868, -1642749564, 1092118040,  736267622,
      -912961762,  -655803685,  -1442065192, 467457903};
  const int32_t ans[M * N] = {
      -1681124598, -1888223806, -1344465868, -912961762,
      -1292122543, -619176567,  -1642749564, -655803685,
      -2028898099, 2129772106,  1092118040,  -1442065192,
      -2045352292, 294845680,   736267622,   467457903};
  SET_MBA0_I32();
  msettilem(M);
  msettilen(N);
  mint32_t ms = mlc_m(src, stride);
  mint32_t md = mtc_m(ms);
  msc_m(md, i32_buffer, stride);
  EXCEPT_I32_ARRAY_EQ(ans, i32_buffer, M * N, "MTC_M I32");
}

static void test_mtc_m_i64() {
  enum { M = 2, N = 2 };
  const size_t stride = N * sizeof(int64_t);
  const int64_t src[M * N] = {-125363293662549438, 7432419646672721022,
                              -2936165396028250924, -3423249875061222608};
  const int64_t ans[M * N] = {-125363293662549438, -2936165396028250924,
                              7432419646672721022, -3423249875061222608};
  SET_MBA0_I64();
  msettilem(M);
  msettilen(N);
  mint64_t ms = mlc_m(src, stride);
  mint64_t md = mtc_m(ms);
  msc_m(md, i64_buffer, stride);
  EXCEPT_I64_ARRAY_EQ(ans, i64_buffer, M * N, "MTC_M I64");
}

static void test_mtc_m_u8() {
  enum { M = 8, N = 8 };
  const size_t stride = N * sizeof(uint8_t);
  const uint8_t src[M * N] = {
      95,  250, 228, 97,  253, 236, 51,  182, 179, 21,  94,  127, 114,
      5,   231, 109, 121, 8,   212, 28,  177, 82,  40,  252, 192, 79,
      3,   221, 157, 195, 107, 24,  106, 202, 78,  144, 127, 57,  86,
      236, 59,  191, 23,  242, 32,  87,  93,  156, 91,  140, 164, 177,
      151, 179, 245, 204, 63,  212, 235, 46,  248, 86,  224, 71};
  const uint8_t ans[M * N] = {
      95,  179, 121, 192, 106, 59,  91,  63,  250, 21,  8,   79, 202,
      191, 140, 212, 228, 94,  212, 3,   78,  23,  164, 235, 97, 127,
      28,  221, 144, 242, 177, 46,  253, 114, 177, 157, 127, 32, 151,
      248, 236, 5,   82,  195, 57,  87,  179, 86,  51,  231, 40, 107,
      86,  93,  245, 224, 182, 109, 252, 24,  236, 156, 204, 71};
  SET_MBA0_I8();
  msettilem(M);
  msettilen(N);
  muint8_t ms = mlc_m(src, stride);
  muint8_t md = mtc_m(ms);
  msc_m(md, u8_buffer, stride);
  EXCEPT_U8_ARRAY_EQ(ans, u8_buffer, M * N, "MTC_M U8");
}

static void test_mtc_m_u16() {
  enum { M = 8, N = 8 };
  const size_t stride = N * sizeof(uint16_t);
  const uint16_t src[M * N] = {
      25570, 41452, 3929,  17553, 39826, 5858,  23422, 25449, 2671,  52676,
      35333, 21801, 51814, 56172, 40104, 14848, 56365, 42600, 26097, 38242,
      58392, 5342,  39285, 36714, 52224, 32844, 59183, 8869,  59262, 75,
      18977, 65231, 53459, 30958, 25173, 23693, 61305, 19295, 16173, 11282,
      64095, 33860, 52416, 20816, 8220,  51630, 27705, 48754, 24673, 46715,
      9824,  28027, 24889, 27056, 11529, 11787, 6899,  5502,  14756, 35893,
      31189, 16010, 47528, 34047};
  const uint16_t ans[M * N] = {
      25570, 2671,  56365, 52224, 53459, 64095, 24673, 6899,  41452, 52676,
      42600, 32844, 30958, 33860, 46715, 5502,  3929,  35333, 26097, 59183,
      25173, 52416, 9824,  14756, 17553, 21801, 38242, 8869,  23693, 20816,
      28027, 35893, 39826, 51814, 58392, 59262, 61305, 8220,  24889, 31189,
      5858,  56172, 5342,  75,    19295, 51630, 27056, 16010, 23422, 40104,
      39285, 18977, 16173, 27705, 11529, 47528, 25449, 14848, 36714, 65231,
      11282, 48754, 11787, 34047};
  SET_MBA0_I16();
  msettilem(M);
  msettilen(N);
  muint16_t ms = mlc_m(src, stride);
  muint16_t md = mtc_m(ms);
  msc_m(md, u16_buffer, stride);
  EXCEPT_U16_ARRAY_EQ(ans, u16_buffer, M * N, "MTC_M U16");
}

static void test_mtc_m_u32() {
  enum { M = 4, N = 4 };
  const size_t stride = N * sizeof(uint32_t);
  const uint32_t src[M * N] = {1853700213, 1224691227, 579172120,  2544000723,
                               3705217052, 3847582878, 2047817136, 2873319869,
                               944198550,  2495765483, 825898774,  920734754,
                               2888440314, 3623959478, 3650400147, 2853937346};
  const uint32_t ans[M * N] = {1853700213, 3705217052, 944198550,  2888440314,
                               1224691227, 3847582878, 2495765483, 3623959478,
                               579172120,  2047817136, 825898774,  3650400147,
                               2544000723, 2873319869, 920734754,  2853937346};
  SET_MBA0_I32();
  msettilem(M);
  msettilen(N);
  muint32_t ms = mlc_m(src, stride);
  muint32_t md = mtc_m(ms);
  msc_m(md, u32_buffer, stride);
  EXCEPT_U32_ARRAY_EQ(ans, u32_buffer, M * N, "MTC_M U32");
}

static void test_mtc_m_u64() {
  enum { M = 2, N = 2 };
  const size_t stride = N * sizeof(uint64_t);
  const uint64_t src[M * N] = {1021280658148866281, 1524863831652402790,
                               1028769707722912970, 1169303502346608508};
  const uint64_t ans[M * N] = {1021280658148866281, 1028769707722912970,
                               1524863831652402790, 1169303502346608508};
  SET_MBA0_I64();
  msettilem(M);
  msettilen(N);
  muint64_t ms = mlc_m(src, stride);
  muint64_t md = mtc_m(ms);
  msc_m(md, u64_buffer, stride);
  EXCEPT_U64_ARRAY_EQ(ans, u64_buffer, M * N, "MTC_M U64");
}

static void test_mtc_m_f16() {
  enum { M = 8, N = 8 };
  const size_t stride = N * sizeof(fp16_t);
  const fp16_t src[M * N] = {
      8.586,  -7.586, -9.125, -5.676,   -0.9844, 2.707,  1.951,  3.246,
      -1.897, 6.254,  -8.48,  2.453,    -1.325,  1.384,  -7.79,  1.528,
      -8.21,  6.438,  -9.484, 9.72,     1.674,   -9.35,  -1.623, 6.336,
      -4.633, 4.223,  3.107,  2.379,    5.547,   -1.083, 5.652,  -7.16,
      4.906,  0.6665, -2.396, -6.883,   6.082,   -7.652, 6.363,  -0.02127,
      7.344,  0.6743, -5.246, -0.773,   1.587,   -7.555, -1.096, 1.293,
      0.439,  -4.9,   -2.473, 4.9,      8.586,   4.92,   9.68,   -4.81,
      -5.55,  -9.805, -6.926, -0.06555, 1.454,   6.69,   -5.918, -0.6934};
  const fp16_t ans[M * N] = {
      8.586,   -1.897, -8.21,  -4.633, 4.906,    7.344,  0.439,  -5.55,
      -7.586,  6.254,  6.438,  4.223,  0.6665,   0.6743, -4.9,   -9.805,
      -9.125,  -8.48,  -9.484, 3.107,  -2.396,   -5.246, -2.473, -6.926,
      -5.676,  2.453,  9.72,   2.379,  -6.883,   -0.773, 4.9,    -0.06555,
      -0.9844, -1.325, 1.674,  5.547,  6.082,    1.587,  8.586,  1.454,
      2.707,   1.384,  -9.35,  -1.083, -7.652,   -7.555, 4.92,   6.69,
      1.951,   -7.79,  -1.623, 5.652,  6.363,    -1.096, 9.68,   -5.918,
      3.246,   1.528,  6.336,  -7.16,  -0.02127, 1.293,  -4.81,  -0.6934};
  SET_MBA0_F16();
  msettilem(M);
  msettilen(N);
  mfloat16_t ms = mlc_m(src, stride);
  mfloat16_t md = mtc_m(ms);
  msc_m(md, f16_buffer, stride);
  EXCEPT_F16_ARRAY_EQ(ans, f16_buffer, M * N, "MTC_M F16");
}

static void test_mtc_m_f32() {
  enum { M = 4, N = 4 };
  const size_t stride = N * sizeof(fp32_t);
  const fp32_t src[M * N] = {-5.317063,  2.6362946, 3.2381415,  6.414423,
                             -2.2291253, 3.17508,   -2.7422917, -9.390402,
                             -6.324778,  4.798573,  0.21453322, 2.723796,
                             0.8812181,  9.901346,  9.690255,   3.6316826};
  const fp32_t ans[M * N] = {-5.317063, -2.2291253, -6.324778,  0.8812181,
                             2.6362946, 3.17508,    4.798573,   9.901346,
                             3.2381415, -2.7422917, 0.21453322, 9.690255,
                             6.414423,  -9.390402,  2.723796,   3.6316826};
  SET_MBA0_F32();
  msettilem(M);
  msettilen(N);
  mfloat32_t ms = mlc_m(src, stride);
  mfloat32_t md = mtc_m(ms);
  msc_m(md, f32_buffer, stride);
  EXCEPT_F32_ARRAY_EQ(ans, f32_buffer, M * N, "MTC_M F32");
}

static void test_mtc_m_f64() {
  enum { M = 2, N = 2 };
  const size_t stride = N * sizeof(fp64_t);
  const fp64_t src[M * N] = {-4.97872986, 4.66672172, 0.18207431, -8.41076301};
  const fp64_t ans[M * N] = {-4.97872986, 0.18207431, 4.66672172, -8.41076301};
  SET_MBA0_F64();
  msettilem(M);
  msettilen(N);
  mfloat64_t ms = mlc_m(src, stride);
  mfloat64_t md = mtc_m(ms);
  msc_m(md, f64_buffer, stride);
  EXCEPT_F64_ARRAY_EQ(ans, f64_buffer, M * N, "MTC_M F64");
}

static void test_mtc_m() {
  test_mtc_m_i8();
  test_mtc_m_i16();
  test_mtc_m_i32();
  test_mtc_m_i64();
  test_mtc_m_u8();
  test_mtc_m_u16();
  test_mtc_m_u32();
  test_mtc_m_u64();
  test_mtc_m_f16();
  test_mtc_m_f32();
  test_mtc_m_f64();
}

static void test_mla_m_msa_m() {
  test_mla_m_msa_m_i8();
  test_mla_m_msa_m_u8();
  test_mla_m_msa_m_i16();
  test_mla_m_msa_m_u16();
  test_mla_m_msa_m_f16();
  test_mla_m_msa_m_i32();
  test_mla_m_msa_m_u32();
  test_mla_m_msa_m_f32();
  test_mla_m_msa_m_i64();
  test_mla_m_msa_m_u64();
  test_mla_m_msa_m_f64();
}

static void test_mlb_m_msb_m() {
  test_mlb_m_msb_m_i8();
  test_mlb_m_msb_m_u8();
  test_mlb_m_msb_m_i16();
  test_mlb_m_msb_m_u16();
  test_mlb_m_msb_m_f16();
  test_mlb_m_msb_m_i32();
  test_mlb_m_msb_m_u32();
  test_mlb_m_msb_m_f32();
  test_mlb_m_msb_m_i64();
  test_mlb_m_msb_m_u64();
  test_mlb_m_msb_m_f64();
}

static void test_mlc_m_msc_m() {
  test_mlc_m_msc_m_i8();
  test_mlc_m_msc_m_u8();
  test_mlc_m_msc_m_i16();
  test_mlc_m_msc_m_u16();
  test_mlc_m_msc_m_f16();
  test_mlc_m_msc_m_i32();
  test_mlc_m_msc_m_u32();
  test_mlc_m_msc_m_f32();
  test_mlc_m_msc_m_i64();
  test_mlc_m_msc_m_u64();
  test_mlc_m_msc_m_f64();
}

static void test_mltr_m_mstr_m() {
  test_mltr_m_mstr_m_i8();
  test_mltr_m_mstr_m_u8();
  test_mltr_m_mstr_m_i16();
  test_mltr_m_mstr_m_u16();
  test_mltr_m_mstr_m_f16();
  test_mltr_m_mstr_m_i32();
  test_mltr_m_mstr_m_u32();
  test_mltr_m_mstr_m_f32();
  test_mltr_m_mstr_m_i64();
  test_mltr_m_mstr_m_u64();
  test_mltr_m_mstr_m_f64();
}

static void test_mlacc_m_msacc_m() {
  test_mlacc_m_msacc_m_i8();
  test_mlacc_m_msacc_m_u8();
  test_mlacc_m_msacc_m_i16();
  test_mlacc_m_msacc_m_u16();
  test_mlacc_m_msacc_m_f16();
  test_mlacc_m_msacc_m_i32();
  test_mlacc_m_msacc_m_u32();
  test_mlacc_m_msacc_m_f32();
  test_mlacc_m_msacc_m_i64();
  test_mlacc_m_msacc_m_u64();
  test_mlacc_m_msacc_m_f64();
}

static void test_mlat_m_msat_m() {
  test_mlat_m_msat_m_i8();
  test_mlat_m_msat_m_u8();
  test_mlat_m_msat_m_i16();
  test_mlat_m_msat_m_u16();
  test_mlat_m_msat_m_f16();
  test_mlat_m_msat_m_i32();
  test_mlat_m_msat_m_u32();
  test_mlat_m_msat_m_f32();
  test_mlat_m_msat_m_i64();
  test_mlat_m_msat_m_u64();
  test_mlat_m_msat_m_f64();
}

static void test_mlbt_m_msbt_m() {
  test_mlbt_m_msbt_m_i8();
  test_mlbt_m_msbt_m_u8();
  test_mlbt_m_msbt_m_i16();
  test_mlbt_m_msbt_m_u16();
  test_mlbt_m_msbt_m_f16();
  test_mlbt_m_msbt_m_i32();
  test_mlbt_m_msbt_m_u32();
  test_mlbt_m_msbt_m_f32();
  test_mlbt_m_msbt_m_i64();
  test_mlbt_m_msbt_m_u64();
  test_mlbt_m_msbt_m_f64();
}

static void test_mlct_m_msct_m() {
  test_mlct_m_msct_m_i8();
  test_mlct_m_msct_m_u8();
  test_mlct_m_msct_m_i16();
  test_mlct_m_msct_m_u16();
  test_mlct_m_msct_m_f16();
  test_mlct_m_msct_m_i32();
  test_mlct_m_msct_m_u32();
  test_mlct_m_msct_m_f32();
  test_mlct_m_msct_m_i64();
  test_mlct_m_msct_m_u64();
  test_mlct_m_msct_m_f64();
}

static void test_load_store() {
  test_mla_m_msa_m();
  test_mlb_m_msb_m();
  test_mlc_m_msc_m();
  test_mltr_m_mstr_m();
  test_mlacc_m_msacc_m();
  test_mlat_m_msat_m();
  test_mlbt_m_msbt_m();
  test_mlct_m_msct_m();
  test_mta_m();
  test_mtb_m();
  test_mtc_m();
}

#endif  // !MATRIX_TESTS_ISA_LOAD_STORE_H_