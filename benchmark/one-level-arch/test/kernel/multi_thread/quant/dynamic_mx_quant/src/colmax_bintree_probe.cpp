// ===========================================================================
// colmax_bintree_probe —— 纯二分树规约(stride 折半),单块 [ROWS,P] → [1,P] 列 abs-max。
//   载入 ROWS 个独立 [1,P] 行 tile → 逐轮两两 TMAX:
//     第1轮 ROWS/2 个独立 TMAX → ROWS/2 行 → … → 1 行,关键路径 = log2(ROWS)。
//   全部编译期常量下标(避开运行期索引 spill)。用来实测二分树的真实约束点:
//   ROWS(峰值活 tile)和 P(单 tile 宽度)分别调,看编译/gfrun 在哪崩、怎么崩。
// ===========================================================================
#include <common/pto_tileop.hpp>

#include <cstdint>
#include <utility>
#include <fcntl.h>
#include <unistd.h>

#include "benchmark.h"

using namespace pto;

#ifndef CBT_ROWS
#define CBT_ROWS 32          // 单块归约行数(峰值活 tile 数),2 的幂
#endif
#ifndef CBT_P
#define CBT_P 512            // 单行/单 tile 宽度(元素)
#endif
#ifndef CBT_A
#define CBT_A 256            // 总行数;块数 = CBT_A / CBT_ROWS(与基线/W 路同场景 256x512)
#endif

#define ALIGN_MASK 0xfffffffffffff000ull
#define ALIGN (4 * 1024)

#ifdef RES_CHECK
static void silent_read(const char *fn, uint8_t *p, size_t n) {
    int fd = open(fn, O_RDWR); if (fd < 0) return;
    size_t got = 0; while (got < n) { int r = read(fd, p + got, n - got); if (r <= 0) break; got += r; }
    close(fd);
}
static void silent_write(const char *fn, const uint8_t *p, size_t n) {
    int fd = open(fn, O_CREAT | O_RDWR | O_TRUNC, 0644); if (fd < 0) return;
    size_t put = 0; while (put < n) { int w = write(fd, p + put, n - put); if (w <= 0) break; put += w; }
    close(fd);
}
#endif

template <int P> using row_t = Tile<Location::Vec, float, 1, P, BLayout::RowMajor>;

// 递归二分:N 个活 tile → 每轮 N/2 个独立 TMAX → 递归 N/2。全常量下标。
template <int N, int P>
static void tree_fold(row_t<P> *v) {
    if constexpr (N > 1) {
        [&]<int... J>(std::integer_sequence<int, J...>) {
            (TMAX(v[J], v[J], v[J + N / 2]), ...);       // 本轮 N/2 个独立 TMAX
        }(std::make_integer_sequence<int, N / 2>{});
        tree_fold<N / 2, P>(v);
    }
}

template <int A, int ROWS, int P>
static void colmax_bintree(float *out_ptr, float *in_ptr) {
    static_assert((ROWS & (ROWS - 1)) == 0 && ROWS >= 2, "ROWS power of two >= 2");
    static_assert(A % ROWS == 0, "A must be a multiple of ROWS");
    constexpr int numKb = A / ROWS;
    using gm_x = global_tensor<float, RowMajor<A, P>>;
    using gm_s = global_tensor<float, RowMajor<numKb, P>>;

    for (int kb = 0; kb < numKb; ++kb) {
        const int row0 = kb * ROWS;
        // 载入本块 ROWS 行 → v[0..ROWS-1]（第1轮之前全部活着，峰值 = ROWS 个 tile）。
        row_t<P> v[ROWS];
        [&]<int... K>(std::integer_sequence<int, K...>) {
            ([&] {
                global_iterator<gm_x, row_t<P>> it(in_ptr + (row0 + K) * P);
                auto g = it(0, 0);
                row_t<P> t; TLOAD(t, g);
                TABS(v[K], t);
            }(), ...);
        }(std::make_integer_sequence<int, ROWS>{});

        tree_fold<ROWS, P>(v);                            // v[0] = 该块列 abs-max

        global_iterator<gm_s, row_t<P>> os(out_ptr + kb * P);
        auto go = os(0, 0);
        TSTORE(go, v[0]);
    }
}

int main() {
    static float input_buffer[CBT_A * CBT_P + 2 * ALIGN];
    static float output_buffer[(CBT_A / CBT_ROWS) * CBT_P + 2 * ALIGN];

    float *input = reinterpret_cast<float *>(
        (reinterpret_cast<uint64_t>(input_buffer) & ALIGN_MASK) + ALIGN);
    float *output = reinterpret_cast<float *>(
        (reinterpret_cast<uint64_t>(output_buffer) & ALIGN_MASK) + ALIGN);

#ifdef RES_CHECK
    silent_read(CHK_DIR "/input.bin", reinterpret_cast<uint8_t *>(input),
                CBT_A * CBT_P * sizeof(float));
#elif defined(CBT_FILL)
    for (int i = 0; i < CBT_A * CBT_P; ++i)
        input[i] = static_cast<float>((i % 97) - 48) * 0.125f;
#endif

    BENCHSTART;
    colmax_bintree<CBT_A, CBT_ROWS, CBT_P>(output, input);
    BENCHEND;

#ifdef RES_CHECK
    silent_write(CHK_DIR "/output.bin", reinterpret_cast<uint8_t *>(output),
                 (CBT_A / CBT_ROWS) * CBT_P * sizeof(float));
#endif
    return 0;
}
