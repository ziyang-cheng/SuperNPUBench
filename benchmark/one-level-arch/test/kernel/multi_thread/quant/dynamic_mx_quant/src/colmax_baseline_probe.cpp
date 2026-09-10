// ===========================================================================
// colmax_baseline_probe —— 非尾轴列 abs-max 归约的 TCOLMAX 基线,用于和
// colmax_tree_probe(行循环 + 多路 TMAX)做同场景性能对照。
//
// 每块整体 load 成 [BS,P] tile,一条 TCOLMAX 沿 BS 行归约 → [1,P]。
// 与树形探针完全同场景:[A,P]=[256,512],BS=32,fp32,静音文件 I/O,
// 默认不填充输入(static 零初始化;max 时序与数据无关,cycle 干净)。
//
// ⚠ 数值:当前 model 存在 #63(TCOLMAX/TCOL* 绑目标几何后只归约 source 第 0 行),
//   故本基线 res_check 会 DIFF(输出 == abs(x[每块第0行]))。它代表"预期快路径"
//   的 cycle 成本口径,不代表可用的正确结果 —— 正确性看 colmax_tree_probe。
// ===========================================================================
#include <common/pto_tileop.hpp>

#include <cstdint>
#include <fcntl.h>
#include <unistd.h>

#include "benchmark.h"

using namespace pto;

#ifndef CMT_A
#define CMT_A 256
#endif
#ifndef CMT_P
#define CMT_P 512
#endif
#ifndef CMT_BS
#define CMT_BS 32
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

template <int A, int P, int BS>
static void colmax_baseline(float *out_ptr, float *in_ptr) {
    static_assert(A % BS == 0, "A must be a multiple of BS");
    constexpr int numKb = A / BS;

    using gm_x   = global_tensor<float, RowMajor<A, P>>;
    using gm_s   = global_tensor<float, RowMajor<numKb, P>>;
    using tile_b = Tile<Location::Vec, float, BS, P, BLayout::RowMajor>;
    using tile_c = Tile<Location::Vec, float, 1, P, BLayout::RowMajor, 1, P>;

    for (int kb = 0; kb < numKb; ++kb) {
        global_iterator<gm_x, tile_b> it(in_ptr + kb * BS * P);
        auto g = it(0, 0);
        tile_b xin, absb;
        TLOAD(xin, g);
        TABS(absb, xin);
        tile_c acc;
        TCOLMAX(acc, absb);                 // 慢轴归约(单条,vs 树形的 BS-1 个 TMAX)

        global_iterator<gm_s, tile_c> os(out_ptr + kb * P);
        auto go = os(0, 0);
        TSTORE(go, acc);
    }
}

int main() {
    static float input_buffer[CMT_A * CMT_P + 2 * ALIGN];
    static float output_buffer[(CMT_A / CMT_BS) * CMT_P + 2 * ALIGN];

    float *input = reinterpret_cast<float *>(
        (reinterpret_cast<uint64_t>(input_buffer) & ALIGN_MASK) + ALIGN);
    float *output = reinterpret_cast<float *>(
        (reinterpret_cast<uint64_t>(output_buffer) & ALIGN_MASK) + ALIGN);

#ifdef RES_CHECK
    silent_read(CHK_DIR "/input.bin", reinterpret_cast<uint8_t *>(input),
                CMT_A * CMT_P * sizeof(float));
#elif defined(CMT_FILL)
    for (int i = 0; i < CMT_A * CMT_P; ++i)
        input[i] = static_cast<float>((i % 97) - 48) * 0.125f;
#endif

    BENCHSTART;
    colmax_baseline<CMT_A, CMT_P, CMT_BS>(output, input);
    BENCHEND;

#ifdef RES_CHECK
    silent_write(CHK_DIR "/output.bin", reinterpret_cast<uint8_t *>(output),
                 (CMT_A / CMT_BS) * CMT_P * sizeof(float));
#endif
    return 0;
}
