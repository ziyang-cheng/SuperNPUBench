// ===========================================================================
// rowmax_tail_probe —— 尾轴(连续轴)abs-max 规约,用 TROWMAX。与非尾轴 colmax /
// 树形探针做等工作量对比:
//   非尾轴 [256,512] bs=32:reduce 每列 32 行 → 8块×512列 = 4096 次规约,长度 32。
//   本尾轴 [512,256] bs=32:reduce 每行连续 32 元素 → 512行×8块 = 4096 次规约,长度 32。
//   总规约次数 / 规约长度 / 总元素(131072)完全一致,只是换成硬件友好的快轴。
//
// 一条 TROWMAX 把 [TileM,BS] tile 沿连续 BS 列归约成 [TileM,1] —— 一次算 TileM 个
// 规约(而非尾轴要么慢轴 TCOLMAX、要么逐行 TMAX 一次只算 1 个)。TileM 越大 op 越少。
// ===========================================================================
#include <common/pto_tileop.hpp>

#include <cstdint>
#include <fcntl.h>
#include <unistd.h>

#include "benchmark.h"

using namespace pto;

#ifndef CTL_A
#define CTL_A 512             // 行数
#endif
#ifndef CTL_P
#define CTL_P 256             // 列数(尾轴);块数 = CTL_P / CTL_BS
#endif
#ifndef CTL_BS
#define CTL_BS 32             // 尾轴规约块长
#endif
#ifndef CTL_TILEM
#define CTL_TILEM 128         // 一次 TROWMAX 覆盖的行数(op 数 = (A/TileM)*(P/BS))
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

template <int A, int P, int BS, int TileM>
static void rowmax_tail(float *out_ptr, float *in_ptr) {
    static_assert(A % TileM == 0, "A must be a multiple of TileM");
    static_assert(P % BS == 0, "P must be a multiple of BS");
    constexpr int numBlk  = P / BS;        // 每行的尾轴块数
    constexpr int numRowT = A / TileM;      // 行方向分块

    using gm_x   = global_tensor<float, RowMajor<A, P>>;
    using gm_s   = global_tensor<float, RowMajor<A, numBlk>>;
    using tile_b = Tile<Location::Vec, float, TileM, BS, BLayout::RowMajor>;
    using tile_r = Tile<Location::Vec, float, TileM, 1,  BLayout::RowMajor, TileM, 1>;

    for (int rt = 0; rt < numRowT; ++rt) {
        for (int b = 0; b < numBlk; ++b) {
            // [TileM, BS] 子块:行 rt*TileM 起、尾轴列 b*BS 起(行间 stride=P)。
            global_iterator<gm_x, tile_b> it(in_ptr + rt * TileM * P + b * BS);
            auto g = it(0, 0);
            tile_b xin, absb;
            TLOAD(xin, g);
            TABS(absb, xin);
            tile_r acc;
            TROWMAX(acc, absb);                 // 沿连续 BS 列归约 → [TileM,1](一次 TileM 个规约)

            global_iterator<gm_s, tile_r> os(out_ptr + rt * TileM * numBlk + b);
            auto go = os(0, 0);
            TSTORE(go, acc);
        }
    }
}

int main() {
    static float input_buffer[CTL_A * CTL_P + 2 * ALIGN];
    static float output_buffer[CTL_A * (CTL_P / CTL_BS) + 2 * ALIGN];

    float *input = reinterpret_cast<float *>(
        (reinterpret_cast<uint64_t>(input_buffer) & ALIGN_MASK) + ALIGN);
    float *output = reinterpret_cast<float *>(
        (reinterpret_cast<uint64_t>(output_buffer) & ALIGN_MASK) + ALIGN);

#ifdef RES_CHECK
    silent_read(CHK_DIR "/input.bin", reinterpret_cast<uint8_t *>(input),
                CTL_A * CTL_P * sizeof(float));
#elif defined(CTL_FILL)
    for (int i = 0; i < CTL_A * CTL_P; ++i)
        input[i] = static_cast<float>((i % 97) - 48) * 0.125f;
#endif

    BENCHSTART;
    rowmax_tail<CTL_A, CTL_P, CTL_BS, CTL_TILEM>(output, input);
    BENCHEND;

#ifdef RES_CHECK
    silent_write(CHK_DIR "/output.bin", reinterpret_cast<uint8_t *>(output),
                 CTL_A * (CTL_P / CTL_BS) * sizeof(float));
#endif
    return 0;
}
