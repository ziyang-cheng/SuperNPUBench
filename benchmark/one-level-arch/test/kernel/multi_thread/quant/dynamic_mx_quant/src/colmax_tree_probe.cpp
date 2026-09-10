// ===========================================================================
// colmax_tree_probe —— 非尾轴列 abs-max 归约。blocksize 外提循环,每行是独立
// [1,P] tile,行间普通 TMAX(不碰 subview / 不碰 TCOLMAX,避 #63)。
//
// 依赖结构可调(CMT_W = 独立累加器路数,须为 2 的幂):
//   W=1 : 线性链  TMAX(acc,acc,a) ×(BS-1)     关键路径 = BS-1(串行,OoO 无从并发)
//   W>1 : W 路独立链 + 末尾 log2(W) 树合并      关键路径 ≈ BS/W-1 + log2(W)
//         W 路互不依赖 → 超标量层内并发;活 tile 仅 ~W+2,不溢出寄存器。
// TMAX 总数恒 = BS-1(与 W 无关),只改依赖结构 → 量 OoO 并发收益。
//
// 固定场景 [A,P]=[256,512],BS=32,fp32(归约=max,dtype 无关,比对干净)。
// 输出 [A/BS,P] 每块列 abs-max。golden = np.abs(x).reshape(A/BS,BS,P).max(1)。
//
// 自带静音文件 I/O(不引 fileop.h,避开其 printf(stdout) 在新 musl 上的 writev 挂起)。
// ===========================================================================
#include <common/pto_tileop.hpp>

#include <cstdint>
#include <utility>
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
#ifndef CMT_W
#define CMT_W 8            // 独立累加器路数(ILP 宽度),2 的幂
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

template <int A, int P, int BS, int W>
static void colmax_tree(float *out_ptr, float *in_ptr) {
    static_assert(A % BS == 0, "A must be a multiple of BS");
    static_assert((W & (W - 1)) == 0 && W >= 1, "W must be a power of two");
    static_assert(BS % W == 0, "BS must be a multiple of W");
    constexpr int numKb = A / BS;

    using gm_x     = global_tensor<float, RowMajor<A, P>>;
    using gm_s     = global_tensor<float, RowMajor<numKb, P>>;
    using tile_row = Tile<Location::Vec, float, 1, P, BLayout::RowMajor>;

    // lane K(编译期常量)load+abs 第 gRow 行到 dst —— tile 索引恒为常量,load 地址运行期。
    auto load_abs = [&](tile_row &dst, int gRow) {
        global_iterator<gm_x, tile_row> it(in_ptr + gRow * P);
        auto g = it(0, 0);
        tile_row t; TLOAD(t, g);
        TABS(dst, t);
    };
    auto fold_one = [&](tile_row &acck, int gRow) {
        global_iterator<gm_x, tile_row> it(in_ptr + gRow * P);
        auto g = it(0, 0);
        tile_row t; TLOAD(t, g);
        tile_row a; TABS(a, t);
        TMAX(acck, acck, a);
    };

    for (int kb = 0; kb < numKb; ++kb) {
        const int row0 = kb * BS;

        // W 路累加器,均以**编译期常量**下标 K 访问(避免运行期索引导致整个 tile 数组 spill)。
        tile_row acc[W];
        [&]<int... K>(std::integer_sequence<int, K...>) {
            (load_abs(acc[K], row0 + K), ...);              // lane K <- row0+K
        }(std::make_integer_sequence<int, W>{});

        // 每一"行组" m 里,W 条 lane 互不依赖(不同常量 K)→ OoO 层内并发发射。
        for (int m = 1; m < BS / W; ++m) {
            const int base = row0 + m * W;
            [&]<int... K>(std::integer_sequence<int, K...>) {
                (fold_one(acc[K], base + K), ...);          // acc[K] = max(acc[K], |row base+K|)
            }(std::make_integer_sequence<int, W>{});
        }

        // 末尾把 acc[1..W-1] 合并进 acc[0](W-1 个 TMAX,下标全常量;W 小,串行即可)。
        [&]<int... K>(std::integer_sequence<int, K...>) {
            (TMAX(acc[0], acc[0], acc[K + 1]), ...);
        }(std::make_integer_sequence<int, W - 1>{});

        global_iterator<gm_s, tile_row> os(out_ptr + kb * P);
        auto go = os(0, 0);
        TSTORE(go, acc[0]);
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
    // 仅在需要真实数据时填充(会引入 A*P 次标量循环拖慢 gfsim)。cycle 测量默认不填充:
    // static 缓冲天然零初始化,max 时序与数据无关,Total Cycles≈kernel 本身。
    for (int i = 0; i < CMT_A * CMT_P; ++i)
        input[i] = static_cast<float>((i % 97) - 48) * 0.125f;
#endif

    BENCHSTART;
    colmax_tree<CMT_A, CMT_P, CMT_BS, CMT_W>(output, input);
    BENCHEND;

#ifdef RES_CHECK
    silent_write(CHK_DIR "/output.bin", reinterpret_cast<uint8_t *>(output),
                 (CMT_A / CMT_BS) * CMT_P * sizeof(float));
#endif
    return 0;
}
