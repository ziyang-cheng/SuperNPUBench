# DynamicMxQuant test entries. Only DEBUGGED configs (op-aligned to AscendC, all
# known issues recorded in RECORD.md) keep a build entry here. UNDEBUGGED kernels
# (tail/nontail OCP-FP8, tail/nontail DynRange-FP4) have had their driver cpp and
# TYPE blocks removed on purpose — see README.md「状态总览」. FP4_PROBE and
# BS512_CUBLAS (a TileSize probe over the debugged cublas kernel) are kept as tooling.

ifeq ($(TYPE), TAIL_CUBLAS_FP8)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/tail_cublas_fp8.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_tail_cublas_fp8.elf
endif

ifeq ($(TYPE), TAIL_CUBLAS_FP8_FP16)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/tail_cublas_fp8_fp16.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_tail_cublas_fp8_fp16.elf
endif

ifeq ($(TYPE), NONTAIL_CUBLAS_FP8_FP16)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/nontail_cublas_fp8_fp16.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_nontail_cublas_fp8_fp16.elf
endif

ifeq ($(TYPE), TAIL_OCP_FP4)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/tail_ocp_fp4.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_tail_ocp_fp4.elf
endif

ifeq ($(TYPE), NONTAIL_CUBLAS_FP8)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/nontail_cublas_fp8.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_nontail_cublas_fp8.elf
endif

ifeq ($(TYPE), NONTAIL_CUBLAS_FP8_BS128)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/nontail_cublas_fp8_bs128.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_nontail_cublas_fp8_bs128.elf
endif

ifeq ($(TYPE), NONTAIL_OCP_FP4)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/nontail_ocp_fp4.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_nontail_ocp_fp4.elf
endif

ifeq ($(TYPE), NONTAIL_OCP_FP4_BS128)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/nontail_ocp_fp4_bs128.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_nontail_ocp_fp4_bs128.elf
endif

ifeq ($(TYPE), FP4_PROBE)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/fp4_probe.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_fp4_probe.elf
endif

ifeq ($(TYPE), DTYPE_PROBE)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/dtype_probe.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_dtype_probe.elf
endif

ifeq ($(TYPE), BS512_CUBLAS)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/bs512_cublas.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_bs512_cublas.elf
endif

ifeq ($(TYPE), PROBE_OCP_FP8)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/probe_ocp_fp8.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_probe_ocp_fp8.elf
endif

ifeq ($(TYPE), PROBE_OCP_FP8_NEWCALC)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/probe_ocp_fp8_newcalc.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_probe_ocp_fp8_newcalc.elf
endif

# 正式 TAIL_OCP_FP8 kernel（固定 SPMD 4-PE，去 probe 化自 newcalc_mt 母本）。
# 必须用 4 线程跑：gfrun -s softcore.multiThreadNum=4 / gfsim --conf fourpe。
ifeq ($(TYPE), TAIL_OCP_FP8)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/tail_ocp_fp8.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_tail_ocp_fp8.elf
endif

# TAIL_OCP_FP8 运行期动态 shape 版（M/N 运行期 tiling 传入，BlockSize/kPeNum 模板）。
# kPeNum=4 时必须用 4 线程跑：gfrun -s softcore.multiThreadNum=4。
ifeq ($(TYPE), TAIL_OCP_FP8_DYN)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/tail_ocp_fp8_dyn.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_tail_ocp_fp8_dyn.elf
endif

# TAIL_OCP_FP4 运行期动态 shape 版（M/N 运行期 tiling 传入；fp4 打包 + 三守卫 recip）。
# kPeNum=4 时必须用 4 线程跑：gfrun -s softcore.multiThreadNum=4。
ifeq ($(TYPE), TAIL_OCP_FP4_DYN)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/tail_ocp_fp4_dyn.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_tail_ocp_fp4_dyn.elf
endif

# TAIL_CUBLAS_FP8 运行期动态 shape 版（M/K 运行期 tiling 传入；cuBLAS scale，fp16 in）。
# kPeNum=4 时必须用 4 线程跑：gfrun -s softcore.multiThreadNum=4。
ifeq ($(TYPE), TAIL_CUBLAS_FP8_DYN)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/tail_cublas_fp8_dyn.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_tail_cublas_fp8_dyn.elf
endif

# 正式 TAIL_CUBLAS_FP8 kernel（固定 SPMD 4-PE，kPeNum=4，fp16 in）。
# 必须用 4 线程跑：gfrun -s softcore.multiThreadNum=4。
ifeq ($(TYPE), TAIL_CUBLAS_FP8_4PE)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/tail_cublas_fp8_4pe.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_tail_cublas_fp8_4pe.elf
endif

# 正式 NONTAIL_CUBLAS_FP8 kernel（固定 SPMD 4-PE，kPeNum=4，按块行切分，fp16 in）。
# 必须用 4 线程跑：gfrun -s softcore.multiThreadNum=4。
ifeq ($(TYPE), NONTAIL_CUBLAS_FP8_4PE)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/nontail_cublas_fp8_4pe.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_nontail_cublas_fp8_4pe.elf
endif

# NONTAIL_CUBLAS_FP8 运行期动态 shape 版（Axis/Post 运行期 tiling 传入；约束 Post%TileN==0）。
# kPeNum=4 时必须用 4 线程跑：gfrun -s softcore.multiThreadNum=4。
ifeq ($(TYPE), NONTAIL_CUBLAS_FP8_DYN)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/nontail_cublas_fp8_dyn.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_nontail_cublas_fp8_dyn.elf
endif

# NONTAIL_CUBLAS_FP8 动态 shape + 按自由轴 N（列 tile）切分 4-PE 版。
ifeq ($(TYPE), NONTAIL_CUBLAS_FP8_SPLITN_DYN)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/nontail_cublas_fp8_splitN_dyn.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_nontail_cublas_fp8_splitN_dyn.elf
endif

# 正式 NONTAIL_OCP_FP4 kernel（固定 SPMD 4-PE，kPeNum=4，按块行切分，fp16 in）。
# 必须用 4 线程跑：gfrun -s softcore.multiThreadNum=4。
ifeq ($(TYPE), NONTAIL_OCP_FP4_4PE)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/nontail_ocp_fp4_4pe.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_nontail_ocp_fp4_4pe.elf
endif

# NONTAIL_OCP_FP4 运行期动态 shape 版（Axis/Post 运行期 tiling 传入；约束 Post%TileN==0）。
# kPeNum=4 时必须用 4 线程跑：gfrun -s softcore.multiThreadNum=4。
ifeq ($(TYPE), NONTAIL_OCP_FP4_DYN)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/nontail_ocp_fp4_dyn.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_nontail_ocp_fp4_dyn.elf
endif

# NONTAIL_OCP_FP4 动态 shape + 按自由轴 N（列 tile）切分 4-PE 版。
ifeq ($(TYPE), NONTAIL_OCP_FP4_SPLITN_DYN)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/nontail_ocp_fp4_splitN_dyn.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_nontail_ocp_fp4_splitN_dyn.elf
endif

ifeq ($(TYPE), ALIGN32_PROBE)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/align32_probe.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_align32_probe.elf
endif

ifeq ($(TYPE), ROW32_PROBE)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/row32_probe.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_row32_probe.elf
endif

ifeq ($(TYPE), TSEL_INPLACE_PROBE)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/tsel_inplace_probe.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_tsel_inplace_probe.elf
endif

ifeq ($(TYPE), TSEL_INPLACE_FALSESRC_PROBE)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/tsel_inplace_falsesrc_probe.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_tsel_inplace_falsesrc_probe.elf
endif

ifeq ($(TYPE), FP4_SHAPE_PROBE)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/fp4_shape_probe.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_fp4_shape_probe.elf
   ifeq ($(WIDEN), on)
     DEFINES += -DWIDEN
   endif
endif

ifeq ($(TYPE), TABS_TROWMAX_PROBE)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/tabs_trowmax_probe.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_tabs_trowmax_probe.elf
endif

ifeq ($(TYPE), TCVT_REDUCE_SHAPE_PROBE)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/tcvt_reduce_shape_probe.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_tcvt_reduce_shape_probe.elf
endif

# 可选 shape 覆盖（4-PE res_check 多 Post 一致性验证，issue #489）：命令行提供时
# 转发为 -D 宏，使同一 driver 可编不同 shape；driver 内 #ifndef 默认值不受影响。
# tail driver 用 PM/PN；nontail driver 用 PAXIS/PPOST。
ifdef PM
   DEFINES += -DPM=$(PM)
endif
ifdef PN
   DEFINES += -DPN=$(PN)
endif
ifdef PAXIS
   DEFINES += -DPAXIS=$(PAXIS)
endif
ifdef PPOST
   DEFINES += -DPPOST=$(PPOST)
endif

ifeq ($(TYPE), ROWMAX_TAIL_PROBE)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/rowmax_tail_probe.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_rowmax_tail_probe.elf
endif
ifdef CTL_TILEM
   DEFINES += -DCTL_TILEM=$(CTL_TILEM)
endif
ifeq ($(TYPE), COLMAX_BINTREE_PROBE)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/colmax_bintree_probe.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_colmax_bintree_probe.elf
endif
ifdef CBT_A
   DEFINES += -DCBT_A=$(CBT_A)
endif
ifdef CBT_ROWS
   DEFINES += -DCBT_ROWS=$(CBT_ROWS)
endif
ifdef CBT_P
   DEFINES += -DCBT_P=$(CBT_P)
endif
ifeq ($(TYPE), COLMAX_BASELINE_PROBE)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/colmax_baseline_probe.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_colmax_baseline_probe.elf
endif
ifeq ($(TYPE), COLMAX_TREE_PROBE)
   SRC_FILE = $(TEST_ROOT)/$(CASE_SRC_DIR)/colmax_tree_probe.cpp
   TARGET = $(ELF_HEAD)/$(TESTCASE)_colmax_tree_probe.elf
endif
ifdef CMT_W
   DEFINES += -DCMT_W=$(CMT_W)
endif
ifdef CMT_MODE
   DEFINES += -DCMT_MODE=$(CMT_MODE)
endif

include ../../../../common/Makefile.common

clean:
	@find $(OBJ_ROOT) -type f -name "*.o" -exec rm -rf {} \;
	@rm -rf $(COMM_OBJ) $(LINK_SCRIPT)
