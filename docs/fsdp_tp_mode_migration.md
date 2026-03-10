# `fsdp_tp` 模式处理与迁移说明

本文档基于 `@sglang`（`slime-qwen/sglang-true-fp32/sglang`）中 `fsdp_tp` 的实现，整理其关键处理链路，并说明在 `@sglang-yi` 中的落地情况。

## 1. `@sglang` 中 `fsdp_tp` 的核心处理链路

`fsdp_tp` 由 `rl_on_policy_target="fsdp_tp"` 触发，关键点如下：

1. 参数入口与模式开关  
   - `server_args.py` 将 `rl_on_policy_target` 扩展为 `["fsdp", "fsdp_tp"]`。  
   - 开启后会强制 deterministic 推理，并关闭 flashinfer allreduce fusion（避免与该模式冲突）。

2. LayerNorm 路径  
   - `layernorm.py` 在 batch-invariant 模式下，`fsdp_tp` 会走 `forward_native`，避免 batch-invariant kernel 与该模式组合带来的不一致。

3. TP all-reduce 路径（通信）  
   - `communicator.py`、`qwen3_moe.py`、`dp_attention.py` 在 `fsdp_tp` 下会走专门分支。  
   - 原始实现中优先尝试 tree all-reduce（依赖 `tensor_model_parallel_tree_all_reduce` / `tp_invariant_ops`），否则回退普通 all-reduce。

4. CUDA graph 下 prefill/decode 的模式切换  
   - `cuda_graph_runner.py` 在 prefill-only deterministic 逻辑里：
     - decode 阶段暂时清空 `rl_on_policy_target`；
     - prefill 阶段恢复为 `"fsdp_tp"`；
     - 并同步 deterministic / allreduce fusion 的相关状态。

5. MoE 路由行为  
   - `qwen3_moe.py` 在 `rl_on_policy_target is not None` 下使用显式 `softmax + topk` 路由输出（`StandardTopKOutput`），保证路由行为在该模式下更可控。

## 2. 在 `@sglang-yi` 的迁移落地

已在 `sglang-yi` 完成以下迁移：

1. `python/sglang/srt/server_args.py`  
   - `RL_ON_POLICY_TARGET_CHOICES` 扩展为 `["fsdp", "fsdp_tp"]`。  
   - 在 deterministic 处理逻辑中补充 `fsdp_tp` 相关分支，显式关闭 flashinfer allreduce fusion 并记录 warning。

2. `python/sglang/srt/layers/layernorm.py`  
   - batch-invariant 条件中加入 `rl_on_policy_target == "fsdp_tp"`，与 `fsdp` 保持一致策略，走 `forward_native`。

3. `python/sglang/srt/model_executor/cuda_graph_runner.py`  
   - prefill-only deterministic 切换中增加 `rl_on_policy_target` 同步：  
     - decode 时置 `None`；  
     - prefill 时置 `"fsdp_tp"`；  
     - 同步更新 `get_global_server_args()`。
   - 与 `@sglang` 对齐的字段同步清单（对应 `cuda_graph_runner.py:502-513` 及其 decode 对偶段）：
     - `enable_deterministic_inference`：decode=False / prefill=True（本地 + 全局）
     - `enable_flashinfer_allreduce_fusion`：decode=True / prefill=False（本地 + 全局）
     - `rl_on_policy_target`：decode=None / prefill="fsdp_tp"（本地 + 全局）
     - `attn_backend.num_splits`：decode=0 / prefill=1
     - `disable_custom_all_reduce`：decode=False / prefill=True
     - `SGLANG_ENABLE_DETERMINISTIC_INFERENCE`：decode=0 / prefill=1
     - `NCCL_ALGO` 或 `ACCL_BINARY_TREE_ENABLE`：按分支切换通信后端
     - `batch_invariant_mode`：decode 关闭 / prefill 开启
     - `tp_invariant_mode`（仅含该模块的分支）：decode 关闭 / prefill 开启

4. `python/sglang/srt/models/qwen3_moe.py`  
   - 增加 `rl_on_policy_target is not None` 下的显式 `softmax + topk` 路由（`StandardTopKOutput`）。  
   - 保留原有 TP all-reduce 行为。

5. `python/sglang/srt/model_executor/model_runner.py`  
   - 对 `fsdp_tp` 增加 `enable_tp_invariant_mode` 的尝试性启用；若当前代码树没有 `tp_invariant_ops`，则记录 warning 并安全跳过。

## 3. 与源实现的差异说明（更新）

本次补充迁移后，以下模块/能力已落地到 `@sglang-yi`：

- `tensor_model_parallel_tree_all_reduce`（`distributed/communication_op.py`）
- `sglang.srt.tp_invariant_ops` 完整包（含 `tree_all_reduce_sum`、`moe_sum_tree_reduce`、`enable_tp_invariant_mode` 等）

并已接入以下关键调用点：

- `model_runner.py`：`fsdp_tp` 下启用 `enable_tp_invariant_mode`
- `communicator.py` / `qwen3_moe.py` / `dp_attention.py`：`fsdp_tp` 下根据 `ACCL_BINARY_TREE_ENABLE` 选择 tree all-reduce 或普通 all-reduce
- `fused_moe_triton/fused_moe.py`：`fsdp_tp` 下走 `moe_sum_tree_reduce`

## 4. 建议验证项

1. 参数可用性  
   - 启动参数中传入 `--rl-on-policy-target fsdp_tp`，确认参数校验通过。

2. 运行时状态切换  
   - 在 prefill-only deterministic 场景下，确认 prefill/decode 期间 `rl_on_policy_target` 切换符合预期。

3. 模型前向稳定性  
   - 使用 `qwen3_moe` 路径进行一次短跑，确认路由输出与 all-reduce 路径无异常。

## 5. `fsdp` 模型差距复核（本轮）

对照 `@sglang` 后，本轮补齐了两处 `fsdp` 相关模型参数对齐点：

1. `python/sglang/srt/models/qwen2_moe.py`  
   - 在 `Qwen2MoeModel` 的最终 `self.norm` 初始化中，补上 `rl_on_policy_target is not None` 时的 `norm_kwargs`：  
     - `cast_x_before_out_mul=True`  
     - `fp32_residual=False`

2. `python/sglang/srt/models/qwen3_moe.py`  
   - 在 `Qwen3MoeAttention` 的 `q_norm/k_norm` 初始化中，补上与 `@sglang` 一致的 `norm_kwargs`（同上两项）。

说明：

- `qwen3_moe` 中 `compatible_with_fused_kv_buffer` 在 `sglang-yi` 仍保持为 `False` 的当前实现（这部分属于 `sglang-yi` 本身路径策略，不仅仅是 `fsdp` 分支），本轮未强行改写，以避免引入额外行为变化。

## 6. 新 `@sglang` 分支同步（本轮）

已将 `cuda_graph_runner.py:502-513` 对应语义同步到 `@sglang`：

- 目标文件：`slime-qwen/sglang-true-fp32/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py`
- 修正内容：
  - 在 prefill-only deterministic patch 的 decode/prefill 双分支中，补齐
    `enable_flashinfer_allreduce_fusion` 的本地与全局同步更新；
  - 使其与 `sglang-yi` 这段状态机字段切换语义保持一致。

