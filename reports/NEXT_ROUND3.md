# Round-3 Runbook — 双卡解耦 + 多 trial 加固 + 配方精扫(公司 2×5090)

目标:把单卡"serve↔train 交替"升级为**双卡解耦**(vLLM 常驻 GPU0 + 训练独占 GPU1 + LoRA 热加载),
顺带把 0.493 用多 trial 钉死、扫一遍最优配方。一切代码已本地写好测好(v16/v17 包)。

## 你(用户)只需做的两件事
1. 周末把公司机器开好,确认两张 5090 都空闲(没人占)。
2. 给我 SSH 连接信息(`ssh -p <port> <user>@<host>` + 密码),其余我远程全包。
   - 公司机器给外部访问前请先确认合规/安全;不放心就把 runbook 发我、你贴命令执行也行。

## 我(远程)的执行流程
0. **传脚本**:scp `grpo_scripts_v17.tgz` → 解压到 `<repo>/scripts/`(含 dual / master_round3 / setup_check)。
1. **诊断**:`bash scripts/grpo/setup_check.sh` —— 摸清 GPU/CUDA、torch 能否看到 5090(sm_120)、transformers 是否 4.57.x、模型/数据在不在、磁盘。
2. **按需搭环境**(机器能联网):
   - 缺依赖 → `pip install -r requirements-training.txt`(已 pin transformers==4.57.6);
     **5090 关键**:torch 必须是支持 Blackwell 的版本(cu128+,torch≥2.7;我们 AutoDL 用的是 2.11+cu130)。版本不对就单独装匹配 CUDA 的 torch + 对应 vLLM。
   - 缺模型 → 从 ModelScope/HF 下 `Qwen2.5-7B-Instruct`。
   - HotpotQA → `datasets` 首次运行自动下(设 `HF_ENDPOINT=https://hf-mirror.com` 更稳)。
3. **smoke(必做,验唯一没在 GPU 上验过的点 = vLLM LoRA 热加载)**:
   ```
   RUN=dual_smoke REWARD=f1 LATA=1 ITERS=1 N_TRAIN_Q=8 N_EVAL_Q=20 \
   VLLM_GPU=0 TRAIN_GPU=1 POLICY_MODEL=<7B路径> WORKDIR=<repo> PYBIN=<python> \
   bash scripts/grpo/run_search_agent_dual.sh
   ```
   看三点:① vLLM 在 GPU0 起来;② 训练在 GPU1(`nvidia-smi` 两卡都动);③ 日志出现
   `[dual] LoRA policy_1 registered`(热加载成功)。失败多半是 `VLLM_ALLOW_RUNTIME_LORA_UPDATING`
   或 `--max-lora-rank` 太小 → 已在脚本里设好,真不行就回退"每轮重启 vLLM"的兼容模式。
4. **跑矩阵**:`POLICY_MODEL=... WORKDIR=... PYBIN=... bash scripts/grpo/master_round3.sh`
   - A `dual_f1lata`:复现最佳配方 + 多 trial(EVAL_TRIALS=5) **钉死 0.493 的置信区间**
   - B `dual_lata_proc`:长度归一 × 过程奖励**组合**,看能否再抬峰
   - C `dual_lata_kl`:长度归一 + 轻 KL(0.02)
   - D/E `dual_kl02 / dual_kl01`:KL 精扫,找"既稳又不压峰"的 sweet spot(对比 0.05 的 0.437)
   - 每个 run 自带 per-iter held-out eval + 末尾 `analyze_search_eval` 出完整曲线。
5. **拉结果**:打包 `outputs/dual_*/`(eval jsonl + MASTER_R3.log + 最佳 adapter)→ 本地存档 → 更新报告。
   公司机器**不关机**(不是租用)。

## 这次想得到什么(成功标准)
- **系统**:双卡解耦跑通,记录每轮耗时 vs 单卡(预期省掉每轮 vLLM 启停 → 明显更快)= 一条 infra 工程亮点。
- **加固**:多 trial 把最佳配方的 pass^1 置信区间收窄,消掉"单 trial 噪声"caveat。
- **配方**:组合/KL 扫描里**只要有一个**稳定 ≥0.49 且终点即最优,就刷新或巩固纪录;
  即使都没超 0.493,"系统化扫了配方、0.493 是稳的上界"本身就是干净结论。
- 诚实优先:不为刷 1-2 个点编故事;曲线 + 显著性怎样就怎样写。

## 关键风险/注意
- **vLLM LoRA 热加载**:唯一没在 GPU 验过的,smoke 先验(见步骤 3)。
- **5090=Blackwell**:torch/vLLM 必须够新(cu128+),否则 `torch.cuda.is_available()=False` 或 sm_120 不支持。
- **transformers==4.57.6**:v5 会让 render-mask 全 0(训练静默失效);grpo_update.py 启动有自检会直接报错拦截。
- **on-policy 正确性**:dual 脚本里 collect_i 用 policy_{i-1}、train 出 policy_i,已对齐。
