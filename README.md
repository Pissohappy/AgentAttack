# AgentAttack

## Title

**CATS: Consistency-Aware Backtracking Tree Search with Skill-Based Memory for Agentic Red-Teaming of Vision-Language Models**

---

## Abstract

多模态大模型（VLM/MLLM）的安全评测正在从单轮提示攻击转向“Agent 化的多轮交互红队”。现有自适应红队框架通常采用线性迭代（失败后继续改写下一步），并以案例轨迹检索作为记忆支撑，导致三个关键瓶颈：**（1）缺乏显式回溯的路径搜索，容易在局部最优附近重复试错；（2）多轮上下文一致性弱，易出现目标漂移、冗余循环与不可诊断的跑偏；（3）记忆多为轨迹级案例回放，缺少可组合、可参数化、可迁移的“技能”表征。**
本研究提出 CATS 框架：以 **回溯式树搜索**作为规划器主干，用**一致性约束与一致性评分**把长程目标维护变成可优化对象，并引入 **skills-based memory（宏技能/选项）**从历史轨迹中抽取可复用子过程、降低推理开销与交互成本。我们将构建可复现的评测协议与指标体系，重点衡量：覆盖度、成本、恢复能力（recovery）、一致性（drift/redundancy）、技能迁移与审计性，为 VLM 安全评测与防御（如前缀警惕/teacher prompting）提供统一的对抗评测基座。

---

## 1. Motivation & Problem Statement

### 1.1 为什么需要“Agentic 路径搜索”

多轮多模态风险往往不是“单次输入触发”，而是经由多轮语义累积、上下文绑定与策略性推进出现。红队 Agent 的本质是：在与模型的交互环境中进行**序贯决策**。但当前常见范式存在不足：

* **线性迭代**：失败后继续改写同一条链路，缺少显式回溯与分支扩展，试错成本高且易陷入局部最优；
* **一致性不可控**：长程交互容易“目标漂移/重复/自相矛盾”，造成评测不可诊断、不可复现；
* **记忆停留在案例级**：存“成功轨迹”≠ 学到“技能”，无法形成可迁移的程序性知识与宏动作。

### 1.2 研究问题（RQs）

* **RQ1（Backtracking）**：如何在黑盒交互下实现可控成本的回溯式路径搜索，使 Agent 能在失败时回到更优历史节点继续展开？
* **RQ2（Consistency）**：如何把“攻击目标在上下文的一致性”显式建模成约束与评分，减少漂移/冗余，并提供可审计诊断？
* **RQ3（Skills memory）**：如何从轨迹中抽取、维护并调用宏技能（skill/option），实现参数化固化、组合复用与跨模型迁移？

---

## 2. Key Contributions（预期贡献）

1. **回溯式树搜索红队规划器**：将 ReAct 单链交互升级为“候选前沿（frontier）+ 回溯扩展”的搜索树/beam 框架，显著降低无效迭代。
2. **一致性优先（Consistency-first）机制与指标**：提出结构化目标表示、漂移检测与冗余惩罚，把一致性从“事后观察”变成“搜索内在优化目标”。
3. **Skills-based memory**：从交互轨迹中自动归纳宏技能（可审计 DSL/状态机形式）、维护适用条件与参数后验，实现“少推理、少轮数、可迁移”的能力支撑。
4. **可复现评测协议**：构建覆盖度/一致性/恢复能力/成本/迁移的综合指标体系，为防御（如前缀警惕）提供更严格的对抗评测。

---

## 3. Method Overview（CATS 框架）

### 3.1 总体架构

我们将红队 Agent 拆成四个松耦合模块，减少“同一个模型既生成又自评”的偏置与循环：

1. **Parser/Tagger（解析与标签）**
   将 victim 输出映射为结构化标签：拒绝类型、是否偏离、是否不确定、是否重复等（仅用安全标签，不生成具体有害内容）。

2. **Planner（回溯树搜索）**
   维护搜索前沿与分支，负责选择下一步“动作”（由原子 operator 或宏 skill 实现）。

3. **Realizer（动作实现器）**
   把抽象动作实例化为下一轮交互输入（文本/图像变体的抽象接口），并调用 victim 进行交互。

4. **Checker（约束与一致性检查器）**
   对候选动作与新节点做约束检查：目标漂移、设定矛盾、无效重复、预算超限等；同时输出一致性评分供 Planner 选择。

> 你可以把它理解为：Planner 做“搜索”，Checker 把“上下文一致性”变成可计算信号，Skills memory 把“经验”从轨迹升级到技能。

---

## 3.2 回溯树搜索：从单链 ReAct 到可回撤路径搜索

### 节点定义

每个搜索节点 (n_t) 包含：

* 观察 (o_t)：对话历史摘要、图像上下文摘要、victim 最新输出标签；
* 结构化状态 (s_t)：目标/子目标、已尝试动作、预算、失败模式统计；
* 动作 (a_t)：本轮采用的 operator/skill（抽象标识，不含敏感文本）；
* 评估分数 (V(n_t))：综合“目标推进 + 一致性 + 覆盖度 + 成本”的值。

### 搜索策略（可实现为 Beam / Best-first / 轻量 MCTS）

* **Frontier 维护**：保留 K 个候选节点（beam）或优先队列（best-first）。
* **选择（Select）**：从 frontier 选取扩展节点（兼顾高分与探索）。
* **扩展（Expand）**：对该节点生成一组候选动作（原子 operator + 宏 skills），形成分支。
* **评估（Evaluate）**：通过 victim 交互与 tagger 得到新节点，计算 (V(n_{t+1}))。
* **回溯（Backtrack）**：当某分支陷入低增益/高漂移/重复循环时，停止扩展并回到 frontier 其他节点继续。

> 与线性迭代相比，这种做法的核心优势是：失败不必“硬着头皮往前改”，而是“把失败当成剪枝信号”，在更优历史节点上继续探索。

---

## 3.3 上下文一致性：把“目标维护”变成搜索内生目标

我们把一致性拆成三类可计算对象：

1. **目标一致性（Goal Consistency）**

* 用结构化目标表示（risk definition → 子目标图/序列）；
* 每轮更新当前子目标索引与完成度；
* 计算漂移：当前轮意图与目标图的匹配度下降则惩罚。

2. **叙事/设定一致性（Narrative/Constraint Consistency）**

* 维护约束集合（不得自相矛盾、不得泄露内部控制信息、预算限制等）；
* Checker 对候选动作做规则/模型双重校验（可实现为轻量分类器 + 规则）。

3. **行为一致性（Anti-redundancy）**

* 记录“动作序列签名”（operator/skill 序列 + 标签序列）；
* 重复签名/低信息增益触发惩罚或剪枝，避免循环。

### 一致性评分融入值函数

[
V(n)=\alpha \cdot \text{Progress}(n) + \beta \cdot \text{Consistency}(n) + \gamma \cdot \text{Coverage}(n) - \delta \cdot \text{Cost}(n)
]
其中 Consistency 可由 drift、矛盾率、冗余率、恢复能力等构成。

---

## 3.4 Skills-based Memory：从“存轨迹”到“存技能/宏动作”

### Skill 的最小可审计定义

一个 skill 是四元组：

* **Precondition**：适用条件（标签/状态特征触发）
* **Policy sketch**：动作骨架（抽象 DSL/状态机，不含敏感文本）
* **Parameters**：可调参数槽位与其统计/后验（“手感”固化）
* **Termination**：终止条件（达到子目标/触发拒绝类型/低增益等）

### Skill discovery（从轨迹自动抽取）

* **轨迹分段**：基于标签变化、子目标切换、得分跃迁做 segmentation；
* **子过程聚类**：把高频、稳定的动作片段聚类成候选宏技能；
* **压缩为 DSL**：仅保留 operator 序列、适用条件与参数槽位（确保可发布、可审计）。

### Skill parameter tuning（参数后验）

* 对每个 skill 的参数维护统计（例如 bandit/贝叶斯更新思路），让“哪类情况下用哪类参数更稳”逐步固化，而不是每次临场猜。

### Skill composition（宏技能组合）

* Planner 优先把宏 skill 作为扩展动作以降低推理开销；
* 失败时回退到原子 operator 再细化（层次化控制）。

### Transfer（跨模型迁移评测）

* 在 N-1 个 victim 上发现/优化 skills，直接迁移到新 victim；
* 衡量“可迁移技能比例与成功率”，而非只看总体成功率。

---

## 4. Experimental Plan

### 4.1 Victim Models

* 选择开源 + 闭源混合（确保可复现与现实意义）：若干主流 VLM/MLLM（具体名单按可获得性确定）。

### 4.2 Baselines

* **Linear ReAct**：无回溯、无 frontier，仅单链迭代；
* **ARMs-style adaptive**：策略库 + 记忆检索（轨迹级）；
* **Ours ablations**：

  * w/o backtracking（只做一致性+skills）
  * w/o consistency（只做回溯+skills）
  * w/o skills（只做回溯+一致性）
  * w/o checker（验证一致性约束的价值）

### 4.3 Metrics（重点体现你的三条主线）

**安全评测结果**

* ASR / violation rate（按公开风险定义与判定协议）
* Success@Budget：固定交互预算下的成功率

**搜索与成本**

* Avg turns / API calls / latency
* Branching efficiency：每次扩展带来的平均增益

**一致性（你们的主打）**

* Goal-drift rate：目标漂移比例
* Redundancy：重复动作/重复标签序列比例
* Recovery@k：出现拒绝/偏离后 k 轮内恢复到目标轨道的概率
* Trajectory validity：违反约束的轨迹比例

**技能（skills memory）**

* Macro-skill reuse rate：宏技能调用占比
* Skill library growth：有效技能数量与稳定性
* Transfer score：leave-one-victim-out 迁移表现
* Parameter stability：同一 skill 的参数方差随经验是否收敛

### 4.4 防御联动（可选扩展，不影响主线）

用同一 CATS 红队框架评测前缀警惕/teacher prompting 的鲁棒性—效用曲线：

* Robustness gain vs Utility drop
* Over-defense rate（对无害请求的误拒绝率）

---

## 5. Expected Outcomes & Why It Matters

我们预计 CATS 相对线性迭代类方法的核心收益在于：

* **更少无效轮数**：回溯+剪枝减少“越改越偏/重复试错”；
* **更可诊断**：一致性指标与 checker 让失败原因可归因；
* **更可迁移**：skills memory 让经验从“案例”上升为“能力单元”；
* **更适合做防御验证**：对抗更强、更结构化，能暴露前缀防御在多轮下的真实弱点与副作用。

---

## 6. Risk & Mitigation（安全与可发布性）

* **避免可滥用细节**：公开版本仅发布抽象 operator/skill DSL、标签序列与统计指标，不发布高风险原始提示与完整对话。
* **负责任披露**：发现高风险漏洞走披露流程；对闭源系统仅报告统计与类型，不扩散可复现利用链。
* **评估可靠性**：采用多 judge + 规则校验 + 人工抽检标定，降低 reward hacking 与 judge 偏差。

---

## 7. Milestones

* **M1（2–4 周）**：定义风险分类/目标图表示；实现 tagger + checker；跑通 Linear ReAct baseline
* **M2（4–8 周）**：实现 frontier/beam 回溯搜索；加入一致性评分与剪枝；完成基础消融
* **M3（8–12 周）**：实现 skill discovery（分段+聚类+DSL）；宏技能调用与回退机制
* **M4（12–16 周）**：迁移评测（leave-one-victim-out）；成本优化（缓存、早停）
* **M5（可选）**：联动评测前缀防御（teacher prompting）与鲁棒性—效用曲线

---

## 8. One-sentence Positioning（写在引言的核心定位）

**我们把多模态红队从“线性自适应改写 + 轨迹检索”推进到“回溯式路径搜索 + 一致性约束 + 技能记忆”的可审计评测范式，从而在覆盖度、成本、可诊断性与迁移性上实现系统提升。**

---
## 9. 代码框架设计与首版实现（MCP + Skills）

下面给出一个可直接扩展的首版工程骨架（已在本仓库实现）：

```text
AgentAttack/
├─ pyproject.toml
├─ README.md
└─ src/agent_attack/
   ├─ core/
   │  ├─ types.py          # 统一数据结构（Goal/Node/Action/Observation）
   │  └─ interfaces.py     # Victim/Parser/Checker/Realizer 抽象接口
   ├─ planner/
   │  └─ search.py         # Frontier best-first + backtracking 规划器
   ├─ memory/
   │  └─ skills.py         # Skill 定义、调用、在线参数更新
   ├─ mcp/
   │  └─ client.py         # MCP Registry 与工具调用抽象
   ├─ skills/
   │  └─ loader.py         # 技能持久化（JSON）
   ├─ runtime/
   │  ├─ components.py     # HeuristicTagger / ConsistencyChecker 等默认组件
   │  └─ engine.py         # CATSAttackEngine 装配入口
   └─ examples/
      └─ run_demo.py       # 本地 demo
```

### 9.1 架构分层

1. **core 层**：
   - 使用 `SearchNode` 表达树搜索节点；
   - `Action` 统一表示原子 operator 与 skill 动作（`source=operator|skill`）；
   - `ObservationTag` 作为一致性与恢复分析的统一标签空间。

2. **planner 层**：
   - `FrontierPlanner` 维护 frontier（优先队列）；
   - 每轮执行 `Select -> Expand -> Evaluate -> Backtrack`；
   - 支持 `beam_width` 与 `max_budget` 预算控制。

3. **checker/tagger 层**：
   - `HeuristicTagger` 把 victim 响应映射为标签；
   - `ConsistencyChecker` 将 progress/refusal/repetition 映射到分值，并提供 prune 规则。

4. **skills memory 层**：
   - `Skill` 采用四元组近似：preconditions、policy_steps、parameters、termination；
   - `SkillLibrary.suggest()` 根据当前标签挑选技能；
   - `observe_transition()` 用在线统计更新参数（示例里更新 `directness`）。

5. **MCP 集成层**：
   - `MCPRegistry` 解耦 planner 与具体工具实现；
   - 可把 parser、judge、external retriever、image transform 等能力注册为 MCP handler。

### 9.2 如何扩展成真实系统

- 将 `MockVictim` 替换为真实模型 API Client（OpenAI/Anthropic/自建 VLM 服务）；
- 将 `HeuristicTagger` 替换为 LLM+规则混合 judge；
- 在 `mcp/client.py` 注册外部工具，例如：
  - 检索历史技能候选；
  - 多裁判一致性评分；
  - 图像变换或 OCR 管线；
- 将 `SkillStore` 与离线 discovery（分段+聚类）联通，持续扩充 skill 库。

### 9.3 运行示例

```bash
python -m agent_attack.examples.run_demo
```

该示例会输出搜索树节点深度、动作名、节点评分，便于验证多轮回溯框架是否跑通。
