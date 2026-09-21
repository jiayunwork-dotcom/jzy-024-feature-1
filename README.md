# 开环频域稳定裕度服务

把被控对象的有理传递函数交给它，它在对数间隔频率网格上扫频并还回
**相位裕度（PM）** 与 **幅值裕度（GM）**；再给一份参数不确定范围，
它就能在整个参数族上求**最坏情况裕度**并判定**鲁棒稳定性**。
Python 3.12 + FastAPI，对象档以本地 JSON 文件持久化，无数据库进程，
单容器一条命令启动。

## 定义

对开环传递函数

```
G(s) = K · num(s)/den(s) · e^(−Ls)
```

把 `s` 换成 `jω`，分子分母在虚轴上各得一个复数，相除即 `G(jω)`；
纯延迟只动相位（相位额外减去 `ωL` 弧度），幅值曲线不变。

- 相位对外一律用**度**；
- 幅值同时给**真值**和**分贝**，且 `dB = 20·log10(|G|)`，两套数严格一致；
- **幅值穿越 ω_c**：`|G|` 穿过 1 的频率；`PM = ∠G(jω_c) + 180°`；
- **相位穿越 ω_π**：相位穿过 −180° 的频率；`GM = 1/|G(jω_π)|`；
- 穿越落在两个网格点之间时，在两点间**对分加密**到相对容差
  `|hi−lo|/|mid| < 1e-10`（`app/margins.py: FREQ_REL_TOL`）才读数，
  绝不拿邻近网格点冒充；
- 找不到有限穿越时显式标记 `exists=false / note="无有限穿越"`，
  裕度字段为 `null`，绝不填 0 或大数；
- 稳定性（最小相位）：已有的两个裕度都为正才判稳定，任一为负即不稳定。

## 鲁棒裕度分析（参数族上的最坏情况）

`POST /api/robust` 在对象描述之外再收一份不确定性说明，把整个参数族扫一遍，
回答「在这段不确定性里，回路是不是始终守得住稳定」。

### 不确定性说明

三类不确定量都可缺省（缺省 = 该维不飘、取名义值）；一个都不给等价于
在名义参数上做一次普通扫频，读数与 `/api/sweep` 逐项一致。

```jsonc
{
  "uncertainty": {
    "K":    {"min": 1.0, "max": 3.0},                 // 开环增益区间（必须为正）
    "L":    {"min": 0.0, "max": 0.2},                 // 纯延迟区间（不得含负值）
    "pole": {"nominal": -1.0, "min": -1.5, "max": -0.5} // 某个实极点的移动范围
  }
}
```

- 区间两端都必须是有限数，且 `max >= min`；
- `pole.nominal` 标识移动的是哪一个极点（必须对应对象的一个真实实极点），
  其余零极点保持标称值不动；极点区间不得越过虚轴进入右半平面
  （`max <= 0`），否则当场拒绝并说明理由，不给任何读数。

### 最坏化怎么做

- **单调维度直接坍缩，不傻扫**：延迟只压相位，PM 随 L 单调下降；增益放大
  把幅值穿越推向高频、压低 PM，且 GM 随 K 严格下降——两个裕度的最坏点
  都在 K、L 区间的**上界**；
- **极点位置非单调，做真一维搜索**：先在区间内 25 点粗扫定位最坏子区间，
  再黄金分割加密，直到参数相对容差 `1e-6`
  （`app/robust.py: PARAM_REL_TOL`），绝不拿粗扫网格点冒充极值点；
- 最坏 PM 与最坏 GM **分别**给出各自的最坏参数点（两者不一定在同一组
  参数上取到）；每个参数点的读数都由与 `/api/sweep` 同一个
  `analyze` 内核算出，口径不分叉。

### 返回（节选）

```json
{
  "worst_phase_margin": {
    "exists": true, "value_deg": -12.32,
    "params": {"K": 3.0, "L": 0.2, "pole": -0.5},
    "omega_c": 1.684, "crossover_coverage": "all", "note": null
  },
  "worst_gain_margin": {
    "exists": true, "value": 0.574, "value_db": -4.81,
    "params": {"K": 3.0, "L": 0.2, "pole": -0.5},
    "omega_pi": 1.261, "crossover_coverage": "all", "note": null
  },
  "robustly_stable": false,
  "first_violation": {"margin": "phase_margin",
                      "params": {"K": 3.0, "L": 0.2, "pole": -0.5},
                      "value_deg": -12.32, "omega_c": 1.684}
}
```

- `crossover_coverage`：`all` 全族有穿越 / `partial` 部分成员无穿越 /
  `none` 全族无穿越。全族无有限穿越时裕度字段为 `null` 并显式标记
  「全族无有限穿越」，绝不填 0 或大数；部分无穿越时最坏值落在有穿越的
  子族上，无穿越成员不会被悄悄丢掉，而是体现在该标记里；
- **鲁棒稳定判定**：最坏 PM 与最坏 GM(dB) 都存在且严格为正才判
  `robustly_stable=true`；任一最坏裕度落到零或负，判不鲁棒并交出
  `first_violation` 里的临界参数点，作为设计需要收紧的证据。

### 正确性不变量（自动化测试锁定）

1. **退化一致性**：所有区间收成单点，最坏读数与直接扫频逐项相等；
2. **区间单调包含**：任一区间放大，最坏裕度只会变差或持平；
3. **延迟最坏端锁定**：纯延迟族的最坏 PM 恰好等于延迟上界的单点读数；
4. **逐点口径对齐**：最坏参数点回代单点扫频复算，两边数值完全一致。

鲁棒分析是一次性读数：不建流水、不写盘，持久化的仍只是对象档本身。

## 模块划分

| 文件 | 职责 |
|---|---|
| `app/validation.py` | 入参检查：有限数、真有理性、网格非空/严格递增/为正；ZPK→多项式 |
| `app/frf.py` | 频响求值：`G(jω)`、延迟相位、相位解卷绕、dB |
| `app/margins.py` | 网格穿越搜索、区间对分加密、裕度与稳定性 |
| `app/uncertainty.py` | 不确定性说明检查、参数族构造（极点搬迁：zpk 替换 / poly 综合除法） |
| `app/robust.py` | 最坏情况搜索（单调坍缩 + 极点一维最坏化）、鲁棒稳定判定 |
| `app/archive.py` | 具名对象档的本地 JSON 读写（临时文件 + 原子替换） |
| `app/models.py` | HTTP 请求体模型 |
| `app/main.py` | FastAPI 路由与启动载入 |
| `app/errors.py` | 统一业务异常（错误响应只给原因，不带任何读数） |

## 一条命令构建并启动

```bash
./run.sh                 # = docker compose up --build -d
# 或
docker compose up --build
```

服务在 `http://127.0.0.1:8000`，Swagger 在 `/docs`。
对象档存在宿主机 `./data/plants/`（挂进容器 `/data/plants`），重启后仍在。

本地开发 / 跑测试：

```bash
pip install -r requirements-dev.txt
python -m pytest          # 71 项
uvicorn app.main:app --port 8000
```

在容器内跑整套测试（镜像与启动仍沿用单容器一条命令的形态）：

```bash
docker compose --profile test run --rm margin-test
# 或：docker build --target test -t open-loop-margin:test . && docker run --rm open-loop-margin:test
```

## HTTP 接口

### 对象档管理

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/plants` | 列出全部档（分子分母或零极点、K、L） |
| POST | `/api/plants/{name}` | 新增（已存在返回 409） |
| PUT | `/api/plants/{name}` | 改完系数整档替换（校验不过旧档不动） |
| GET | `/api/plants/{name}` | 取单个档 |
| DELETE | `/api/plants/{name}` | 删除 |

### 扫频

`POST /api/sweep`，`plant` 给档名字符串或直接内联对象描述；
`K` / `L` 可在当次请求中覆盖档内值（只对本次有效，不改档）。

### 鲁棒裕度分析

`POST /api/robust`，`plant` 用法同扫频，另加可选的 `uncertainty`
不确定性说明（见上文「鲁棒裕度分析」）；返回两个最坏裕度、各自的最坏
参数点与穿越频率、全族穿越覆盖标记、鲁棒稳定判定与第一处越界参数点。

对象描述两种形式：

```jsonc
// 多项式系数（降幂）
{"type": "poly", "num": [10.0], "den": [1.0, 11.0, 10.0, 0.0], "K": 1.0, "L": 0.0}

// 零点、极点（实数或 [实部, 虚部]）+ 增益
{"type": "zpk", "zeros": [], "poles": [[0,0],[-1,0],[-10,0]], "gain": 10.0, "K": 1.0, "L": 0.0}
```

示例：

```bash
curl -s -X POST http://127.0.0.1:8000/api/sweep \
  -H 'Content-Type: application/json' \
  -d '{"plant":"startup_plant",
       "freqs":[0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50,100]}'

# 鲁棒分析：K∈[1,3]、L∈[0,0.2]、极点 -1 在 [-1.5,-0.5] 上飘
curl -s -X POST http://127.0.0.1:8000/api/robust \
  -H 'Content-Type: application/json' \
  -d '{"plant":"startup_plant",
       "freqs":[0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50,100],
       "uncertainty":{"K":{"min":1,"max":3},"L":{"min":0,"max":0.2},
                      "pole":{"nominal":-1,"min":-1.5,"max":-0.5}}}'
```

返回（节选）：

```json
{
  "bode": {"omega": [...], "magnitude": [...], "magnitude_db": [...], "phase_deg": [...]},
  "gain_crossover":  {"exists": true,  "omega_c": 0.7844, "note": null},
  "phase_crossover": {"exists": true,  "omega_pi": 3.1623, "note": null},
  "phase_margin_deg": 47.40,
  "gain_margin": 11.0,
  "gain_margin_db": 20.83,
  "stable": true
}
```

## 启动对象（可手算核对）

```
G0(s) = 10 / [ s (s+1) (s+10) ]
```

- 相位在 −90°−180° 之间，−180° 穿越：ω_π = √10 ≈ 3.1623，GM = 11（20.83 dB）；
- |G|=1 给出 ω_c ≈ 0.7844，PM ≈ 47.4°（为正、几十度量级）。

## 错误即停，不给假读数

档名对不上（404）、非真有理、空/非递增/含非正频率的网格、系数非有限数、
非法不确定区间（上界小于下界、延迟含负、增益非正、极点区间越过虚轴、
极点 nominal 对不上对象的真实极点）等，一律返回 `{"error": "中文原因"}`，
响应体里不含任何裕度/曲线数据。
