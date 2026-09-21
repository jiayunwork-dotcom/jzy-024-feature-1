# 开环频域稳定裕度服务

把被控对象的有理传递函数交给它，它在对数间隔频率网格上扫频并还回
**相位裕度（PM）** 与 **幅值裕度（GM）**。Python 3.12 + FastAPI，
对象档以本地 JSON 文件持久化，无数据库进程，单容器一条命令启动。

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

## 模块划分

| 文件 | 职责 |
|---|---|
| `app/validation.py` | 入参检查：有限数、真有理性、网格非空/严格递增/为正；ZPK→多项式 |
| `app/frf.py` | 频响求值：`G(jω)`、延迟相位、相位解卷绕、dB |
| `app/margins.py` | 网格穿越搜索、区间对分加密、裕度与稳定性 |
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
python -m pytest          # 33 项
uvicorn app.main:app --port 8000
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

档名对不上（404）、非真有理、空/非递增/含非正频率的网格、系数非有限数等，
一律返回 `{"error": "中文原因"}`，响应体里不含任何裕度/曲线数据。
