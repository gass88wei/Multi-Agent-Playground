# Multi-Agent Playground

一个多智能体协作流本地 Playground。

## 项目结构

```
Multi-Agent-Playground/
├── backend/          # Python FastAPI 后端
│   ├── app/         # 应用代码
│   │   └── workflows/  # 基于Langgraph 5 种工作流实现
│   ├── skills/      # 已安装的技能
│   ├── .venv/       # Python 虚拟环境
│   └── requirements.txt
├── frontend/         # Vue 3 前端
│   └── src/
├── desktop/         # Electron 桌面端打包
└── .env             # 根目录配置文件（后端从这里读取配置）
```

## 快速开始

### 1. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，至少需要设置：

```
OPENAI_API_KEY=sk-...
```

### 2. 后端

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. 前端

```bash
cd frontend
npm install
```

## 启动服务

**后端**（端口 8011）：

```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8011 --reload
```

**前端**（代理到 `http://127.0.0.1:8011`）：

```bash
cd frontend
npm run dev
```

## 工作流类型

| 类型 | 说明                          |
|------|-----------------------------|
| `single_agent_chat` | 单智能体对话，可选最终合成器              |
| `router_specialists` | 路由器选择最佳专家智能体                |
| `planner_executor` | 规划器 → 验证器 → 分发器 → 执行器 → 合成器 |
| `supervisor_dynamic` | 监督者动态分配任务                   |
| `peer_handoff` | 智能体相互交接                     |

## 桌面端打包

将应用打包成独立的 Electron 桌面端（后端通过 PyInstaller 打包）。

前置条件：

```bash
# 在后端 venv 中安装 PyInstaller
cd backend
source .venv/bin/activate
pip install -r requirements-desktop.txt
```

**macOS**（本地未签名构建）：

```bash
cd desktop
npm install
npm run dist:mac
```

**macOS**（签名发布构建）：

```bash
cd desktop
npm run dist:mac:signed
```

**Windows**：

```bash
cd desktop
npm install
npm run dist:win
```

构建产物输出到 `desktop/release/`。

## 注意事项

- 后端配置只能从项目根目录的 `.env` 读取
- 技能（Skills）存放在 `backend/skills/`
- SQLite 数据库位于 `backend/data/playground.db`
