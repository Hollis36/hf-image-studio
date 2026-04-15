# HF Image Studio — 部署指南

从零到本地运行或 Hugging Face Spaces 上线的安全部署流程。

---

## 前置准备

### 1. 注册 Hugging Face 账号

前往 https://huggingface.co/join 注册（免费）。

### 2. 获取 Access Token

1. 登录后访问 https://huggingface.co/settings/tokens
2. 点击 **Create new token**
3. 命名为 `image-studio`，权限选 **Read**（足够调用 Inference API）
4. 复制生成的 `hf_xxxxxxxx` Token，后续要用

> 不要把真实 token 提交到 Git 仓库，也不要写进 README、脚本或截图里。

### 3. 安装 Git 和 Git LFS

```bash
# macOS
brew install git-lfs

# Ubuntu/Debian
sudo apt install git-lfs

# 初始化 LFS
git lfs install
```

### 4. 本地环境变量约定

推荐使用未提交的本地环境文件或 shell 环境变量管理 `HF_TOKEN`：

```bash
cp .env.example .env
# 然后自行把 HF_TOKEN 写入本地 .env，不要提交 .env
```

如果你习惯直接在 shell 里设置，也可以：

```bash
export HF_TOKEN="your_token_here"
```

---

## 方式零：本地运行

```bash
pip install -r requirements.txt
python app.py
```

默认访问地址：

```text
http://localhost:7860
```

如果启动后接口全部报认证错误，优先检查 `HF_TOKEN` 是否已经在当前环境中生效。

---

## 方式一：通过 Web UI 部署（最简单）

### Step 1 — 创建 Space

1. 访问 https://huggingface.co/new-space
2. 填写信息：
   - **Owner**: 你的用户名
   - **Space name**: `hf-image-studio`
   - **License**: MIT
   - **SDK**: Gradio
   - **Hardware**: CPU basic（免费）
3. 点击 **Create Space**

### Step 2 — 上传文件

进入 Space 页面，点击 **Files** 标签 → **Add file** → **Upload files**，上传以下文件：

```
app.py
requirements.txt
README.md
```

> 注意：README.md 顶部的 YAML 元数据块（`---` 之间的内容）是 HF Spaces 的配置，不要删除。

### Step 3 — 配置 Token

1. 进入 Space 页面 → **Settings**
2. 滚动到 **Repository secrets**
3. 点击 **New secret**：
   - Name: `HF_TOKEN`
   - Value: 粘贴你的 Access Token
4. 保存

> `HF_TOKEN` 应该放在 Space secrets 中，而不是写进 `app.py` 或仓库文件。

### Step 4 — 等待构建

Space 会自动检测到文件变化并开始构建。在 **Logs** 标签可查看构建日志。
构建完成后即可通过 `https://huggingface.co/spaces/你的用户名/hf-image-studio` 访问。

---

## 方式二：通过 Git 命令行部署（推荐）

### Step 1 — 创建 Space

同方式一的 Step 1，在 Web UI 创建一个空 Space。

### Step 2 — 克隆并推送代码

```bash
# 克隆空的 Space 仓库（替换为你的用户名）
git clone https://huggingface.co/spaces/你的用户名/hf-image-studio
cd hf-image-studio

# 将项目文件复制进来（或直接在此目录开发）
cp /path/to/your/app.py .
cp /path/to/your/requirements.txt .
cp /path/to/your/README.md .

# 提交并推送
git add .
git commit -m "Initial commit: HF Image Studio"
git push
```

> 推送时会要求输入用户名和密码，密码使用你的 Access Token。

### Step 3 — 配置 Token

同方式一的 Step 3，在 Space Settings 中添加 `HF_TOKEN` Secret。

---

## 方式三：使用 huggingface_hub Python API 部署

```python
from huggingface_hub import HfApi

api = HfApi()

# 创建 Space（首次）
api.create_repo(
    repo_id="你的用户名/hf-image-studio",
    repo_type="space",
    space_sdk="gradio",
    private=False,
)

# 上传整个目录
api.upload_folder(
    repo_id="你的用户名/hf-image-studio",
    repo_type="space",
    folder_path="/path/to/hf-image-studio",
)
```

然后去 Web UI 配置 `HF_TOKEN` Secret。

---

## 部署后验证

### 检查清单

- [ ] Space 状态显示为 **Running**（绿色）
- [ ] 页面正常加载，显示 4 个顶级标签页（生成/编辑/理解/智能）
- [ ] **生成 > 文生图**：输入 `a cat on the moon` → 能生成图片
- [ ] **生成 > Prompt 助手**：输入简单描述 → 能输出优化 Prompt
- [ ] **编辑 > 风格转换**：上传图 + 输入 `oil painting` → 能生成风格化图片
- [ ] **编辑 > 超分辨率**：上传小图 → 输出放大后的图片
- [ ] **编辑 > 背景移除**：上传图片 → 输出透明背景图
- [ ] **理解 > 图像描述**：上传图片 → 输出英文描述
- [ ] **理解 > 视觉问答**：上传图片 + 提问 → 能回答
- [ ] **理解 > OCR**：上传含文字图片 → 提取文字
- [ ] **理解 > 图像分割**：上传图片 → 输出彩色分割叠加图
- [ ] **智能 > AI Agent**：输入自然语言指令 → Agent 自动编排执行
- [ ] **智能 > 批量处理**：上传多图 + 选操作 → 批量输出结果

### 常见问题

| 问题 | 原因 | 解决方法 |
|------|------|----------|
| `401 Unauthorized` | HF_TOKEN 未设置或过期 | 检查 Settings → Secrets |
| `Model is loading` | 模型冷启动中 | 等待 30-60 秒后重试 |
| `Rate limit exceeded` | API 调用次数超限 | 等待一段时间，或切换更稳的部署方案 |
| 页面加载但按钮无响应 | 构建未完成 | 查看 Logs 标签确认构建状态 |
| `503 Service Unavailable` | 模型暂时不可用 | 切换到其他模型选项 |

---

## 升级选项

### CPU → ZeroGPU（推荐进阶）

如果你是 HF PRO 用户（$9/月），可以将 Space 升级为 ZeroGPU：

1. Space Settings → Hardware 选择 **ZeroGPU**
2. 在需要 GPU 的函数上加装饰器：

```python
import spaces

@spaces.GPU
def text_to_image(prompt, ...):
    # 使用本地模型而非 API
    ...
```

ZeroGPU 的优势是可以把部分推理迁移到 Space 侧，减少对远程 API 的依赖。

### 绑定自定义域名

PRO 用户可以在 Space Settings 中绑定自己的域名。

---

## 项目文件说明

```
hf-image-studio/
├── README.md           # Space 配置 + 项目说明（YAML 头部是 HF 配置）
├── app.py              # Gradio 应用主代码（13 个功能模块，~750 行）
├── requirements.txt    # Python 依赖（gradio, huggingface_hub, numpy, smolagents）
├── .gitattributes      # Git LFS 配置（HF 标准）
├── .gitignore          # Git 忽略规则
├── DEPLOY.md           # 本部署文档
└── ROADMAP.md          # 后续研究方向
```

---

## 公开前检查

- [ ] `.env` 没有提交，仓库里只保留 `.env.example`
- [ ] Space secrets 中使用的是有效且可撤销的 token
- [ ] README、DEPLOY、日志和截图里没有出现真实 token
- [ ] 如果启用了额外模型或代理工作流，确认它们适合公开发布
