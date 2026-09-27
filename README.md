# 树莓派智能视觉系统

> 基于 **树莓派 5 + YOLO 本地检测 + DashScope Qwen-VL 云端理解** 的边缘视觉智能系统
> 智能嵌入技术课程期末大作业 —— 题目 2「面向真实场景的边缘视觉智能系统」

一套可直接运行的**闭环**系统：摄像头采集 → 本地实时目标检测 → 事件告警与快照 → 数据落库 → 统计分析 → 历史回放 → 拍照问答（多模态大模型）。

---

## 一、系统架构

采用 **「本地实时感知 + 云端智能理解」** 的分层协同架构：

```
┌───────────────────────────────────────────────────────────────┐
│  ① 视频采集层  backend/camera_capture.py                       │
│     USB 摄像头(OpenCV) / CSI 摄像头(picamera2)                 │
│     独立守护线程持续读帧，互斥锁保护「只保留最新一帧」不积压      │
│     断连自愈：5 次重试 + 5 秒冷却 + OpenCV ↔ picamera2 互相回退  │
└───────────────────────────┬───────────────────────────────────┘
                            │ BGR 帧
┌───────────────────────────▼───────────────────────────────────┐
│  ② 本地视觉层  backend/yolo_detector.py                        │
│     Ultralytics YOLO11n 逐帧推理（树莓派5 约 10-11 FPS）        │
│     输出 {类别, 置信度, 边界框, 归一化中心点, 面积占比}          │
│     模型查找四级回退：指定路径 → .onnx → CWD/models → 自动下载   │
│     把中心点坐标翻译成中文方位（中央/左上/右下）用于构造 Prompt   │
└───────────────────────────┬───────────────────────────────────┘
                            │
        ┌───────────────────┼────────────────────┬──────────────────┐
        ▼                   ▼                    ▼                  ▼
  绘制检测框          写入 SQLite            事件判断           生成文本摘要
 (Streamlit 替换刷新) (database.py)    (连续3帧 + 5秒去重)   (get_summary)
                                              │                    │
                                    蜂鸣告警 + 快照落盘            │
                                    (utils.py + pygame)           │
                                                                  ▼
┌───────────────────────────────────────────────────────────────┐
│  ③ 云端理解层  backend/dashscope_client.py                     │
│     DashScope MultiModalConversation → qwen-vl-plus / max      │
│     仅在用户提问时按需调用，YOLO 摘要注入 Prompt 作为上下文      │
│     支持同步调用与线程+队列的异步调用（analyze / analyze_async）│
└───────────────────────────────────────────────────────────────┘
```

**设计取舍**：时效性敏感的实时显示全部在本地完成，延迟可预测；语义分析按需触发、由云端大模型完成，避免主线程阻塞。

---

## 二、功能清单

| # | 功能 | 实现位置 |
|---|---|---|
| 1 | 摄像头实时视频流 + YOLO 目标检测（含 ONNX 导出脚本） | `pages/1_实时检测.py`、`tools/export_onnx.py` |
| 2 | 特定事件识别（连续 3 帧命中同一目标才判定为事件，抗误报） | `pages/1_实时检测.py` |
| 3 | 识别到目标即播放警告声（pygame 循环蜂鸣，目标消失自动停） | `backend/utils.py` |
| 4 | 检测结果持久化到 SQLite | `backend/database.py` |
| 5 | 日 / 周 / 月统计图表页 | `pages/3_数据分析.py` |
| 6 | 设置窗口（sidebder：API Key、模型、置信度、IoU、视频源、告警类别） | `app.py` |
| 7 | 事件历史时间线 + 筛选 + 快照回放 + CSV 导出 | `pages/4_历史回放.py` |
| 8 | 拍照识别 + 智能多轮问答（数量/存在性/类别枚举/位置关系/面积占比） | `pages/2_拍照问答.py` |

---

## 三、目录结构

```
树莓派智能视觉系统/
├── app.py                      # Streamlit 主入口（页面配置 + 全局侧边栏）
├── pages/                      # Streamlit 多页面
│   ├── 1_实时检测.py            #   摄像头实时检测 + 告警 + 快照
│   ├── 2_拍照问答.py            #   拍照/上传 → YOLO → Qwen-VL 问答
│   ├── 3_数据分析.py            #   日/周/月统计图表
│   └── 4_历史回放.py            #   事件时间线 + 快照回放 + CSV 导出
├── backend/                    # 后端模块（纯 Python，与 Web 层解耦）
│   ├── config.py               #   全局配置 + .env 加载 + session_state 初始化
│   ├── camera_capture.py       #   摄像头采集（CSI / USB，独立线程）
│   ├── yolo_detector.py        #   YOLO 检测、画框、文本摘要
│   ├── dashscope_client.py     #   Qwen-VL 多模态调用（同步 / 异步）
│   ├── database.py             #   SQLite 持久化与统计查询
│   ├── utils.py                #   Base64、快照、告警声、图像工具
│   └── logging_setup.py        #   日志初始化（控制台 + logs/ 文件）
├── tools/
│   └── export_onnx.py          # YOLO → ONNX 导出与推理自检
├── models/                     # YOLO 权重（首次运行自动下载，不入库）
├── data/                       # SQLite 数据库（运行时生成，不入库）
├── snapshots/                  # 检测快照，按日期分目录（运行时生成，不入库）
├── logs/                       # 运行日志，按日期分文件（运行时生成，不入库）
├── assets/                     # 告警声音等静态资源（可选）
├── requirements.txt
├── .env.example                # 环境变量模板
└── .gitignore
```

---

## 四、快速开始

### 4.1 环境要求

- Python 3.9+（树莓派上推荐 Raspberry Pi OS Bookworm / Trixie，64-bit）
- 摄像头：USB 摄像头（OpenCV）或 CSI 摄像头（picamera2，树莓派系统预装）
- 可选：能访问阿里云百炼的 DashScope API Key（拍照问答功能需要）

### 4.2 安装依赖

```bash
# 建议使用虚拟环境
python3 -m venv env
source env/bin/activate          # Windows: env\Scripts\activate

pip install -r requirements.txt
```

> **树莓派提示**：`opencv-python` 在树莓派上编译较慢，建议改用系统源安装
> `sudo apt install python3-opencv`；`picamera2` 由系统预装，不在 pip 清单中。

### 4.3 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，把 DASHSCOPE_API_KEY 换成真实密钥
```

Key 的读取优先级：**环境变量 → Streamlit secrets → `.env` 文件 → 页面侧边栏输入**。
也可以完全不建 `.env`，直接在左侧边栏粘贴 Key，然后点「应用配置」。

### 4.4 启动

```bash
streamlit run app.py
```

浏览器会自动打开（默认 `http://localhost:8501`）。用「启动检测」按钮开始实时检测。

### 4.5 首次运行会发生什么

| 时机 | 行为 |
|---|---|
| 首次检测 | `models/` 无权重时，ultralytics 自动下载 `yolo11n.pt` 并**自动复制回 `models/` 目录**作为本地缓存，仅首次需要联网 |
| 首次检测 | 自动创建 `data/detections.db` 并建表建索引 |
| 触发告警 | 自动创建 `snapshots/YYYY-MM-DD/`，写入快照（需勾选「自动保存快照」） |
| 启动应用 | 创建 `logs/app_YYYYMMDD.log` |

> 想跳过下载：把任意 `.pt`（或 ONNX）权重放进 `models/` 并命名为 `yolo11n.pt` 即可。

---

## 五、摄像头配置

在左侧边栏「摄像头设置 → 视频源」切换，切换后重新点「启动检测」：

| 选项 | 后端 | 适用 |
|---|---|---|
| USB 摄像头 (0) | OpenCV `VideoCapture(0)` | 大多数 USB 摄像头 |
| USB 摄像头 (1) | OpenCV `VideoCapture(1)` | 第二个 USB 摄像头 |
| CSI 摄像头 (picamera2) | picamera2 | 树莓派官方排线摄像头 |

**CSI 摄像头排障**：若报 `Pipeline handler in use`，通常是上次进程异常退出残留了
libcamera 管线锁。程序已内置清理（`pkill libcamera` + `fuser -k /dev/video0` + 重置
CameraManager），若仍失败可手动执行：

```bash
sudo pkill -9 -f libcamera
sudo fuser -k /dev/video0
```

---

## 六、模型：训练与 ONNX 导出

### 6.1 使用预训练模型（开箱即用）

默认加载 ultralytics 官方 `yolo11n.pt`（COCO 80 类），无需训练。

### 6.2 自己训练并导出 ONNX

```bash
# 1) 训练（以自建数据集 data.yaml 为例）
yolo detect train data=datasets/data.yaml model=yolo11n.pt epochs=100 imgsz=640

# 2) 导出 ONNX（本仓库提供脚本，含导出后推理自检）
python tools/export_onnx.py --model models/best.pt --imgsz 320 --check

# 3) 让后端使用 ONNX：backend/config.py 中设置 YOLO_USE_ONNX = True
```

导出脚本还支持：

```bash
python tools/export_onnx.py                      # 导出默认模型 yolo11n
python tools/export_onnx.py --half               # FP16 半精度
python tools/export_onnx.py --outdir models      # 指定产物目录
```

> ONNX 推理在树莓派 5 上通常比 PyTorch 快 20%~40%，CPU 占用更低。
> 需要更快可继续转 NCNN：`yolo export model=models/yolo11n.pt format=ncnn`。

---

## 七、数据存储

SQLite 数据库位于 `data/detections.db`（WAL 模式，已建时间/类别索引）。

| 表 | 写入时机 | 内容 |
|---|---|---|
| `detections` | 每帧检测到目标时 | 时间戳、类别、置信度、边界框、快照路径 |
| `events` | 告警事件触发时 | 时间戳、事件类型、类别、描述、最高置信度、快照路径、视频片段路径（预留） |

常用查询：

```sql
-- 各表行数
SELECT 'detections', COUNT(*) FROM detections
UNION ALL SELECT 'events', COUNT(*) FROM events;

-- 清理 7 天前的逐帧检测记录（长期运行时建议定期执行）
DELETE FROM detections WHERE timestamp < datetime('now', '-7 days');
VACUUM;
```

> ⚠️ `detections` 表按帧记录，10 FPS 下持续检测约产生 125 行/秒。长时间运行请
> 定期清理，或把 `pages/1_实时检测.py` 中的 `db.save_detections(detections)` 改为
> 隔帧（如 `if frame_count % 10 == 0`）采样写入。

---

## 八、关键参数

全部集中在 `backend/config.py`：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `YOLO_MODEL_SHORT` | `yolo11n` | 模型短名，可换 `yolo11s` / `yolo8n` |
| `YOLO_CONF_THRESHOLD` | `0.35` | 置信度阈值（侧边栏可实时调） |
| `YOLO_IOU_THRESHOLD` | `0.45` | NMS IoU 阈值 |
| `YOLO_USE_ONNX` | `False` | 置 `True` 后优先加载同名 `.onnx` |
| `CAMERA_WIDTH/HEIGHT` | `640×480` | 采集分辨率 |
| `CAMERA_FPS` | `15` | 采集线程目标帧率 |
| `EVENT_CONSECUTIVE_FRAMES` | `3` | 连续多少帧命中才算事件（抗误报） |
| `EVENT_MIN_INTERVAL` | `5` | 同类事件最小记录间隔（秒） |
| `ALERT_CLASSES` | `person/dog/cat` | 默认告警类别 |
| `API_IMAGE_MAX_WIDTH` | `640` | 送云端前压缩到的最大宽度 |

---

## 九、常见问题

| 现象 | 原因与处理 |
|---|---|
| 页面提示「DashScope 未配置」 | 没有 `.env` 也没在侧边栏填 Key。参考 §4.3 |
| 告警没有声音 | 未安装 `pygame`（`pip install pygame`）；或系统无音频设备。代码在此情况下静默降级，不影响检测 |
| 检测框画出来了但没有画面 | 检查摄像头是否被其它程序（如 `cheese`、上一次残留进程）占用 |
| 提示 YOLO 模型下载失败 | 树莓派无外网时，手动把 `.pt` 放到 `models/`，或改用 ONNX |
| 数据分析页空白 | `data/detections.db` 还没有数据，先去「实时检测」页跑一段 |
| 日志在哪 | `logs/app_YYYYMMDD.log`，同时输出到启动 `streamlit` 的终端 |
| 快照保存失败 | 已修复：`cv2.imwrite` 在路径含中文时会**静默失败**（返回 `False` 且不建文件）。`backend/utils.py::imwrite_unicode()` 改为先 `imencode` 到内存再用 Python `open()` 落盘，兼容任意 Unicode 路径 |
| 模型每次都重新下载 | 已修复：权重缓存目录原先写死 Linux 的 `~/.cache/ultralytics`，Windows 等平台定位不到下载产物，导致缓存回写静默失效。现按「ultralytics settings → 平台路径 → CWD」多目录查找 |

---

## 十、已知限制

- **无视频片段录制**：`events.video_clip_path` 字段已预留，但尚未实现录制，历史回放目前以快照为主。
- **逐帧全量落库**：`detections` 表按帧写入，长时间运行需人工清理（见 §7）。
- **问答为单轮+上下文摘要**：每次提问携带 YOLO 检测摘要，但历史对话不参与请求。
- **异步接口未启用**：`DashScopeClient.analyze_async` 已实现，页面当前使用同步调用。

---

## 十一、参考资料

- [Ultralytics YOLO 文档](https://docs.ultralytics.com/)
- [Picamera2 手册](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf)
- [阿里云百炼 DashScope](https://bailian.console.aliyun.com/) / [多模态 API 文档](https://help.aliyun.com/zh/model-studio/)
- [Streamlit 文档](https://docs.streamlit.io/)
