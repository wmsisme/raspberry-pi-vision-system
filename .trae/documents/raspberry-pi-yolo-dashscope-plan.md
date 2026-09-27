# 树莓派 YOLO + DashScope + Streamlit 完整系统实施计划

## 摘要

基于项目架构文档，为树莓派5实现一个完整的边缘智能视觉系统。包含：
- YOLO 本地实时目标检测 + 动作检测
- DashScope 云 API（Qwen-VL）智能语义问答
- Streamlit 全功能 Web 界面（实时检测、数据分析、历史回放、拍照问答、设置面板）
- SQLite 检测结果持久化存储
- 可部署在树莓派5上运行

---

## 当前状态分析

- 工作区仅含 3 个文件：作业说明 `.md`、架构文档 `.md`、考核表 `.doc`
- 无任何代码文件，需从零构建
- 架构文档已提供清晰的模块划分和实现方向，作为编码参考

---

## 需要创建的文件

### 1. `backend/config.py` — 全局配置文件

**作用**：集中管理所有配置项，避免硬编码。

**内容**：
- DashScope API Key（从环境变量读取，提供默认占位）
- YOLO 模型路径、置信度阈值、NMS 阈值
- 摄像头参数（分辨率 640x480、FPS 目标值）
- SQLite 数据库路径
- Streamlit 页面配置
- 告警声音文件路径
- 支持的检测类别列表

---

### 2. `backend/camera_capture.py` — 摄像头采集模块

**作用**：统一摄像头接口，支持 CSI（picamera2）和 USB（OpenCV）摄像头。

**内容**：
- `CameraCapture` 类
  - `__init__()`：自动检测摄像头类型，初始化对应后端
  - `get_frame()`：返回单帧（BGR numpy array）
  - `get_frame_base64()`：返回 Base64 编码的 JPEG
  - `release()`：释放资源
  - 自动重连机制（断连后 3-5 次重试）
- 使用独立线程持续读取，减少主线程阻塞

---

### 3. `backend/yolo_detector.py` — YOLO 本地检测模块

**作用**：加载 YOLO 模型，执行目标检测，返回结构化结果。

**内容**：
- `YOLODetector` 类
  - `__init__(model_path, conf_threshold)`：加载模型
  - `detect(frame)`：返回 `[{class, confidence, bbox, area}, ...]`
  - `draw_boxes(frame, detections)`：在图像上绘制检测框，返回标注后的图像
  - `get_summary(detections)`：生成结构化文本摘要（用于构造 Prompt）
  - 支持 ONNX/NCNN 导出后的模型加载
- 预加载 COCO 类别名映射
- 支持动作检测的扩展（可选，基于 pose estimation）

---

### 4. `backend/dashscope_client.py` — DashScope 云 API 客户端

**作用**：封装 Qwen-VL 多模态模型调用，异步非阻塞。

**内容**：
- `DashScopeClient` 类
  - `__init__(api_key, model)`：初始化客户端
  - `analyze(image_base64, prompt)`：发送图像 + Prompt，返回模型回答
  - `analyze_async(image_base64, prompt, callback)`：异步调用（线程+队列）
  - 支持的问题类型：
    - 数量统计类（几个/多少）
    - 存在性判断类（有没有/是否）
    - 类别枚举类（有哪些）
    - 位置关系类（左边/右边/中间/上下）
    - 面积占比类（最大/最多/占据空间）
  - 异常处理：超时重试、网络异常降级
  - 图片预处理：压缩分辨率以降低传输开销

---

### 5. `backend/database.py` — SQLite 数据库模块

**作用**：持久化存储检测结果，支持按时间查询和统计。

**内容**：
- `DetectionDatabase` 类
  - `init_db()`：创建表结构
    - `detections` 表：id, timestamp, class_name, confidence, bbox_x1/y1/x2/y2, image_path, event_type
    - `events` 表：id, timestamp, event_type, description, snapshot_path, video_clip_path
  - `save_detection()`：保存单帧检测结果
  - `save_event()`：保存事件记录
  - `get_daily_stats(date)`：日统计
  - `get_weekly_stats(start_date)`：周统计
  - `get_monthly_stats(year, month)`：月统计
  - `get_events_by_timerange(start, end)`：按时间范围查询事件
  - `get_all_detections()`：获取全部检测记录

---

### 6. `backend/utils.py` — 工具函数模块

**作用**：通用辅助函数。

**内容**：
- `frame_to_base64(frame)`：numpy 数组转 Base64 JPEG 字符串
- `generate_timestamp()`：生成格式化时间戳
- `ensure_dir(path)`：确保目录存在
- `play_alert_sound(sound_path)`：播放告警声音（使用 pygame 或系统命令）
- `save_snapshot(frame, dir)`：保存快照图片
- `compress_image_for_api(frame, max_size)`：压缩图像以适应 API 传输

---

### 7. `app.py` — Streamlit 主入口

**作用**：Streamlit 多页面应用入口，配置全局布局和导航。

**内容**：
- 页面配置（标题、图标、布局）
- 侧边栏导航
- 全局状态初始化（摄像头、YOLO 模型、数据库、DashScope 客户端）
- 页面路由

---

### 8. `pages/1_实时检测.py` — 实时视频流检测页面

**作用**：核心检测页面，实时显示摄像头画面 + YOLO 检测结果。

**内容**：
- 摄像头视频流展示（使用 `st.image()` 循环刷新）
- 检测框和标签实时叠加
- 检测结果文本实时显示
- 特定事件告警（检测到指定物体时播放声音）
- 手动/自动保存检测快照
- 启停控制按钮



### 9. `pages/2_拍照问答.py` — 拍照识别 + 智能多轮问答

**作用**：拍照上传图像 → YOLO 检测 → 用户提问 → Qwen-VL 回答。

**内容**：
- 拍照/上传图像区域
- YOLO 检测结果显示（检测框 + 类别列表）
- 检测结果文本摘要（自动生成）
- 用户自由提问输入框
- Qwen-VL 智能回答展示
- 支持多轮对话（保持上下文）
- 预设快速提问按钮（数量统计/存在判断/类别枚举/位置关系/面积占比）
- 回答语音生成（可选，使用 pyttsx3）

---

### 10. `pages/3_数据分析.py` — 数据统计与分析页面

**作用**：展示日/周/月检测统计图表。

**内容**：
- 时间范围选择器（日/周/月）
- 各类别检测数量柱状图
- 检测数量趋势折线图（按小时/天）
- 置信度分布直方图
- 事件类型占比饼图
- 使用 plotly 或 matplotlib 绘图

---

### 11. `pages/4_历史回放.py` — 事件历史与回放页面

**作用**：时间线界面，用户选择日期时间，回放历史事件。

**内容**：
- 日期选择器
- 时间线展示（按时间段列出事件）
- 事件详情展示（缩略图、类别、时间、置信度）
- 事件快照查看
- 视频片段回放（如有录制）
- 事件筛选（按类别/置信度）

---

### 12. `requirements.txt` — Python 依赖清单

```
opencv-python
ultralytics>=8.3.0
dashscope>=1.20.0
streamlit>=1.28.0
plotly>=5.18.0
pandas>=2.0.0
pillow>=10.0.0
pygame>=2.5.0
numpy>=1.24.0
```

树莓派特有依赖（如 picamera2）由系统预装，不列入 pip 清单。

---

## 数据流设计

```
摄像头 → camera_capture.py → 帧(BGR)
                                  ↓
                            yolo_detector.py → 检测结果[{class, conf, bbox}, ...]
                                  ↓
                    ┌─────────────┼──────────────┐
                    ↓             ↓              ↓
              绘制检测框      存入 SQLite      触发事件判断
              (streamlit显示)  (database.py)   (告警/快照)
                    ↓                            ↓
              用户拍照提问              dashscope_client.py
              (拍照问答页面)           (异步 API 调用)
                    ↓                            ↓
              Qwen-VL 回答             结果显示在 Streamlit
```

---

## 关键实现决策

1. **Streamlit 中摄像头流的处理**：使用 `st.empty()` + 循环刷新，通过 session_state 传递最新帧
2. **YOLO 模型**：默认使用 `yolov11n.pt`，首次运行时自动从 Ultralytics 下载；也可手动放置 ONNX/NCNN 模型
3. **DashScope API**：异步调用，使用 Python `threading` + `queue.Queue` 实现生产者-消费者模型
4. **告警声音**：使用 `pygame.mixer` 播放 wav 文件
5. **数据库**：SQLite，文件存储在项目 `data/` 目录下
6. **快照存储**：保存在 `snapshots/` 目录，按日期分子目录

---

## 目录结构（最终）

```
期末作业/
├── backend/
│   ├── config.py
│   ├── camera_capture.py
│   ├── yolo_detector.py
│   ├── dashscope_client.py
│   ├── database.py
│   └── utils.py
├── pages/
│   ├── 1_实时检测.py
│   ├── 2_拍照问答.py
│   ├── 3_数据分析.py
│   └── 4_历史回放.py
├── app.py
├── requirements.txt
├── models/              (存放 YOLO 模型文件)
├── data/                (SQLite 数据库)
├── snapshots/           (检测快照)
├── logs/                (运行日志)
├── assets/              (告警声音等静态资源)
├── 26春-智能嵌入技术期末大作业 .md
├── 智能嵌入技术成绩考核表.doc
└── 项目架构.md
```

---

## 验证步骤

1. 在树莓派上安装依赖：`pip install -r requirements.txt`
2. 配置 DashScope API Key：设置环境变量 `DASHSCOPE_API_KEY`
3. 启动应用：`streamlit run app.py`
4. 验证实时检测页面能显示摄像头画面和检测框
5. 验证拍照问答能正确调用 YOLO + Qwen-VL
6. 验证数据分析和历史回放页面数据正确
7. 验证告警声音功能
