# 香连止痢丸检测系统

## 项目概述

基于 Python Flask + OpenCV 的香连止痢丸检测系统。通过摄像头拍摄放置在黑线框内的香连止痢丸，自动检测数量和尺寸。支持透视校正、广角摄像头畸变补偿等功能。

## 技术栈

| 类型 | 技术 | 版本 |
|------|------|------|
| 语言 | Python | 3.11+ |
| Web框架 | Flask | 3.0 |
| 图像处理 | OpenCV | 4.9 |
| 数值计算 | NumPy | 1.26 |
| 图片处理 | Pillow | 10.2 |

## 目录结构

```
.
├── app.py              # Flask主应用 (Web界面 + API)
├── detector.py         # 香连止痢丸检测算法模块
├── requirements.txt    # Python依赖
├── templates/
│   └── index.html      # Web界面模板
├── assets/             # 临时文件目录
└── AGENTS.md           # 本文档
```

## 快速开始

### 安装依赖

```bash
# 使用默认源
pip install -r requirements.txt

# 使用清华镜像源（推荐，国内更快）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 使用阿里云镜像源
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 使用中科大镜像源
pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/
```

### 启动服务

```bash
# 开发环境启动
python app.py

# 服务将运行在 http://localhost:5000
```

## 服务端口

- **端口**: 5000

## API接口

### GET /

主页 - Web界面

### GET /api/health

健康检查接口

**响应示例:**
```json
{
  "status": "ok"
}
```

### POST /api/detect

香连止痢丸检测接口

**请求参数:**
```json
{
  "image": "base64图像字符串",
  "threshold": 50,
  "minSize": 0.5,
  "maxSize": 3.0,
  "calibrationWidth": 60,
  "calibrationHeight": 60,
  "offsetX": 0,
  "offsetY": 0,
  "frameWidth": 50,
  "frameHeight": 50,
  "distortionCompensation": 1.0,
  "sizeCalibration": 1.25,
  "cameraType": "standard"
}
```

**响应示例:**
```json
{
  "success": true,
  "count": 10,
  "min_size": 1.49,
  "max_size": 1.71,
  "avg_size": 1.62,
  "balls": [
    {"id": 1, "x": 100, "y": 200, "diameter": 1.68, "area": 324},
    {"id": 2, "x": 150, "y": 180, "diameter": 1.71, "area": 338}
  ],
  "imageWithOverlay": "base64结果图",
  "intermediateImages": {...}
}
```

### POST /api/detect-frame

自动检测黑框位置

## 检测算法

### 流程步骤

1. **ROI提取** - 提取检测框区域
2. **灰度化** - 彩色图像转灰度
3. **高斯模糊** - 降噪处理
4. **二值化** - 阈值分割提取前景
5. **形态学操作** - 腐蚀+膨胀去噪
6. **轮廓检测** - 查找连通区域
7. **圆整度过滤** - 过滤非圆形物体
8. **尺寸计算** - 像素到毫米转换

### 圆整度公式

```
圆整度 = 4π × 面积 / 周长²
```

- 完美圆形 = 1
- 检测阈值 ≥ 0.2

### 透视校正

支持手持拍摄时的透视变形校正：
- 自动检测黑框四角
- 透视变换校正为正视图
- 提高测量精度

## 参数说明

### 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| threshold | 50 | 二值化阈值 (20-100) |
| minSize | 0.5 | 最小直径 (mm) |
| maxSize | 3.0 | 最大直径 (mm) |
| calibrationWidth | 60 | 标定框宽度 (mm) |
| calibrationHeight | 60 | 标定框高度 (mm) |
| distortionCompensation | 1.0 | 畸变补偿系数 |
| sizeCalibration | 1.25 | 尺寸校准系数 |

### 摄像头类型

| 类型 | distortionCompensation |
|------|------------------------|
| 标准摄像头 | 1.0 |
| 广角摄像头 | 0.85-0.95 |

## 使用说明

1. 选择摄像头类型（标准/广角）
2. 将实际黑框对准画面中的红框
3. 点击"拍照"按钮捕获图像
4. 确认黑框与红框重合，点击"开始检测"
5. 查看检测结果和中间处理图像
6. 导出CSV/JSON格式数据

## 常见问题

### 检测不到香连止痢丸？

1. 调整二值化阈值（20-100）
2. 放宽尺寸范围（最小/最大直径）
3. 检查标定框尺寸是否正确
4. 确保光照均匀

### 检测尺寸不准？

1. 调整尺寸校准系数（默认1.25）
2. 确保黑框完全在画面内
3. 使用支架固定摄像头，保持正对拍摄
