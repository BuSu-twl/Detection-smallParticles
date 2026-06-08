# 丸子检测系统

## 项目概述

基于Python Flask + OpenCV的丸子检测系统。通过摄像头拍摄放置在2cm×2cm黑线框内的丸子，自动检测数量和尺寸。

## 技术栈

- **语言**: Python 3.11
- **Web框架**: Flask 3.0
- **图像处理**: OpenCV 4.9
- **数值计算**: NumPy 1.26

## 目录结构

```
.
├── app.py              # Flask主应用 (Web界面 + API)
├── detector.py         # 丸子检测算法模块
├── requirements.txt    # Python依赖
└── AGENTS.md           # 本文档
```

## 服务端口

- **端口**: 5000

## API接口

### GET /
主页 - Web界面

### GET /api/health
健康检查

### POST /api/detect
丸子检测接口

**请求参数:**
```json
{
  "image": "base64图像",
  "threshold": 100,
  "minSize": 0.6,
  "maxSize": 1.5,
  "calibrationWidth": 20,
  "calibrationHeight": 20,
  "offsetX": 0,
  "offsetY": 0,
  "frameWidth": 50,
  "frameHeight": 50,
  "distortionCompensation": 1.0
}
```

### POST /api/detect-frame
自动检测黑框位置

## 检测算法

### 流程步骤

1. **灰度化** - 彩色图像转灰度
2. **高斯模糊** - 降噪处理
3. **二值化** - 阈值分割提取前景
4. **形态学操作** - 腐蚀+膨胀去噪
5. **轮廓检测** - 查找连通区域
6. **圆形度过滤** - 过滤非圆形物体
7. **尺寸计算** - 像素到毫米转换

### 圆形度公式

```
圆形度 = 4π × 面积 / 周长²
```

- 完美圆形 = 1
- 检测阈值 ≥ 0.2

## 启动命令

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
python app.py
```

## 使用说明

1. 将丸子放入 2cm×2cm 黑色框内
2. 用广角摄像头对准黑框拍摄
3. 点击"拍照"按钮捕获图像
4. 点击"开始检测"分析结果
5. 导出CSV/JSON格式数据
