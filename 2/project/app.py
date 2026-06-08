#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
                    种子检测系统 - Seed Detection System
===============================================================================

功能说明：
    通过摄像头拍摄放置在2cm×2cm黑线框内的丸子，自动检测数量和尺寸。

检测范围：
    - 丸子直径：0.6mm ~ 1.5mm
    - 检测框尺寸：20mm × 20mm (2cm × 2cm)

技术栈：
    - Python 3.11
    - OpenCV (图像处理)
    - Flask (Web服务)

使用方法：
    1. 运行程序：python app.py
    2. 打开浏览器访问：http://localhost:5000
    3. 点击"拍照"按钮拍摄
    4. 点击"检测"按钮分析
    5. 导出CSV/JSON数据

作者：AI Assistant
版本：2.0.0

===============================================================================
"""

import os
import sys
import base64
import json
from datetime import datetime
from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np

# 导入检测模块
from detector import BallDetector

# =============================== 配置 =================================

# 服务端口 - 强制使用 5000 端口（沙箱环境规范）
# 注意：沙箱环境必须使用 5000 端口
PORT = 5000

# =============================== Flask应用 ===============================

app = Flask(__name__)

# 检测器实例
detector = BallDetector()

# =============================== 工具函数 ===============================

def base64_to_image(base64_string):
    """将Base64字符串转换为OpenCV图像"""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    image_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    return image


def image_to_base64(image, format='jpeg'):
    """将OpenCV图像转换为Base64字符串"""
    _, buffer = cv2.imencode(f'.{format}', image)
    base64_string = base64.b64encode(buffer).decode('utf-8')
    return f'data:image/{format};base64,{base64_string}'


# =============================== API路由 ===============================

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({
        'status': 'ok',
        'service': '种子检测系统',
        'version': '2.0.0'
    })


@app.route('/api/detect', methods=['POST'])
def detect():
    """丸子检测API"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': '缺少图像数据'}), 400
        
        # 解码图像
        image = base64_to_image(data['image'])
        if image is None:
            return jsonify({'success': False, 'error': '图像解码失败'}), 400
        
        # 设置参数
        detector.set_params(
            threshold=data.get('threshold', 100),
            min_size=data.get('minSize', 0.6),
            max_size=data.get('maxSize', 1.5),
            calibration_width=data.get('calibrationWidth', 20),
            calibration_height=data.get('calibrationHeight', 20),
            offset_x=data.get('offsetX', 0),
            offset_y=data.get('offsetY', 0),
            frame_width=data.get('frameWidth', 50),
            frame_height=data.get('frameHeight', 50),
            distortion_compensation=data.get('distortionCompensation', 1.0)
        )
        
        # 执行检测
        result = detector.detect(image)
        
        return jsonify({
            'success': True,
            'count': result['count'],
            'minSize': result['min_size'],
            'maxSize': result['max_size'],
            'avgSize': result['avg_size'],
            'balls': result['balls'],
            'imageWithOverlay': result['image_with_overlay'],
            'framePixelSize': result.get('frame_pixel_size')
        })
        
    except Exception as e:
        print(f"检测错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/detect-frame', methods=['POST'])
def detect_frame():
    """黑框检测API"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'found': False, 'error': '缺少图像数据'}), 400
        
        image = base64_to_image(data['image'])
        if image is None:
            return jsonify({'found': False, 'error': '图像解码失败'}), 400
        
        result = detector.detect_black_frame(image)
        return jsonify(result)
        
    except Exception as e:
        print(f"黑框检测错误: {e}")
        return jsonify({'found': False, 'error': str(e)}), 500


# =============================== 主程序 ===============================

if __name__ == '__main__':
    print("=" * 60)
    print("🔴 种子检测系统 v2.0")
    print("=" * 60)
    print(f"📍 服务地址: http://localhost:{PORT}")
    print(f"📋 检测范围: 0.5mm ~ 10mm")
    print(f"📐 标定框尺寸: 60mm × 60mm (6cm × 6cm)")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=PORT, debug=True, threaded=True)
