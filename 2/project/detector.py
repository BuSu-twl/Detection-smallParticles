#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
                    丸子检测器 - Ball Detector Module
===============================================================================

使用OpenCV实现图像处理和丸子检测算法

检测流程:
    1. 灰度化 - 彩色图像转灰度
    2. 高斯模糊 - 降噪处理
    3. 二值化 - 阈值分割提取前景
    4. 形态学操作 - 腐蚀+膨胀去噪
    5. 轮廓检测 - 查找连通区域
    6. 圆形度过滤 - 过滤非圆形物体
    7. 尺寸计算 - 像素到毫米的转换

广角摄像头畸变补偿:
    - 通过畸变补偿系数调整边缘测量误差
    - 使用标准件进行校准可提高精度

===============================================================================
"""

import cv2
import numpy as np
import base64


class BallDetector:
    """
    丸子检测器类
    
    属性:
        threshold: 二值化阈值 (0-255)
        min_size: 最小直径 (mm)
        max_size: 最大直径 (mm)
        calibration_width: 标定框实际宽度 (mm)
        calibration_height: 标定框实际高度 (mm)
        offset_x: 标定框水平偏移 (%)
        offset_y: 标定框垂直偏移 (%)
        frame_width: 标定框宽度占比 (%)
        frame_height: 标定框高度占比 (%)
        distortion_compensation: 畸变补偿系数
    """
    
    def __init__(self):
        """初始化检测器，设置默认参数"""
        # 二值化阈值
        self.threshold = 100
        
        # 种子尺寸范围 (mm) - 根据常见种子大小调整
        self.min_size = 0.5   # 最小 0.5mm
        self.max_size = 10.0  # 最大 10mm
        
        # 标定框实际尺寸 (mm) - 6cm × 6cm
        self.calibration_width = 60   # 6cm
        self.calibration_height = 60  # 6cm
        
        # 标定框位置和大小 (百分比)
        self.offset_x = 0
        self.offset_y = 0
        self.frame_width = 50
        self.frame_height = 50
        
        # 广角镜头畸变补偿系数
        self.distortion_compensation = 1.0
    
    def set_params(self, **kwargs):
        """
        设置检测参数
        
        参数:
            threshold: 二值化阈值
            min_size: 最小直径 (mm)
            max_size: 最大直径 (mm)
            calibration_width: 标定框宽度 (mm)
            calibration_height: 标定框高度 (mm)
            offset_x: 水平偏移 (%)
            offset_y: 垂直偏移 (%)
            frame_width: 框宽度 (%)
            frame_height: 框高度 (%)
            distortion_compensation: 畸变补偿系数
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    # =========================================================================
    #                           黑框检测算法
    # =========================================================================
    
    def detect_black_frame(self, image):
        """
        自动检测图像中的黑色矩形框
        
        改进算法：使用霍夫直线检测 + 边缘检测来精确定位黑框
        
        参数:
            image: OpenCV图像 (BGR格式)
        
        返回:
            dict: {
                'found': bool,           # 是否找到
                'offsetX': float,        # 水平偏移 (%)
                'offsetY': float,        # 垂直偏移 (%)
                'frameWidth': float,     # 框宽度 (%)
                'frameHeight': float,    # 框高度 (%)
                'pixelWidth': int,       # 像素宽度
                'pixelHeight': int       # 像素高度
            }
        """
        height, width = image.shape[:2]
        
        # 步骤1: 灰度化
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 步骤2: 使用自适应阈值检测深色线条
        # 对于深色黑框，使用较低的阈值
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        
        # 步骤3: 形态学操作 - 增强线条
        kernel_h = np.ones((1, 15), np.uint8)  # 水平核
        kernel_v = np.ones((15, 1), np.uint8)  # 垂直核
        
        # 分别检测水平和垂直线条
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_h)
        vertical = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_v)
        
        # 合并
        combined = cv2.bitwise_or(horizontal, vertical)
        
        # 再做一次闭操作连接断裂
        kernel = np.ones((5, 5), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 步骤4: 查找轮廓 - 使用combined结果
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # 回退到binary
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'found': False, 'error': '未找到轮廓'}
        
        # 步骤5: 找最大的矩形轮廓
        best_rect = None
        max_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # 过滤太小的区域 - 根据6cm框调整
            if area < 10000:
                continue
            
            # 获取最小外接矩形
            x, y, w, h = cv2.boundingRect(contour)
            
            # 检查宽高比 (接近正方形，允许更大偏差)
            aspect_ratio = w / h if h > 0 else 0
            
            # 黑框应该占画面较大比例，且接近正方形
            frame_area_ratio = (w * h) / (width * height)
            
            if (0.3 < aspect_ratio < 3.0 and 
                w > 150 and h > 150 and 
                frame_area_ratio > 0.1):
                if area > max_area:
                    max_area = area
                    best_rect = (x, y, w, h)
        
        if best_rect is None:
            return {'found': False, 'error': '未找到有效的矩形框'}
        
        x, y, w, h = best_rect
        
        # 略微缩小检测框，确保在黑框内部
        margin = int(min(w, h) * 0.02)  # 2%边距
        x += margin
        y += margin
        w -= 2 * margin
        h -= 2 * margin
        
        # 强制正方形 - 取宽高的较小值
        square_size = min(w, h)
        
        # 重新计算中心点
        center_x_img = x + w / 2
        center_y_img = y + h / 2
        
        # 计算中心点和百分比参数 - 使用正方形尺寸
        center_x = center_x_img / width * 100 - 50  # 转换为偏移百分比
        center_y = center_y_img / height * 100 - 50
        frame_size_pct = square_size / max(width, height) * 100  # 使用较大的维度作为基准
        
        return {
            'found': True,
            'offsetX': round(center_x, 1),
            'offsetY': round(center_y, 1),
            'frameWidth': round(frame_size_pct, 1),  # 正方形：宽高相同
            'frameHeight': round(frame_size_pct, 1),  # 正方形：宽高相同
            'pixelWidth': square_size,
            'pixelHeight': square_size
        }
    
    # =========================================================================
    #                           丸子检测算法
    # =========================================================================
    
    def detect(self, image):
        """
        检测图像中的丸子
        
        算法流程:
            1. 定义ROI区域 (根据标定框参数)
            2. 灰度化
            3. 高斯模糊 (降噪)
            4. 二值化 (阈值分割)
            5. 形态学操作 (腐蚀+膨胀)
            6. 轮廓检测
            7. 圆形度过滤
            8. 尺寸计算和过滤
            9. 绘制结果
        
        参数:
            image: OpenCV图像 (BGR格式)
        
        返回:
            dict: {
                'count': int,            # 数量
                'min_size': float,       # 最小直径 (mm)
                'max_size': float,       # 最大直径 (mm)
                'avg_size': float,       # 平均直径 (mm)
                'balls': list,           # 丸子列表
                'image_with_overlay': str,  # 标注图像 (Base64)
                'frame_pixel_size': dict    # 检测框像素尺寸
            }
        """
        height, width = image.shape[:2]
        
        # =====================================================================
        # 步骤1: 定义检测区域 (ROI)
        # =====================================================================
        scale_w = self.frame_width / 100
        scale_h = self.frame_height / 100
        roi_width = int(width * scale_w)
        roi_height = int(height * scale_h)
        
        # 计算ROI中心位置 (考虑偏移)
        center_x = int(width / 2 + (width * self.offset_x / 100))
        center_y = int(height / 2 + (height * self.offset_y / 100))
        roi_x = max(0, center_x - roi_width // 2)
        roi_y = max(0, center_y - roi_height // 2)
        
        # 边界检查
        roi_width = min(roi_width, width - roi_x)
        roi_height = min(roi_height, height - roi_y)
        
        # =====================================================================
        # 步骤2: 提取ROI区域
        # =====================================================================
        roi = image[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
        
        # =====================================================================
        # 步骤3: 灰度化
        # =====================================================================
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # =====================================================================
        # 步骤4: 高斯模糊 (降噪)
        # =====================================================================
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # =====================================================================
        # 步骤5: 二值化
        # =====================================================================
        # THRESH_BINARY_INV: 深色物体 (丸子) 变为白色
        _, binary = cv2.threshold(
            blurred, 
            self.threshold, 
            255, 
            cv2.THRESH_BINARY_INV
        )
        
        # =====================================================================
        # 步骤6: 形态学操作
        # =====================================================================
        kernel = np.ones((3, 3), np.uint8)
        
        # 腐蚀: 去除细小噪点
        eroded = cv2.erode(binary, kernel, iterations=1)
        
        # 膨胀: 恢复物体大小
        dilated = cv2.dilate(eroded, kernel, iterations=1)
        
        # =====================================================================
        # 步骤7: 轮廓检测
        # =====================================================================
        contours, _ = cv2.findContours(
            dilated, 
            cv2.RETR_EXTERNAL,  # 只检测外轮廓
            cv2.CHAIN_APPROX_SIMPLE  # 压缩轮廓点
        )
        
        # =====================================================================
        # 步骤8: 计算像素到毫米的比例
        # =====================================================================
        mm_per_pixel_x = self.calibration_width / roi_width
        mm_per_pixel_y = self.calibration_height / roi_height
        mm_per_pixel = ((mm_per_pixel_x + mm_per_pixel_y) / 2) * self.distortion_compensation
        
        # =====================================================================
        # 步骤9: 分析每个轮廓
        # =====================================================================
        balls = []
        
        for contour in contours:
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            
            # 过滤太小的区域 (噪点) - 根据6cm框调整
            if area < 20:
                continue
            
            # 计算周长
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter == 0:
                continue
            
            # =============================================================
            # 计算圆形度
            # 公式: 圆形度 = 4π × 面积 / 周长²
            # 完美圆形的圆形度 = 1
            # =============================================================
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            
            # 过滤非圆形物体 (阈值 0.15，种子可能不规则)
            if circularity < 0.15:
                continue
            
            # 计算等效直径 (假设物体为圆形)
            equivalent_diameter = np.sqrt(4 * area / np.pi)
            
            # 计算实际直径 (mm)
            real_diameter = equivalent_diameter * mm_per_pixel
            
            # 过滤尺寸范围
            if real_diameter < self.min_size or real_diameter > self.max_size:
                continue
            
            # 计算中心点 (转换回原图坐标)
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00']) + roi_x
                cy = int(M['m01'] / M['m00']) + roi_y
            else:
                continue
            
            # 计算半径
            radius = equivalent_diameter / 2
            
            balls.append({
                'id': len(balls) + 1,
                'x': float(cx),
                'y': float(cy),
                'radius': float(radius),
                'diameter': round(real_diameter, 3),
                'area': float(area),
                'circularity': round(circularity, 3)
            })
        
        # =====================================================================
        # 步骤10: 绘制检测结果
        # =====================================================================
        result_image = image.copy()
        
        # 绘制检测框 (红色)
        cv2.rectangle(
            result_image, 
            (roi_x, roi_y), 
            (roi_x + roi_width, roi_y + roi_height),
            (68, 68, 255),  # 红色 (BGR)
            2
        )
        
        # 绘制每个丸子
        for ball in balls:
            center = (int(ball['x']), int(ball['y']))
            radius = int(ball['radius'])
            
            # 绘制圆圈 (绿色)
            cv2.circle(result_image, center, radius, (0, 255, 0), 2)
            
            # 绘制中心点 (红色)
            cv2.circle(result_image, center, 2, (0, 0, 255), -1)
            
            # 准备标注文本
            label = f"#{ball['id']}: {ball['diameter']:.2f}mm"
            
            # 计算文本位置
            text_x = center[0] - 40
            text_y = center[1] - radius - 10
            
            # 绘制文本背景
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                result_image,
                (text_x - 5, text_y - text_height - 5),
                (text_x + text_width + 5, text_y + 5),
                (0, 0, 0),
                -1
            )
            
            # 绘制文本 (绿色)
            cv2.putText(
                result_image, 
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6,
                (0, 255, 0),
                2
            )
        
        # 编码结果图像为Base64
        image_with_overlay = self._image_to_base64(result_image)
        
        # =====================================================================
        # 步骤11: 计算统计数据
        # =====================================================================
        diameters = [ball['diameter'] for ball in balls]
        
        return {
            'count': len(balls),
            'min_size': round(min(diameters), 3) if diameters else 0,
            'max_size': round(max(diameters), 3) if diameters else 0,
            'avg_size': round(sum(diameters) / len(diameters), 3) if diameters else 0,
            'balls': balls,
            'image_with_overlay': image_with_overlay,
            'frame_pixel_size': {
                'width': roi_width,
                'height': roi_height
            }
        }
    
    def _image_to_base64(self, image, format='jpeg'):
        """将OpenCV图像转换为Base64字符串"""
        _, buffer = cv2.imencode(f'.{format}', image)
        base64_string = base64.b64encode(buffer).decode('utf-8')
        return f'data:image/{format};base64,{base64_string}'


# =========================================================================
#                           测试代码
# =========================================================================

if __name__ == '__main__':
    print("丸子检测器模块")
    print("=" * 50)
    print("使用方法:")
    print("  from detector import BallDetector")
    print("  detector = BallDetector()")
    print("  result = detector.detect(image)")
    print("=" * 50)
