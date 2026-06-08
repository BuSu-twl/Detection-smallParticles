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
        # 二值化阈值 - 较低的阈值适合检测黑色丸子
        self.threshold = 50
        
        # 丸子尺寸范围 (mm) - 根据丸子实际大小调整
        # 实际丸子直径约1.3-1.8mm，但检测值可能因拍摄条件有偏差
        self.min_size = 0.3   # 最小 0.3mm（放宽范围）
        self.max_size = 3.0   # 最大 3.0mm（放宽范围）
        
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
        
        # 尺寸校准系数 - 用于修正检测直径的偏差
        # 如果检测值偏小，可设置 > 1.0 的值进行修正
        self.size_calibration = 1.0
    
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
            size_calibration: 尺寸校准系数 (用于修正直径偏差)
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
        
        算法：通过扫描图像边缘，检测黑框的四条边界线
        
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
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 方法1：检测黑色边界线
        # 黑框是深色线条，在图像中形成明显的灰度变化
        # 扫描中间行和中间列来找到边界
        
        # 水平扫描 - 找左右边界
        mid_y = height // 2
        h_profile = gray[mid_y, :]
        
        # 垂直扫描 - 找上下边界
        mid_x = width // 2
        v_profile = gray[:, mid_x]
        
        # 计算梯度来找边界（黑线是灰度下降的区域）
        h_gradient = np.abs(np.diff(h_profile.astype(float)))
        v_gradient = np.abs(np.diff(v_profile.astype(float)))
        
        # 找到显著的梯度变化点（黑线边缘）
        threshold = np.mean(h_gradient) + np.std(h_gradient) * 2
        
        # 左边界 - 从左边找第一个显著变化
        left_x = 0
        for i in range(len(h_gradient)):
            if h_gradient[i] > threshold:
                left_x = i
                break
        
        # 右边界 - 从右边找第一个显著变化
        right_x = width - 1
        for i in range(len(h_gradient) - 1, 0, -1):
            if h_gradient[i] > threshold:
                right_x = i
                break
        
        # 上边界 - 从上边找第一个显著变化
        threshold_v = np.mean(v_gradient) + np.std(v_gradient) * 2
        top_y = 0
        for i in range(len(v_gradient)):
            if v_gradient[i] > threshold_v:
                top_y = i
                break
        
        # 下边界 - 从下边找第一个显著变化
        bottom_y = height - 1
        for i in range(len(v_gradient) - 1, 0, -1):
            if v_gradient[i] > threshold_v:
                bottom_y = i
                break
        
        # 计算框的尺寸
        frame_w = right_x - left_x
        frame_h = bottom_y - top_y
        
        # 验证检测结果的合理性
        # 黑框应该占图像的一定比例，且接近正方形
        min_size = min(width, height) * 0.2
        max_size = min(width, height) * 0.95
        
        if frame_w < min_size or frame_h < min_size or frame_w > max_size or frame_h > max_size:
            # 尝试备选方法
            return self._detect_frame_by_edges(image, gray)
        
        # 检查宽高比（应该是接近正方形）
        aspect_ratio = max(frame_w, frame_h) / min(frame_w, frame_h)
        if aspect_ratio > 1.5:
            # 宽高比不合理，使用备选方法
            return self._detect_frame_by_edges(image, gray)
        
        # 略微缩小检测框，确保在黑框内部（留5%边距）
        margin = int(min(frame_w, frame_h) * 0.05)
        final_x = left_x + margin
        final_y = top_y + margin
        final_w = frame_w - 2 * margin
        final_h = frame_h - 2 * margin
        
        # 强制正方形 - 取宽高的较小值
        square_size = min(final_w, final_h)
        
        # 计算中心点
        center_x_img = final_x + final_w / 2
        center_y_img = final_y + final_h / 2
        
        # 计算百分比参数
        center_x = center_x_img / width * 100 - 50
        center_y = center_y_img / height * 100 - 50
        frame_size_pct = square_size / max(width, height) * 100
        
        return {
            'found': True,
            'offsetX': round(float(center_x), 1),
            'offsetY': round(float(center_y), 1),
            'frameWidth': round(float(frame_size_pct), 1),
            'frameHeight': round(float(frame_size_pct), 1),
            'pixelWidth': int(square_size),
            'pixelHeight': int(square_size)
        }
    
    def _detect_frame_by_edges(self, image, gray=None):
        """通过边缘检测定位黑框（备选方案2）"""
        height, width = image.shape[:2]
        
        if gray is None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 使用Canny边缘检测
        edges = cv2.Canny(gray, 30, 100)
        
        # 检测直线
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=30)
        
        if lines is None or len(lines) < 4:
            # 最终备选：使用整个图像中心区域
            square_size = min(width, height) * 0.7
            return {
                'found': False,
                'offsetX': 0,
                'offsetY': 0,
                'frameWidth': 70,
                'frameHeight': 70,
                'pixelWidth': int(square_size),
                'pixelHeight': int(square_size)
            }
        
        # 分离水平和垂直线
        horizontal_lines = []  # (y, x1, x2)
        vertical_lines = []    # (x, y1, y2)
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) > abs(y2 - y1):  # 水平线
                if abs(x2 - x1) > width * 0.3:  # 只保留足够长的水平线
                    horizontal_lines.append((y1, min(x1, x2), max(x1, x2)))
            else:  # 垂直线
                if abs(y2 - y1) > height * 0.3:  # 只保留足够长的垂直线
                    vertical_lines.append((x1, min(y1, y2), max(y1, y2)))
        
        if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
            square_size = min(width, height) * 0.7
            return {
                'found': False,
                'offsetX': 0,
                'offsetY': 0,
                'frameWidth': 70,
                'frameHeight': 70,
                'pixelWidth': int(square_size),
                'pixelHeight': int(square_size)
            }
        
        # 找最上和最下的水平线
        horizontal_lines.sort(key=lambda x: x[0])
        top_line = horizontal_lines[0]
        bottom_line = horizontal_lines[-1]
        
        # 找最左和最右的垂直线
        vertical_lines.sort(key=lambda x: x[0])
        left_line = vertical_lines[0]
        right_line = vertical_lines[-1]
        
        # 计算框的位置
        top_y = top_line[0]
        bottom_y = bottom_line[0]
        left_x = left_line[0]
        right_x = right_line[0]
        
        frame_w = right_x - left_x
        frame_h = bottom_y - top_y
        
        # 验证结果
        min_size = min(width, height) * 0.2
        if frame_w < min_size or frame_h < min_size:
            square_size = min(width, height) * 0.7
            return {
                'found': False,
                'offsetX': 0,
                'offsetY': 0,
                'frameWidth': 70,
                'frameHeight': 70,
                'pixelWidth': int(square_size),
                'pixelHeight': int(square_size)
            }
        
        # 缩小检测框
        margin = int(min(frame_w, frame_h) * 0.05)
        final_x = left_x + margin
        final_y = top_y + margin
        final_w = frame_w - 2 * margin
        final_h = frame_h - 2 * margin
        
        # 强制正方形
        square_size = min(final_w, final_h)
        
        center_x_img = final_x + final_w / 2
        center_y_img = final_y + final_h / 2
        
        center_x = center_x_img / width * 100 - 50
        center_y = center_y_img / height * 100 - 50
        frame_size_pct = square_size / max(width, height) * 100
        
        return {
            'found': True,
            'offsetX': round(center_x, 1),
            'offsetY': round(center_y, 1),
            'frameWidth': round(frame_size_pct, 1),
            'frameHeight': round(frame_size_pct, 1),
            'pixelWidth': int(square_size),
            'pixelHeight': int(square_size)
        }

    def _detect_actual_black_frame(self, image):
        """
        检测图像中实际黑框的位置（用于在结果图上标注）
        
        支持透视变形：检测四边形的四个角点
        
        返回:
            dict: {
                'found': bool,
                'x': int, 'y': int, 'w': int, 'h': int,
                'corners': list  # 四个角点 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
            }
        """
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 使用边缘检测 + 轮廓检测找四边形
        # 先用高斯模糊降噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 使用Canny边缘检测
        edges = cv2.Canny(blurred, 30, 100)
        
        # 形态学操作连接断开的边缘
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'found': False, 'x': 0, 'y': 0, 'w': 0, 'h': 0, 'corners': None}
        
        # 按面积排序，找最大的几个轮廓
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for contour in contours[:5]:  # 只检查前5个最大轮廓
            area = cv2.contourArea(contour)
            if area < (width * height) * 0.05:  # 太小，跳过
                continue
            if area > (width * height) * 0.95:  # 太大（可能是整个图像），跳过
                continue
            
            # 多边形近似
            perimeter = cv2.arcLength(contour, True)
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 检查是否为四边形
            if len(approx) == 4:
                # 获取四个角点
                corners = approx.reshape(4, 2)
                
                # 排序角点：左上、右上、右下、左下
                corners = self._order_points(corners)
                
                # 计算边界框
                x, y, w, h = cv2.boundingRect(approx)
                
                # 验证是否接近正方形（允许一定变形）
                aspect_ratio = max(w, h) / min(w, h)
                if aspect_ratio < 1.5:  # 允许一定程度的透视变形
                    return {
                        'found': True, 
                        'x': x, 
                        'y': y, 
                        'w': w, 
                        'h': h,
                        'corners': corners.tolist()
                    }
        
        # 如果没找到四边形，回退到梯度扫描方法
        return self._detect_frame_by_gradient(image, gray)
    
    def _order_points(self, pts):
        """
        将四个点按照左上、右上、右下、左下的顺序排列
        
        参数:
            pts: numpy数组，形状为(4, 2)
        
        返回:
            排序后的点数组
        """
        # 按x坐标排序
        x_sorted = pts[np.argsort(pts[:, 0]), :]
        
        # 左边两个点和右边两个点
        left_most = x_sorted[:2, :]
        right_most = x_sorted[2:, :]
        
        # 按y坐标排序左边两点，上面的为左上
        left_most = left_most[np.argsort(left_most[:, 1]), :]
        (tl, bl) = left_most
        
        # 右边两点按y排序
        right_most = right_most[np.argsort(right_most[:, 1]), :]
        (tr, br) = right_most
        
        return np.array([tl, tr, br, bl], dtype="float32")
    
    def _detect_frame_by_gradient(self, image, gray=None):
        """
        使用梯度扫描检测黑框边界（回退方法）
        """
        height, width = image.shape[:2]
        
        if gray is None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 水平扫描 - 找左右边界（扫描多条线取平均）
        scan_lines = [height//4, height//2, 3*height//4]
        left_edges = []
        right_edges = []
        
        for y in scan_lines:
            if y >= height:
                continue
            h_profile = gray[y, :]
            h_gradient = np.abs(np.diff(h_profile.astype(float)))
            threshold = np.mean(h_gradient) + np.std(h_gradient) * 2
            
            for i in range(len(h_gradient)):
                if h_gradient[i] > threshold:
                    left_edges.append(i)
                    break
            
            for i in range(len(h_gradient) - 1, 0, -1):
                if h_gradient[i] > threshold:
                    right_edges.append(i)
                    break
        
        # 垂直扫描 - 找上下边界
        scan_cols = [width//4, width//2, 3*width//4]
        top_edges = []
        bottom_edges = []
        
        for x in scan_cols:
            if x >= width:
                continue
            v_profile = gray[:, x]
            v_gradient = np.abs(np.diff(v_profile.astype(float)))
            threshold = np.mean(v_gradient) + np.std(v_gradient) * 2
            
            for i in range(len(v_gradient)):
                if v_gradient[i] > threshold:
                    top_edges.append(i)
                    break
            
            for i in range(len(v_gradient) - 1, 0, -1):
                if v_gradient[i] > threshold:
                    bottom_edges.append(i)
                    break
        
        if not left_edges or not right_edges or not top_edges or not bottom_edges:
            return {'found': False, 'x': 0, 'y': 0, 'w': 0, 'h': 0, 'corners': None}
        
        left_x = int(np.mean(left_edges))
        right_x = int(np.mean(right_edges))
        top_y = int(np.mean(top_edges))
        bottom_y = int(np.mean(bottom_edges))
        
        frame_w = right_x - left_x
        frame_h = bottom_y - top_y
        
        min_size = min(width, height) * 0.15
        
        if frame_w < min_size or frame_h < min_size:
            return {'found': False, 'x': 0, 'y': 0, 'w': 0, 'h': 0, 'corners': None}
        
        # 构造四个角点（矩形）
        corners = [
            [left_x, top_y],      # 左上
            [right_x, top_y],     # 右上
            [right_x, bottom_y],  # 右下
            [left_x, bottom_y]    # 左下
        ]
        
        return {
            'found': True, 
            'x': left_x, 
            'y': top_y, 
            'w': frame_w, 
            'h': frame_h,
            'corners': corners
        }
    
    def _perspective_correction(self, image, corners, output_size):
        """
        对图像进行透视校正，将四边形区域变换为正方形
        
        参数:
            image: 原始图像
            corners: 四个角点 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]，顺序为左上、右上、右下、左下
            output_size: 输出图像的尺寸（正方形）
        
        返回:
            校正后的图像
        """
        # 源点（检测到的四边形）
        src_pts = np.array(corners, dtype="float32")
        
        # 目标点（正方形）
        dst_pts = np.array([
            [0, 0],
            [output_size - 1, 0],
            [output_size - 1, output_size - 1],
            [0, output_size - 1]
        ], dtype="float32")
        
        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # 执行透视变换
        warped = cv2.warpPerspective(image, M, (output_size, output_size))
        
        return warped, M
    
    def _detect_frame_by_hough(self, image, gray=None):
        """
        使用霍夫直线检测黑框边界（备选方法）
        """
        height, width = image.shape[:2]
        
        if gray is None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 使用Canny边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 检测直线
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, 
                                minLineLength=min(width, height) * 0.3, maxLineGap=20)
        
        if lines is None or len(lines) < 4:
            return {'found': False, 'x': 0, 'y': 0, 'w': 0, 'h': 0}
        
        # 分离水平和垂直线
        horizontal_lines = []  # (y, x1, x2)
        vertical_lines = []    # (x, y1, y2)
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) > abs(y2 - y1):  # 水平线
                if abs(x2 - x1) > width * 0.2:
                    horizontal_lines.append((y1, min(x1, x2), max(x1, x2)))
            else:  # 垂直线
                if abs(y2 - y1) > height * 0.2:
                    vertical_lines.append((x1, min(y1, y2), max(y1, y2)))
        
        if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
            return {'found': False, 'x': 0, 'y': 0, 'w': 0, 'h': 0}
        
        # 找最上和最下的水平线
        horizontal_lines.sort(key=lambda x: x[0])
        top_y = horizontal_lines[0][0]
        bottom_y = horizontal_lines[-1][0]
        
        # 找最左和最右的垂直线
        vertical_lines.sort(key=lambda x: x[0])
        left_x = vertical_lines[0][0]
        right_x = vertical_lines[-1][0]
        
        frame_w = right_x - left_x
        frame_h = bottom_y - top_y
        
        # 验证
        min_size = min(width, height) * 0.15
        if frame_w < min_size or frame_h < min_size:
            return {'found': False, 'x': 0, 'y': 0, 'w': 0, 'h': 0}
        
        return {
            'found': True, 
            'x': left_x, 
            'y': top_y, 
            'w': frame_w, 
            'h': frame_h
        }

    # =========================================================================
    #                           丸子检测算法
    # =========================================================================
    
    def detect(self, image):
        """
        检测图像中的丸子
        
        算法流程:
            1. 检测实际黑框位置（四点检测，支持透视变形）
            2. 对黑框区域进行透视校正（消除变形）
            3. 灰度化
            4. 高斯模糊 (降噪)
            5. 二值化 (阈值分割)
            6. 形态学操作 (腐蚀+膨胀)
            7. 轮廓检测
            8. 圆形度过滤
            9. 尺寸计算和过滤（使用校正后的正方形计算比例）
            10. 绘制结果
        
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
                'frame_pixel_size': dict,   # 检测框像素尺寸
                'intermediate_images': dict  # 中间结果图像
            }
        """
        height, width = image.shape[:2]
        
        # 存储中间结果图像
        intermediate_images = {}
        
        # =====================================================================
        # 步骤1: 检测实际黑框位置（支持透视变形的四点检测）
        # =====================================================================
        detected_frame = self._detect_actual_black_frame(image)
        
        # 透视变换矩阵（用于将坐标映射回原图）
        perspective_matrix = None
        
        if detected_frame['found'] and detected_frame.get('corners'):
            # 检测到黑框，进行透视校正
            corners = detected_frame['corners']
            
            # 计算校正后的输出尺寸（使用较大的边）
            roi_width = detected_frame['w']
            roi_height = detected_frame['h']
            square_size = max(roi_width, roi_height)
            
            # 进行透视校正
            roi, perspective_matrix = self._perspective_correction(image, corners, square_size)
            
            # 校正后，整个图像就是正方形的ROI
            roi_x, roi_y = 0, 0
            frame_pixel_size = square_size
            use_detected_frame = True
            
            # 保存校正后的ROI显示
            roi_display = image.copy()
            # 绘制检测到的四边形（黄色）
            corners_int = np.array(corners, dtype=np.int32)
            cv2.polylines(roi_display, [corners_int], True, (0, 255, 255), 3)  # 黄色
            # 标注四个角点
            for i, (cx, cy) in enumerate(corners):
                cv2.circle(roi_display, (int(cx), int(cy)), 5, (0, 255, 0), -1)
            # 标注尺寸
            cv2.putText(roi_display, f"{self.calibration_width}mm x {self.calibration_height}mm", 
                       (int(corners[0][0]), int(corners[0][1]) - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            intermediate_images['roi'] = self._image_to_base64(roi_display)
        else:
            # 回退到手动设置的区域
            scale_w = self.frame_width / 100
            scale_h = self.frame_height / 100
            roi_width = int(width * scale_w)
            roi_height = int(height * scale_h)
            
            # 强制使用正方形 - 取宽高的较小值
            square_size = min(roi_width, roi_height)
            
            # 计算ROI中心位置 (考虑偏移)
            center_x = int(width / 2 + (width * self.offset_x / 100))
            center_y = int(height / 2 + (height * self.offset_y / 100))
            
            # 以中心点为基准，计算正方形ROI的位置
            roi_x = max(0, center_x - square_size // 2)
            roi_y = max(0, center_y - square_size // 2)
            
            # 边界检查
            roi_x = min(roi_x, width - square_size)
            roi_y = min(roi_y, height - square_size)
            
            roi_width = square_size
            roi_height = square_size
            frame_pixel_size = square_size
            use_detected_frame = False
            
            # 提取ROI区域
            roi = image[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
            
            # 在原图上绘制ROI区域
            roi_display = image.copy()
            cv2.rectangle(roi_display, (roi_x, roi_y), (roi_x + square_size, roi_y + square_size), (255, 255, 0), 3)
            cv2.putText(roi_display, f"ROI {square_size}x{square_size}px", (roi_x, roi_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            intermediate_images['roi'] = self._image_to_base64(roi_display)
        
        # =====================================================================
        # 步骤3: 灰度化 (只处理ROI区域)
        # =====================================================================
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # 转换为BGR以便显示 - 显示完整ROI区域
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        intermediate_images['gray'] = self._image_to_base64(gray_bgr)
        
        # =====================================================================
        # 步骤4: 高斯模糊 (降噪) - 只处理ROI区域
        # =====================================================================
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        blurred_bgr = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
        intermediate_images['blurred'] = self._image_to_base64(blurred_bgr)
        
        # =====================================================================
        # 步骤5: 二值化 - 只处理ROI区域
        # =====================================================================
        # THRESH_BINARY_INV: 深色物体 (丸子) 变为白色
        _, binary = cv2.threshold(
            blurred, 
            self.threshold, 
            255, 
            cv2.THRESH_BINARY_INV
        )
        binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        intermediate_images['binary'] = self._image_to_base64(binary_bgr)
        
        # =====================================================================
        # 步骤6: 形态学操作 - 只处理ROI区域
        # =====================================================================
        kernel = np.ones((3, 3), np.uint8)
        
        # 腐蚀: 去除细小噪点
        eroded = cv2.erode(binary, kernel, iterations=1)
        eroded_bgr = cv2.cvtColor(eroded, cv2.COLOR_GRAY2BGR)
        intermediate_images['eroded'] = self._image_to_base64(eroded_bgr)
        
        # 膨胀: 恢复物体大小
        dilated = cv2.dilate(eroded, kernel, iterations=1)
        dilated_bgr = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
        intermediate_images['dilated'] = self._image_to_base64(dilated_bgr)
        
        # =====================================================================
        # 步骤7: 轮廓检测 - 在ROI区域内
        # =====================================================================
        contours, _ = cv2.findContours(
            dilated, 
            cv2.RETR_EXTERNAL,  # 只检测外轮廓
            cv2.CHAIN_APPROX_SIMPLE  # 压缩轮廓点
        )
        
        # 绘制所有检测到的轮廓 - 在ROI区域上绘制
        contour_image = roi.copy()
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
        intermediate_images['contours'] = self._image_to_base64(contour_image)
        
        # =====================================================================
        # 步骤8: 计算像素到毫米的比例
        # 关键改进：使用检测到的实际黑框像素尺寸计算比例
        # =====================================================================
        # 如果检测到黑框，使用黑框的实际像素尺寸
        # 黑框实际尺寸 = calibration_width (mm)，检测到的像素尺寸 = frame_pixel_size
        mm_per_pixel = self.calibration_width / frame_pixel_size * self.distortion_compensation
        
        # =====================================================================
        # 步骤9: 分析每个轮廓
        # =====================================================================
        balls = []
        
        for contour in contours:
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            
            # 过滤太小的区域 (噪点) - 放宽到10
            if area < 10:
                continue
            
            # 过滤太大的区域（可能是黑框边缘）
            if area > 1000:
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
            
            # 过滤非圆形物体 (阈值 0.3，丸子应该接近圆形)
            if circularity < 0.3:
                continue
            
            # 计算等效直径 (假设物体为圆形)
            equivalent_diameter = np.sqrt(4 * area / np.pi)
            
            # 计算实际直径 (mm) - 应用尺寸校准系数
            real_diameter = equivalent_diameter * mm_per_pixel * self.size_calibration
            
            # 过滤尺寸范围（使用更宽松的范围）
            if real_diameter < self.min_size or real_diameter > self.max_size:
                continue
            
            # 计算中心点（在校正后的图像坐标系中）
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx_roi = int(M['m10'] / M['m00'])
                cy_roi = int(M['m01'] / M['m00'])
            else:
                continue
            
            # 将坐标转换回原图坐标系
            if perspective_matrix is not None and use_detected_frame:
                # 使用逆透视变换将坐标映射回原图
                inv_matrix = cv2.invert(perspective_matrix)[1]
                # 齐次坐标
                pt = np.array([[[cx_roi, cy_roi]]], dtype=np.float32)
                pt_original = cv2.perspectiveTransform(pt, inv_matrix)
                cx = int(pt_original[0][0][0])
                cy = int(pt_original[0][0][1])
            else:
                # 简单的偏移转换
                cx = cx_roi + roi_x
                cy = cy_roi + roi_y
            
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
        
        # 绘制检测到的黑框（橙色四边形）
        if use_detected_frame and detected_frame['found'] and detected_frame.get('corners'):
            corners = detected_frame['corners']
            corners_int = np.array(corners, dtype=np.int32)
            # 绘制四边形
            cv2.polylines(result_image, [corners_int], True, (0, 165, 255), 3)  # 橙色
            # 标注四个角点
            for i, (cx, cy) in enumerate(corners):
                cv2.circle(result_image, (int(cx), int(cy)), 8, (0, 255, 0), -1)
                cv2.putText(result_image, str(i+1), (int(cx)+10, int(cy)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # 标注尺寸
            cv2.putText(result_image, f"{self.calibration_width}mm (Perspective Corrected)", 
                       (int(corners[0][0]), int(corners[0][1]) - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        # 绘制ROI检测框 (红色) - 仅当未自动检测到黑框时显示
        if not use_detected_frame:
            cv2.rectangle(
                result_image, 
                (roi_x, roi_y), 
                (roi_x + square_size, roi_y + square_size),
                (0, 0, 255),  # 红色 (BGR)
                3
            )
            
            # 绘制尺寸标注 - 四边都标注
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_thickness = 2
            
            # 尺寸文本
            size_text = f"{self.calibration_width}mm"
            (text_w, text_h), _ = cv2.getTextSize(size_text, font, font_scale, font_thickness)
            
            # 顶部标注
            cv2.rectangle(
                result_image,
                (roi_x + square_size//2 - text_w//2 - 8, roi_y - text_h - 15),
                (roi_x + square_size//2 + text_w//2 + 8, roi_y - 5),
                (0, 0, 255),
                -1
            )
            cv2.putText(
                result_image,
                size_text,
                (roi_x + square_size//2 - text_w//2, roi_y - 10),
                font, font_scale, (255, 255, 255), font_thickness
            )
            
            # 底部标注
            cv2.rectangle(
                result_image,
                (roi_x + square_size//2 - text_w//2 - 8, roi_y + square_size + 5),
                (roi_x + square_size//2 + text_w//2 + 8, roi_y + square_size + text_h + 15),
                (0, 0, 255),
                -1
            )
            cv2.putText(
                result_image,
                size_text,
                (roi_x + square_size//2 - text_w//2, roi_y + square_size + text_h + 10),
                font, font_scale, (255, 255, 255), font_thickness
            )
            
            # 左侧标注（竖直）
            cv2.rectangle(
                result_image,
                (roi_x - text_h - 25, roi_y + square_size//2 - text_w//2 - 8),
                (roi_x - 5, roi_y + square_size//2 + text_w//2 + 8),
                (0, 0, 255),
                -1
            )
            # 竖直文本需要特殊处理
            left_text = size_text
            for i, char in enumerate(left_text):
                cv2.putText(
                    result_image,
                    char,
                    (roi_x - text_h - 20, roi_y + square_size//2 - text_w//2 + i * 20 + 15),
                    font, font_scale * 0.7, (255, 255, 255), font_thickness - 1
                )
        
            # 右侧标注（竖直）
            cv2.rectangle(
                result_image,
                (roi_x + square_size + 5, roi_y + square_size//2 - text_w//2 - 8),
                (roi_x + square_size + text_h + 25, roi_y + square_size//2 + text_w//2 + 8),
                (0, 0, 255),
                -1
            )
            for i, char in enumerate(size_text):
                cv2.putText(
                    result_image,
                    char,
                    (roi_x + square_size + 10, roi_y + square_size//2 - text_w//2 + i * 20 + 15),
                    font, font_scale * 0.7, (255, 255, 255), font_thickness - 1
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
                'width': square_size,
                'height': square_size
            },
            'intermediate_images': intermediate_images
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
