# -*- coding: utf-8 -*-

import cv2
import numpy as np
import base64


class BallDetector:
    """
    香连止痢丸检测器类
    
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
        # 二值化阈值 - 较低的阈值适合检测黑色香连止痢丸
        self.threshold = 50
        
        # 香连止痢丸尺寸范围 (mm) - 根据香连止痢丸实际大小调整
        # 实际香连止痢丸直径约1.3-1.8mm，但检测值可能因拍摄条件有偏差
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
        
        优化版本v2：更稳定的多策略检测
        
        返回:
            dict: {
                'found': bool,
                'x': int, 'y': int, 'w': int, 'h': int,
                'corners': list  # 四个角点 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
            }
        """
        height, width = image.shape[:2]
        
        # 收集所有检测到的四边形
        all_quads = []
        
        # =============================================================
        # 策略1：多阈值Canny边缘检测
        # =============================================================
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        canny_params = [
            (20, 60),
            (30, 100),
            (50, 150),
            (40, 80),
        ]
        
        for low, high in canny_params:
            edges = cv2.Canny(blurred, low, high)
            # 形态学增强
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < (width * height) * 0.005:  # 降低阈值到0.5%
                    continue
                if area > (width * height) * 0.95:
                    continue
                
                perimeter = cv2.arcLength(contour, True)
                if perimeter < 20:
                    continue
                
                epsilon = 0.015 * perimeter  # 更精细的多边形近似
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if 4 <= len(approx) <= 8:  # 允许4-8个顶点的近似四边形
                    all_quads.extend(self._refine_to_quad(approx, contour, width, height))
        
        # =============================================================
        # 策略2：颜色空间分析 - 检测深色区域
        # =============================================================
        # 转换为HSV检测深色（黑色）区域
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # 黑色的HSV范围
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])
        mask_black = cv2.inRange(hsv, lower_black, upper_black)
        
        # 形态学操作
        kernel = np.ones((5, 5), np.uint8)
        mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_CLOSE, kernel)
        mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_OPEN, kernel)
        
        # 找轮廓
        black_contours, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in black_contours:
            area = cv2.contourArea(contour)
            if area < (width * height) * 0.01:
                continue
            if area > (width * height) * 0.8:
                continue
            
            perimeter = cv2.arcLength(contour, True)
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if 4 <= len(approx) <= 8:
                all_quads.extend(self._refine_to_quad(approx, contour, width, height))
        
        # =============================================================
        # 策略3：霍夫直线检测辅助
        # =============================================================
        edges_hough = cv2.Canny(blurred, 50, 150)
        lines = cv2.HoughLinesP(edges_hough, 1, np.pi/180, 
                                  threshold=50, 
                                  minLineLength=min(width, height) * 0.2,
                                  maxLineGap=20)
        
        if lines is not None:
            hough_quads = self._lines_to_quad(lines, width, height)
            all_quads.extend(hough_quads)
        
        # =============================================================
        # 选择最佳四边形
        # =============================================================
        if not all_quads:
            return self._detect_frame_by_gradient_v2(image, gray)
        
        # 去重：合并相似的四边形
        unique_quads = self._merge_similar_quads(all_quads, width, height)
        
        # 按得分排序
        unique_quads.sort(key=lambda x: x['score'], reverse=True)
        
        # 选择最佳候选
        best = unique_quads[0]
        
        if best['score'] < 0.5:  # 得分阈值
            return self._detect_frame_by_gradient_v2(image, gray)
        
        # 验证结果
        corners = best['corners']
        x, y, w, h = best['x'], best['y'], best['w'], best['h']
        
        # 确保边界有效
        if w < 50 or h < 50:
            return self._detect_frame_by_gradient_v2(image, gray)
        
        return {
            'found': True,
            'x': int(x),
            'y': int(y),
            'w': int(w),
            'h': int(h),
            'corners': corners.tolist() if isinstance(corners, np.ndarray) else corners
        }
    
    def _refine_to_quad(self, approx, contour, width, height):
        """
        将近似多边形精细化为四边形候选
        """
        results = []
        
        if len(approx) == 4:
            corners = approx.reshape(4, 2).astype(float)
            corners = self._order_points(corners)
            
            x, y, w, h = cv2.boundingRect(approx)
            area = cv2.contourArea(contour)
            
            # 计算得分
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 10
            area_ratio = area / (w * h) if w * h > 0 else 0
            image_area_ratio = area / (width * height)
            
            # 角点锐度评估
            sharpness = self._evaluate_corner_sharpness(contour)
            
            score = (image_area_ratio * 100) * (1.0 / aspect_ratio if aspect_ratio < 2.0 else 0.3) * (area_ratio * 2) * sharpness
            
            results.append({
                'corners': corners,
                'x': x, 'y': y, 'w': w, 'h': h,
                'score': score,
                'area': area
            })
        
        return results
    
    def _evaluate_corner_sharpness(self, contour):
        """
        评估轮廓角点的锐度（四边形特征）
        返回0-1之间的分数
        """
        if len(contour) < 4:
            return 0
        
        # 计算凸包
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(contour)
        
        if hull_area == 0:
            return 0
        
        # 凸包面积比接近1表示形状接近凸形（如矩形）
        solidity = contour_area / hull_area
        
        # 检查最小外接矩形的宽高比
        rect = cv2.minAreaRect(contour)
        rect_area = rect[1][0] * rect[1][1] if rect[1][0] > 0 and rect[1][1] > 0 else 0
        
        if rect_area > 0:
            fill_ratio = contour_area / rect_area
        else:
            fill_ratio = 0
        
        # 综合得分
        return (solidity * 0.5 + fill_ratio * 0.5)
    
    def _lines_to_quad(self, lines, width, height):
        """
        从霍夫直线转换为四边形
        """
        if lines is None or len(lines) < 4:
            return []
        
        results = []
        
        # 分离水平和垂直线
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) > abs(y2 - y1):  # 水平线
                horizontal_lines.append((min(y1, y2), min(x1, x2), max(x1, x2)))
            else:  # 垂直线
                vertical_lines.append((min(x1, x2), min(y1, y2), max(y1, y2)))
        
        if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
            return []
        
        # 排序
        horizontal_lines.sort(key=lambda x: x[0])  # 按y坐标排序
        vertical_lines.sort(key=lambda x: x[0])    # 按x坐标排序
        
        # 取最外侧的四条线
        top_line = horizontal_lines[0]
        bottom_line = horizontal_lines[-1]
        left_line = vertical_lines[0]
        right_line = vertical_lines[-1]
        
        # 计算角点
        corners = np.array([
            [left_line[0], top_line[0]],      # 左上
            [right_line[0], top_line[0]],     # 右上
            [right_line[0], bottom_line[0]],   # 右下
            [left_line[0], bottom_line[0]]     # 左下
        ], dtype=float)
        
        corners = self._order_points(corners)
        
        x, y, w, h = cv2.boundingRect(corners.astype(int))
        area = w * h
        image_area_ratio = area / (width * height)
        
        # 霍夫直线检测的置信度较低
        score = image_area_ratio * 50
        
        results.append({
            'corners': corners,
            'x': x, 'y': y, 'w': w, 'h': h,
            'score': score,
            'area': area
        })
        
        return results
    
    def _merge_similar_quads(self, quads, width, height):
        """
        合并相似的四边形，去除重复检测
        """
        if not quads:
            return []
        
        # 按得分排序
        quads.sort(key=lambda x: x['score'], reverse=True)
        
        merged = []
        used_indices = set()
        
        for i, quad in enumerate(quads):
            if i in used_indices:
                continue
            
            # 查找相似的四边形
            similar_group = [quad]
            used_indices.add(i)
            
            for j, other in enumerate(quads[i+1:], start=i+1):
                if j in used_indices:
                    continue
                
                # 检查是否相似（角点距离相近）
                if self._quads_similar(quad['corners'], other['corners'], threshold=0.15):
                    similar_group.append(other)
                    used_indices.add(j)
            
            # 合并：取平均角点位置，加权平均得分
            if len(similar_group) == 1:
                merged.append(quad)
            else:
                weights = [q['score'] for q in similar_group]
                total_weight = sum(weights)
                
                avg_corners = np.zeros((4, 2))
                for q, w in zip(similar_group, weights):
                    avg_corners += q['corners'] * (w / total_weight)
                
                # 计算合并后的边界
                x, y, w, h = cv2.boundingRect(avg_corners.astype(int))
                area = w * h
                
                merged.append({
                    'corners': self._order_points(avg_corners),
                    'x': x, 'y': y, 'w': w, 'h': h,
                    'score': max(q['score'] for q in similar_group),  # 使用最高得分
                    'area': area
                })
        
        return merged
    
    def _quads_similar(self, corners1, corners2, threshold=0.1):
        """
        检查两个四边形是否相似（角点距离在阈值内）
        """
        if corners1 is None or corners2 is None:
            return False
        
        corners1 = np.array(corners1).reshape(4, 2)
        corners2 = np.array(corners2).reshape(4, 2)
        
        # 计算所有角点对之间的距离
        min_dist = float('inf')
        for c1 in corners1:
            for c2 in corners2:
                dist = np.sqrt(np.sum((c1 - c2) ** 2))
                min_dist = min(min_dist, dist)
        
        # 计算四边形的平均尺寸作为归一化参考
        avg_size = np.sqrt(np.sum(corners1[0] - corners1[2])**2 + np.sum(corners1[1] - corners1[3])**2) / 2
        
        if avg_size == 0:
            return False
        
        # 归一化距离
        normalized_dist = min_dist / avg_size
        
        return normalized_dist < threshold
    
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
    
    def _detect_frame_by_gradient_v2(self, image, gray=None):
        """
        优化的梯度扫描方法检测黑框（版本2）
        
        改进点：
        1. 多阈值检测，取最稳定的结果
        2. 使用累积分布确定边界
        3. 添加置信度评估
        """
        height, width = image.shape[:2]
        
        if gray is None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 使用多条扫描线
        scan_lines_h = np.linspace(0.25, 0.75, 5) * height  # 5条水平扫描线
        scan_lines_v = np.linspace(0.25, 0.75, 5) * width   # 5条垂直扫描线
        
        left_edges = []
        right_edges = []
        top_edges = []
        bottom_edges = []
        
        # 水平扫描 - 找左右边界
        for y in scan_lines_h:
            y = int(y)
            if y >= height:
                continue
            profile = gray[y, :]
            gradient = np.abs(np.diff(profile.astype(float)))
            
            # 多种阈值策略
            thresholds = [
                np.percentile(gradient, 90),  # 90分位
                np.mean(gradient) + np.std(gradient) * 1.5,
                np.max(gradient) * 0.5
            ]
            
            left_candidates = []
            right_candidates = []
            
            for thresh in thresholds:
                for i in range(len(gradient)):
                    if gradient[i] > thresh and len(left_candidates) < 3:
                        left_candidates.append(i)
                        break
                
                for i in range(len(gradient) - 1, 0, -1):
                    if gradient[i] > thresh and len(right_candidates) < 3:
                        right_candidates.append(i)
                        break
            
            # 取中值
            if left_candidates:
                left_edges.append(np.median(left_candidates))
            if right_candidates:
                right_edges.append(np.median(right_candidates))
        
        # 垂直扫描 - 找上下边界
        for x in scan_lines_v:
            x = int(x)
            if x >= width:
                continue
            profile = gray[:, x]
            gradient = np.abs(np.diff(profile.astype(float)))
            
            thresholds = [
                np.percentile(gradient, 90),
                np.mean(gradient) + np.std(gradient) * 1.5,
                np.max(gradient) * 0.5
            ]
            
            top_candidates = []
            bottom_candidates = []
            
            for thresh in thresholds:
                for i in range(len(gradient)):
                    if gradient[i] > thresh and len(top_candidates) < 3:
                        top_candidates.append(i)
                        break
                
                for i in range(len(gradient) - 1, 0, -1):
                    if gradient[i] > thresh and len(bottom_candidates) < 3:
                        bottom_candidates.append(i)
                        break
            
            if top_candidates:
                top_edges.append(np.median(top_candidates))
            if bottom_candidates:
                bottom_edges.append(np.median(bottom_candidates))
        
        # 需要至少有一定数量的有效边缘
        if len(left_edges) < 2 or len(right_edges) < 2 or len(top_edges) < 2 or len(bottom_edges) < 2:
            return {'found': False, 'x': 0, 'y': 0, 'w': 0, 'h': 0, 'corners': None}
        
        # 计算边界（使用中值减少异常值影响）
        left_x = int(np.median(left_edges))
        right_x = int(np.median(right_edges))
        top_y = int(np.median(top_edges))
        bottom_y = int(np.median(bottom_edges))
        
        # 确保边界有效
        if left_x >= right_x or top_y >= bottom_y:
            return {'found': False, 'x': 0, 'y': 0, 'w': 0, 'h': 0, 'corners': None}
        
        frame_w = right_x - left_x
        frame_h = bottom_y - top_y
        
        # 验证尺寸合理性
        min_size = min(width, height) * 0.1
        max_size = min(width, height) * 0.95
        
        if frame_w < min_size or frame_h < min_size or frame_w > max_size or frame_h > max_size:
            return {'found': False, 'x': 0, 'y': 0, 'w': 0, 'h': 0, 'corners': None}
        
        # 构造四个角点
        corners = [
            [float(left_x), float(top_y)],      # 左上
            [float(right_x), float(top_y)],     # 右上
            [float(right_x), float(bottom_y)],  # 右下
            [float(left_x), float(bottom_y)]    # 左下
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
    #                           香连止痢丸检测算法
    # =========================================================================
    
    def detect(self, image):
        """
        检测图像中的香连止痢丸
        
        算法流程:
            1. 检测实际黑框位置（四点检测，支持透视变形）
            2. 对黑框区域进行透视校正（消除变形）
            3. 灰度化
            4. 高斯模糊 (降噪)
            5. 二值化 (阈值分割)
            6. 形态学操作 (腐蚀+膨胀)
            7. 轮廓检测
            8. 圆整度过滤
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
                'balls': list,           # 香连止痢丸列表
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
        # 步骤5: 自适应二值化 - 基于直方图统计
        # =====================================================================
        # 统计ROI区域内所有像素的灰度分布
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten()
        
        # 计算直方图的累计分布
        total_pixels = gray.shape[0] * gray.shape[1]
        cumsum = np.cumsum(hist)
        
        # 方法1: Otsu自动阈值
        # 遍历所有可能的阈值，计算类间方差，找最优分割点
        max_variance = 0
        otsu_threshold = self.threshold
        
        for t in range(1, 255):
            # 前景（< t）和背景（>= t）的像素比例
            w0 = cumsum[t] / total_pixels
            w1 = 1 - w0
            
            if w0 == 0 or w1 == 0:
                continue
            
            # 计算前景和背景的平均灰度
            mu0 = np.sum(np.arange(t) * hist[:t]) / cumsum[t] if cumsum[t] > 0 else 0
            mu1 = np.sum(np.arange(t, 256) * hist[t:]) / (total_pixels - cumsum[t]) if (total_pixels - cumsum[t]) > 0 else 0
            
            # 类间方差
            variance = w0 * w1 * (mu0 - mu1) ** 2
            
            if variance > max_variance:
                max_variance = variance
                otsu_threshold = t
        
        # 方法2: 基于直方图谷底的自适应阈值
        # 找直方图的双峰（背景峰 + 前景峰），阈值在谷底
        # 使用简单的高斯平滑
        def gaussian_kernel(size, sigma):
            x = np.arange(size) - size // 2
            kernel = np.exp(-x**2 / (2 * sigma**2))
            return kernel / kernel.sum()
        
        kernel = gaussian_kernel(7, 3)  # 核大小7，sigma=3
        hist_smooth = np.convolve(hist.astype(float), kernel, mode='same')
        
        # 找两个主要峰的位置
        peak_threshold_val = np.max(hist_smooth) * 0.1  # 峰的高度阈值
        peaks = []
        for i in range(2, 254):
            if hist_smooth[i] > peak_threshold_val and hist_smooth[i] > hist_smooth[i-1] and hist_smooth[i] > hist_smooth[i+1]:
                peaks.append((i, hist_smooth[i]))
        
        # 如果找到两个峰，阈值在它们之间
        if len(peaks) >= 2:
            peaks.sort(key=lambda x: x[1], reverse=True)
            peak1 = peaks[0][0]  # 最大峰（背景）
            peak2 = peaks[1][0]  # 次大峰
            
            if peak1 > peak2:
                peak1, peak2 = peak2, peak1
            
            # 在两个峰之间找最小值作为阈值
            valley_threshold = peak1 + np.argmin(hist_smooth[peak1:min(peak2+1, 256)])
        else:
            valley_threshold = otsu_threshold
        
        # 综合两种方法：取Otsu和谷底的平均值
        auto_threshold = int((otsu_threshold + valley_threshold) / 2)
        
        # 如果用户设置了阈值，使用加权平均
        if self.threshold > 0:
            final_threshold = int(self.threshold * 0.3 + auto_threshold * 0.7)
        else:
            final_threshold = auto_threshold
        
        # THRESH_BINARY_INV: 深色物体 (香连止痢丸) 变为白色
        _, binary = cv2.threshold(
            blurred, 
            final_threshold, 
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
            cv2.RETR_LIST,  # 检测所有轮廓，不建立父子关系
            cv2.CHAIN_APPROX_SIMPLE  # 压缩轮廓点
        )
        
        # 绘制所有检测到的轮廓 - 在ROI区域上绘制
        contour_image = roi.copy()
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
        intermediate_images['contours'] = self._image_to_base64(contour_image)
        
        # =====================================================================
        # 步骤8: 创建掩码排除黑框边缘区域
        # 关键修复：黑框边缘可能被误检测为丸子，需要排除
        # =====================================================================
        # 透视校正后的roi图像中，排除边缘区域
        roi_height, roi_width = roi.shape[:2]
        margin = int(min(roi_width, roi_height) * 0.08)  # 边缘留8%的安全区域
        
        # 创建掩码：边缘区域为0，中心区域为255
        mask = np.ones((roi_height, roi_width), dtype=np.uint8) * 255
        # 将边缘区域置零
        mask[:margin, :] = 0
        mask[-margin:, :] = 0
        mask[:, :margin] = 0
        mask[:, -margin:] = 0
        
        # =====================================================================
        # 步骤9: 计算像素到毫米的比例
        # 关键改进：使用检测到的实际黑框像素尺寸计算比例
        # =====================================================================
        # 如果检测到黑框，使用黑框的实际像素尺寸
        # 黑框实际尺寸 = calibration_width (mm)，检测到的像素尺寸 = frame_pixel_size
        # 注意：透视校正后的图像已经排除了黑框边框，所以比例计算要准确
        # 使用校正后图像的尺寸（排除边框后的可用区域）
        usable_size = min(roi_width, roi_height) - 2 * margin
        mm_per_pixel = self.calibration_width / usable_size * self.distortion_compensation
        
        # =====================================================================
        # 步骤10: 分析每个轮廓
        # =====================================================================
        balls = []
        
        # 调试统计：记录每个过滤步骤排除的轮廓数量
        debug_stats = {
            'total_contours': len(contours),
            'filtered_no_moment': 0,
            'filtered_mask': 0,
            'filtered_area_min': 0,
            'filtered_area_max': 0,
            'filtered_perimeter': 0,
            'filtered_circularity': 0,
            'filtered_size_range': 0,
            'passed': 0
        }
        
        for contour in contours:
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            
            # 计算中心点（必须在掩码区域内）
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx_roi = int(M['m10'] / M['m00'])
                cy_roi = int(M['m01'] / M['m00'])
            else:
                debug_stats['filtered_no_moment'] += 1
                continue
            
            # 关键修复：检查中心点是否在有效掩码区域内
            # 如果中心点在边缘区域，跳过（可能是黑框边缘）
            if mask[cy_roi, cx_roi] == 0:
                debug_stats['filtered_mask'] += 1
                continue
            
            # 过滤太小的区域 (噪点) - 适当降低以检测浅色丸子
            if area < 15:
                debug_stats['filtered_area_min'] += 1
                continue
            
            # 过滤太大的区域（可能是黑框边缘或背景）
            # 使用roi区域面积的5%作为上限
            max_area = (min(roi_width, roi_height) * 0.42) ** 2 * np.pi  # 约等于直径为42%边长的圆面积
            if area > max_area:
                debug_stats['filtered_area_max'] += 1
                continue
            
            # 计算周长
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter == 0:
                debug_stats['filtered_perimeter'] += 1
                continue
            
            # =============================================================
            # 计算圆整度
            # 公式: 圆整度 = 4π × 面积 / 周长²
            # 完美圆形的圆整度 = 1
            # =============================================================
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            
            # 过滤非圆形物体 (阈值0.35，平衡黑框边缘和浅色丸子)
            if circularity < 0.35:
                debug_stats['filtered_circularity'] += 1
                continue
            
            # =============================================================
            # 使用最小外接圆法计算直径（不受二值化阈值影响）
            # =============================================================
            (fit_cx, fit_cy), fit_radius = cv2.minEnclosingCircle(contour)
            
            # 优先使用圆拟合的直径，更稳定
            equivalent_diameter = fit_radius * 2
            
            # 计算实际直径 (mm) - 应用尺寸校准系数
            real_diameter = equivalent_diameter * mm_per_pixel * self.size_calibration
            
            # 暂时禁用尺寸范围过滤，显示所有检测到的丸子
            # if real_diameter < self.min_size or real_diameter > self.max_size:
            #     debug_stats['filtered_size_range'] += 1
            #     print(f"[DEBUG] 丸子被尺寸过滤: 计算直径={real_diameter:.3f}mm, 范围=[{self.min_size}, {self.max_size}], 像素面积={area}", flush=True)
            #     continue
            
            # 使用圆拟合的中心点
            cx = int(fit_cx)
            cy = int(fit_cy)
            
            # 使用圆拟合的半径
            radius = float(fit_radius)
            
            balls.append({
                'id': len(balls) + 1,
                'x': float(cx),
                'y': float(cy),
                'radius': float(radius),
                'diameter': round(real_diameter, 3),
                'area': float(area),
                'circularity': round(circularity, 3)
            })
        
        debug_stats['passed'] = len(balls)
        
        # 打印调试信息到日志文件和控制台
        import sys
        diameters = [ball['diameter'] for ball in balls]
        debug_msg = (
            f"[DEBUG] 轮廓过滤统计: {debug_stats}\n"
            f"[DEBUG] 尺寸参数: mm_per_pixel={mm_per_pixel:.6f}, "
            f"usable_size={usable_size}, roi_size={roi_width}x{roi_height}, "
            f"calibration_width={self.calibration_width}mm, "
            f"size_calibration={self.size_calibration}\n"
            f"[DEBUG] 检测结果: count={len(balls)}, "
            f"min_diameter={min(diameters):.3f}mm, "
            f"max_diameter={max(diameters):.3f}mm, "
            f"avg_diameter={sum(diameters)/len(diameters):.3f}mm"
        )
        print(debug_msg, flush=True)
        try:
            with open('/app/work/logs/bypass/app.log', 'a', encoding='utf-8') as f:
                f.write(debug_msg + '\n')
        except:
            pass
        
        # =====================================================================
        # 步骤10: 绘制检测结果
        # =====================================================================
        result_image = image.copy()
        
        # 创建标注图像（在透视校正后的图像上绘制，避免坐标转换误差）
        overlay_roi = roi.copy()
        
        # 保存透视变换信息用于将标注贴回原图
        overlay_matrix = None
        overlay_corners = None
        
        # 绘制检测到的黑框（橙色四边形）
        if use_detected_frame and detected_frame['found'] and detected_frame.get('corners'):
            corners = detected_frame['corners']
            corners_int = np.array(corners, dtype=np.int32)
            
            # 在原图上绘制四边形
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
            
            # 保存透视变换信息，用于将roi标注贴回原图
            # roi是透视校正后的图像，corners是原图中的四边形角点
            overlay_corners = corners
            roi_h, roi_w = roi.shape[:2]
            # 校正后图像的四个角
            overlay_dst_corners = np.array([
                [0, 0],
                [roi_w - 1, 0],
                [roi_w - 1, roi_h - 1],
                [0, roi_h - 1]
            ], dtype=np.float32)
            # 计算从roi角点到原图角点的透视变换矩阵
            overlay_matrix = cv2.getPerspectiveTransform(overlay_dst_corners, np.array(corners, dtype=np.float32))
        
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
        
        # 绘制每个香连止痢丸 - 在roi图像上绘制
        for ball in balls:
            # 使用roi图像坐标系中的坐标
            # 如果是透视校正后的图像，坐标直接在roi中
            # 如果是手动设置的ROI，需要加上roi_x, roi_y的偏移
            if use_detected_frame:
                # 透视校正情况下，坐标已经在roi坐标系中（校正后的图像）
                center_roi = (int(ball['x']), int(ball['y']))
            else:
                # 非透视校正情况，加上roi偏移
                center_roi = (int(ball['x'] + roi_x + margin), int(ball['y'] + roi_y + margin))
            
            radius = int(ball['radius'])
            
            # 在roi图像上绘制圆圈 (绿色)
            cv2.circle(overlay_roi, center_roi, radius, (0, 255, 0), 2)
            
            # 绘制中心点 (绿色)
            cv2.circle(overlay_roi, center_roi, 2, (0, 255, 0), -1)
            
            # 准备标注文本
            label = f"#{ball['id']}: {ball['diameter']:.2f}mm"
            
            # 计算文本位置
            text_x = center_roi[0] - 40
            text_y = center_roi[1] - radius - 10
            
            # 绘制文本背景 (黑色)
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                overlay_roi,
                (text_x - 5, text_y - text_height - 5),
                (text_x + text_width + 5, text_y + 5),
                (0, 0, 0),
                -1
            )
            
            # 绘制文本 (绿色)
            cv2.putText(
                overlay_roi, 
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6,
                (0, 255, 0),
                2
            )
        
        # 将标注后的roi图像透视变换贴回原图
        if use_detected_frame and overlay_corners is not None:
            # 获取校正后图像的尺寸
            roi_h, roi_w = overlay_roi.shape[:2]
            
            # 方法1: 直接将校正后的ROI图像叠加到原图的对应区域
            # 计算原图中四边形的包围盒
            corners_arr = np.array(overlay_corners)
            min_x = max(0, int(np.min(corners_arr[:, 0])))
            min_y = max(0, int(np.min(corners_arr[:, 1])))
            max_x = min(result_image.shape[1], int(np.max(corners_arr[:, 0])) + 1)
            max_y = min(result_image.shape[0], int(np.max(corners_arr[:, 1])) + 1)
            
            # 计算ROI图像中对应区域的大小
            patch_w = max_x - min_x
            patch_h = max_y - min_y
            
            # 确保尺寸匹配
            if patch_w > 0 and patch_h > 0:
                # 缩放校正后的ROI图像以匹配目标区域
                overlay_resized = cv2.resize(overlay_roi, (patch_w, patch_h))
                
                # 创建掩码：白色背景 + 黑色边框
                # 将校正后图像的边缘区域（可能是黑框边缘）设为黑色
                border_size = int(min(patch_w, patch_h) * 0.05)  # 5%边缘
                mask_patch = np.ones((patch_h, patch_w), dtype=np.uint8) * 255
                if border_size > 0:
                    mask_patch[:border_size, :] = 0
                    mask_patch[-border_size:, :] = 0
                    mask_patch[:, :border_size] = 0
                    mask_patch[:, -border_size:] = 0
                
                # 叠加
                for c in range(3):
                    result_image[min_y:max_y, min_x:max_x, c] = np.where(
                        mask_patch > 0,
                        overlay_resized[:, :, c],
                        result_image[min_y:max_y, min_x:max_x, c]
                    )
            
            # 重新绘制黑框（在最上层）
            cv2.polylines(result_image, [np.array(overlay_corners, dtype=np.int32)], 
                         True, (0, 165, 255), 3)
            # 标注四个角点
            for i, (cx, cy) in enumerate(overlay_corners):
                cv2.circle(result_image, (int(cx), int(cy)), 8, (0, 255, 0), -1)
                cv2.putText(result_image, str(i+1), (int(cx)+10, int(cy)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        elif not use_detected_frame:
            # 非透视校正情况，直接用roi区域替换
            result_image[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width] = overlay_roi
        
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
                'width': usable_size,
                'height': usable_size,
                'margin': margin
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
    print("香连止痢丸检测器模块")
    print("=" * 50)
    print("使用方法:")
    print("  from detector import BallDetector")
    print("  detector = BallDetector()")
    print("  result = detector.detect(image)")
    print("=" * 50)
