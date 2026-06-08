# -*- coding: utf-8 -*-
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

PORT = 9000
app = Flask(__name__)

# 确保assets目录存在
os.makedirs('assets', exist_ok=True)


# 添加静态文件路由
@app.route('/assets/<path:filename>')
def serve_asset(filename):
    """提供assets目录下的文件访问"""
    from flask import send_from_directory
    return send_from_directory('assets', filename)


# 检测器实例
detector = BallDetector()


# 工具函数

def base64_to_image(base64_string):
    """将Base64字符串转换为OpenCV图像"""
    try:
        if base64_string is None or len(base64_string) == 0:
            print("错误: Base64字符串为空")
            return None

        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]

        image_data = base64.b64decode(base64_string)
        if len(image_data) == 0:
            print("错误: Base64解码后数据为空")
            return None

        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            print("错误: OpenCV解码图像失败")
            return None

        return image
    except Exception as e:
        print(f"Base64转图像错误: {e}")
        return None


def image_to_base64(image, format='jpeg'):
    """将OpenCV图像转换为Base64字符串"""
    _, buffer = cv2.imencode(f'.{format}', image)
    base64_string = base64.b64encode(buffer).decode('utf-8')
    return f'data:image/{format};base64,{base64_string}'


# API路由

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({
        'status': 'ok',
        'service': '香连止痢丸检测系统',
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
            distortion_compensation=data.get('distortionCompensation', 1.0),
            size_calibration=data.get('sizeCalibration', 1.0)
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
            'framePixelSize': result.get('frame_pixel_size'),
            'intermediateImages': result.get('intermediate_images', {})
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


@app.route('/api/export', methods=['POST'])
def export_data():
    """导出数据到服务器指定目录"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'success': False, 'error': '缺少数据'}), 400

        export_format = data.get('format', 'csv')
        balls = data.get('balls', [])
        metadata = data.get('metadata', {})

        if not balls:
            return jsonify({'success': False, 'error': '没有可导出的数据'}), 400

        # 生成文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if export_format == 'csv':
            filename = f'检测数据_{timestamp}.csv'
            filepath = os.path.join('assets', filename)

            # 写入CSV文件
            with open(filepath, 'w', encoding='utf-8-sig') as f:
                # 写入元数据
                f.write(f"# 香连止痢丸检测数据\n")
                f.write(f"# 导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# 标定框尺寸: {metadata.get('width', 60)}mm x {metadata.get('height', 60)}mm\n")
                f.write(f"# 检测数量: {len(balls)}\n")
                f.write(f"# 最小直径: {metadata.get('minSize', 0):.2f}mm\n")
                f.write(f"# 最大直径: {metadata.get('maxSize', 0):.2f}mm\n")
                f.write(f"# 平均直径: {metadata.get('avgSize', 0):.2f}mm\n")
                f.write("\n")

                # 写入表头
                f.write("序号,直径(mm),X坐标(px),Y坐标(px),半径(px),面积(px),圆整度\n")

                # 写入数据
                for i, ball in enumerate(balls, 1):
                    f.write(
                        f"{i},{ball.get('diameter', 0):.3f},{ball.get('x', 0):.1f},{ball.get('y', 0):.1f},{ball.get('radius', 0):.1f},{ball.get('area', 0):.1f},{ball.get('circularity', 0):.3f}\n")

            mime_type = 'text/csv'
        else:
            filename = f'香连止痢丸检测数据_{timestamp}.json'
            filepath = os.path.join('assets', filename)

            import json
            export_data = {
                '导出时间': datetime.now().isoformat(),
                '标定框尺寸': {
                    '宽度_mm': metadata.get('width', 60),
                    '高度_mm': metadata.get('height', 60)
                },
                '检测结果': {
                    '数量': len(balls),
                    '最小直径_mm': metadata.get('minSize', 0),
                    '最大直径_mm': metadata.get('maxSize', 0),
                    '平均直径_mm': metadata.get('avgSize', 0)
                },
                '丸子列表': balls
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)

            mime_type = 'application/json'

        # 返回文件URL
        file_url = f'/assets/{filename}'

        return jsonify({
            'success': True,
            'message': f'文件已保存到 {filepath}',
            'filename': filename,
            'filepath': filepath,
            'url': file_url
        })

    except Exception as e:
        print(f"导出错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# =============================== 主程序 ===============================

if __name__ == '__main__':
    print("=" * 60)
    print("香连止痢丸检测系统")
    print(f"服务地址: http://localhost:{PORT}")

    app.run(host='0.0.0.0', port=PORT, debug=True, threaded=True)
