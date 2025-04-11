from flask import Flask, request, jsonify, render_template, Response
import torch
import cv2
import numpy as np
from PIL import Image
import io
import base64
from torchvision import transforms, models

app = Flask(__name__)

# 加载YOLOv5模型（实时检测）
model_yolov5 = torch.hub.load('ultralytics/yolov5', 'custom',
                              path="C:\\Users\\admin\\Desktop\\best.pt")
model_yolov5.conf = 0.2  # 置信度阈值

# 加载ResNet18模型（图片分类）
model_resnet18 = models.resnet18(pretrained=False)
num_ftrs = model_resnet18.fc.in_features
model_resnet18.fc = torch.nn.Linear(num_ftrs, 10)  # 10分类
checkpoint = torch.load("C:\\Users\\admin\\Desktop\\best_model.pth", map_location='cpu')
model_resnet18.load_state_dict(checkpoint["model_state_dict"])
model_resnet18.eval()

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 类别名称和所属类别
class_info = {
    "电池": {"name": "电池", "category": "有害垃圾"},
    "厨余": {"name": "厨余", "category": "厨余垃圾"},
    "纸板": {"name": "纸板", "category": "可回收垃圾"},
    "衣服": {"name": "衣服", "category": "可回收垃圾"},
    "玻璃": {"name": "玻璃", "category": "可回收垃圾"},
    "金属": {"name": "金属", "category": "可回收垃圾"},
    "纸张": {"name": "纸张", "category": "可回收垃圾"},
    "塑料": {"name": "塑料", "category": "可回收垃圾"},
    "鞋子": {"name": "鞋子", "category": "可回收垃圾"},
    "可回收": {"name": "可回收", "category": "可回收垃圾"}
}

# 定义类别名称列表，确保顺序与模型输出一致
class_names = list(class_info.keys())

@app.route('/')
def index():
    return render_template('before.html')


# 处理图片上传分类（ResNet18）
@app.route('/classify', methods=['POST'])
def classify_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'})

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model_resnet18(img_tensor)

        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        max_prob, pred_idx = torch.max(probabilities, 1)
        max_prob = max_prob.item()

        # 获取类别名称和所属类别
        class_name = class_info.get(class_names[pred_idx.item()], {"name": "未知", "category": "其他垃圾"})
        return jsonify({
            'class': class_name["name"],
            'category': class_name["category"],
            'probability': round(max_prob, 4)
        })
    except Exception as e:
        return jsonify({'error': str(e)})


# 处理摄像头实时检测（YOLOv5）
def generate_frames():
    cap = cv2.VideoCapture(0)  # 打开默认摄像头
    # 设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # 尝试设置摄像头的FOV（视野）
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # 使用MJPEG编码

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # YOLOv5检测
            results = model_yolov5(frame)
            rendered_frame = np.squeeze(results.render())  # 绘制检测框

            # 编码为JPEG
            ret, buffer = cv2.imencode('.jpg', rendered_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)