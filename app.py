import io
import math
from decimal import ROUND_HALF_UP, Decimal

import av
import requests
import streamlit as st
import torch
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from streamlit_webrtc import webrtc_streamer

model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
truetype_url = "https://github.com/JotJunior/PHP-Boleto-ZF2/blob/master/public/assets/fonts/arial.ttf?raw=true"
r = requests.get(truetype_url, allow_redirects=True)
classes = list(model.model.names.values())


def object_detection(image: Image) -> Image:
    # 物体検出の実行
    pred = model(image)
    # 色の一覧を作成
    cmap = plt.get_cmap("hsv", len(model.model.names))

    # フォントサイズ設定
    sqrt = math.sqrt(image.size[0] * image.size[1] / 10000)
    size = int(sqrt * 5)
    font = ImageFont.truetype(io.BytesIO(r.content), size=size)
    # BBoxの線の太さ設定
    rec_width = int(sqrt / 2)

    # 検出結果の描画
    for detections in pred.xyxy:
        for detection in detections:
            class_id = int(detection[5])
            class_name = str(model.model.names[class_id])
            bbox = [int(x) for x in detection[:4].tolist()]
            conf = float(detection[4])
            # 閾値以上のconfidenceの場合のみ描画
            if conf >= threshold:
                color = cmap(class_id, bytes=True)
                draw = ImageDraw.Draw(image)
                draw.rectangle(bbox, outline=color, width=rec_width)
                conf_str = Decimal(str(conf * 100)).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
                draw.text(
                    [bbox[0] + 5, bbox[1] + 10],
                    f"{class_name} {conf_str}%",
                    fill=color,
                    font=font,
                )

    return image


def callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_image()

    img = object_detection(image=img)

    return av.VideoFrame.from_image(img)


# Streamlitの画面設定
st.set_page_config(page_title="Real-time object detection", page_icon=":shark:")

# サイドバー表示
classes_str = "\n".join(f"- {item}" for item in classes)
st.sidebar.markdown(f"データセットに含まれるクラス一覧:\n{classes_str}")

# メイン画面表示
st.title("Real-time object detection")
threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)

webrtc_streamer(
    key="object_detection",
    video_frame_callback=callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)
