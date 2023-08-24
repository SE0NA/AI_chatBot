# 이미지에 대한 설명 출력

import pandas as pd
import requests

url = ""     # 원하는 이미지의 url 주소 입력
# url = "https://farm9.staticflickr.com/8149/7659476094_a091f8a5c8_z.jpg"  # COCO datasets-북극곰 이미지

# image viewer
import matpotlib.pyplot as plt
from PIL import Image
image = Image.open(requests.get(url, stream=True).raw)
plt.imshow(image)

from transformers import pipeline
image_classifier = pipeline("image-classfication", model="google/vit-base-patch16-224")

preds = image_classifier(image)
preds_df = pd.DataFrame(preds)

pnit(">>", preds_df.label[0])  # 결과 리스트 중 가장 label 값이 높은 첫번째를 출력

plt.show()
