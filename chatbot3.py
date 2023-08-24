# 입력한 카테고리 중 이미지와 가장 가까운 카테고리 선택

from PIL import Image
import requests

from transformers import CLIPProcessro, ClIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

url = input("이미지 URL 입력: ")
image = Image.open(requests.get(url, stream=True).raw)

num_text = int(input("텍스트 입력 수: "))
li_text = []
for i in range(num_text):
  li_text.append(input(f"category({i}) > "))

inputs = processor(li_text, images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image      # this is the image-text siilarity score
probs = logits_per_image.softmax(dim=1)          # we can take the softmax to get the label probabilities

high = 0
for i in range(len(probs[0])):
  if(probs[0][high]<probs[0][i]):
    high = i


print(f">> {li_text[high]}")
