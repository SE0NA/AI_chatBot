# 입력한 사진이 강아지인지, 고양이인지 판단하는 기능

from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import requests

class DialoGPT:
  def __init__(self, model_name: str='openai/clip-vit-large-patch14', ):
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-pathch14")

  def __call__(self, image:Image)-> bool:
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-pathch14")
    inputs = processor(text=["cat", "dog"], images=image, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logit_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    print(probs)
    iscat = probs[0]0]>probs[0][1]  # 고양이인지 아닌지 bool 형태로 입력. probs에서 값만 추출하여 비교

    return iscat

  def run(self):
    while True:
      url = input("image url: ")
      if(url == "end"):
        break

      image = Image.open(request.get(url, stream=True).raw)
      if(self(image)):
        print(">> 해당 사진은 *고양이*에 가깝습니다.")
      else:
        print(">> 해당 사진은 *강아지*에 가깝습니다.")


if __name__ == "__main__":
  bot = DialoGPT()
  bot.run()
