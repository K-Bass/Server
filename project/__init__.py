from io import BytesIO
import numpy as np
import tensorflow as tf
from PIL import ImageOps, Image
from fastapi import FastAPI, UploadFile, Request


def create_app() -> FastAPI:
    print("[+] Initializing app...")
    app = FastAPI()

    model = tf.keras.models.load_model('service/keras_model.h5')
    print("[+] Loading model...")

    @app.post("/predict_image")
    async def predict_image(file: UploadFile, request: Request):

        request_object_content = await file.read()
        image = Image.open(BytesIO(request_object_content)).convert("RGB")

        # resizing the image to be at least 224x224 and then cropping from the center
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

        # turn the image into a numpy array
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # contents = await file.read()

        # print(request.)

        # image = tf.io.decode_image(contents, channels=3)
        # image = tf.image.resize(image, [224, 224])
        # image = tf.cast(image, tf.float32) / 255.0
        # image = tf.expand_dims(image, axis=0)
        #
        # prediction = model(image)
        prediction = model.predict(data)

        class_id = np.argmax(prediction[0])

        is_not_good = False
        description = 'Description'

        if class_id == 0:
            is_not_good = True
            description = '큰 입 배스는 한국에도 식용으로 수입되었으나 현재는 야생화되고 개체수가 지나치게 늘어나서 생태계교란 생물로 지정되었다.'
        elif class_id == 1:
            is_not_good = True
            description = '블루길(영어: bluegill, Lepomis macrochirus)은 북아메리카가 원산지인 외래종으로, 물살이 빠르지 않은 하천에 사는 민물고기이다.[1] 대한민국에서는 황소개구리, 배스, 미국가재, 뉴트리아와 함께 생태계교란종으로 지정하여 특별 관리되고 있다. 사람에게 물리적인 해를 입히지는 않으나, 날카로운 등지느러미 가시에 찔리면 상당히 아프다.'
        else:
            is_not_good = False
            description = """잉어목에 속하는 민물고기. 추어(鰍魚)라고도 부르며 식용으로 쓴다.

방언으로는 미꾸리라고 하지만, 미꾸리는 다른 어류 종의 이름이기도 하다.
미꾸라지 두부숙회
재료 및 분량 (5인분)
미꾸라지: 600 g
두부: 3 모
양념장: 적당량
만드는 법
살아 있는 미꾸라지를 물을 바꾸어 주면서 2∼3 일 진흙을 토해 내도록 한다.
두부는 큰 것을 통째로 솥에 넣고, 미꾸라지는 물에서 건져서 함께 넣는다.
뚜껑을 닫고 불에 올려 가열하면, 미꾸라지는 뜨거워서 두부 속으로 기어든다.
더 뜨거워지면 두부 속의 미꾸라지는 약이 바싹 오르면서 죽어 간다.
이것을 썰어서 산초가루를 넣은 양념장에 찍어 먹는다. 또는 참기름에 지져서 먹기도 하고 탕을 끓여 먹기도 한다."""

        return {
            "prediction": str(class_id),
            "is_not_good": is_not_good,
            "description": description
        }

    @app.get("/")
    async def root():
        return {"message": "Hello World"}

    return app
