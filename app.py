import io
import cv2
import streamlit as st
from PIL import Image
from models import model


# Заголовок приложения
st.title("Container Damage Classifier")

# Инструкция для пользователя
st.write("Загрузите изображение, чтобы получить предсказание модели")

# Загрузка изображения
uploaded_files = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

names = model.names
if uploaded_files is not None:
    images = []
    for uploaded_file in uploaded_files:
        # Отображение загруженного изображения
        image = Image.open(uploaded_file)
        with io.BytesIO() as output:
            image.save(output, format="JPEG")  # Adjust format if needed
            contents = output.getvalue()
            with open('temp.jpg', 'wb') as f:
                f.write(contents)

        images.append(image)
        st.image(image, caption="Загруженное изображение", use_column_width=True)

        # image_bbs = cv2.imread('temp.jpg')
        # image_rgb = cv2.cvtColor(image_bbs, cv2.COLOR_BGR2RGB)

    # Кнопка для запуска классификации
    ct = 0
    if st.button("Классифицировать"):
        for image in images:
            im_name = uploaded_files[ct]
            im_name = im_name.name

            image_bbs = cv2.imread('temp.jpg')
            image_rgb = cv2.cvtColor(image_bbs, cv2.COLOR_BGR2RGB)
            # normalized_image = transform(image).unsqueeze(0)
            # normalized_image = image_rgb / 255.0
            # normalized_image = normalized_image.astype(np.float32)  # Convert to float32

            # Предсказание
            results = model.predict(source=image_rgb, conf=0.25)

            for result in results:
                boxes = result.boxes  # Получите ограничивающие рамки
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]  # Координаты bbox
                    conf = box.conf[0]  # Уверенность
                    cls = box.cls[0]  # Класс
                    class_name = names[int(cls)]

                    # Нарисуйте bbox на изображении
                    cv2.rectangle(image_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Зеленый цвет
                    cv2.putText(image_rgb, f'Class: {class_name}, Conf: {conf:.2f}', (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            st.image(image, caption="Загруженное изображение", use_column_width=True)

            # Отображение результатов
            st.write("Результат для:", im_name)
            st.image(image_rgb, use_column_width=True)

