import io
import base64
import os


def image_to_base64(image, format="PNG"):
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer_bytes = buffer.getvalue()

    base64_str = base64.b64encode(buffer_bytes).decode("utf-8")

    return base64_str


def read_image_files(folder_path):
    image_extensions = [".png", ".jpg", ".jpeg", ".webp"]
    image_files = []

    for file_name in os.listdir(folder_path):
        file_ext_lower = os.path.splitext(file_name)[1].lower()
        if file_ext_lower in image_extensions:
            file_path = os.path.join(folder_path, file_name)
            image_files.append(file_path)

    images_base64 = []
    for file_path in image_files:
        with open(file_path, "rb") as image_file:
            image_data = image_file.read()
            encoded_data = base64.b64encode(image_data)
            base64_string = encoded_data.decode("utf-8")
            images_base64.append(base64_string)

    return images_base64
