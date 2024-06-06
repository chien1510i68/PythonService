import base64
import os

import cv2
import joblib
import numpy as np
import torch
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from mtcnn import MTCNN
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from student_manager.FaceDetection import FaceDetection

output_folder = "output"
trained_model = "train"
recognizer = cv2.face_LBPHFaceRecognizer.create()
path_cascade = "../../static/file/haarcascade_frontalface_default.xml"


class FaceRecognizer:
    # @staticmethod
    # def training_model():
    #     faces = []
    #     labels = []
    #     if not os.path.exists(trained_model):
    #         os.makedirs(trained_model)
    #     for image in os.listdir(output_folder):
    #         face = cv2.imread(os.path.join(output_folder, image), cv2.IMREAD_GRAYSCALE)
    #         faces.append(face)
    #         label = image.split(".")[0].split("_")[0]
    #         image_dm = image
    #         labels.append(label)
    #         print(label)
    #     labels = np.array(labels, dtype=np.int32)
    #
    #     recognizer.train(faces, labels)
    #
    #     recognizer.write("train/train.yml")
    #     pass

    # @staticmethod
    # def training_model_classroom(student_ids, file_name):
    #     faces = []
    #     labels = []
    #     filtered_files = filter(lambda file: any(file.startswith(name) for name in student_ids),
    #                             os.listdir('output'))
    #     for image in filtered_files:
    #         face = cv2.imread(os.path.join(output_folder, image), cv2.IMREAD_GRAYSCALE)
    #         faces.append(face)
    #         label = image.split(".")[0].split("_")[0]
    #         # image_dm = image
    #         labels.append(label)
    #         print(label)
    #     labels = np.array(labels, dtype=np.int32)
    #
    #     recognizer.train(faces, labels)
    #
    #     recognizer.write(os.path.join("train", f"{file_name}.yml"))
    #     pass

    # @staticmethod
    # def training_models_by_users(user_code):
    #     return None

    # @staticmethod
    # def training_model_user(user_code):
    # faces = []
    # labels = []
    #
    # filtered_files = filter(lambda file: file.startswith(user_code), os.listdir("output"))
    #
    # for image in filtered_files:
    #     face = cv2.imread(os.path.join(output_folder, image), cv2.IMREAD_GRAYSCALE)
    #     faces.append(face)
    #     label = image.split(".")[0].split("_")[0]
    #     labels.append(label)
    # labels = np.array(labels, dtype=np.int32)
    # recognizer.train(faces, labels)
    # recognizer.write(os.path.join("train", f"{user_code}.yml"))
    @staticmethod
    def training_model_user(user_code):

        X_train = []  # Danh sách chứa các đặc trưng từ ảnh
        y_train = []
        filtered_files = filter(lambda file: file.startswith(user_code), os.listdir("output"))

        for image_file in filtered_files:
            # Đọc ảnh từ tệp tin
            image_path = os.path.join("output", image_file)
            image = cv2.imread(image_path)

            # Chuyển đổi ảnh thành ảnh xám
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Trích xuất đặc trưng từ ảnh xám
            features = gray_image.flatten()

            # Thêm đặc trưng và nhãn vào danh sách
            X_train.append(features)
            label = image_file.split(".")[0].split("_")[0]
            y_train.append(label)

        # Chuyển đổi danh sách sang mảng numpy
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Tạo và huấn luyện mô hình KNN
        knn_model = KNeighborsClassifier(n_neighbors=3)
        knn_model.fit(X_train, y_train)

        # Lưu mô hình vào tệp tin
        file_name = f"train/{user_code}.pkl"
        joblib.dump(knn_model, file_name)

    # @staticmethod
    # def predict_model(image, file_name):
    #     recognition = cv2.face.LBPHFaceRecognizer_create()
    #     recognition.read(f'train/{file_name}.yml')
    #     # Load the Haar Cascade classifier for face detection
    #     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    #     faces = face_cascade.detectMultiScale(image,scaleFactor=1.3, minNeighbors=10, minSize=(30, 30))
    #
    #     for (x, y, w, h) in faces:
    #         face_roi = cv2.resize(image[y:y + h, x:x + w], (224, 224))
    #         cv2.imwrite("static/face_roi.jpg", face_roi)
    #         label, confidence = recognition.predict(face_roi)
    #         return confidence

    @staticmethod
    def preprocess_image(image):
        # Preprocess the image
        image = cv2.resize(image, (160, 160))
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))
        return image

    @staticmethod
    def get_embedding(image):
        model = InceptionResnetV1(pretrained='vggface2').eval()
        detector = MTCNN()
        faces = detector.detect_faces(image)
        if faces:
            for face in faces:
                x, y, w, h = face['box']
                face_image = image[y:y + h, x:x + w]
                preprocessed_image = FaceDetection.preprocess_image(face_image)
                preprocessed_image_tensor = torch.tensor(preprocessed_image, dtype=torch.float32)
                preprocessed_image_tensor = preprocessed_image_tensor.unsqueeze(0)
                embedding = model(preprocessed_image_tensor).detach().numpy().flatten()
                return embedding
        return None

    # @staticmethod
    # def verify(known_embeddings, known_labels, test_img, threshold=0.5):
    #     test_embedding = FaceRecognizer.get_embedding(test_img)
    #     if not test_embedding:
    #         return None
    #     distances = [tf.reduce_sum(tf.square(a - b)) for a, b in zip(known_embeddings, [test_embedding])]
    #     min_index = np.argmin(distances)
    #     normalized_distance = distances[min_index] / threshold
    #     confidence = 1 / (1 + tf.exp(normalized_distance))  # Hàm sigmoid
    #     if confidence > 0.5:
    #         return known_labels[min_index], confidence
    #     else:
    #         return None

    @staticmethod
    def predict_model(image, file_name):
        # model_path = "train/undefined.pkl"
        model_path = f"train/{file_name}.pkl"
        if not os.path.exists(model_path):
            return {"success": False, "message": 500}

        try:
            # Tải mô hình đã lưu
            predict_model = joblib.load(model_path)

        except Exception as e:
            return {"success": False, "message": 501}
        # Tải mô hình đã lưu
        # predict_model = joblib.load(model_path)
        embedding = FaceRecognizer.get_embedding(image)
        if embedding is None:
            return {"success": False, "message": 502}
        try:
            embedding_2d = embedding.reshape(1, -1)
            probabilities = predict_model.predict_proba(embedding_2d)[0]
            value_max = max(probabilities)
            return value_max * 100
        except Exception as e:
            return {"success": False, "message": "Có lỗi khi nhận dạng hình ảnh"}

    @staticmethod
    def get_images_by_user_code(user_code):
        # images = []
        # image_response = []
        # for filename in os.listdir("output"):
        #     if filename.startswith(user_code):
        #         images.append(filename)
        #
        # for image in images:
        #     img = cv2.imread("output/" + image)
        #     cv2.imwrite(os.path.join("static/face_images/", image), img)  # Lưu ảnh vào thư mục static/images
        #
        #     with open(os.path.join("output/", image), 'rb') as file:
        #         encoded_image = base64.b64encode(file.read()).decode("utf-8")  # Mã hóa ảnh dưới dạng base64
        #
        #     image_response.append({
        #         "file_name": f"output/{image}",
        #         "encoded_image": encoded_image
        #     })
        #
        # return image_response
        image_response = []
        user_folder = f"output/{user_code}"
        if not os.path.exists(user_folder):
            return {"success": False, "message": "User code does not exist"}

        for image in os.listdir(user_folder):
            image_path = os.path.join(user_folder, image)
            if os.path.isfile(image_path):  # Kiểm tra xem có phải là tệp tin không
                with open(image_path, 'rb') as file:
                    encoded_image = base64.b64encode(file.read()).decode("utf-8")  # Mã hóa ảnh dưới dạng base64

                image_response.append({
                    "file_name": f"{user_code}/{image}",
                    "encoded_image": encoded_image
                })

        return {"success": True, "data": image_response}

    @staticmethod
    def delete_images(filename):
        images = []
        results = []
        for file_path in os.listdir("output"):
            if file_path.startswith(filename):
                images.append(file_path)
        for image in images:
            file_path = os.path.join("output/", image)
            if os.path.exists(file_path):
                os.remove(file_path)
                results.append(file_path.split("/")[1])

        return results
        pass

    @staticmethod
    def replace_file(filename, image):
        FaceRecognizer.delete_images(filename)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            face_roi = cv2.resize(image[y:y + h, x:x + w], (224, 224))
            cv2.imwrite(os.path.join("output", filename), face_roi)
            return "Successfully replaced"
