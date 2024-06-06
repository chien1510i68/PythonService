import os

import cv2
import joblib
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face
from mtcnn import MTCNN
from sklearn.svm import SVC
import torch


class FaceDetection:
    @staticmethod
    def face_detection(image):

        path_cascade = "../static/file/haarcascade_frontalface_default.xml"
        if not os.path.exists('../static/file/haarcascade_frontalface_default.xml'):
            return "File haarcascade_frontalface_default not found"
        else:
            face_cascade = cv2.CascadeClassifier(path_cascade)

            img = cv2.imread(image)
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            for i, (x, y, w, h) in enumerate(faces):
                face_roi = cv2.resize(gray_image[y:y + h, x:x + w], (224, 224))
                return face_roi

    @staticmethod
    def preprocess_image(image):
        # Tiền xử lý ảnh
        image = cv2.resize(image, (160, 160))  # Chỉnh kích thước ảnh về 160x160
        image = image / 255.0  # Chuẩn hóa giá trị pixel về khoảng [0,1]
        # Chuyển đổi định dạng ảnh thành đúng định dạng tensor PyTorch: [channels, height, width]
        image = np.transpose(image, (2, 0, 1))
        return image

    @staticmethod
    def extract_image(input_folder, label):
        detector = MTCNN()

        # Load pre-trained FaceNet model
        model = InceptionResnetV1(pretrained='vggface2').eval()

        # Khởi tạo danh sách embeddings và nhãn
        X_train = []
        y_train = []
        face_index = 1
        output_folder = os.path.join("output", label)
        os.makedirs(output_folder, exist_ok=True)

        # Lặp qua các hình ảnh trong thư mục
        for image_name in os.listdir(input_folder):
            image_path = os.path.join(input_folder, image_name)
            image = cv2.imread(image_path)
            if image is not None:
                # Phát hiện khuôn mặt trong ảnh
                faces = detector.detect_faces(image)
                if faces:
                    # Lặp qua mỗi khuôn mặt được phát hiện trong ảnh
                    for face in faces:

                        # Trích xuất khuôn mặt từ ảnh
                        x, y, w, h = face['box']
                        face_image = image[y:y + h, x:x + w]
                        face_filename = f"{face_index}.jpg"
                        face_path = os.path.join(output_folder, face_filename)
                        cv2.imwrite(face_path, face_image)
                        face_index += 1

                        # Tiền xử lý và chuẩn bị ảnh để đưa vào mô hình
                        preprocessed_image = FaceDetection.preprocess_image(face_image)
                        # Chuyển đổi thành tensor PyTorch
                        preprocessed_image_tensor = torch.tensor(preprocessed_image, dtype=torch.float32)
                        preprocessed_image_tensor = preprocessed_image_tensor.unsqueeze(0)

                        # Trích xuất embedding của khuôn mặt sử dụng FaceNet
                        embeddings = model(preprocessed_image_tensor)

                        X_train.append(embeddings.squeeze().detach().numpy())
                        # Nhãn của mỗi khuôn mặt
                        y_train.append(label)

        return X_train, y_train

    @staticmethod
    def face_detections(input_folder, user_code):
        path_fake_data = "static/data_fake"

        X_data, y_data = FaceDetection.extract_image(input_folder, user_code)
        X_fake_data, y_fake_data = FaceDetection.extract_image(path_fake_data,"111111")
        X_train = np.array(X_data + X_fake_data)  # Kết hợp dữ liệu từ hình ảnh thật và giả
        y_train = np.array(y_data + y_fake_data)

        # Khởi tạo và huấn luyện mô hình nhận dạng (ví dụ: SVM)
        clf = SVC(kernel='linear', probability=True)
        clf.fit(X_train, y_train)
        joblib.dump(clf, os.path.join("train/", f'{user_code}.pkl'))


    # @staticmethod
    # def face_detections(input_folder):
    #     output_folder = "output"
    #     path_cascade = "../static/file/haarcascade_frontalface_default.xml"
    #
    #     try:
    #         # Tạo thư mục output nếu chưa tồn tại
    #         if not os.path.exists(output_folder):
    #             os.makedirs(output_folder)
    #
    #         # Kiểm tra tồn tại của file phân loại khuôn mặt
    #         if not os.path.exists(path_cascade):
    #             raise ValueError("File not found ".format(path_cascade))
    #
    #         face_cascade = cv2.CascadeClassifier(path_cascade)
    #
    #         for filename in os.listdir(input_folder):
    #             if filename.endswith(('.jpg', '.jpeg', '.png')):  # Lọc các loại tệp ảnh
    #                 image_path = os.path.join(input_folder, filename)
    #
    #                 try:
    #                     # Đọc ảnh
    #                     img = cv2.imread(image_path)
    #                     gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #                     faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
    #                                                           minSize=(40, 40))
    #
    #                     for i, (x, y, w, h) in enumerate(faces):
    #                         face_roi = cv2.resize(gray_image[y:y + h, x:x + w], (224, 224))
    #                         output_path = os.path.join("output", f"{filename}")
    #                         cv2.imwrite(output_path, face_roi)
    #
    #                 except cv2.error as e:  # Bắt lỗi liên quan đến OpenCV
    #                     print(f"Error in filename {filename}: {e}")
    #
    #     except Exception as e:  # Bắt các lỗi khác
    #         print(f"Error: {e}")
