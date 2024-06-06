# views.py
import json
import os
import shutil
import tempfile
import time
import zipfile
import cv2
import numpy as np
from mtcnn import MTCNN
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .FaceDetection import FaceDetection
from .FaceRecognizer import FaceRecognizer
# from .ImageResponse import ImageResponse

face_recognizer = FaceRecognizer()
face_detection = FaceDetection()
# image_response = ImageResponse()


# @api_view(['GET'])
# def training_model(request):
#     try:
#         face_recognizer.training_model()
#         return Response({'message': 'Model training successful'})
#     except Exception as e:
#         return Response({'error': str(e)}, status=500)
#

# @api_view(['POST'])
# def predict_model(request):
#     try:
#         image_data = request.FILES['image'].read()
#         image_name = request.POST['file_name']
#         image_array = np.frombuffer(image_data, dtype='uint8')  # Chuyển đổi dữ liệu thành mảng NumPy
#         input_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
#         detector = MTCNN()
#
#         faces = detector.detect_faces(input_image)
#         for i, face_info in enumerate(faces):
#             x, y, w, h = face_info['box']
#             x, y, w, h = max(x, 0), max(y, 0), max(w, 0), max(h, 0)
#             face_roi = cv2.resize(input_image[y:y + h, x:x + w], (224, 224))
#             face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
#             predictions = face_recognizer.predict_model(face_roi_gray, image_name)
#             # return Response({'predictions': "demo"})
#         return Response({'predictions': "successes"})
#
#
#     except Exception as e:
#         return Response({'error': str(e)}, status=500)


@api_view(['POST'])
def predict_model(request):
    try:
        image_data = request.FILES['image'].read()
        image_name = request.POST['file_name']
        image_array = np.frombuffer(image_data, dtype='uint8')  # Chuyển đổi dữ liệu thành mảng NumPy
        input_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        prediction = face_recognizer.predict_model(input_image, image_name)
        return Response({'predictions': prediction})

    except Exception as e:
        return Response({'error': str(e)}, status=500)


@api_view(['GET'])
def demo01(request):
    return Response({'message': 'Phan hoi tu doan code demo 01 '})


@api_view(['POST'])
def face_detections(request):
    input_folder = request.FILES['input_folder']
    user_code = request.POST['user_code']
    extract_folder = f"static/images/{user_code}"
    # Mở file ZIP
    with zipfile.ZipFile(input_folder, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
        try:
            face_detection.face_detections(extract_folder, user_code)
        finally:
            shutil.rmtree(extract_folder)
        return Response({'message': 'Thanh cong'})


# @api_view(['POST'])
# def training_classroom(request):
#     data = json.loads(request.body)
#     student_ids = data['ids']
#     file_name = data['file_name']
#     faces = face_recognizer.training_model_classroom(student_ids, file_name)
#     print(student_ids)
#     return Response({"message": "Ids received successfully"})


@api_view(['POST'])
def get_images_by_user_code(request):
    user_code = json.loads(request.body)["user_code"]
    images = face_recognizer.get_images_by_user_code(user_code)
    return Response(images)


# @api_view(['POST'])
# def training_user(request):
#     data = json.loads(request.body)
#     user_code = data['user_code']
#     face_recognizer.training_model_user(user_code)
#     return Response({"message": "Successfully trained"})


@api_view(['POST'])
def delete_image(request):
    data = json.loads(request.body)
    file_name = data['file_name']
    res = face_recognizer.delete_images(file_name)
    return Response({"success": res})


# @api_view(['POST'])
# def training_list_user(request):
#     data = json.loads(request.body)
#     student_ids = data['user_code']
#     # file_name = data['file_name']
#     face_recognizer.training_models_by_users(student_ids)
#
#     return Response({"message": "Ids received successfully"})


@api_view(['POST'])
def replace_images(request):
    image_data = request.FILES['image'].read()
    file_name = request.POST['file_name']
    image_array = np.frombuffer(image_data, dtype='uint8')  # Chuyển đổi dữ liệu thành mảng NumPy
    input_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    message = face_recognizer.replace_file(file_name, gray_image)
    return Response({"message": message})


# @api_view(['POST'])
# def get_images_from_video(request):
#     try:
#         video_data = request.FILES['video'].read()
#         user_code = request.POST['user_code']
#
#         with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
#             temp_file.write(video_data)
#             temp_file.flush()
#             cam = cv2.VideoCapture(temp_file.name)
#             if not cam.isOpened():
#                 return Response({"message": "Error: Cannot open video file."}, status=400)
#
#             image_response.extract_frames(cam, user_code)
#             cam.release()
#
#         return Response({"message": "true"})
#     except Exception as e:
#         return Response({"message": f"Error: {str(e)}"}, status=400)


