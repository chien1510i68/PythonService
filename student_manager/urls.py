"""
URL configuration for student_manager project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from .views import *

urlpatterns = [
    # path('demo01', test.demo01),
    # path('demo01/', demo01),
    # path('training/', training_model),
    path('predict/', predict_model),
    path('detections/', face_detections),
    # path('training_claasroom/', training_classroom),
    # path('training_user/', training_user),
    path('get_images/', get_images_by_user_code),
    path('del_image/', delete_image),
    # path('training_users/', training_list_user),
    path('replace_image/', replace_images),
    # path('test/', get_images_from_video),

]
