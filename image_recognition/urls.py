from django.urls import path  
from . import views
urlpatterns=[ 
    path('',views.homepage),
    path('img_recog',views.img_recognition)    
]