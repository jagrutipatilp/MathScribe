from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('scan/', views.scan, name='scan'),
    path('gesture/', views.gesture, name='gesture'),
    path('about-us/', views.about_us, name='about_us'),
    path('contact-us/', views.contact_us, name='contact_us'),
    path('get_frame/', views.get_frame, name='get_frame'),
    
    path('upload/', views.upload_image, name='upload_image'),
    path('scanSolution/<str:solution_id>/', views.show_solution, name='show_solution'),


    path('solve/', views.solve, name='solve'),
    

]
