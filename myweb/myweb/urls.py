"""myweb URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
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
from django.conf.urls import url
from myapp import views

urlpatterns = [
    url(r'^predict_gender/$', views.predict_gender, name='predict_gender'),
    url(r'^predict_character/$', views.predict_character, name='predict_character'),
    url(r'^predict_rating/$', views.predict_rating, name='predict_rating'),
    #url(r'^show/$', views.list),
    url(r'^admin/', admin.site.urls),
    url(r'^test/$', views.test, name='test'),


]
