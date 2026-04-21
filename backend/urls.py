from .views import index, inference_api
from django.urls import path, re_path
from django.conf import settings
from django.views.static import serve

FRONTEND_DIST = settings.BASE_DIR / 'frontend' / 'dist'

urlpatterns = [
    path('api/inference', inference_api),
    re_path(r'^assets/(?P<path>.*)$', serve, {'document_root': FRONTEND_DIST / 'assets'}),
    re_path(r'^static/(?P<path>.*)$', serve, {'document_root': settings.BASE_DIR / 'static'}),
    re_path(r'^(?P<path>favicon\.ico|robots\.txt|.*\.(?:png|jpg|jpeg|svg|webp|ico))$', serve, {'document_root': FRONTEND_DIST}),
    re_path(r'^.*$', index),
]
