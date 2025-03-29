from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.views.static import serve
from django.conf import settings
from django.urls import path, include
from django.contrib import admin
from django.urls import path
from resumes.views import process_uploaded_resume, get_top_candidates
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('resumes.urls')),  # Ensure 'your_app' matches your app name
    path('media/<path:path>', serve, {'document_root': settings.MEDIA_ROOT}),
    path('process_uploaded_resume/', process_uploaded_resume, name='process_uploaded_resume'),
    path('get-top-candidates/', get_top_candidates, name='get_top_candidates'),
]


# Serve uploaded files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
