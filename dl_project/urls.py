# dl_project/dl_project/urls.py
from django.contrib import admin
from django.urls import path, include # Make sure 'include' is imported

urlpatterns = [
    path('admin/', admin.site.urls),
    path('mood/', include('mood_tester.urls')), # Add this line to include your app's URLs
    # You can add other paths for your project here later
    # For example, if you wanted something at the root URL, you might add:
    # from mood_tester import views as mood_tester_views # if you want to point root to your app
    # path('', mood_tester_views.test_mood_view, name='home_page'), # Example: makes test page the home page
]
