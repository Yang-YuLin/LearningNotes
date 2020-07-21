### 第三步:看Django REST framework教程https://www.django-rest-framework.org/tutorial/quickstart/
- Serializers:
```python
from django.contrib.auth.models import User, Group
from rest_framework import serializers


class UserSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = User
        fields = ('url', 'username', 'email', 'groups')


class GroupSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Group
        fields = ('url', 'name')
```
- Views:
```python
from django.contrib.auth.models import User, Group
from rest_framework import viewsets
from quickstart.serializers import UserSerializer, GroupSerializer


class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all().order_by('-date_joined')
    serializer_class = UserSerializer


class GroupViewSet(viewsets.ModelViewSet):
    queryset = Group.objects.all()
    serializer_class = GroupSerializer
```
- URLs:
```python
from django.contrib import admin
from django.urls import include, path
from rest_framework import routers
from quickstart import views


router = routers.DefaultRouter()
router.register(r'users', views.UserViewSet)
router.register(r'groups', views.GroupViewSet)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include(router.urls)),
    path('api-auth/', include('rest_framework.urls', namespace='rest_framework'))
]
```
- Pagination(分页)：allow you to control how many objects per page are returned.

  settings.py

  ```python
  REST_FRAMEWORK = {
      'DEFAULT_PAGINATION_CLASS':'rest_framework.pagination.PageNumberPagination',
      'PAGE_SIZE': 10
  }
  ```

- Settings：Add `'rest_framework'` to `INSTALLED_APPS`. 

  The settings module will be in `tutorial/settings.py`

  ```c++
  INSTALLED_APPS = [
      ...
      'rest_framework',
  ]
  ```

- Testing our API

  启动服务器：python manage.py runserver

  `http://127.0.0.1:8000/users/`...

  ![](http://ww1.sinaimg.cn/large/8eb0608fly1g4i2kh4egij20wo0br0tc.jpg)
  ![](http://ww1.sinaimg.cn/large/8eb0608fly1g4i2l61e3rj20w20dfgm5.jpg)