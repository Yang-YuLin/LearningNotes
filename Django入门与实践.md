### 第一步:看Django入门与实践的视频
- Python的高级Web框架
- 向目标URL发送了一个HTTP请求，服务器把页面响应给浏览器
- 浏览器 (发送HTTP请求) 网站服务器 处理请求 (返回HTTP响应) HTML文档 浏览器
- django-admin startproject myblog 创建项目
- cd myblog
- python manage.py runserver  启动服务器
- Django会以这个默认配置启动开发服务器：http://127.0.0.1:8000/ 或者 localhost:8000
- python manage.py runserver 9999  更换端口
- python manage.py runserver 0:8000 更换IP （0 是 0.0.0.0 的简写）

#### 创建应用
- 打开命令行，进入项目中manage.py同级目录
- 命令行输入：python manage.py startapp blog
- 添加应用名到settings.py中的INSTALLED_APPS里

#### 创建第一个页面（响应）
- 编辑blog.views
    - 每个响应对应一个函数，函数必须返回一个响应
    
    - 函数必须存在一个参数，一般约定为request
    
    - 每个响应函数对应一个URL
    
    ```python
      from django.http import HttpResponse
      def index(request):
      	return HttpResponse("Hello,world!")
    ```
- 编辑urls.py
    - 每个URL都以url的形式写出来
    - url函数放在urlpatterns列表中

#### 开发第一个Template
- 在APP根目录下创建名叫templates的目录
- 在该目录下创建index.HTML文件
- 在views.py中返回render()

- 在模板中使用{{参数名}}来直接使用

#### Model
- 一个Model对应数据库的一张数据表
- Django中Models以类的形式表现
- 它包含了一些基本字段以及数据的一些行为
- ORM:对象关系映射
    - 实现了对象和数据库之间的映射
    - 不需要编写SQL语句
- 编写Models步骤：
    - 在应用根目录下创建models.py，并引入models模块
    - 创建类，继承models.Model，该类即是一张数据表
    - 在类中创建字段(即属性、变量)
    - 如何把Model映射成一张数据表？
        - python manage.py makemigrations app名(可选)
        - 再执行python manage.py migrate 
        - Django会自动在app/migrations/目录下生成移植文件
        - 执行python manage.py sqlmigrate 应用名 文件id 查看SQL语句
        - 默认sqlite3的数据库在项目根目录下db.sqlite3
    - 页面呈现数据：
        ##### 后台步骤
        - views.py中import models
        - article = models.Article.objects.get(pk=1)
        - render(request, 'blog/index.html', {'article': article})
        ##### 前端步骤
        - index.html中添加{{ article.title }}  {{ article.content }}

#### View
- URL是Web服务的入口，用户通过浏览器发送过来的任何请求，都是发送到一个指定的URL地址，然后被响应

#### Admin
- Django自带的一个功能强大的自动化数据库管理界面
- 创建用户：python manage.py createsuperuser 创建超级用户/管理员账号
- localhost:8000/admin/ Admin入口
- 修改setting.py中LANGUAGE_CODE = 'zh-Hans'
- 修改数据默认显示名称：
    - 在Article类下添加一个方法
    - _str_(self)
    - return self.title

#### Templates过滤器
- 写在模板中，是Django模板语言
- 可以修改模板中的变量，从而显示不同的内容

#### 查看Django版本 
```python
import django
print(django.get_version())
 Django (2.2.2)
 django-extensions (2.1.9)
 djangorestframework (3.9.4)
```

#### 总结:
1、搭建完整的Django开发环境。
2、创建项目及应用。
3、了解项目目录下各文件的含义和作用。
4、了解并学会开发Templates。
5、了解并学会开发Models。
6、掌握Admin的基本配置方法。
7、学会项目URL的配置方法。
8、开发一个由三个页面组成的简易博客网站。
        
        