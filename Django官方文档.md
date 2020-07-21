### 第二步:看Django官方文档https://docs.djangoproject.com/zh-hans/2.2/intro/overview/

#### 设计模型
- Django无需数据库便可使用,它提供了ORM,可以使用Python代码来描述数据库结构
- 例子: mysite/news/models.py
```python
from django.db import models

class Reporter(models.Model):
    full_name = models.CharField(max_length=70)

    def __str__(self):
        return self.full_name

class Article(models.Model):
    pub_date = models.DateField()
    headline = models.CharField(max_length=200)
    content = models.TextField()
    reporter = models.ForeignKey(Reporter, on_delete=models.CASCADE)

    def __str__(self):
        return self.headline
```

 #### 应用数据模型
 - 运行Django命令行工具来自动创建数据库表 

   ```python
   #查找所有可用的models，为任意一个在数据库中不存在对应数据表的model创建migrations脚本文件
   python manage.py makemigrations 
   #运行这些migrations自动创建数据库表
   python manage.py migrate
   ```

 #### 享受便捷的API
 - 我们可以使用一套便捷而丰富的Python API访问数据
 这些 API 是即时创建的，而不用显式的生成代码

 ```python
 # Import the models we created from our "news" app
 from news.models import Article,Reporter
 # No reporters are in the system yet
 Reporter.objects.all()
 # Create a new Reporter
 r = Reporter(full_name='John Smith')
 # Save the object into the database.
 r.save()
 # Now it has an ID
 r.id		1
 # Now the new reporter is in the database
 Reporter.objects.all()		<QuerySet [<Reporter: John Smith>]>
 r.full_name   'John Smith'
 # Django provides a rich database lookup API.
 Reporter.objects.get(id=1)		<Reporter: John Smith>
 Reporter.objects.get(full_name__startswith='John')		<Reporter: John Smith>
 Reporter.objects.get(full_name__contains='mith')		<Reporter: John Smith>
 ```

 -  learn = models.ManyToManyField(TranslateUser,null=True)
 这一句会自动生成一个关系表
 对于word, 想查关系表就用learn字段
对于User,会自动生成一个字段,名字叫word_set,就是那个关系表

  ```python
 class User(models.Model):
    username = models.TextField(primary_key=True)
    password = models.TextField()

    def __str__(self):
        return self.username


class Word(models.Model):
    query = models.TextField(primary_key=True)
    is_CET4 = models.BooleanField(default=False)
    is_CET6 = models.BooleanField(default=False)
    is_UNGEE = models.BooleanField(default=False)
    learn = models.ManyToManyField(TranslateUser,null=True)

    def __str__(self):
        return self.query
  ```

#### 一个动态管理接口：并非徒有其表
- 定义完模型之后,Django就会自动生成一个**管理接口**
一个允许认证用户添加、修改和删除对象的Web站点
我们只需简单的在admin站点上注册模型即可
- 例子: mysite/news/models.py
```python
from django.db import models

class Article(models.Model):
    pub_date = models.DateField()
    headline = models.CharField(max_length=200)
    content = models.TextField()
    reporter = models.ForeignKey(Reporter, on_delete=models.CASCADE)
```
mysite/news/admin.py  
```python
from django.contrib import admin

from . import models

admin.site.register(models.Article) # 注册模型
```
- 创建Django应用的典型流程:
1. 先建立数据模型
2. 然后搭建管理站点
3. 之后向网站里填充数据

#### 规划URLs
- 为了设计自己的URLconf，需要创建一个叫做URLconf的Python模块
- 这是网站的目录，它包含了一张URL和Python回调函数(视图)之间的映射表
- 例子: mysite/news/urls.py
```python
from django.urls import path

from . import views

# 当用户请求了这样的 URL "/articles/2005/05/39323/",Django会调用 news.views.article_detail(request, year=2005, month=5, pk=39323)
urlpatterns = [
    path('articles/<int:year>/', views.year_archive),
    path('articles/<int:year>/<int:month>/', views.month_archive),
    path('articles/<int:year>/<int:month>/<int:pk>/', views.article_detail),
]
```

#### 编写视图

- 视图函数的执行结果只可能有两种：
  - 返回一个包含请求页面元素的HttpResponse对象
  - 或者是抛出Http404这类异常
- 视图的工作：从参数获取数据，装载一个模板，然后将根据获取的数据对模板进行渲染

- 例子: mysite/news/views.py
```python
from django.shortcuts import render

from .models import Article


def year_archive(request, year):
    a_list = Article.objects.filter(pub_date__year=year)
    context = {'year': year, 'article_list': a_list}
    return render(request, 'news/year_archive.html', context)
```

#### 设计模板

- 上面的代码加载了 news/year_archive.html 模板
- 模板继承: {% extends "base.html" %}

#### 编写第一个Django应用: 基本的投票应用程序
- django-admin startproject mysite
- python manage.py runserver
- python manage.py startapp polls
- 编写第一个视图:
    - polls/views.py
    ```python
    from django.http import HttpResponse
    
    
    def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")
    ```

    - 在polls目录里新建一个urls.py文件
    
      ```python
      from django.urls import path
      from . import views
      
      urlpatterns = [
          path('', views.index, name='index')
  ]
      ```
    
    - mysite/urls.py文件的urlpatterns列表里插入一个 include()


```python
from django.contrib import admin
from django.urls import path, includes

urlpatterns = [
    path('admin/', admin.site.urls),
    path('polls/', include('polls.urls')),
]
    # 函数path()具有四个参数，两个必须参数：route和view，两个可选参数：kwargs和name。
    # route是一个匹配URL的准则.当Django响应一个请求时，它会从urlpatterns的第一项开始，按顺序依次匹配列表中的项，直到找到匹配的项。这些准则不会匹配 GET 和 POST 参数或域名。
    # path:当Django找到了一个匹配的准则，就会调用这个特定的视图函数，并传入一个HttpRequest对象作为第一个参数，被“捕获”的参数以关键字参数的形式传入
```
- 数据库配置: mysite/settings.py使用SQLite作为默认数据库
- 使用其他数据库:
    - ENGINE
    - NAME 数据库的名称,NAME应该是此文件的绝对路径，包括文件名
    - 默认值 os.path.join(BASE_DIR, 'db.sqlite3') 将会把数据库文件储存在项目的根目录
- python manage.py migrate为INSTALLED_APPS里声明了的应用进行数据库迁移
- 创建模型:
    - polls/models.py
    ```python
    from django.db import models

    class Question(models.Model):
        question_text = models.CharField(max_length=200)
        pub_date = models.DateTimeField('date published')
        
    class Choice(models.Model):
        question = models.ForeignKey(Question, on_delete=models.CASCADE) # 使用ForeignKey定义了一个关系
        choice_text = models.CharField(max_length=200)
        votes = models.IntegerField(default=0)
    ```

- 在配置类 INSTALLED_APPS 中添加设置'polls.apps.PollsConfig',
- python manage.py makemigrations polls
- python manage.py migrate
- 打开Python命令行:python manage.py shell
```python

In [3]: from polls.models import Question,Choice

In [4]: Question.objects.all()
Out[4]: <QuerySet [<Question: What is up?>]>

In [5]: from django.utils import timezone

In [6]: q = Question(question_text="What's new?",pub_date = timezone.now())

In [7]: q.save()

In [8]: q.id
Out[8]: 2

In [9]: q.question_text
Out[9]: "What's new?"

In [10]: q.pub_date
Out[10]: datetime.datetime(2019, 6, 29, 0, 48, 23, 877952, tzinfo=<UTC>)

In [11]: Question.objects.all()
Out[11]: <QuerySet [<Question: What is up?>, <Question: What's new?>]>

In [12]: Question.objects.filter(id=1)
Out[12]: <QuerySet [<Question: What is up?>]>

In [15]: Question.objects.filter(question_text__startswith='What')
Out[15]: <QuerySet [<Question: What is up?>, <Question: What's new?>]>

In [17]:current_year = timezone.now().year
>>> Question.objects.get(pub_date__year=current_year)
<Question: What's up?> **只能返回一个值,如果满足条件的有多个,则报错**

In [22]: Question.objects.get(pk=1) **pk=primary key**
Out[22]: <Question: What is up?>

In [23]: Question.objects.get(pk=2)
Out[23]: <Question: What's new?>

In [25]: q =  Question.objects.get(pk=2)
    ...: q.choice_set.all()
    ...: 
Out[25]: <QuerySet []>

In [27]: q.choice_set.create(choice_text='Not much',votes=0)
Out[27]: <Choice: Not much>

In [28]: q.choice_set.create(choice_text='The sky',votes=0)
Out[28]: <Choice: The sky>

In [29]: c = q.choice_set.create(choice_text='Just hacking again',votes=0)

In [30]: c.question
Out[30]: <Question: What's new?>

In [31]: q.choice_set.all()
Out[31]: <QuerySet [<Choice: Not much>, <Choice: The sky>, <Choice: Just hacking again>]>

In [32]: q.choice_set.count()
Out[32]: 3

In [33]: Choice.objects.filter(question__pub_date__year=current_year)
Out[33]: <QuerySet [<Choice: Not much>, <Choice: The sky>, <Choice: Just hacking again>]>

In [34]: c = q.choice_set.filter(choice_text__startswith='Just')

In [35]: c.delete()
Out[35]: (1, {'polls.Choice': 1})

In [36]: q.choice_set.all()
Out[36]: <QuerySet [<Choice: Not much>, <Choice: The sky>]>

```
- Django管理页面:
    - 创建一个管理员账号:python manage.py createsuperuser
    - sername: admin
    - Email address: admin@example.com
    - Password: 9个*
    Password (again): 9个*
    PSuperuser created successfully.
    - 启动开发服务器:python manage.py runserver
    - 打开浏览器，http://127.0.0.1:8000/admin/
    - 向管理页面中加入投票应用:
    ```python
    from django.contrib import admin
    
    from .models import Question
    
    admin.site.register(Question)
    ```
- 编写更多视图:
    - polls/views.py
    ```python
    def detail(request, question_id):
        return HttpResponse("You're looking at question %s." % question_id)
    
    def results(request, question_id):
        response = "You're looking at the results of question %s."
        return HttpResponse(response % question_id)
    
    def vote(request, question_id):
        return HttpResponse("You're looking at the results of question %s." % question_id)
    ```

- 把这些新视图添加进polls.urls模块里，只要添加几个url()函数调用就行
```python
from django.urls import path
from . import views

urlpatterns = [
    # ex: /polls/
    path('', views.index, name='index'),
    # ex: /polls/5/
    path('<int:question_id>/', views.detail, name='detail'),
    # ex: /polls/5/results/
    path('<int:question_id>/results/', views.results, name='results'),
    # ex: /polls/5/vote/
    path('<int:question_id>/vote/', views.vote, name='vote'),
]
```
- 写一个真正有用的视图:polls/views.py
```python
from django.http import HttpResponse

from .models import Question


def index(request):
    latest_question_list = Question.objects.order_by('-pub_date')[:5]
    output = ', '.join([q.question_text for q in latest_question_list])
    return HttpResponse(output)
```
- polls/templates/polls/index.html
```python
{% if latest_question_list %}
    <ul>
    {% for question in latest_question_list %}
        <li><a href="/polls/{{ question.id }}/">{{ question.question_text }}</a></li>
    {% endfor %}
    </ul>
{% else %}
    <p>No polls are available.</p>
{% endif %}
```
- 更新polls/views.py里的index视图来使用模板
```python
from django.http import HttpResponse
from django.template import loader

from .models import Question


def index(request):
    latest_question_list = Question.objects.order_by('-pub_date')[:5]
    template = loader.get_template('polls/index.html')
    # 字典,将模板内的变量映射为Python对象
    context = {'latest_question_list': latest_question_list,}
    return HttpResponse(template.render(context, request))
```
- 一个快捷函数:render()
    - 「载入模板，填充上下文，再返回由它生成的HttpResponse对象」
    - 重写index()视图,polls/views.py
    ```python
    from django.shortcuts import render

    from .models import Question
    
    
    def index(request):
        latest_question_list = Question.objects.order_by('-pub_date')[:5]
        context = {'latest_question_list': latest_question_list}
        return render(request, 'polls/index.html', context)
    ```

- 编写一个简单的表单:polls/templates/polls/detail.html
```python
<h1>{{ question.question_text }}</h1>

{% if error_message %}<p><strong>{{ error_message }}</strong></p>{% endif %}

<form action="{% url 'polls:vote' question.id %}" method="post">
{% csrf_token %}
{% for choice in question.choice_set.all %}
    <input type="radio" name="choice" id="choice{{ forloop.counter }}" value="{{ choice.id }}">
    <label for="choice{{ forloop.counter }}">{{ choice.choice_text }}</label><br>
{% endfor %}
<input type="submit" value="Vote">
</form>
```
- 当需要创建一个改变服务器端数据的表单时，请使用 ``method="post"
- polls/views.py
```python
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import get_object_or_404, render
from django.urls import reverse

from .models import Choice, Question

def vote(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    try:
        selected_choice = question.choice_set.get(pk=request.POST['choice']) **request.POST的值永远是字符串**
    except (KeyError, Choice.DoesNotExist):
        # Redisplay the question voting form.
        return render(request, 'polls/detail.html', {
            'question': question,
            'error_message': "You didn't select a choice.",
        })
    else:
        selected_choice.votes += 1
        selected_choice.save()
        # Always return an HttpResponseRedirect after successfully dealing
        # with POST data. This prevents data from being posted twice if a
        # user hits the Back button.
        return HttpResponseRedirect(reverse('polls:results', args=(question.id,)))
```
```python
from django.shortcuts import get_object_or_404, render


def results(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    return render(request, 'polls/results.html', {'question': question})
```
- polls/results.html模板
```python
<h1>{{ question.question_text }}</h1>

<ul>
{% for choice in question.choice_set.all %}
    <li>{{ choice.choice_text }} -- {{ choice.votes }} vote{{ choice.votes|pluralize }}</li>
{% endfor %}
</ul>

<a href="{% url 'polls:detail' question.id %}">Vote again?</a>
```
- 改良视图：删除旧的index,detail和result 视图,用Django的通用视图代替  polls/views.py
```python
from django.http import HttpResponseRedirect
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.views import generic

from .models import Choice, Question


class IndexView(generic.ListView):
    template_name = 'polls/index.html'
    context_object_name = 'latest_question_list'

    def get_queryset(self):
        """Return the last five published questions."""
        return Question.objects.order_by('-pub_date')[:5]


class DetailView(generic.DetailView):
    model = Question
    template_name = 'polls/detail.html'


class ResultsView(generic.DetailView):
    model = Question
    template_name = 'polls/results.html'


def vote(request, question_id):
    ... # same as above, no changes needed.
```