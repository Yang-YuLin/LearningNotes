### �ڶ���:��Django�ٷ��ĵ�https://docs.djangoproject.com/zh-hans/2.2/intro/overview/

#### ���ģ��
- Django�������ݿ���ʹ��,���ṩ��ORM,����ʹ��Python�������������ݿ�ṹ
- ����: mysite/news/models.py
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

 #### Ӧ������ģ��
 - ����Django�����й������Զ��������ݿ�� 

   ```python
   #�������п��õ�models��Ϊ����һ�������ݿ��в����ڶ�Ӧ���ݱ��model����migrations�ű��ļ�
   python manage.py makemigrations 
   #������Щmigrations�Զ��������ݿ��
   python manage.py migrate
   ```

 #### ���ܱ�ݵ�API
 - ���ǿ���ʹ��һ�ױ�ݶ��ḻ��Python API��������
 ��Щ API �Ǽ�ʱ�����ģ���������ʽ�����ɴ���

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
 ��һ����Զ�����һ����ϵ��
 ����word, ����ϵ�����learn�ֶ�
����User,���Զ�����һ���ֶ�,���ֽ�word_set,�����Ǹ���ϵ��

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

#### һ����̬����ӿڣ�����ͽ�����
- ������ģ��֮��,Django�ͻ��Զ�����һ��**����ӿ�**
һ��������֤�û���ӡ��޸ĺ�ɾ�������Webվ��
����ֻ��򵥵���adminվ����ע��ģ�ͼ���
- ����: mysite/news/models.py
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

admin.site.register(models.Article) # ע��ģ��
```
- ����DjangoӦ�õĵ�������:
1. �Ƚ�������ģ��
2. Ȼ������վ��
3. ֮������վ���������

#### �滮URLs
- Ϊ������Լ���URLconf����Ҫ����һ������URLconf��Pythonģ��
- ������վ��Ŀ¼����������һ��URL��Python�ص�����(��ͼ)֮���ӳ���
- ����: mysite/news/urls.py
```python
from django.urls import path

from . import views

# ���û������������� URL "/articles/2005/05/39323/",Django����� news.views.article_detail(request, year=2005, month=5, pk=39323)
urlpatterns = [
    path('articles/<int:year>/', views.year_archive),
    path('articles/<int:year>/<int:month>/', views.month_archive),
    path('articles/<int:year>/<int:month>/<int:pk>/', views.article_detail),
]
```

#### ��д��ͼ

- ��ͼ������ִ�н��ֻ���������֣�
  - ����һ����������ҳ��Ԫ�ص�HttpResponse����
  - �������׳�Http404�����쳣
- ��ͼ�Ĺ������Ӳ�����ȡ���ݣ�װ��һ��ģ�壬Ȼ�󽫸��ݻ�ȡ�����ݶ�ģ�������Ⱦ

- ����: mysite/news/views.py
```python
from django.shortcuts import render

from .models import Article


def year_archive(request, year):
    a_list = Article.objects.filter(pub_date__year=year)
    context = {'year': year, 'article_list': a_list}
    return render(request, 'news/year_archive.html', context)
```

#### ���ģ��

- ����Ĵ�������� news/year_archive.html ģ��
- ģ��̳�: {% extends "base.html" %}

#### ��д��һ��DjangoӦ��: ������ͶƱӦ�ó���
- django-admin startproject mysite
- python manage.py runserver
- python manage.py startapp polls
- ��д��һ����ͼ:
    - polls/views.py
    ```python
    from django.http import HttpResponse
    
    
    def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")
    ```

    - ��pollsĿ¼���½�һ��urls.py�ļ�
    
      ```python
      from django.urls import path
      from . import views
      
      urlpatterns = [
          path('', views.index, name='index')
  ]
      ```
    
    - mysite/urls.py�ļ���urlpatterns�б������һ�� include()


```python
from django.contrib import admin
from django.urls import path, includes

urlpatterns = [
    path('admin/', admin.site.urls),
    path('polls/', include('polls.urls')),
]
    # ����path()�����ĸ��������������������route��view��������ѡ������kwargs��name��
    # route��һ��ƥ��URL��׼��.��Django��Ӧһ������ʱ�������urlpatterns�ĵ�һ�ʼ����˳������ƥ���б��е��ֱ���ҵ�ƥ������Щ׼�򲻻�ƥ�� GET �� POST ������������
    # path:��Django�ҵ���һ��ƥ���׼�򣬾ͻ��������ض�����ͼ������������һ��HttpRequest������Ϊ��һ���������������񡱵Ĳ����Թؼ��ֲ�������ʽ����
```
- ���ݿ�����: mysite/settings.pyʹ��SQLite��ΪĬ�����ݿ�
- ʹ���������ݿ�:
    - ENGINE
    - NAME ���ݿ������,NAMEӦ���Ǵ��ļ��ľ���·���������ļ���
    - Ĭ��ֵ os.path.join(BASE_DIR, 'db.sqlite3') ��������ݿ��ļ���������Ŀ�ĸ�Ŀ¼
- python manage.py migrateΪINSTALLED_APPS�������˵�Ӧ�ý������ݿ�Ǩ��
- ����ģ��:
    - polls/models.py
    ```python
    from django.db import models

    class Question(models.Model):
        question_text = models.CharField(max_length=200)
        pub_date = models.DateTimeField('date published')
        
    class Choice(models.Model):
        question = models.ForeignKey(Question, on_delete=models.CASCADE) # ʹ��ForeignKey������һ����ϵ
        choice_text = models.CharField(max_length=200)
        votes = models.IntegerField(default=0)
    ```

- �������� INSTALLED_APPS ���������'polls.apps.PollsConfig',
- python manage.py makemigrations polls
- python manage.py migrate
- ��Python������:python manage.py shell
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
<Question: What's up?> **ֻ�ܷ���һ��ֵ,��������������ж��,�򱨴�**

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
- Django����ҳ��:
    - ����һ������Ա�˺�:python manage.py createsuperuser
    - sername: admin
    - Email address: admin@example.com
    - Password: 9��*
    Password (again): 9��*
    PSuperuser created successfully.
    - ��������������:python manage.py runserver
    - ���������http://127.0.0.1:8000/admin/
    - �����ҳ���м���ͶƱӦ��:
    ```python
    from django.contrib import admin
    
    from .models import Question
    
    admin.site.register(Question)
    ```
- ��д������ͼ:
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

- ����Щ����ͼ��ӽ�polls.urlsģ���ֻҪ��Ӽ���url()�������þ���
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
- дһ���������õ���ͼ:polls/views.py
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
- ����polls/views.py���index��ͼ��ʹ��ģ��
```python
from django.http import HttpResponse
from django.template import loader

from .models import Question


def index(request):
    latest_question_list = Question.objects.order_by('-pub_date')[:5]
    template = loader.get_template('polls/index.html')
    # �ֵ�,��ģ���ڵı���ӳ��ΪPython����
    context = {'latest_question_list': latest_question_list,}
    return HttpResponse(template.render(context, request))
```
- һ����ݺ���:render()
    - ������ģ�壬��������ģ��ٷ����������ɵ�HttpResponse����
    - ��дindex()��ͼ,polls/views.py
    ```python
    from django.shortcuts import render

    from .models import Question
    
    
    def index(request):
        latest_question_list = Question.objects.order_by('-pub_date')[:5]
        context = {'latest_question_list': latest_question_list}
        return render(request, 'polls/index.html', context)
    ```

- ��дһ���򵥵ı�:polls/templates/polls/detail.html
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
- ����Ҫ����һ���ı�����������ݵı�ʱ����ʹ�� ``method="post"
- polls/views.py
```python
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import get_object_or_404, render
from django.urls import reverse

from .models import Choice, Question

def vote(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    try:
        selected_choice = question.choice_set.get(pk=request.POST['choice']) **request.POST��ֵ��Զ���ַ���**
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
- polls/results.htmlģ��
```python
<h1>{{ question.question_text }}</h1>

<ul>
{% for choice in question.choice_set.all %}
    <li>{{ choice.choice_text }} -- {{ choice.votes }} vote{{ choice.votes|pluralize }}</li>
{% endfor %}
</ul>

<a href="{% url 'polls:detail' question.id %}">Vote again?</a>
```
- ������ͼ��ɾ���ɵ�index,detail��result ��ͼ,��Django��ͨ����ͼ����  polls/views.py
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