- C++ 是一种静态类型的、编译式的、通用的、**大小写敏感**的、不规则的编程语言。

- 静态类型的编程语言是在编译时执行类型检查，而不是在运行时执行类型检查。

- C++ 完全支持面向对象的程序设计，包括抽象、封装、继承、多态四大特性。

- 可以用 "\n" 代替代码里的 endl。

- ```c++
  std::cout << "Hello World!\n";
  
  using namespace std;//告诉编译器使用std命名空间
  cout << "Hello world!" << endl;
  ```

- C++ 不以行末作为结束符的标识，因此，您可以在一行上放置多个语句。

- C++标识符：用来标识变量、函数、类、模块，或任何其他用户自定义项目的名称。

  以字母 A-Z 或 a-z 或下划线 _ 开始，后跟零个或多个字母、下划线和数字（0-9）。

- ```C++
  typedef short int wchar_t;//wchar_t 实际上的空间是和 short int 一样
  ```

- 枚举类型(enumeration)是C++中的一种派生数据类型，它是由用户定义的若干枚举常量的集合。

  所谓"枚举"是指将变量的值一一列举出来，变量的值只能在列举出来的值的范围内。

  如果一个变量只有几种可能的值，可以定义为枚举类型。

  ```c++
  enum color { red, green, blue } c;
  c = blue;
  ```

- ```c++
  #include <iostream>
  using namespace std;
  
  //变量声明
  extern int a,b;
  extern int c;
  extern float f;
  
  int main()
  {
      //变量定义
      int a, b;
      int c;
      float f;
      
      //实际初始化
      a = 10;
      b = 20;
      c = a+b;
      
      cout << c << endl;
       
      f = 70.0/3.0;
      cout << f << endl;
      
      return 0;
  }
  ```

- ```c++
  //函数声明:提供一个函数名，参数的名称并不重要，只有参数的类型是必须的
  //当您在一个源文件中定义函数且在另一个文件中调用函数时，函数声明是必需的。在这种情况下，您应该在调用函数的文件顶部声明函数。
  int func();
  
  int main()
  {
      //函数调用
      int i = func();
  }
  
  //函数定义:可以在任何地方进行
  int func()
  {
      return 0;
  }
  ```

- 定义局部变量时，系统不会对其初始化，必须自行对其初始化。

  定义全局变量时，系统会自动初始化为下列值：

  | 数据类型 | 初始化默认值 |
  | -------- | ------------ |
  | int      | 0            |
  | char     | '\0'         |
  | float    | 0            |
  | double   | 0            |
  | pointer  | NULL         |

- 定义常量：使用**#define**预处理器  or  使用**const**关键字    最好把常量定义为大写字母形式！

- 修饰符 **volatile** 告诉编译器不需要优化volatile声明的变量，让程序可以直接从内存中读取变量。对于一般的变量编译器会对变量进行优化，将内存中的变量值放在寄存器中以加快读写效率。

- 由 **restrict** 修饰的指针是唯一一种访问它所指向的对象的方式。只有C99增加了新的类型限定符 restrict。

- i++：先人后己

  ++i：先己后人

- | 循环类型        | 描述                                                         |
  | --------------- | ------------------------------------------------------------ |
  | while 循环      | 当给定条件为真时，重复语句或语句组。它会在执行循环主体之前测试条件。 |
  | for 循环        | 多次执行一个语句序列，简化管理循环变量的代码。               |
  | do...while 循环 | 除了它是在循环主体结尾测试条件外，其他与 while 语句类似。    |

- | 控制语句      | 描述                                                         |
  | ------------- | ------------------------------------------------------------ |
  | break 语句    | 终止 **loop** 或 **switch** 语句，程序流将继续执行紧接着 loop 或 switch 的下一条语句。 |
  | continue 语句 | 引起循环跳过主体的剩余部分，立即重新开始测试条件。           |

- Lambda函数/表达式形式：

  ```c++
  [capture](parameters)->return-type{body}
  例如：[](int x, int y){ return x < y ; }
  ```

- ```c++
  #include <iostream>
  #include <ctime>
  #include <cstdlib>
  
  using namespace std;
  
  int main()
  {
  	int i, j;
  
  	cout << time(NULL) << endl;
  	cout << (unsigned)time(NULL) << endl;
  
  	//设置种子
  	srand((unsigned)time(NULL));
  
  	//生成10个随机数
  	for (i = 0; i < 10; i++)
  	{
  		//生成实际的随机数
  		j = rand();
  		cout << "随机数: " << j << endl;
  	}
  
  	return 0;
  }
  ```

- | 概念           | 描述                                                         |
  | -------------- | ------------------------------------------------------------ |
  | 多维数组       | C++ 支持多维数组。多维数组最简单的形式是二维数组。           |
  | 指向数组的指针 | 您可以通过指定不带索引的数组名称来生成一个指向数组中第一个元素的指针。 |
  | 传递数组给函数 | 您可以通过指定不带索引的数组名称来给函数传递一个指向数组的指针。 |
  | 从函数返回数组 | C++ 允许从函数返回数组。                                     |

- C++ 提供了以下两种类型的字符串表示形式：

  - C 风格字符串
  - C++ 引入的 string 类类型

- 每一个变量都有一个内存位置，每一个内存位置都定义了可使用连字号（&）运算符访问的地址，它表示了在内存中的一个地址。

  ```c++
  #include <iostream>
  using namespace std;
  
  int main()
  {
  	int var1;
  	char var2[10];
  	 
  	cout << "var1 变量的地址：";
  	cout << &var1 << endl;
  
  	cout << "var2 变量的地址：";
  	cout << &var2 << endl;
  
  	return 0;
  }
  
  /*
  var1 变量的地址：00000038F217FB04
  var2 变量的地址：00000038F217FB28
  */
  ```

  C++指针：动态内存分配，没有指针是无法执行的。

  指针是一个变量，其值是另一个变量的地址，即：内存位置的直接地址。

  就像其他变量或常量一样，必须在使用指针存储其他变量地址之前，对其进行声明。

  ```c++
  int    *ip;    /* 一个整型的指针 */
  double *dp;    /* 一个 double 型的指针 */
  float  *fp;    /* 一个浮点型的指针 */
  char   *ch;    /* 一个字符型的指针 */
  
  //所有指针的值的实际数据类型，不管是整型、浮点型、字符型，还是其他的数据类型，都是一样的，都是一个代表内存地址的长的十六进制数。不同数据类型的指针之间唯一的不同是，指针所指向的变量或常量的数据类型不同。
  ```

  使用指针时会频繁进行以下几个操作：

  - 定义一个指针变量

  - 把变量地址赋值给指针

  - 访问指针变量中可用地址的值

    ```c++
    #include <iostream>
    using namespace std;
    
    int main()
    {
    	int var = 20;	//实际变量的声明
    	int *ip;		//指针变量的声明
        
        ip = &var;		//在指针变量中存储var的地址
     
    	cout << "Value of var variable：";
    	cout << var << endl;
    
    	//输出在指针变量中存储的地址
    	cout << "Address stored in ip variable：";
    	cout << ip << endl;
    
    	//访问指针中地址的值
    	cout << "Value of *ip variable：";
    	cout << *ip << endl;
    
    	return 0;
    }
    
    /*
    Value of var variable：20
    Address stored in ip variable：0000007CB5AFF814
    Value of *ip variable：20
    */
    ```

    | 概念               | 描述                                                         |
    | ------------------ | ------------------------------------------------------------ |
    | C++ Null 指针      | C++ 支持空指针。NULL 指针是一个定义在标准库中的值为零的常量。 |
    | C++ 指针的算术运算 | 可以对指针进行四种算术运算：++、--、+、-                     |
    | C++ 指针 vs 数组   | 指针和数组之间有着密切的关系。                               |
    | C++ 指针数组       | 可以定义用来存储指针的数组。                                 |
    | C++ 指向指针的指针 | C++ 允许指向指针的指针。                                     |
    | C++ 传递指针给函数 | 通过引用或地址传递参数，使传递的参数在调用函数中被改变。     |
    | C++ 从函数返回指针 | C++ 允许函数返回指针到局部变量、静态变量和动态内存分配。     |

- C++引用：引用变量是一个别名

  引用和指针的不同：

  - 不存在空引用。引用必须连接到一块合法的内存。

  - 一旦引用被初始化为一个对象，就不能被指向到另一个对象。指针可以在任何时候指向到另一个对象。

  - 引用必须在创建时被初始化。指针可以在任何时间被初始化。

    ```c++
    a#include <iostream>
    using namespace std;
    
    int main()
    {
    	//声明简单的变量
    	int        i;
    	double d;
    
    	//声明引用变量  在这些声明中,& 读作引用。
    	int&       r = i;		//r 是一个初始化为 i 的整型引用
    	double& s = d;	//s 是一个初始化为 d 的 double 型引用
    
    	i = 5;
    	cout << "Value of i：" << i << endl;
    	cout << "Value of i reference：" << r << endl;
    
    	d = 11.7;
    	cout << "Value of d：" << d << endl;
    	cout << "Value of d reference：" << s << endl;
    
    	return 0;
    }
    
    /*
    Value of i：5
    Value of i reference：5
    Value of d：11.7
    Value of d reference：11.7
    */
    ```

    | 概念             | 描述                                                     |
    | ---------------- | -------------------------------------------------------- |
    | 把引用作为参数   | C++ 支持把引用作为参数传给函数，这比传一般的参数更安全。 |
    | 把引用作为返回值 | 可以从 C++ 函数中返回引用，就像返回其他数据类型一样。    |

- C/C++ 数组允许定义可存储相同类型数据项的变量，但是**结构**是 C++ 中另一种用户自定义的可用的数据类型，它允许您存储不同类型的数据项。

  ```c++
  //在结构定义的末尾，最后一个分号之前，您可以指定一个或多个结构变量，这是可选的
  //声明一个结构体类型Books，结构变量为book
  struct Books
  {
  	char title[50];
  	char author[50];
  	char subject[100];
  	int book_id;
  }book;
  ```

  结构作为函数参数：

  ```c++
  #include <iostream>
  #include <cstring>
  
  using namespace std;
  void printBook(struct Books book);
  
  //声明一个结构体类型Books
  struct Books
  {
  	char title[50];
  	char author[50];
  	char subject[100];
  	int book_id;
  };
  
  int main()
  {
  	Books Book1;	//定义结构体类型Books的变量Book1
  	Books Book2;	//定义结构体类型Books的变量Book2
  
  	//Book1详述
  	strcpy(Book1.title, "C++教程");
  	strcpy(Book1.author, "Runoob");
  	strcpy(Book1.subject, "编程语言");
  	Book1.book_id = 12345;
  
  	//Book2详述
  	strcpy(Book2.title, "CSS教程");
  	strcpy(Book2.author, "Runoob");
  	strcpy(Book2.subject, "前端技术");
  	Book2.book_id = 123456;
  
  	//输出Book1信息
  	cout << "第一本书标题：" << Book1.title << endl;
  	cout << "第一本书作者：" << Book1.author << endl;
  	cout << "第一本书类目：" << Book1.subject << endl;
  	cout << "第一本书ID：" << Book1.book_id << endl;
  
  	printBook(Book1);
  
  	//输出Book2信息
  	cout << "第二本书标题：" << Book2.title << endl;
  	cout << "第二本书作者：" << Book2.author << endl;
  	cout << "第二本书类目：" << Book2.subject << endl;
  	cout << "第二本书ID：" << Book2.book_id << endl;
  
  	printBook(Book2);
  
  	return 0;
  }
  
  //结构作为函数参数
  void printBook(struct Books book)
  {
  	cout << "书标题：" << book.title << endl;
  	cout << "书作者：" << book.author << endl;
  	cout << "书类目：" << book.subject << endl;
  	cout << "书ID：" << book.book_id << endl;
  }
  
  /*
  第一本书标题：C++教程
  第一本书作者：Runoob
  第一本书类目：编程语言
  第一本书ID：12345
  书标题：C++教程
  书作者：Runoob
  书类目：编程语言
  书ID：12345
  第二本书标题：CSS教程
  第二本书作者：Runoob
  第二本书类目：前端技术
  第二本书ID：123456
  书标题：CSS教程
  书作者：Runoob
  书类目：前端技术
  书ID：123456
  */
  ```

  指向结构的指针：

  ```c++
  #include <iostream>
  #include <cstring>
  
  using namespace std;
  void printBook(struct Books *book);
  
  //声明一个结构体类型Books
  struct Books
  {
  	char title[50];
  	char author[50];
  	char subject[100];
  	int book_id;
  };
  
  int main()
  {
  	Books Book1;	//定义结构体类型Books的变量Book1
  	Books Book2;	//定义结构体类型Books的变量Book2
  
  	//Book1详述
  	strcpy(Book1.title, "C++教程");
  	strcpy(Book1.author, "Runoob");
  	strcpy(Book1.subject, "编程语言");
  	Book1.book_id = 12345;
  
  	//Book2详述
  	strcpy(Book2.title, "CSS教程");
  	strcpy(Book2.author, "Runoob");
  	strcpy(Book2.subject, "前端技术");
  	Book2.book_id = 123456;
  
  	//输出Book1信息
  	cout << "第一本书标题：" << Book1.title << endl;
  	cout << "第一本书作者：" << Book1.author << endl;
  	cout << "第一本书类目：" << Book1.subject << endl;
  	cout << "第一本书ID：" << Book1.book_id << endl;
  
  	printBook(&Book1);
  
  	//输出Book2信息
  	cout << "第二本书标题：" << Book2.title << endl;
  	cout << "第二本书作者：" << Book2.author << endl;
  	cout << "第二本书类目：" << Book2.subject << endl;
  	cout << "第二本书ID：" << Book2.book_id << endl;
  
  	printBook(&Book2);
  
  	return 0;
  }
  
  //指向结构的指针
  void printBook(struct Books *book)
  {
  	cout << "书标题：" << book->title << endl;
  	cout << "书作者：" << book->author << endl;
  	cout << "书类目：" << book->subject << endl;
  	cout << "书ID：" << book->book_id << endl;
  }
  
  /*
  第一本书标题：C++教程
  第一本书作者：Runoob
  第一本书类目：编程语言
  第一本书ID：12345
  书标题：C++教程
  书作者：Runoob
  书类目：编程语言
  书ID：12345
  第二本书标题：CSS教程
  第二本书作者：Runoob
  第二本书类目：前端技术
  第二本书ID：123456
  书标题：CSS教程
  书作者：Runoob
  书类目：前端技术
  书ID：123456
  */
  ```

- 函数重载：函数的名称相同，但是函数的形式参数(指参数的个数、类型或者顺序)不同

- 运算符重载：

  ```c++
  #include <iostream>
  using namespace std;
  
  class Box
  {
  	public:
  		double getVolume(void)
  		{
  			return length*breadth*height;
  		}
  		void setLength(double len)
  		{
  			length = len;
  		}
  		void setBreadth(double bre)
  		{
  			breadth = bre;
  		}
  		void setHeight(double hei)
  		{
  			height = hei;
  		}
  		//重载+运算符，用于把两个Box对象相加
  		//函数名是由关键字 operator 和其后要重载的运算符符号构成的
  		Box operator+(const Box& b)
  		{
  			Box box;
  			box.length = this->length + b.length;
  			box.breadth = this->breadth + b.breadth;
  			box.height = this->height + b.height;
  			return box;
  		}
  	private:
  		double length;
  		double breadth;
  		double height;
  };
  
  int main()
  {
  	Box Box1;
  	Box Box2;
  	Box Box3;
  	double volume = 0.0;		//把体积存储在该变量中
  
  	Box1.setLength(6.0);
  	Box1.setBreadth(7.0);
  	Box1.setHeight(5.0);
  
  	Box2.setLength(12.0);
  	Box2.setBreadth(13.0);
  	Box2.setHeight(10.0);
  
  	volume = Box1.getVolume();
  	cout << "Volume of Box1：" << volume << endl;
  
  	volume = Box2.getVolume();
  	cout << "Volume of Box2：" << volume << endl;
  
  	Box3 = Box1 + Box2;
  
  	volume = Box3.getVolume();
  	cout << "Volume of Box3：" << volume << endl;
  
  	return 0;
  }
  
  /*
  Volume of Box1：210
  Volume of Box2：1560
  Volume of Box3：5400
  */
  ```
