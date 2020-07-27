- 访问矩阵：数据是按光栅扫描顺序存储的

- 平滑处理/模糊处理：用来减少图像上的噪声或者失真  降低图像分辨率

- in place：输入图像和输出图像是同一个图像

- **双边滤波可视为高斯平滑**，对相似的像素赋予较高的权重，不相似的像素赋予较小的权重，典型效果就是使处理过的图像看上去更像是一幅源图的水彩画，可用于**图像的分割**。

- 图像形态学：改变物体的形状
  
- 膨胀(求局部最大值)、腐蚀(求局部最小值)
  
- 图像金字塔：运用了颜色融合(根据依赖于颜色相互之间的相似性度量)实现**图像分割**
  - 高斯金字塔 ：向下降采样图像
  - 拉普拉斯金字塔：从金字塔低层图像中向上采样重建一个图像

- 直方图均衡化：平衡亮度值

- 分水岭算法：**分割图像**

- 无监督的学习算法叫做聚类算法，不需要对数据进行训练和学习。聚类就是看能把数据分成几类。
  
- K均值
  
- 随机森林：可以通过收集很多树的子节点对各个类别的投票，然后选择获得最多投票的类别作为判断结果。

- 随机森林和AdaBoost：需要建立多棵树

- 决策树：只需要建立一棵树

- 快速完成训练：最近邻算法、正态贝叶斯和决策树

- 考虑内存因素：决策树、神经网络

- 不需要很快训练，需要很快判断：神经网络、正态贝叶斯、SVM

- 不需要训练很快，需要精确度很高，有很多数据：boosting、随机森林

- 当数据集比较小：支持向量机

- 选取的特征比较好，仅仅需要一个简单易懂的分类器：决策树、最近邻算法

- 获得最好的性能：boosting、随机森林

- 基于树的算法：决策树、随机森林、boosting支持类别变量和**数值变量**

- 评估分类器性能：

- opencv

  - 交叉验证或者与之相近的自抽样法√(验证集是从测试集中随机选取的，选择的点仅用于测试)
  - 画ROC曲线图和填充混淆矩阵

- opencv模块架构：

  | 模块名称                  | 模块功能                                                     |
  | ------------------------- | ------------------------------------------------------------ |
  | calib3d(calibration校准)  | 相机标定与立体视觉(物体位姿估计、三维重建、摄像头标定)       |
  | core(核心功能)            | opencv基本数据结构、绘图函数、数组操作相关函数、动态数据结构 |
  | dnn(深度学习)             | 构建神经网络、加载序列化网络模型，仅适用于正向传递计算(测试网络)，不支持反向计算(训练网络) |
  | features2d                | 处理图像特征点(特征检测、描述与匹配)                         |
  | flann                     | 高维的近似近邻快速搜索算法库与聚类                           |
  | gapi                      | 加速常规的图像处理                                           |
  | highgui                   | 高层GUI图形用户界面，包含创建和操作显示图像的窗口、处理鼠标事件以及键盘命令、提供图形交互可视化界面等 |
  | imgcodecs                 | 图像文件读取与保存模块                                       |
  | imgproc                   | 图像处理(图像滤波、几何变换、直方图、特征检测与目标检测)     |
  | ml(机器学习)              | 统计分类、回归和数据聚类                                     |
  | objdetect(目标检测)       | 图像目标检测(haar特征)                                       |
  | photo(计算摄影)           | 图像修复和去噪                                               |
  | stitching(图像拼接)       | 特征点寻找与匹配图像、估计旋转、自动校准、接缝估计等         |
  | video(视频分析)           | 运动估计、背景分离、对象跟踪等视频处理                       |
  | videoio(视频输入输出模块) | 读取与写入视频或者图像序列                                   |

- Mat类：保存矩阵类型的数据信息(包括向量、矩阵、灰度或彩色图像等数据)

  Mat类分为**矩阵头**(包含矩阵的尺寸、存储方法、地址和引用次数等。矩阵头的大小是一个常数，不随尺寸改变)和**指向存储数据的矩阵指针**两部分。

  ```c++
  #include <opencv2/opencv.hpp>
  using namespace cv;
  
  int main(int argc, char** argv)
  {
  	/*
  	创建Mat类 读取图像文件
  	虽然image、image1有各自的矩阵头，但是其矩阵指针指向的是同一个矩阵数据，通过任意一个矩阵头     修改矩阵中的数据，另一个矩阵头指向的数据也会跟着发生改变。
  	但是当删除image变量时，image1变量并不会指向一个空数据，只有当两个变量都删除后，才会释放矩     阵数据。因为矩阵头中引用次数标记了引用某个矩阵数据的次数，只有当矩阵数据引用次数为0的时候才	 会释放矩阵数据。用这种方式可以避免仍有某个变量引用数据时将这个数据删除造成程序崩溃的问题。
  	*/
  	Mat image;//创建一个名为image的矩阵头
  	image = imread("‪D:/opencv/Projects/Test1/Test1/lena.jpg");//向image中赋值图像数据，矩阵指针指向像素数据
  	Mat image1 = image;//赋值矩阵头，并命名为image1
  	
  	/*
  	声明一个指定类型的Mat类变量
  	*/
  	Mat a = Mat_<double>(3, 3);//创建一个3*3的矩阵用于存放double类型数据
  
  	/*
  	通过opencv数据类型创建Mat类
  	*/
  	Mat b(3, 3, CV_8UC1);//创建一个3*3的8位无符号整数的单通道矩阵,单通道矩阵C1标识可以省略
  	Mat c(640, 480, CV_8UC3);//创建一个640*480的3通道矩阵用于存放彩色图像
  
  	return 0;
  }
  ```

- Mat类矩阵元素的读取方式：

  - 通过at方法进行读取

    ```c++
    #include <opencv2/opencv.hpp>
    using namespace cv;
    
    int main(int argc, char** argv)
    {
    	//单通道图像是一个二维矩阵，因此在at方法的最后给出二维平面坐标即可访问对应位置元素
    	//枚举赋值法：将矩阵中所有的元素都一一枚举出，并用数据流的形式赋值给Mat类
    	Mat a = (Mat_<uchar>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9);
    	//通过at方法读取元素需要在后面跟上"<数据类型>"
    	//该方法以坐标的形式给出需要读取的元素坐标(行数，列数)
    	//如果矩阵定义的是uchar类型的数据，在需要输入数据的时候，需要强制转换成int类型的数据进行输出
    	int value = (int)a.at<uchar>(0, 0);
    	
    	//多通道矩阵中每一个元素坐标处都是多个数据，因此引入Vec3b等变量表示同一元素多个数据
    	//构造时赋值，将每个元素想要赋予的值放入Scalar结构中即可，用此方法会将图像中的每个元素赋值相同的数值
    	Mat b(3, 4, CV_8UC3, Scalar(0, 0, 1));//创建一个3通道矩阵，每个像素都是0,0,255
    	Vec3b vc3 = b.at<Vec3b>(0, 0);
    	int first = (int)vc3.val[0];
    	int second = (int)vc3.val[1];
    	int third = (int)vc3.val[2];
    
    	return 0;
    }
    ```

  - 通过指针ptr进行读取

    ```c++
    #include <opencv2/opencv.hpp>
    #include <iostream>
    
    using namespace cv;
    using namespace std;
    
    int main(int argc, char** argv)
    {
    	Mat b(3, 4, CV_8UC3, Scalar(0, 0, 1));
    	for (int i = 0; i < b.rows; i++)
    	{
    		uchar* ptr = b.ptr<uchar>(i);
    		//用于输出矩阵中每一行所有通道的数据
    		for (int j = 0; j < b.cols*b.channels(); j++)
    		{
    			cout << (int)ptr[j] << endl;
    		}
    	}
    
    	return 0;
    }
    ```

  - 通过迭代器进行读取

    ```c++
    #include <opencv2/opencv.hpp>
    #include <iostream>
    
    using namespace cv;
    using namespace std;
    
    int main(int argc, char** argv)
    {
    	//Mat类变量同时也是一个容器变量，所以Mat类变量拥有迭代器，用于访问Mat类变量中的数据
    	Mat a = (Mat_<uchar>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9);
    	MatIterator_<uchar> it = a.begin<uchar>();
    	MatIterator_<uchar> it_end = a.end<uchar>();
    	for (int i = 0; it != it_end; it++)
    	{
    		cout << (int)(*it) << " ";
    		if ((++i % a.cols) == 0)
    		{
    			cout << endl;
    		}
    	}
    
    	return 0;
    }
    
    /*
    1 2 3
    4 5 6
    7 8 9
    */
    ```

  - 通过矩阵元素地址定位方式访问元素

    ```c++
    //row：某个数据所在元素的行数
    //col：某个数据所在元素的列数
    //channel：某个数据所在元素的通道数
    //将首个数据的地址指针移动若干位后指向需要读取的数据
    //这种方式可以通过直接给出行、列和通道数进行读取，前三种都需要知道Mat类矩阵存储数据的类型
    (int)(*(b.data + b.step[0] * row + b.step[1] * col + channel));
    ```

- 图像的读取与显示

  ```c++
  #include <opencv2/opencv.hpp>
  #include <iostream>
  
  using namespace cv;
  using namespace std;
  
  int main()
  {
  	//读取图像文件
  	Mat img = imread("lena.jpg");
  
  	//通过判断返回矩阵的data属性是否为空或者empty()函数是否为真来判断是否成功读取图像
  	if (img.empty())
  	{
  		cout << "Could not open or find the image" << endl;
  		cin.get();
  		return -1;
  	}
  
  	//窗口的名字
  	String windowName = "Test";
  	//创建一个窗口
  	namedWindow(windowName);
  	//在创建的窗口里展示图像
  	imshow(windowName, img);
  	//用于将程序暂停一段时间，以毫秒计。参数缺省或者为0表示等待用户按键结束该函数
  	waitKey(0);
  	//关闭创建的窗口
  	destroyWindow(windowName);
  	return 0;
  }
  ```
  
- 视频加载与摄像头调用

  ```c++
  #include <opencv2/opencv.hpp>
  #include <iostream>
  
  using namespace cv;
  using namespace std;
  
  int main()
  {
  	//更改输出界面颜色
  	system("color F0");		
  	//视频读取函数
  	//VideoCapture video("music.mp4");
  	//摄像头的直接调用
  	VideoCapture video(0);
  	if (video.isOpened())
  	{
  		//get()函数查看视频属性
  		cout << "视频中图像的宽度=" << video.get(CAP_PROP_FRAME_WIDTH) << endl;
  		cout << "视频中图像的高度=" << video.get(CAP_PROP_FRAME_HEIGHT) << endl;
  		cout << "视频帧率=" << video.get(CAP_PROP_FPS) << endl;
  		cout << "视频的总帧数=" << video.get(CAP_PROP_FRAME_COUNT) << endl;
  	}
  	else {
  		cout << "请确认视频文件名称是否正确" << endl;
  		return -1;
  	}
  	while (1)
  	{
  		Mat frame;
  		//通过">>"运算符将图像按照视频顺序由VideoCapture类变量赋值给Mat类变量
  		video >> frame;
  		//当VideoCapture类变量中所有的图像都赋值给Mat类变量后，再次赋值的时候Mat类变量会变为空矩阵，因此可以通过empty()判断VideoCapture类变量中是否所有图像都已经读取完毕
  		if (frame.empty())
  		{
  			break;
  		}
  		imshow("video", frame);
  		waitKey(1000 / video.get(CAP_PROP_FPS));
  	}
  	waitKey();
  	return 0;
  }
  ```
  
- 图像的保存

  ```c++
  #include <opencv2/opencv.hpp>
  #include <iostream>
  
  using namespace cv;
  using namespace std;
  
  void AlphaMat(Mat &mat) 
  {
  	CV_Assert(mat.channels() == 4);
  	for (int i = 0; i < mat.rows; ++i)
  	{
  		for (int j = 0; j < mat.cols; ++j)
  		{
  			Vec4b& bgra = mat.at<Vec4b>(i, j);
  			//蓝色通道
  			bgra[0] = UCHAR_MAX;//255		
  			//绿色通道  saturate_cast<uchar>主要是为了防止颜色溢出操作
  			bgra[1] = saturate_cast<uchar>((float(mat.cols - j)) / ((float)mat.cols) * UCHAR_MAX);		
  			//红色通道
  			bgra[2] = saturate_cast<uchar>((float(mat.rows - i)) / ((float)mat.rows) * UCHAR_MAX);
  			//Alpha通道
  			bgra[3] = saturate_cast<uchar>(0.5 * (bgra[1] + bgra[2]));
  		}
  	}
  }
  
  int main()
  {
  	//imwrite()函数用于将Mat类矩阵保存成图像文件
  	//生成带有Alpha通道(4通道)的矩阵，并保存成PNG格式图像
  	Mat mat(480, 640, CV_8UC4);
  	AlphaMat(mat);
  	vector<int> compression_params;
  	//PNG格式图像压缩标志
  	compression_params.push_back(IMWRITE_PNG_COMPRESSION);
  	//设置最高压缩质量
  	compression_params.push_back(9);
  	bool result = imwrite("alpha.png", mat, compression_params);
  	if (!result)
  	{
  		cout << "保存成PNG格式图像失败" << endl;
  		return -1;
  	}
  	cout << "保存成功" << endl;
  	return 0;
  }
  ```

- 视频的保存

  ```c++
  #include <opencv2/opencv.hpp>
  #include <iostream>
  
  using namespace cv;
  using namespace std;
  
  int main()
  {
  	Mat img;
  	//摄像头的直接调用
  	//VideoCapture video(0);
  	//读取视频
  	VideoCapture video;
  	video.open("music.mp4");
  
  	//判断是否调用成功
  	if (!video.isOpened())
  	{
  		cout << "打开摄像头失败，请确认摄像头是否安装成功";
  		return -1;
  	}
  
  	//获取图像  通过“>>”操作符从文件中读取数据
  	video >> img;
  	//检测是否成功获取图像
  	if (img.empty()) {
  		cout << "没有获取到图像" << endl;
  		return -1;
  	}
  
  	//VideoWrite()类用于实现多张图像保存成视频文件
  	VideoWriter writer;
  	//保存的视频文件名称
  	string filename = "live.avi";
  	//选择编码格式
  	int codec = VideoWriter::fourcc('M', 'J', 'P', 'G');
  	//设置视频帧率，即视频中每秒图像的张数
  	double fps = 25.0;
  	//判断相机（视频）类型是否为彩色
  	bool isColor = (img.type() == CV_8UC3);
  
  	//创建保存视频
  	writer.open(filename, codec, fps, img.size(), isColor);
  
  	//判断视频流是否创建成功
  	if (!writer.isOpened())
  	{
  		cout << "打开视频文件失败，请确认是否为合法输入" << endl;
  		return -1;
  	}
  
  	while (1)
  	{
  		//检测是否执行完毕
  		if (!video.read(img))		//判断能继续从摄像头或者视频文件中读出一帧图像
  		{
  			cout << "摄像头断开连接或者视频读取完成" << endl;
  			break;
  		}
  		//把图像写入视频流
  		writer.write(img);
  		//writer << img;	//通过“<<”操作符将数据写入文件中
  		//显示图像
  		imshow("Live", img);
  		char c = waitKey(50);
  		//按ESC按键退出视频保存
  		if (c == 27)
  		{
  			break;
  		}
  	}
  	//退出程序时自动关闭视频流
  	video.release();
  	writer.release();
  	return 0;
  }
  ```
  
- 保存和读取XML和YMAL文件

  - 除了图像数据之外，有时程序中的尺寸较小的Mat类矩阵、字符串、数组等数据也需要进行保存，这些数据通常保存成XML文件或者YAML文件。
  - 程序中使用write()函数和“<<”操作符两种方式向文件中写入数据，使用迭代器和“[]”地址两种方式从文件中读取数据。
  
  ```c++
  #include <opencv2/opencv.hpp>
  #include <iostream>
  #include <string>
  
  using namespace cv;
  using namespace std;
  
  int main(int argc, char** argv)
  {
  	//修改运行程序背景和文字颜色
  	system("color F0");
  	//文件的名称
  	string fileName = "datas.yaml";
  	//以写入的模式打开文件
  	FileStorage fwrite(fileName, FileStorage::WRITE);
  
  	//存入矩阵Mat类型的数据 eye()：构建一个单位矩阵
  	Mat mat = Mat::eye(3, 3, CV_8U);
  	//使用write()函数写入数据
  	fwrite.write("mat", mat);
  	//存入浮点型数据，节点名称为x
  	float x = 100;
  	fwrite << "x" << x;
  	//存入字符串型数据，节点名称为str
  	String str = "Learn OpenCV 4";
  	fwrite << "str" << str;
  	//存入数组，节点名称为number_array
  	fwrite << "number_array" << "[" << 4 << 5 << 6 << "]";
  	//存入多node节点数据，主名称为multi_nodes
  	fwrite << "multi_nodes" << "{" << "month" << 8 << "day" << 28 << "year" << 2019 << "time" << "[" << 0 << 1 << 2 << 3 << "]" << "}";
  	//关闭文件
  	fwrite.release();
  	//以读取的模式打开文件
  	FileStorage fread(fileName, FileStorage::READ);
  	//判断是否成功打开文件
  	if (!fread.isOpened())
  	{
  		cout << "打开文件失败，请确认文件名称是否正确！" << endl;
  		return -1;
  	}
  
  	//读取文件中的数据
  	float xRead;
  	//读取浮点型数据
  	fread["x"] >> xRead;
  	cout << "x=" << xRead << endl;
  
  	//读取字符串数据
  	string strRead;
  	fread["str"] >> strRead;
  	cout << "str=" << strRead << endl;
  
  	//读取含多个数据的number_array节点
  	FileNode fileNode = fread["number_array"];
  	cout << "number_array=[";
  	//循环遍历每个数据
  	for (FileNodeIterator i = fileNode.begin(); i != fileNode.end(); i++)
  	{
  		float a;
  		*i >> a;
  		cout << a << " ";
  	}
  	cout << "]" << endl;
  
  	//读取Mat类型数据
  	Mat matRead;
  	fread["mat"] >> matRead;
  	cout << "mat=" << matRead << endl;
  
  	//读取含有多个子节点的节点数据，不使用FileNode和迭代器进行读取
  	FileNode FileNode1 = fread["multi_nodes"];
  	int month = (int)FileNode1["month"];
  	int day = (int)FileNode1["day"];
  	int year = (int)FileNode1["year"];
  	cout << "multi_nodes:" << endl
  		<< " month=" << month << " day=" << day << " year=" << year;
  	cout << " time=[";
  	for (int i = 0; i < 4; i++)
  	{
  		int a = (int)FileNode1["time"][i];
  		cout << a << " ";
  	}
  	cout << "]" << endl;
  	//关闭文件
  	fread.release();
  	return 0;
  }
  ```
  
- 颜色模型与转换

  - RGB颜色模型：如果三种颜色分量都为0，则表示为黑色。
  
    ​						  如果三种颜色的分类相同且都为最大值，则表示为白色。
  
    ​						  RGB取值范围均为0~255。
  
  - YUV颜色模型：像素的宽度(Y)、红色分量与亮度的信号差值(U)、蓝色分量与亮度的信号差值(V)
  
  - HSV颜色模型：色度(Hue)**颜色**、饱和度(Saturation)**深浅**、亮度(Value)**亮暗**
  
  - Lab颜色模型：L表示亮度，a和b是两个颜色通道，两者的取值区间都是-128到+127。
  
    ​						其中a通道数值由小到大对应的颜色是从绿色变成红色，b通道数值由小到大对应的						颜色是从蓝色变成黄色。
  
  - GRAY颜色模型：是灰度图像的模型，灰度图像只有单通道，灰度值根据图像位数不同由0到最大   
  
    ​                            依次表示由黑到白
  
  - 不同颜色模型间的互相转换：
  
    如果转换过程中添加了alpha通道（RGB模型中第四个通道，表示透明度），则其值将设置为相应通道范围的最大值：CV_8U为255，CV_16U为65535，CV_32F为1。
  
    ```c++
    #include <opencv2/opencv.hpp>
    #include <iostream>
    #include <vector>
    
    using namespace cv;
    using namespace std;
    
    int main()
    {
    	Mat img = imread("lena.jpg");
    	if (img.empty())
    	{
    		cout << "请确认图像文件名称是否正确" << endl;
    		return -1;
    	}
    	Mat gray, HSV, YUV, Lab, img32;
    	//为了防止转换后出现数组越界的情况，将CV_8U类型转换成CV_32F类型
    	img.convertTo(img32, CV_32F, 1.0 / 255);
    	cvtColor(img32, HSV, COLOR_BGR2HSV);
    	cvtColor(img32, YUV, COLOR_BGR2YUV);
    	cvtColor(img32, Lab, COLOR_BGR2Lab);
    	cvtColor(img32, gray, COLOR_BGR2GRAY);
    	imshow("原图", img32);
    	imshow("HSV", HSV);
    	imshow("YUV", YUV);
    	imshow("Lab", Lab);
    	imshow("gray", gray);
    	waitKey(0);
    	return 0;
    }
    ```
  
- 图像像素统计

  - Point(x,y)对应于图像的行和列表示为Point(列数，行数)
  
  - 寻找图像像素最大值与最小值
  
    ```c++
    #include <opencv2/opencv.hpp>
    #include <iostream>
    #include <vector>
    
    using namespace cv;
    using namespace std;
    
    int main()
    {
    	//更改输出界面颜色
    	system("color F0");
    	float a[12] = { 1,2,3,4,5,10,6,7,8,9,10,0 };
    	//单通道矩阵
    	Mat img = Mat(3, 4, CV_32FC1, a);
    	//多通道矩阵
    	Mat imgs = Mat(2, 3, CV_32FC2, a);
    	//用于存放矩阵中的最大值和最小值
    	double minVal, maxVal;
    	//用于存放矩阵中的最大值和最小值在矩阵中的位置
    	Point minIdx, maxIdx;
    
    	//寻找单通道矩阵中的最值
    	minMaxLoc(img, &minVal, &maxVal, &minIdx, &maxIdx);
    	cout << "img中最大值是：" << maxVal << "  " << "在矩阵中的位置：" << maxIdx << endl;
    	cout << "img中最小值是：" << minVal << "  " << "在矩阵中的位置：" << minIdx << endl;
    
    	//寻找多通道矩阵中的最值
    	Mat imgs_re = imgs.reshape(1, 4);	//将多通道矩阵变成单通道矩阵,第一个参数是转换后矩阵的通道数，第二个参数是转换后矩阵的行数，如果参数为零，则转换后行数与转换前相同
    	minMaxLoc(imgs_re, &minVal, &maxVal, &minIdx, &maxIdx);
    	cout << "imgs中最大值是：" << maxVal << "  " << "在矩阵中的位置：" << maxIdx << endl;
    	cout << "imgs中最小值是：" << minVal << "  " << "在矩阵中的位置：" << minIdx << endl;
    	return 0;
    }
    ```
  
  - 计算图像的均值和标准方差
  
    	- 图像的均值表示图像整体的亮暗程度，均值越大图像整体越亮
    	- 图像的标准方差表示图像中明暗变化的对比程度，标准方差越大图像中明暗变化越明显
  
  ```c++
  #include <opencv2/opencv.hpp>
  #include <iostream>
  #include <vector>
  
  using namespace cv;
  using namespace std;
  
  int main()
  {
  	//更改输出界面颜色
  	system("color F0");
  	float a[12] = { 1,2,3,4,5,10,6,7,8,9,10,0 };
  	//单通道矩阵
  	Mat img = Mat(3, 4, CV_32FC1, a);
  	//多通道矩阵
  	Mat imgs = Mat(2, 3, CV_32FC2, a);
  
  	//用于存放矩阵中的最大值和最小值
  	double minVal, maxVal;
  	//用于存放矩阵中的最大值和最小值在矩阵中的位置
  	Point minIdx, maxIdx;
  
  	//寻找单通道矩阵中的最值
  	minMaxLoc(img, &minVal, &maxVal, &minIdx, &maxIdx);
  	cout << "img中最大值是：" << maxVal << "  " << "在矩阵中的位置：" << maxIdx << endl;
  	cout << "img中最小值是：" << minVal << "  " << "在矩阵中的位置：" << minIdx << endl;
  
  	//寻找多通道矩阵中的最值
  	Mat imgs_re = imgs.reshape(1, 4);	//将多通道矩阵变成单通道矩阵,第一个参数是转换后矩阵的通道数，第二个参数是转换后矩阵的行数，如果参数为零，则转换后行数与转换前相同
  	minMaxLoc(imgs_re, &minVal, &maxVal, &minIdx, &maxIdx);
  	cout << "imgs中最大值是：" << maxVal << "  " << "在矩阵中的位置：" << maxIdx << endl;
  	cout << "imgs中最小值是：" << minVal << "  " << "在矩阵中的位置：" << minIdx << endl;
  	
  	//用mean()求取图像的均值
  	Scalar myMean;
  	myMean = mean(imgs);
  	cout << "imgs均值=" << myMean << endl;
  	cout << "imgs第一个通道的均值=" << myMean[0] << "   "
  	    	<< "imgs第二个通道的均值=" << myMean[1] << endl << endl;
  	
  	//用meanStdDev()同时求取图像的均值和标准方差
  	Mat myMeanMat, myStddevMat;
  	meanStdDev(img, myMeanMat, myStddevMat);
  	cout << "img均值=" << myMeanMat << "    " << endl;
  	cout << "img标准方差=" << myStddevMat << endl << endl;
  	meanStdDev(imgs, myMeanMat, myStddevMat);
  	cout << "imgs均值=" << myMeanMat << "    " << endl << endl;
  	cout << "imgs标准方差=" << myStddevMat << endl;
  	return 0;
  }
  ```
  
- 两图像间的像素操作

  - 两张图像的比较运算
  
    ```c#
    #include <opencv2/opencv.hpp>
    #include <iostream>
    #include <vector>
    
    using namespace cv;
    using namespace std;
    
    int main()
    {
    	float a[12] = { 1, 2, 3.3f, 4, 5, 9, 5, 7, 8.2f, 9, 10, 2 };
    	float b[12] = { 1, 2.2f, 3, 1, 3, 10, 6, 7, 8, 9.3f, 10, 1 };
    	Mat imga = Mat(3, 4, CV_32FC1, a);
    	Mat imgb = Mat(3, 4, CV_32FC1, b);
    	Mat imgas = Mat(2, 3, CV_32FC2, a);
    	Mat imgbs = Mat(2, 3, CV_32FC2, b);
    
    	//对两个单通道矩阵进行比较运算
    	Mat myMax, myMin;
    	max(imga, imgb, myMax);
    	min(imga, imgb, myMin);
    	
    	//对两个多通道矩阵进行比较运算
    	Mat myMaxs, myMins;
    	max(imgas, imgbs, myMaxs);
    	min(imgas, imgbs, myMins);
    
    	//对两张彩色图像进行比较运算
    	Mat img0 = imread("lena.jpg");
    	Mat img1 = imread("mitu.jpg");
    	if (img0.empty() || img1.empty())
    	{
    		cout << "请确认图像文件名称是否正确" << endl;
    		return -1;
    	}
    	Mat comMin, comMax;
    	max(img0, img1, comMax);
    	min(img0, img1, comMin);
    	imshow("comMin", comMin);
    	imshow("comMax", comMax);
    
    	//与掩模进行比较运算，可以实现抠图或者选择通道的效果
    	Mat src1 = Mat::zeros(Size(512, 512), CV_8UC3);
    	Rect rect(100, 100, 300, 300);		//(x,y,width,height)
    	//生成一个低通300*300的掩模 掩模是用于设置图像或矩阵中逻辑运算的范围
    	src1(rect) = Scalar(255, 255, 255);
    	imshow("src1", src1);
    	Mat comsrc1, comsrc2;
    	min(img0, src1, comsrc1);
    	imshow("comsrc1", comsrc1);
    
    	//生成一个显示红色通道的低通掩模
    	Mat src2 = Mat(512, 512, CV_8UC3, Scalar(0, 0, 255));
    	imshow("src2", src2);
    	min(img0, src2, comsrc2);
    	imshow("comsrc2", comsrc2);
    
    	//对两张灰度图像进行比较运算
    	Mat img0G, img1G, comMinG, comMaxG;
    	cvtColor(img0, img0G, COLOR_BGR2GRAY);
    	cvtColor(img1, img1G, COLOR_BGR2GRAY);
    	max(img0G, img1G, comMaxG);
    	min(img0G, img1G, comMinG);
    	imshow("comMinG", comMinG);
    	imshow("comMaxG", comMaxG);
    	waitKey(0);
    	return 0;
    }
    ```
  
  - 两张图像的逻辑运算
  
    ```c++
    #include <opencv2/opencv.hpp>
    #include <iostream>
    #include <vector>
    
    using namespace cv;
    using namespace std;
    
    int main()
    {
    	Mat img = imread("lena.jpg");
    	if (img.empty())
    	{
    		cout << "请确认图像文件名称是否正确" << endl;
    		return -1;
    	}
    
    	//创建两个黑白图像
    	Mat img0 = Mat::zeros(200, 200, CV_8UC1);
    	Mat img1 = Mat::zeros(200, 200, CV_8UC1);
    	Rect rect0(50, 50, 100, 100);
    	img0(rect0) = Scalar(255);
    	Rect rect1(100, 100, 100, 100);
    	img1(rect1) = Scalar(255);
    	imshow("img0", img0);
    	imshow("img1", img1);
    
    	//进行逻辑运算
    	Mat myAnd, myOr, myXor, myNot, imgNot;
    	//像素求与运算
    	bitwise_and(img0, img1, myAnd);
    	//像素求或运算
    	bitwise_or(img0, img1, myOr);
    	//像素求异或运算
    	bitwise_xor(img0, img1, myXor);
    	//像素求非运算
    	bitwise_not(img0, myNot);
    	bitwise_not(img, imgNot);
    
    	imshow("myAnd", myAnd);
    	imshow("myOr", myOr);
    	imshow("myXor", myXor);
    	imshow("myNot", myNot);
    	imshow("img", img);
    	imshow("imgNot", imgNot);
    	waitKey(0);
    	return 0;
    }
    ```
  
- 图像LUT查找表：需要与多个阈值进行比较时使用。LUT查找表简单来说就是一个像素灰度值的映射表，它以像素灰度值作为索引，以灰度值映射后的数值作为表中的内容。

  ```c++
  #include <opencv2/opencv.hpp>
  #include <iostream>
  
  using namespace cv;
  using namespace std;
  
  int main(int argc,char** argv)
  {
  	//LUT查找表第一层
  	uchar lutFirst[256];
  	for (int i = 0; i < 256; i++)
  	{
  		if (i <= 100)
  			lutFirst[i] = 0;
  		if (i > 100 && i <= 200)
  			lutFirst[i] = 100;
  		if (i > 200)
  			lutFirst[i] = 255;
  	}
  	Mat lutOne(1, 256, CV_8UC1, lutFirst);
  
  	//LUT查找表第二层
  	uchar lutSecond[256];
  	for (int i = 0; i < 256; i++)
  	{
  		if (i <= 100)
  			lutSecond[i] = 0;
  		if (i > 100 && i <= 150)
  			lutSecond[i] = 100;
  		if (i > 150 && i <= 200)
  			lutSecond[i] = 150;
  		if (i > 200)
  			lutSecond[i] = 255;
  	}
  	Mat lutTwo(1, 256, CV_8UC1, lutSecond);
  
  	//LUT查找表第三层
  	uchar lutThird[256];
  	for (int i = 0; i < 256; i++)
  	{
  		if (i <= 100)
  			lutSecond[i] = 0;
  		if (i > 100 && i <= 200)
  			lutSecond[i] = 200;
  		if (i > 200)
  			lutSecond[i] = 255;
  	}
  	Mat lutThree(1, 256, CV_8UC1, lutThird);
  
  	//拥有三通道的LUT查找表矩阵
  	vector<Mat> mergeMats;
  	mergeMats.push_back(lutOne);
  	mergeMats.push_back(lutTwo);
  	mergeMats.push_back(lutThree);
  	Mat LutTree;
  	merge(mergeMats, LutTree);
  
  	//计算图像的查找表
  	Mat img = imread("lena.jpg");
  	if (img.empty())
  	{
  		cout << "请确认图像文件名称是否正确" << endl;
  		return -1;
  	}
  
  	Mat gray, out0, out1, out2;
  	cvtColor(img, gray, COLOR_BGR2GRAY);
  	//LUT函数的第一个输入参数要求的数据类型必须是CV_8U类型，但可以是多通道的图像矩阵
  	//第二个参数是一个1*256的矩阵，其中存放着每个像素灰度值映射后的数值
  	//函数输出图像的数据类型不与原图像的数据类型保持一致，而是和LUT查找表的数据类型保持一致
  	LUT(gray, lutOne, out0);
  	LUT(img, lutTwo, out1);
  	LUT(img, LutTree, out2);
  	imshow("out0", out0);
  	imshow("out1", out1);
  	imshow("out2", out2);
  	waitKey(0);
  	return 0;
  }
  ```

- 图像多通道分离与合并

  ```c++
  #include <opencv2/opencv.hpp>
  #include <iostream>
  #include <vector>
  
  using namespace cv;
  using namespace std;
  
  int main()
  {
  	Mat img = imread("lena.jpg");
  	if (img.empty())
  	{
  		cout << "请确认图像文件名称是否正确" << endl;
  		return -1;
  	}
  	Mat HSV;
  	cvtColor(img, HSV, COLOR_RGB2HSV);
  	//用于存放数组类型的结果
  	Mat imgs0, imgs1, imgs2;
  	//用于存放vector类型的结果
  	Mat imgv0, imgv1, imgv2;
  	//多通道合并的结果
  	Mat result0, result1, result2;
  
  	//输入数组参数的多通道分离与合并
  	Mat imgs[3];
  	split(img, imgs);
  	imgs0 = imgs[0];
  	imgs1 = imgs[1];
  	imgs2 = imgs[2];
  	//显示分离后R通道的像素值
  	imshow("RGB-R通道", imgs0);
  	//显示分离后G通道的像素值
  	imshow("RGB-G通道", imgs1);
  	//显示分离后B通道的像素值
  	imshow("RGB-B通道", imgs2);
  	//将数组中的图像通道数变成不统一
  	imgs[2] = img;
  	//合并图像
  	merge(imgs, 3, result0);
  	Mat zero = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
  	imgs[0] = zero;
  	imgs[2] = zero;
  	//用于还原G通道的真实情况，合并结果为绿色
  	merge(imgs, 3, result1);
  	//显示合并结果
  	imshow("result1", result1);
  
  	//输入vector参数的多通道分离与合并
  	vector<Mat> imgv;
  	split(HSV, imgv);
  	imgv0 = imgv.at(0);
  	imgv1 = imgv.at(1);
  	imgv2 = imgv.at(2);
  	//显示分离后H通道的像素值
  	imshow("HSV-H通道", imgv0);
  	//显示分离后S通道的像素值
  	imshow("HSV-S通道", imgv1);
  	//显示分离后V通道的像素值
  	imshow("HSV-V通道", imgv2);
  	//将vector中的图像通道数变成不统一
  	imgv.push_back(HSV);
  	//合并图像
  	merge(imgv, result2);
  	waitKey(0);
  	return 0;
  }
  ```

- 图像仿射变换：就是图像的旋转、平移和缩放操作的统称。

  实现图像的旋转首先需要确定旋转角度和旋转中心，之后确定旋转矩阵，最终通过仿射变换实现图像旋转。
  
  ```c++
  #include <opencv2/opencv.hpp>
  #include <iostream>
  #include <vector>
  
  using namespace cv;
  using namespace std;
  
  int main()
  {
  	Mat img = imread("lena.jpg");
  	if (img.empty())
  	{
  		cout << "请确认图像文件名称是否正确" << endl;
  		return -1;
  	}
  
  	Mat rotation0, rotation1, img_warp0, img_warp1;
  	//设置图像旋转的角度
  	double angle = 30;
  	//设置输出图像的尺寸
  	Size dst_size(img.rows, img.cols);
  	//设置图像的旋转中心
  	Point2f center(img.rows / 2.0, img.cols / 2.0);
  	//计算仿射变换矩阵
  	rotation0 = getRotationMatrix2D(center, angle, 1);
  	//进行仿射变换
  	warpAffine(img, img_warp0, rotation0, dst_size);
  	imshow("img_warp0", img_warp0);
  	//根据定义的三个点进行仿射变换
  	Point2f src_points[3];
  	Point2f dst_points[3];
  	//原始图像中的三个点
  	src_points[0] = Point2f(0, 0);
  	src_points[1] = Point2f(0, (float)(img.cols-1));
  	src_points[2] = Point2f((float)(img.rows - 1), (float)(img.cols - 1));
  	//仿射变换后图像中的三个点
  	dst_points[0] = Point2f((float)(img.rows)*0.11, (float)(img.cols)*0.20);
  	dst_points[1] = Point2f((float)(img.rows)*0.15, (float)(img.cols)*0.70);
  	dst_points[2] = Point2f((float)(img.rows)*0.81, (float)(img.cols)*0.85);
  	//根据对应点求取仿射变换矩阵
  	rotation1 = getAffineTransform(src_points, dst_points);
  	//进行仿射变换
  	warpAffine(img, img_warp1, rotation1, dst_size);
  	imshow("img_warp1", img_warp1);
  	waitKey(0);
  	return 0;
  }
  ```
  
  ![image-20200727161141003](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20200727161141003.png)
  
- 图像透视变换：按照物体成像投影规律进行变换，即将物体重新投影到新的成像平面，通过图像的透视变换实现对物体图像的校正

  ```c++
  //二维码图像透视变换
  #include <opencv2/opencv.hpp>
  #include <iostream>
  #include <vector>
  
  using namespace cv;
  using namespace std;
  
  int main()
  {
  	Mat img = imread("qrcode.jpg");
  	if (img.empty())
  	{
  		cout << "请确认图像文件名称是否正确" << endl;
  		return -1;
  	}
  
  	//根据定义的四个点进行透视变换
  	Point2f src_points[4];
  	Point2f dst_points[4];
  	//原始二维码图像的四个角点坐标
  	src_points[0] = Point2f(0.0, 0.0);
  	src_points[1] = Point2f(627.0, 0.0);
  	src_points[2] = Point2f(0.0, 627.0);
  	src_points[3] = Point2f(627.0, 627.0);
  	//期望透视变换后二维码图像四个角点坐标
  	dst_points[0] = Point2f(94.0, 374.0);
  	dst_points[1] = Point2f(507.0, 380.0);
  	dst_points[2] = Point2f(1.0, 623.0);
  	dst_points[3] = Point2f(627.0, 627.0);
  	Mat rotation, img_warp;
  	//根据对应点求取透视变换矩阵
  	rotation = getPerspectiveTransform(src_points, dst_points);
  	//进行透视变换
  	warpPerspective(img, img_warp, rotation, img.size());
  	imshow("img", img);
  	imshow("img_warp", img_warp);
  	waitKey(0);
  	return 0;
  }
  ```
  
  ![image-20200727163918034](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20200727163918034.png)
  
- 

- 

- 

- 

  