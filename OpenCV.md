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

- 图像连接：将两个具有相同高度或者宽度的图像连接在一起

  ```c++
  #include <opencv2/opencv.hpp>
  #include <iostream>
   
  using namespace cv;
  using namespace std;
  
  int main()
  {
  	//矩阵数组的横竖连接
  	Mat matArray[] = { Mat(1,2,CV_32FC1,cv::Scalar(1)),
  			Mat(1,2,CV_32FC1,cv::Scalar(2)) };
  	Mat vout, hout;
  	vconcat(matArray, 2, vout);
  	cout << "图像数组竖向连接：" << endl << vout << endl;
  	hconcat(matArray, 2, hout);
  	cout << "图像数组横向连接：" << endl << hout << endl;
  
  
  	//矩阵的横竖拼接
  	Mat A = (cv::Mat_<float>(2, 2) << 1, 7, 2, 8);
  	Mat B = (cv::Mat_<float>(2, 2) << 4, 10, 5, 11);
  	Mat vC, hC;
  	vconcat(A, B, vC);
  	cout << "多个图像竖向连接：" << endl << vC << endl;
  	hconcat(A, B, hC);
  	cout << "多个图像横向连接：" << endl << hC << endl;
  
  	//读取4个子图像，00表示左上角，01表示右上角，10表示左下角，11表示右下角
  	Mat img00 = imread("lena00.jpg");
  	Mat img01 = imread("lena01.jpg");
  	Mat img10 = imread("lena10.jpg");
  	Mat img11 = imread("lena11.jpg");
  	if (img00.empty() || img01.empty() || img10.empty() || img11.empty())
  	{
  		cout << "请确认图像文件名称是否正确" << endl;
  		return -1;
  	}
  	//显示4个子图像
  	imshow("img00", img00);
  	imshow("img01", img01);
  	imshow("img10", img10);
  	imshow("img11", img11);
  
  	//图像连接
  	Mat img, img0, img1;
  	//图像横向连接
  	hconcat(img00, img01, img0);
  	hconcat(img10, img11, img1);
  	//横向连接结果再进行竖向连接
  	vconcat(img0, img1, img);
  
  	//显示连接图像的结果
  	imshow("img0", img0);
  	imshow("img1", img1);
  	imshow("img", img);
  	waitKey(0);
  	return 0;
  }
  ```

- 图像极坐标变换：将图像在直角坐标系与极坐标系中互相变换

  对表盘图像进行极坐标正变换和逆变换，选取表盘的中心作为极坐标的原点

  ```c++
  #include <opencv2/opencv.hpp>
  #include <iostream>
   
  using namespace cv;
  using namespace std;
  
  int main()
  {
  	Mat img = imread("dial.png");
  	if (!img.data)
  	{
  		cout << "请检查图像文件名称是否正确" << endl;
  		return -1;
  	}
  	Mat img1, img2;
  	//极坐标在图像中的原点
  	Point2f center = Point2f(img.cols / 2, img.rows / 2);
  	cout << center << endl; //[71,71]
  	//正极坐标变换
  	warpPolar(img, img1, Size(300, 600), center, center.x, INTER_LINEAR + WARP_POLAR_LINEAR);
  	//逆极坐标变换
  	warpPolar(img1, img2, Size(img.rows, img.cols), center, center.x, INTER_LINEAR + WARP_POLAR_LINEAR + WARP_INVERSE_MAP);
  	imshow("原表盘图", img);
  	imshow("表盘极坐标变换结果", img1);
  	imshow("逆变换结果", img2);
  	waitKey(0);
  	return 0;
  }
  ```

  ![image-20200728160307438](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20200728160307438.png)

- 图像上绘制几何图形

  ```c++
  #include <opencv2/opencv.hpp>
  #include <iostream>
   
  using namespace cv;
  using namespace std;
  
  int main()
  {
  	//生成一个黑色图像用于绘制几何图形
  	Mat img = Mat::zeros(Size(512, 512), CV_8UC3);
  	//绘制圆形
  	circle(img, Point(50, 50), 25, Scalar(255, 255, 255), -1);	//绘制一个实心圆
  	circle(img, Point(100, 50), 20, Scalar(255, 255, 255), 4);	//绘制一个空心圆
  	//绘制直线
  	line(img, Point(100, 100), Point(200, 100), Scalar(255, 255, 255), 2, LINE_4, 0);
  	//绘制椭圆
  	ellipse(img, Point(300, 255), Size(100, 70), 0, 0, 100, Scalar(255, 255, 255), -1);
  	ellipse(img, RotatedRect(Point2f(150, 100), Size2f(30, 20), 0), Scalar(0, 0, 255), 2);
  	vector<Point> points;
  	//用一些点来近似一个椭圆 用于输出椭圆的边界的像素坐标，但是不会在图像中绘制椭圆
  	ellipse2Poly(Point(200, 400), Size(100, 70), 0, 0, 360, 2, points);
  	//用直线把这个椭圆画出来
  	for (int i = 0; i < points.size() - 1; i++)
  	{
  		if (i == points.size() - 1)
  		{
  			//椭圆中后于一个点与第一个点连线
  			line(img, points[0], points[i], Scalar(255, 255, 255), 2);
  			break;
  		}
  		//当前点与后一个点连线
  		line(img, points[i], points[i + 1], Scalar(255, 255, 255), 2);
  	}
  	//绘制矩形
  	rectangle(img, Point(50, 400), Point(100, 450), Scalar(125, 125, 125), -1);
  	rectangle(img, Rect(400, 450, 60, 50), Scalar(0, 125, 125), 2);
  	//绘制多边形
  	Point pp[2][6];
  	pp[0][0] = Point(72, 200);
  	pp[0][1] = Point(142, 204);
  	pp[0][2] = Point(226, 263);
  	pp[0][3] = Point(172, 310);
  	pp[0][4] = Point(117, 319);
  	pp[0][5] = Point(15, 260);
  	pp[1][0] = Point(359, 339);
  	pp[1][1] = Point(447, 351);
  	pp[1][2] = Point(504, 349);
  	pp[1][3] = Point(484, 433);
  	pp[1][4] = Point(418, 449);
  	pp[1][5] = Point(354, 402);
  	Point pp2[5];
  	pp2[0] = Point(350, 83);
  	pp2[1] = Point(463, 90);
  	pp2[2] = Point(500, 171);
  	pp2[3] = Point(421, 194);
  	pp2[4] = Point(338, 141);
  	//pts变量的生成
  	const Point* pts[3] = { pp[0],pp[1],pp2 };
  	//顶点个数数组的生成
  	int npts[] = { 6,6,5 };
  	//绘制3个多边形
  	fillPoly(img, pts, npts, 3, Scalar(125, 125, 125), 8);
  	//生成文字
  	putText(img, "Learn OpenCV 4", Point(100,400), 2,1,Scalar(255,255,255));
  	imshow("", img);
  	waitKey(0);
  	return 0;
  }
  ```

  ![image-20200728170621371](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20200728170621371.png)

- 图像金字塔：通过多个分辨率表示图像的一种有效且简单的结构。

  金字塔的底部是待处理图像的高分辨率表示，而顶部是低分辨率的表示。

  - 高斯金字塔：是解决尺度不确定性的一种常用方法。通过底层图像构建上层图像。

    - 通过下采样不断的将图像的尺寸缩小，进而在金字塔中包含多个尺度的图像。

    - 一般情况下，高斯金字塔的底部为图像的原图，每上一层就会通过下采样缩小一次图像的尺寸，通过情况尺寸会缩小为原来的一半。常见的层数为3到6层。
    - pyrDown()用于实现图像模糊并对其进行下采样计算，最终实现尺寸缩小的下采样图像。

    ![image-20200729205330337](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20200729205330337.png)

  - 拉普拉斯金字塔：具有预测残差的作用，需与高斯金字塔一起使用。通过上层小尺寸的图像构建下层大尺寸的图像。

    - pyrUp()用于实现图像的上采样。

  ```c++
  #include <opencv2/opencv.hpp>
  #include <iostream>
  
  using namespace cv;
  using namespace std;
  
  int main()
  {
  	Mat img = imread("lena.jpg");
  	if (img.empty())
  	{
  		cout << "请检查图像文件名称是否正确" << endl;
  		return -1;
  	}
  
  	//高斯金字塔和拉普拉斯金字塔
  	vector<Mat> Gauss, Lap;
  	//高斯金字塔下采样次数
  	int level = 3;
  	//将原图作为高斯金字塔的第0层
  	Gauss.push_back(img);
  	//构建高斯金字塔
  	for (int i = 0; i < level; i++)
  	{
  		Mat gauss;
  		//下采样
  		pyrDown(Gauss[i], gauss);
  		Gauss.push_back(gauss);
  	}
  	//构建拉普拉斯金字塔
  	for (int i = Gauss.size() - 1; i > 0; i--)
  	{
  		Mat lap, upGauss;
  		//如果是高斯金字塔中的最上面一层图像
  		if (i == Gauss.size() - 1)
  		{
  			Mat down;
  			pyrDown(Gauss[i], down);
  			pyrUp(down, upGauss);
  			lap = Gauss[i] - upGauss;
  			Lap.push_back(lap);
  		}
  		pyrUp(Gauss[i], upGauss);
  		lap = Gauss[i - 1] - upGauss;
  		Lap.push_back(lap);
  	}
  
  	//查看两个金字塔中的图像
  	for (int i = 0; i < Gauss.size(); i++)
  	{
  		String name = to_string(i);
  		imshow('G' + name, Gauss[i]);
  		imshow('L' + name, Lap[i]);
  	}
  	waitKey(0);
  	return 0;
  }
  ```

- 窗口交互操作

  - 图像窗口滑动条：能够改变参数数值的滑动条

    通过滑动条改变图像亮度，程序中滑动条控制图像亮度系数，将图像原始灰度值乘以亮度系数得到最终的图像。

    ```c++
    #include <opencv2/opencv.hpp>
    #include <iostream>
    
    using namespace cv;
    using namespace std;
    
    //为了能在被调函数中使用，所以设置成全局的
    int value;
    //滑动条回调函数
    void callBack(int, void*);
    Mat img1,img2;
    
    int main()
    {
    	img1 = imread("lena.jpg");
    	if (!img1.data)
    	{
    		cout << "请确认是否输入正确的图像文件" << endl;
    		return -1;
    	}
    	namedWindow("滑动条改变图像亮度");
    	imshow("滑动条改变图像亮度", img1);
    	//滑动条创建时的初值
    	value = 100;
    	//创建滑动条
    	createTrackbar("亮度值百分比", "滑动条改变图像亮度", &value, 600, callBack, 0);
    	waitKey();
    }
    
    static void callBack(int, void*)
    {
    	float a = value / 100.0;
    	img2 = img1*a;
    	imshow("滑动条改变图像亮度", img2);
    }
    ```

  - 鼠标交互响应

    绘制鼠标移动轨迹

    ```c++
    #include <opencv2/opencv.hpp>
    #include <iostream>
    
    using namespace cv;
    using namespace std;
    
    //全局的图像
    Mat img, imgPoint;
    //前一时刻鼠标的坐标，用于绘制直线
    Point prePoint;
    void mouse(int event, int x, int y, int flags, void*);
    
    int main()
    {
    	img = imread("lena.jpg");
    	if (!img.data)
    	{
    		cout << "请确认输入图像名称是否正确！" << endl;
    		return -1;
    	}
    	img.copyTo(imgPoint);
    	imshow("图像窗口1", img);
    	imshow("图像窗口2", imgPoint);
    	//鼠标影响
    	setMouseCallback("图像窗口1", mouse, 0);
    	waitKey(0);
    	return 0;
    }
    
    void mouse(int event, int x, int y, int flags, void*)
    {
    	//单击右键
    	if (event == EVENT_RBUTTONDOWN)
    	{
    		cout << "点击鼠标左键才可以绘制轨迹" << endl;
    	}
    	//单机左键，输出坐标
    	if (event == EVENT_LBUTTONDOWN)
    	{
    		prePoint = Point(x, y);
    		cout << "轨迹起始坐标" << prePoint << endl;
    	}
    	//鼠标按住左键移动
    	if (event == EVENT_MOUSEMOVE && (flags&EVENT_FLAG_LBUTTON))
    	{
    		//通过改变图像像素显示鼠标移动轨迹
    		//回调函数有一定的执行时间，因此当鼠标移动较快时绘制的图像轨迹会出现断点
    		imgPoint.at<Vec3b>(y, x) = Vec3b(0, 0, 255);
    		imgPoint.at<Vec3b>(y, x - 1) = Vec3b(0, 0, 255);
    		imgPoint.at<Vec3b>(y, x + 1) = Vec3b(0, 0, 255);
    		imgPoint.at<Vec3b>(y - 1, x) = Vec3b(0, 0, 255);
    		imgPoint.at<Vec3b>(y + 1, x) = Vec3b(0, 0, 255);
    		imshow("图像窗口2", imgPoint);
    
    		//通过绘制直线显示鼠标移动轨迹
    		//这种方式是在前一时刻和当前时刻鼠标位置间绘制直线，可以避免因鼠标移动过快而带来的轨迹出现断点的问题
    		Point pt(x, y);
    		line(img, prePoint, pt, Scalar(0, 0, 255), 2, 5, 0);
    		prePoint = pt;
    		imshow("图像窗口1", img);
    	}
    }
    ```

- 图像直方图绘制：

  - 图像直方图是图像处理中非常重要的像素统计结果，图像直方图不再表征任何的图像纹理信息，而是对图像像素的统计。具有平移不变性、放缩不变性。

  - 图像直方图简单来说就是统计图像中每个灰度值的个数，之后将图像灰度值作为横轴，以灰度值个数或者灰度值所占比率作为纵轴绘制的统计图。

  - 通过直方图可以看出图像中哪些灰度值数目较多，哪些较少，可以通过一定的方法将灰度值较为集中的区域映射到较为稀疏的区域，从而使得图像在像素灰度值上分布更加符合期望状态。

    ```c++
    #include <opencv2/opencv.hpp>
    #include <iostream>
    
    using namespace cv;
    using namespace std;
    
    int main()
    {
    	Mat img = imread("apple.jpg");
    	if (img.empty())
    	{
    		cout << "请确认图像文件名称是否正确" << endl;
    		return -1;
    	}
    	Mat gray;
    	cvtColor(img, gray, COLOR_BGR2GRAY);
    
    	//设置提取直方图的相关变量
    	Mat hist;		//用于存放直方图计算结果
    	const int channels[1] = { 0 };		//通道索引
    	float inRanges[2] = { 0,255 };
    	const float*ranges[1] = { inRanges };		//像素灰度值范围
    	const int bins[1] = { 256 };		//直方图的维度，其实就是像素灰度值的最大值
    	calcHist(&img, 1, channels, Mat(), hist, 1, bins, ranges);	//计算图像直方图
    	//准备绘制直方图
    	int hist_w = 512;
    	int hist_h = 400;
    	int width = 2;
    	Mat histImage = Mat::zeros(hist_h, hist_w, CV_8UC3);
    	for (int i = 1; i <= hist.rows; i++)
    	{
    		rectangle(histImage, Point(width*(i - 1), hist_h - 1),
    			Point(width*i - 1, hist_h - cvRound(hist.at<float>(i - 1) / 20)),
    			Scalar(255, 255, 255), -1);
    	}
    	namedWindow("histImage", WINDOW_AUTOSIZE);
    	imshow("histImage", histImage);
    	imshow("gray", gray);
    	waitKey(0);
    	return 0;
    }
    ```

    ![image-20200730191019789](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20200730191019789.png)

- 直方图操作

  - 直方图归一化

    - 用每个灰度值像素的数目占一幅图像中所有像素数目的比例来表示某个灰度值数目的多少

    - 寻找统计结果中最大数值，把所有结果除以这个最大数值，从而将所有数据都缩放到0到1之间

      ```c++
      #include <opencv2/opencv.hpp>
      #include <iostream>
      
      using namespace cv;
      using namespace std;
      
      int main()
      {
      	//更改输出界面颜色
      	system("color F0");
      	vector<double> positiveData = { 2.0,8.0,10.0 };
      	vector<double> normalized_L1, normalized_L2, normalized_Inf, normalized_L2SQR;
      	//测试不同归一化方法
      	normalize(positiveData, normalized_L1, 1.0, 0.0, NORM_L1);	//绝对值求和归一化
      	cout << "normalized_L1=[" << normalized_L1[0] << ","
      		<< normalized_L1[1] << "," << normalized_L1[2] << "]" << endl;
      
      	normalize(positiveData, normalized_L2, 1.0, 0.0, NORM_L2);	//模长归一化
      	cout << "normalized_L2=[" << normalized_L2[0] << ","
      		<< normalized_L2[1] << "," << normalized_L2[2] << "]" << endl;
      
      	normalize(positiveData, normalized_Inf, 1.0, 0.0, NORM_INF);		//最大值归一化
      	cout << "normalized_Inf=[" << normalized_Inf[0] << ","
      		<< normalized_Inf[1] << "," << normalized_Inf[2] << "]" << endl;
      
      	normalize(positiveData, normalized_L2SQR, 1.0, 0.0, NORM_MINMAX);		//偏移归一化
      	cout << "normalized_MINMAX=[" << normalized_L2SQR[0] << ","
      		<< normalized_L2SQR[1] << "," << normalized_L2SQR[2] << "]" << endl;
      
      	//将图像直方图归一化
      	Mat img = imread("apple.jpg");
      	if (img.empty())
      	{
      		cout << "请确认图像文件名称是否正确" << endl;
      		return -1;
      	}
      	Mat gray, hist;
      	cvtColor(img, gray, COLOR_BGR2GRAY);
      	const int channels[1] = { 0 };		//通道索引
      	float inRanges[2] = { 0,255 };
      	const float*ranges[1] = { inRanges };		//像素灰度值范围
      	const int bins[1] = { 256 };		//直方图的维度，其实就是像素灰度值的最大值
      	calcHist(&gray, 1, channels, Mat(), hist, 1, bins, ranges);	//计算图像直方图
      	//准备绘制直方图
      	int hist_w = 512;
      	int hist_h = 400;
      	int width = 2;
      	Mat histImage_L1 = Mat::zeros(hist_h, hist_w, CV_8UC3);
      	Mat histImage_Inf = Mat::zeros(hist_h, hist_w, CV_8UC3);
      	Mat hist_L1, hist_Inf;
      	normalize(hist, hist_L1, 1, 0, NORM_L1, -1, Mat());
      	for (int i = 1; i <= hist_L1.rows; i++)
      	{
      		rectangle(histImage_L1, Point(width*(i - 1), hist_h - 1),
      			Point(width*i - 1, hist_h - cvRound(30*hist_h*hist_L1.at<float>(i - 1) )-1),
      			Scalar(255, 255, 255), -1);
      	}
      	normalize(hist, hist_Inf, 1, 0, NORM_INF, -1, Mat());
      	for (int i = 1; i <= hist_Inf.rows; i++)
      	{
      		rectangle(histImage_Inf, Point(width*(i - 1), hist_h - 1),
      			Point(width*i - 1, hist_h - cvRound(hist_h*hist_Inf.at<float>(i - 1) )- 1),
      			Scalar(255, 255, 255), -1);
      	}
      	imshow("histImage_L1", histImage_L1);
      	imshow("histImage_Inf", histImage_Inf);
      	waitKey(0);
      	return 0;
      }
      ```

    ![image-20200730213425946](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20200730213425946.png)

  - 直方图比较

    - 可以通过比较两张图像的直方图特性比较两张图像的相似程度

    - 通过比较两张图像的直方图分布相似性对图像进行初步的筛选与识别

    - 通过观看直方图趋势可以发现即使将图像尺寸缩小，两张图像的直方图分布也有一定的相似性

      ```c++
      #include <opencv2/opencv.hpp>
      #include <iostream>
      
      using namespace cv;
      using namespace std;
      
      //归一化并绘制直方图函数
      void drawHist(Mat &hist, int type, string name)
      {
      	int hist_w = 512;
      	int hist_h = 400;
      	int width = 2;
      	Mat histImage = Mat::zeros(hist_h, hist_w, CV_8UC3);
      	normalize(hist, hist, 1, 0, type, -1, Mat());
      	for (int i = 1; i <= hist.rows; i++)
      	{
      		rectangle(histImage, Point(width*(i - 1), hist_h - 1),
      			Point(width*i - 1, hist_h - cvRound(30 * hist_h*hist.at<float>(i - 1)) - 1),
      			Scalar(255, 255, 255), -1);
      	}
      	imshow(name, histImage);
      }
      
      int main()
      {
      	//更改输出界面颜色
      	system("color F0");
      	Mat img = imread("apple.jpg");
      	if (img.empty())
      	{
      		cout << "请确认图像文件名称是否正确" << endl;
      		return -1;
      	}
      	Mat gray, hist,gray2,hist2,gray3,hist3;
      	//将读取的图像转成灰度图像
      	cvtColor(img, gray, COLOR_BGR2GRAY);
      	//将图像缩小为原来尺寸的一半
      	resize(gray, gray2, Size(), 0.5, 0.5);
      	gray3 = imread("lena.jpg", IMREAD_GRAYSCALE);
      	const int channels[1] = { 0 };		//通道索引
      	float inRanges[2] = { 0,255 };
      	const float*ranges[1] = { inRanges };		//像素灰度值范围
      	const int bins[1] = { 256 };		//直方图的维度，其实就是像素灰度值的最大值
      	calcHist(&gray, 1, channels, Mat(), hist, 1, bins, ranges);	//计算图像直方图
      	calcHist(&gray2, 1, channels, Mat(), hist2, 1, bins, ranges);	//计算图像直方图
      	calcHist(&gray3, 1, channels, Mat(), hist3, 1, bins, ranges);	//计算图像直方图
      	drawHist(hist, NORM_INF, "hist");
      	drawHist(hist2, NORM_INF, "hist2");
      	drawHist(hist3, NORM_INF, "hist3");
      	//原图直方图与原图直方图的相关系数
      	double hist_hist = compareHist(hist, hist, HISTCMP_CORREL);
      	cout << "apple_apple=" << hist_hist << endl;		//apple_apple = 1
      	//原图直方图与缩小原图直方图的相关系数
      	double hist_hist2 = compareHist(hist, hist2, HISTCMP_CORREL);
      	cout << "apple_apple256=" << hist_hist2 << endl;		//apple_apple256 = 0.999968
      	//两张不同图像直方图相关系数
      	double hist_hist3 = compareHist(hist, hist3, HISTCMP_CORREL);
      	cout << "apple_lena=" << hist_hist3 << endl;		//apple_lena = -0.0996329
      	waitKey(0);
      	return 0;
      }
      ```

  ![image-20200730221324738](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20200730221324738.png)

- 直方图应用

  - 直方图均衡化

    - 如果图像的直方图都集中在一个区域，则整体图像的对比度较小，不便于图像中纹理的识别。

    - 如果通过映射关系，将图像中灰度值的范围扩大，增加原来两个灰度值之间的差值，就可以提高图像的对比度，进而将图像中的纹理突出显现出来，这个过程称为图像直方图均衡化。
    
    - 可以自动的改变图像直方图的分布形式。
    
      ```c++
      #include <opencv2/opencv.hpp>
      #include <iostream>
      
      using namespace cv;
      using namespace std;
      
      //归一化并绘制直方图函数
      void drawHist(Mat &hist, int type, string name)
      {
      	int hist_w = 512;
      	int hist_h = 400;
      	int width = 2;
      	Mat histImage = Mat::zeros(hist_h, hist_w, CV_8UC3);
      	normalize(hist, hist, 1, 0, type, -1, Mat());
      	for (int i = 1; i <= hist.rows; i++)
      	{
      		rectangle(histImage, Point(width*(i - 1), hist_h - 1),
      			Point(width*i - 1, hist_h - cvRound(30 * hist_h*hist.at<float>(i - 1)) - 1),
      			Scalar(255, 255, 255), -1);
      	}
      	imshow(name, histImage);
      }
      
      int main()
      {
      	Mat img = imread("gearwheel.jpg");
      	if (img.empty())
      	{
      		cout << "请确认图像文件名称是否正确" << endl;
      		return -1;
      	}
      	Mat gray, hist, hist2;
      	//将读取的图像转成灰度图像
      	cvtColor(img, gray, COLOR_BGR2GRAY);
      	Mat equalImg;
      	//将直方图图像均衡化  该函数只能对单通道的灰度图进行直方图均衡化
      	equalizeHist(gray, equalImg);
      	const int channels[1] = { 0 };		//通道索引
      	float inRanges[2] = { 0,255 };
      	const float*ranges[1] = { inRanges };		//像素灰度值范围
      	const int bins[1] = { 256 };		//直方图的维度，其实就是像素灰度值的最大值
      	calcHist(&gray, 1, channels, Mat(), hist, 1, bins, ranges);	//计算图像直方图
      	calcHist(&equalImg, 1, channels, Mat(), hist2, 1, bins, ranges);	//计算图像直方图
      	drawHist(hist, NORM_INF, "hist");
      	drawHist(hist2, NORM_INF, "hist2");
      	imshow("原图", gray);
      	imshow("均衡化后的图像", equalImg);
      	waitKey(0);
      	return 0;
      }
      ```
    
      ![image-20200801120432113](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20200801120432113.png)
    
  - 直方图匹配

    - 将直方图映射成指定分布形式的算法称为直方图匹配或者直方图规定化。

    - 直方图匹配与直方图均衡化相似，都是对图像的直方图分布形式进行改变，只是直方图均衡化后的图像直方图是均匀分布的，而直方图匹配后的直方图可以随意指定分布形式。

    - 直方图匹配操作能够有目的的增强某个灰度区间，相比于直方图均衡化操作，该算法虽然多了一个输入，但是其变换后的结果也更加灵活。

    - 通过构建原直方图累积概率与目标直方图累积概率之间的差值表，寻找原直方图中灰度值n的累积概率与目标直方图中所有灰度值累积概率差值的最小值，这个最小值对应的灰度值r就是n匹配后的灰度值。

    - 程序中待匹配的原图是一个图像整体偏暗的图像，目标直方图分配形式来自于一张较为明亮的图像，经过图像直方图匹配操作之后，提高了图像的整体亮度，图像直方图分布也更加均匀。

      ```c++
      #include <opencv2/opencv.hpp>
      #include <iostream>
      
      using namespace cv;
      using namespace std;
      
      //归一化并绘制直方图函数
      void drawHist(Mat &hist, int type, string name)
      {
      	int hist_w = 512;
      	int hist_h = 400;
      	int width = 2;
      	Mat histImage = Mat::zeros(hist_h, hist_w, CV_8UC3);
      	normalize(hist, hist, 1, 0, type, -1, Mat());
      	for (int i = 1; i <= hist.rows; i++)
      	{
      		rectangle(histImage, Point(width*(i - 1), hist_h - 1),
      			Point(width*i - 1, hist_h - cvRound(20 * hist_h*hist.at<float>(i - 1)) - 1),
      			Scalar(255, 255, 255), -1);
      	}
      	imshow(name, histImage);
      }
      
      int main()
      {
      	Mat img1 = imread("histMatch.jpg");
      	Mat img2 = imread("bright.jpg");
      	if (img1.empty() || img2.empty())
      	{
      		cout << "请确认图像文件名称是否正确" << endl;
      		return -1;
      	}
      	Mat hist1, hist2;
      	//计算两张图像直方图
      	const int channels[1] = { 0 };
      	float inRanges[2] = { 0,255 };
      	const float* ranges[1] = { inRanges };
      	const int bins[1] = { 256 };
      	calcHist(&img1, 1, channels, Mat(), hist1, 1, bins, ranges);	//计算图像直方图
      	calcHist(&img2, 1, channels, Mat(), hist2, 1, bins, ranges);	//计算图像直方图
      	//归一化两张图像的直方图
      	drawHist(hist1, NORM_L1, "hist1");
      	drawHist(hist2, NORM_L1, "hist2");
      	//计算两张图像直方图的累积概率
      	float hist1_cdf[256] = { hist1.at<float>(0) };
      	float hist2_cdf[256] = { hist2.at<float>(0) };
      	for (int i = 1; i < 256; i++)
      	{
      		hist1_cdf[i] = hist1_cdf[i - 1] + hist1.at<float>(i);
      		hist2_cdf[i] = hist2_cdf[i - 1] + hist2.at<float>(i);
      	}
      	//构建累计概率误差矩阵
      	float diff_cdf[256][256];
      	for (int i = 0; i < 256; i++)
      	{
      		for (int j = 0; j < 256; j++)
      		{
      			diff_cdf[i][j] = fabs(hist1_cdf[i] - hist2_cdf[j]);
      		}
      	}
      
      	//生成LUT查找表
      	Mat lut(1, 256, CV_8U);
      	for (int i = 0; i < 256; i++)
      	{
      		//查找源灰度级为i的映射灰度
      		//和i的累积概率差值最小的规定化灰度
      		float min = diff_cdf[i][0];
      		int index = 0;
      		//寻找累积概率误差矩阵中每一行中的最小值
      		for (int j = 1; j < 256; j++)
      		{
      			if (min > diff_cdf[i][j])
      			{
      				min = diff_cdf[i][j];
      				index = j;
      			}
      		}
      		lut.at<uchar>(i) = (uchar)index;
      	}
      	Mat result, hist3;
      	LUT(img1, lut, result);
      	imshow("待匹配图像", img1);
      	imshow("匹配的模板图像", img2);
      	imshow("直方图匹配结果", result);
      	calcHist(&result, 1, channels, Mat(), hist3, 1, bins, ranges);
      	//绘制匹配后的图像直方图
      	drawHist(hist3, NORM_L1, "hist3");
      	waitKey(0);
      	return 0;
      }
      ```

      ![image-20200810145601973](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20200810145601973.png)

      ![image-20200810145709266](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20200810145709266.png)

  - 直方图反向投影

    - 如果一张图像的某个区域中显示的是一种结构纹理或者一个独特的形状，那么这个区域的直方图就可以看作是这个结构或者形状的概率函数，在图像中寻找这种概率分布就是在图像中寻找该结构纹理或者独特形状。
    
    - 反向投影是一种记录给定图像中的像素点如何适应直方图模型像素分布方式的一种方法。
    
    - 反向投影就是首先计算某一特征的直方图模型，然后使用模型去寻找图像中是否存在该特征的方法。
    
    - 该函数用于在输入图像中寻找与特定图像最匹配的点或者区域，即对图像进行反向投影。
    
    - 该函数输入参数与计算图像直方图函数calcHist()大致相似，都需要输入图像和需要进行反向投影的通道索引数目。区别之处在于该函数需要输入模板图像的直方图统计结果，并返回的是一张图像，而不是直方图统计结果。
    
    - ```c++
      #include<opencv2/opencv.hpp>
      #include <iostream>
      
      using namespace cv;
      using namespace std;
      
      //归一化并绘制直方图函数
      void drawHist(Mat &hist, int type, string name)
      {
      	int hist_w = 512;
      	int hist_h = 400;
      	int width = 2;
      	Mat histImage = Mat::zeros(hist_h, hist_w, CV_8UC3);
      	normalize(hist, hist, 255, 0, type, -1, Mat());
      	namedWindow(name, WINDOW_NORMAL);
      	imshow(name, hist);
      }
      
      int main()
      {
      	Mat img = imread("apple.jpg");
      	Mat sub_img = imread("sub_apple.jpg");
      	Mat img_HSV, sub_HSV, hist, hist2;
      	if (img.empty() || sub_img.empty())
      	{
      		cout << "请确认图像文件名称是否正确" << endl;
      		return -1;
      	}
      
      	imshow("img", img);
      	imshow("sub_img", sub_img);
      	//转成HSV空间，提取H、S两个通道
      	cvtColor(img, img_HSV, COLOR_BGR2HSV);
      	cvtColor(sub_img, sub_HSV, COLOR_BGR2HSV);
      	int h_bins = 32;
      	int s_bins = 32;
      	int histSize[] = { h_bins,s_bins };
      	//H通道值的范围由0到179
      	float h_ranges[] = { 0,180 };
      	//S通道的范围由0到255
      	float s_ranges[] = { 0,256 };
      	//每个通道的范围
      	const float* ranges[] = { h_ranges,s_ranges };
      	//统计的通道索引
      	int channels[] = { 0,1 };
      	//绘制H-S二维直方图
      	calcHist(&sub_HSV, 1, channels, Mat(), hist, 2, histSize, ranges, true, false);
      	//直方图归一化并绘制直方图
      	drawHist(hist, NORM_INF, "hist");
      	Mat backproj;
      	//直方图反向投影
      	calcBackProject(&img_HSV, 1, channels, hist, backproj, ranges, 1.0);
      	imshow("反向投影结果", backproj);
      	waitKey(0);
      	return 0;
      }
      ```
    
      ![image-20200810203430856](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20200810203430856.png)

- 图像卷积

  - 虽然卷积前后图像内容一致，但是图像整体变得模糊一些，可见图像卷积具有对图像模糊的作用。

  ```c++
  #include <opencv2/opencv.hpp>
  #include <iostream>
  
  using namespace cv;
  using namespace std;
  
  int main()
  {
  	//待卷积矩阵
  	uchar points[25] = { 1,2,3,4,5,
  	6,7,8,9,10,
  	11,12,13,14,15,
  	16,17,18,19,20,
  	21,22,23,24,25 };
  	Mat img(5, 5, CV_8UC1, points);
  	//卷积模板
  	Mat kernel = (Mat_<float>(3, 3) << 1, 2, 1,
  		2, 0, 2,
  		1, 2, 1);
  	//卷积模板归一化
  	Mat kernel_norm = kernel / 12;
  	//未归一化卷积结果和归一化卷积结果
  	Mat result, result_norm;
  	filter2D(img, result, CV_32F, kernel, Point(-1, -1), 2, BORDER_CONSTANT);
  	filter2D(img, result_norm, CV_32F, kernel_norm, Point(-1, -1), 2, BORDER_CONSTANT);
  	cout << "result：" << endl << result << endl;
  	cout << "result_norm：" << endl << result_norm << endl;
  	//图像卷积
  	Mat lena = imread("lena.jpg");
  	if (lena.empty())
  	{
  		cout << "请确认图像文件名称是否正确" << endl;
  		return -1;
  	}
  	Mat lena_filter;
  	filter2D(lena, lena_filter, -1, kernel_norm, Point(-1, -1), 2, BORDER_CONSTANT);
  	imshow("lena", lena);
  	imshow("lena_filter", lena_filter);
  	waitKey(0);
  	return 0;
  }
  ```

- 图像噪声的种类与生成

  - 图像中常见的噪声主要有四种：高斯噪声、椒盐噪声(脉冲噪声)、泊松噪声、乘性噪声

  - 椒盐噪声：会随机改变图像中的像素值，随机出现在图像中的任意位置

    ```c++
    //在图像中添加椒盐噪声
    #include <opencv2/opencv.hpp>
    #include <iostream>
    
    using namespace cv;
    using namespace std;
    
    //盐噪声函数
    void saltAndPepper(cv::Mat image, int n)
    {
    	for (int k = 0; k < n / 2; k++)
    	{
    		//随机确定图像中位置
    		int i, j;
    		//取余数运算，保证在图像的列数内
    		i = std::rand() % image.cols;
    		//取余数运算，保证在图像的行数内
    		j = std::rand() % image.rows;
    		//判定为白色噪声还是黑色噪声的变量
    		int write_black = std::rand() % 2;
    		//添加白色噪声
    		if (write_black == 0)
    		{
    			//处理灰度图像
    			if (image.type() == CV_8UC1)
    			{
    				//白色噪声
    				image.at<uchar>(j, i) = 255;
    			}
    			//处理彩色图像
    			else if (image.type() == CV_8UC3)
    			{
    				//Vec3b为opencv定义的3个值的向量类型  
    				//[]指定通道，B:0，G:1，R:2 
    				image.at<Vec3b>(j, i)[0] = 255;
    				image.at<Vec3b>(j, i)[1] = 255;
    				image.at<Vec3b>(j, i)[2] = 255;
    			}
    		}
    		//添加黑色噪声
    		else
    		{
    			//处理灰度图像
    			if (image.type() == CV_8UC1)
    			{
    				//白色噪声
    				image.at<uchar>(j, i) = 0;
    			}
    			//处理彩色图像
    			else if (image.type() == CV_8UC3)
    			{
    				//Vec3b为opencv定义的3个值的向量类型  
    				image.at<Vec3b>(j, i)[0] = 0;
    				image.at<Vec3b>(j, i)[1] = 0;
    				image.at<Vec3b>(j, i)[2] = 0;
    			}
    		}
    	}
    }
    
    int main()
    {
    	Mat lena = imread("lena.jpg");
    	Mat equalLena = imread("equalLena.jpg", IMREAD_ANYDEPTH);
    	if (lena.empty() || equalLena.empty())
    	{
    		cout << "请确认图像文件名称是否正确" << endl;
    		return -1;
    	}
    	imshow("lena原图", lena);
    	imshow("equalLena原图", equalLena);
    	//彩色图像添加椒盐噪声
    	saltAndPepper(lena, 10000);
    	//灰度图像添加椒盐噪声
    	saltAndPepper(equalLena, 10000);
    	imshow("lena添加噪声", lena);
    	imshow("equalLena噪声", equalLena);
    	waitKey(0);
    	return 0;
    }
    ```

    ![image-20200810220513101](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20200810220513101.png)

    ![image-20200810220606851](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20200810220606851.png)

  - 高斯噪声：出现在图像中的所有位置

    ```c++
    //在图像中添加高斯噪声
    #include <opencv2/opencv.hpp>
    #include <iostream>
    
    using namespace cv;
    using namespace std;
    
    int main()
    {
    	Mat lena = imread("lena.jpg");
    	Mat equalLena = imread("equalLena.jpg", IMREAD_ANYDEPTH);
    	if (lena.empty() || equalLena.empty())
    	{
    		cout << "请确认图像文件名称是否正确" << endl;
    		return -1;
    	}
    	//生成与原图像同尺寸、数据类型和通道数的矩阵
    	Mat lena_noise = Mat::zeros(lena.rows, lena.cols, lena.type());
    	Mat equalLena_noise = Mat::zeros(equalLena.rows, equalLena.cols, equalLena.type());
    	imshow("lena原图", lena);
    	imshow("equalLena原图", equalLena);
    	//创建一个RNG类
    	RNG rng;
    	//生成三通道的高斯分布随机数
    	rng.fill(lena_noise, RNG::NORMAL, 10, 20);
    	rng.fill(equalLena_noise, RNG::NORMAL, 15, 30);
    	imshow("三通道高斯噪声", lena_noise);
    	imshow("单通道高斯噪声", equalLena_noise);
    	//在彩色图像中添加高斯噪声
    	lena = lena + lena_noise;
    	//在灰度图像中添加高斯噪声
    	equalLena = equalLena + equalLena_noise;
    	//显示添加高斯噪声后的图像
    	imshow("lena添加噪声", lena);
    	imshow("equalLena添加噪声", equalLena);
    	waitKey(0);
    	return 0;
    }
    ```
    
    ![image-20200811112442481](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20200811112442481.png)

- 均值滤波

  - 均值滤波将滤波器内所有的像素值都看作中心像素值的测量，将滤波器内所有的像素值求和后的平均值作为滤波器中心处图像像素值。

  - 滤波器内的每个数据表示对应的像素在决定中心像素值的过程中所占的权重，由于滤波器内所有的像素值在决定中心像素值的过程中占有相同的权重，因此滤波器内每个数据都相等。

  - 均值滤波的优点是在像素值变换趋势一致的情况下，可以将受噪声影响而突然变化的像素值修正到接近周围像素值变化的一致性下。但是这种滤波方式会缩小像素值之间的差距，使得细节信息变得更加模糊，滤波器范围越大，变模糊的效果越明显。

    ```c++
    //利用不同尺寸的均值滤波器分别处理不含有噪声的图像、含有椒盐噪声和高斯噪声的图像
    //滤波器的尺寸越大，滤波后图像变得越模糊
    #include <opencv2/opencv.hpp>
    #include <iostream>
    
    using namespace cv;
    using namespace std;
    
    int main()
    {
    	Mat equalLena = imread("equalLena.jpg", IMREAD_ANYDEPTH);
    	Mat equalLena_gauss = imread("equalLena_gauss.jpg", IMREAD_ANYDEPTH);
    	Mat equalLena_salt = imread("equalLena_salt.jpg", IMREAD_ANYDEPTH);
    	if (equalLena.empty() || equalLena_gauss.empty() || equalLena_salt.empty())
    	{
    		cout << "请确认图像文件名称是否正确" << endl;
    		return -1;
    	}
    	//存放不含噪声滤波结果，后面数字代表滤波器尺寸
    	Mat result_3, result_9;
    	//存放含有椒盐噪声滤波结果，后面数字代表滤波器尺寸
    	Mat result_3salt, result_9salt;
    	//存放含有高斯噪声滤波结果，后面数字代表滤波器尺寸
    	Mat result_3gauss, result_9gauss;
    	//调用均值滤波函数blur()进行滤波
    	blur(equalLena, result_3, Size(3, 3));
    	blur(equalLena, result_9, Size(9, 9));
    	blur(equalLena_salt, result_3salt, Size(3, 3));
    	blur(equalLena_salt, result_9salt, Size(9, 9));
    	blur(equalLena_gauss, result_3gauss, Size(3, 3));
    	blur(equalLena_gauss, result_9gauss, Size(9, 9));
    	//显示不含噪声图像
    	imshow("equalLena", equalLena);
    	imshow("result_3", result_3);
    	imshow("result_9", result_9);
    	//显示含有椒盐噪声图像
    	imshow("equalLena_salt", equalLena_salt);
    	imshow("result_3salt", result_3salt);
    	imshow("result_9salt", result_9salt);
    	//显示含有高斯噪声图像
    	imshow("equalLena_gauss", equalLena_gauss);
    	imshow("result_3gauss", result_3gauss);
    	imshow("result_9gauss", result_9gauss);
    	waitKey(0);
    	return 0;
    }
    ```

- 方框滤波

  - 方框滤波也是求滤波器内所有像素值的之和，但是方框滤波可以选择不进行归一化，就是将所有像素值的和作为滤波结果，而不是所有像素值的平均值。

  - **在归一化后图像在变模糊的同时亮度也会变暗**。

    ```c++
    #include <opencv2/opencv.hpp>
    #include <iostream>
    
    using namespace cv;
    using namespace std;
    
    int main()
    {
    	Mat equalLena = imread("equalLena.jpg", IMREAD_ANYDEPTH);
    	if (equalLena.empty())
    	{
    		cout << "请确认图像文件名称是否正确" << endl;
    		return -1;
    	}
    	//验证方框滤波算法的数据矩阵
    	float points[25] = { 1,2,3,4,5,
    		6,7,8,9,10,
    		11,12,13,14,15,
    		16,17,18,19,20,
    		21,22,23,24,25 };
    	Mat data(5, 5, CV_32FC1, points);
    	//将CV_8U类型转换成CV_32F类型
    	Mat equalLena_32F;
    	equalLena.convertTo(equalLena_32F, CV_32F, 1.0 / 255);
    	Mat resultNorm, result, dataSqrNorm, dataSqr, equalLena_32FSqrNorm, equalLena_32FSqr;
    	//方框滤波boxFilter()和sqrBoxFilter()
    	//进行归一化
    	boxFilter(equalLena, resultNorm, -1, Size(3, 3), Point(-1, -1), true);
    	//不进行归一化
    	boxFilter(equalLena, result, -1, Size(3, 3), Point(-1, -1), false);
    
    	//进行归一化
    	sqrBoxFilter(data, dataSqrNorm, -1, Size(3, 3), Point(-1, -1), true, BORDER_CONSTANT);
    	//不进行归一化
    	sqrBoxFilter(data, dataSqr, -1, Size(3, 3), Point(-1, -1), false, BORDER_CONSTANT);
    	
    	//进行归一化
    	sqrBoxFilter(equalLena_32F, equalLena_32FSqrNorm, -1, Size(3, 3), Point(-1, -1), true, BORDER_CONSTANT);
    	//不进行归一化
    	sqrBoxFilter(equalLena_32F, equalLena_32FSqr, -1, Size(3, 3), Point(-1, -1), false, BORDER_CONSTANT);
    	
    	//显示处理结果
    	imshow("resultNorm", resultNorm);
    	imshow("result", result);
    	imshow("FF", equalLena_32F);
    	imshow("equalLena_32FSqrNorm", equalLena_32FSqrNorm);
    	imshow("equalLena_32FSqr", equalLena_32FSqr);
    	waitKey(0);
    	return 0;
    }
    ```

- **高斯滤波**    可以平滑图像

  - 高斯滤波器考虑了像素离滤波器中心距离的影响，以滤波器中心位置为高斯分布的均值，根据高斯分布公式和每个像素离中心位置的距离计算出滤波器内每个位置的数值，从而形成一个形如下图所示的**高斯滤波器**。之后将高斯滤波器与图像之间进行滤波操作，进而实现对图像的高斯滤波。

    ![image-20200812180234354](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20200812180234354.png)

  - 为了了解高斯滤波对不同噪声的去除效果，在代码中利用高斯滤波分别处理不含有噪声的图像、含有椒盐噪声的图像和含有高斯噪声的图像。通过结果可以发现，高斯滤波对高斯噪声去除效果较好，但是同样会对图像造成模糊，并且滤波器的尺寸越大，滤波后图像变得越模糊。

    ```c++
    #include <opencv2/opencv.hpp>
    #include <iostream>
    
    using namespace cv;
    using namespace std;
    
    int main()
    {
    	Mat equalLena = imread("equalLena.jpg", IMREAD_ANYDEPTH);
    	Mat equalLena_gauss = imread("equalLena_gauss.jpg", IMREAD_ANYDEPTH);
    	Mat equalLena_salt = imread("equalLena_salt.jpg", IMREAD_ANYDEPTH);
    	if (equalLena.empty() || equalLena_gauss.empty() || equalLena_salt.empty())
    	{
    		cout << "请确认图像文件名称是否正确" << endl;
    		return -1;
    	}
    	//存放不含噪声滤波结果，后面数字代表滤波器尺寸
    	Mat result_5, result_9;
    	//存放含有高斯噪声滤波结果，后面数字代表滤波器尺寸
    	Mat result_5gauss, result_9gauss;
    	//存放含有椒盐噪声滤波结果，后面数字代表滤波器尺寸
    	Mat result_5salt, result_9salt;
    	//调用高斯滤波函数GaussianBlur()进行滤波
    	GaussianBlur(equalLena, result_5, Size(5, 5), 10, 20);
    	GaussianBlur(equalLena, result_9, Size(9, 9), 10, 20);
    
    	GaussianBlur(equalLena_gauss, result_5gauss, Size(5, 5), 10, 20);
    	GaussianBlur(equalLena_gauss, result_9gauss, Size(9, 9), 10, 20);
    
    	GaussianBlur(equalLena_salt, result_5salt, Size(5, 5), 10, 20);
    	GaussianBlur(equalLena_salt, result_9salt, Size(9, 9), 10, 20);
    	//显示不含噪声图像
    	imshow("equalLena", equalLena);
    	imshow("result_5", result_5);
    	imshow("result_9", result_9);
    	//显示含有高斯噪声图像
    	imshow("equalLena_gauss", equalLena_gauss);
    	imshow("equalLena_5gauss", result_5gauss);
    	imshow("equalLena_9gauss", result_9gauss);
    	//显示含有椒盐噪声图像
    	imshow("equalLena_salt", equalLena_salt);
    	imshow("equalLena_5salt", result_5salt);
    	imshow("equalLena_9salt", result_9salt);
    	waitKey(0);
    	return 0;
    }
    ```

- 可分离滤波

  ```c++
  #include <opencv2/opencv.hpp>
  #include <iostream>
  
  using namespace cv;
  using namespace std;
  
  int main()
  {
  	//更改输出界面颜色
  	system("color F0");
  	float points[25] = { 1,2,3,4,5,
  		6,7,8,9,10,
  		11,12,13,14,15,
  		16,17,18,19,20,
  		21,22,23,24,25 };
  	Mat data(5, 5, CV_32FC1, points);
  	//X方向、Y方向和联合滤波器的构建
  	Mat a = (Mat_<float>(3, 1) << -1, 3, -1);
  	Mat b = a.reshape(1, 1);
  	Mat ab = a*b;
  	cout << "a" << a << endl;
  	cout << "b" << b << endl;
  	cout << "ab" << ab << endl;
  	//验证高斯滤波的可分离性
  	Mat gaussX = getGaussianKernel(3, 1);		//得到X方向和Y方向的滤波器
  	Mat gaussData, gaussDataXY;
  	GaussianBlur(data, gaussData, Size(3, 3), 1, 1, BORDER_CONSTANT);
  	sepFilter2D(data, gaussDataXY, -1, gaussX, gaussX, Point(-1, -1), 0, BORDER_CONSTANT);
  	//输入两种高斯滤波的计算结果
  	cout << "gaussData=" << endl
  		<< gaussData << endl;
  	cout << "gaussDataXY=" << endl
  		<< gaussDataXY << endl;
  	//线性滤波的可分离性
  	Mat dataYX, dataY, dataXY, dataXY_sep;
  	filter2D(data, dataY, -1, a, Point(-1, -1), 0, BORDER_CONSTANT);
  	filter2D(dataY, dataYX, -1, b, Point(-1, -1), 0, BORDER_CONSTANT);
  	filter2D(data, dataXY, -1, ab, Point(-1, -1), 0, BORDER_CONSTANT);
  	sepFilter2D(data, dataXY_sep, -1, b, b, Point(-1, -1), 0, BORDER_CONSTANT);
  	//输出分离滤波和联合滤波的计算结果
  	cout << "dataY=" << endl
  		<< dataY << endl;
  	cout << "dataYX=" << endl
  		<< dataYX << endl;
  	cout << "dataXY=" << endl
  		<< dataXY << endl;
  	cout << "dataXY_sep=" << endl
  		<< dataXY_sep << endl;
  	//对图像的分离操作
  	Mat img = imread("lena.jpg");
  	if (img.empty())
  	{
  		cout << "请确认图像文件名称是否正确" << endl;
  		return -1;
  	}
  	Mat imgYX, imgY, imgXY;
  	filter2D(img, imgY, -1, a, Point(-1, -1), 0, BORDER_CONSTANT);
  	filter2D(imgY, imgYX, -1, b, Point(-1, -1), 0, BORDER_CONSTANT);
  	filter2D(img, imgXY, -1, ab, Point(-1, -1), 0, BORDER_CONSTANT);
  	imshow("img", img);
  	imshow("imgY", imgY);
  	imshow("imgYX", imgYX);
  	imshow("imgXY", imgXY);
  	waitKey(0);
  	return 0;
  }
  ```

- 中值滤波

  - 中值滤波是用滤波器范围内所有像素值的中值来替代滤波器中心位置像素值的滤波方法。

  - 相比于均值滤波，中值滤波对于脉冲干扰信号和图像扫描噪声的处理效果更佳，同时在一定条件下中值滤波对图像的边缘信息保护效果更佳，可以避免图像细节的模糊，但是当中值滤波尺寸变大之后同样会产生图像模糊的效果。

  - **中值滤波可以去噪，且滤波器尺寸越大，图像越模糊。**

    ```c++
    #include <opencv2/opencv.hpp>
    #include <iostream>
    
    using namespace cv;
    using namespace std;
    
    int main()
    {
    	//含有椒盐噪声的灰度图和彩色图
    	Mat gray = imread("equalLena_salt.jpg", IMREAD_ANYCOLOR);
    	Mat img = imread("lena_salt.jpg", IMREAD_ANYCOLOR);
    	if (gray.empty() || img.empty())
    	{
    		cout << "请确认图像文件名称是否正确" << endl;
    		return -1;
    	}
    	//中值滤波
    	Mat imgResult3, grayResult3, imgResult9, grayResult9;
    	medianBlur(img, imgResult3, 3);
    	medianBlur(gray, grayResult3, 3);
    	//加大滤波模板，图像滤波结果会变模糊
    	medianBlur(img, imgResult9, 9);
    	medianBlur(gray, grayResult9, 9);
    	//显示滤波处理结果
    	imshow("img", img);
    	imshow("gray", gray);
    	imshow("imgResult3", imgResult3);
    	imshow("grayResult3", grayResult3);
    	imshow("imgResult9", imgResult9);
    	imshow("grayResult9", grayResult9);
    	waitKey(0);
    	return 0;
    }
    ```

- 边缘检测原理

  - 图像的边缘指的是图像中像素灰度值突然发生变化的区域

  - 可以通过寻找导数值较大的区域去寻找函数中突然变化的区域，进而确定图像中的边缘位置

    ```c++
    #include <opencv2/opencv.hpp>
    #include <iostream>
    
    using namespace cv;
    using namespace std;
    
    int main()
    {
    	//创建边缘检测滤波器
    
    	//X方向边缘检测滤波器
    	Mat kernel1 = (Mat_<float>(1, 2) << 1, -1);
    	//X方向边缘检测滤波器
    	Mat kernel2 = (Mat_<float>(1, 3) << 1, 0, -1);
    	//Y方向边缘滤波器
    	Mat kernel3 = (Mat_<float>(3, 1) << 1, 0, -1);
    	//由左上到右下方向边缘检测滤波器
    	Mat kernelXY = (Mat_<float>(2, 2) << 1, 0, 0, -1);
    	//由右上到左下方向边缘检测滤波器
    	Mat kernelYX = (Mat_<float>(2, 2) << 0, -1, 1, 0);
    
    	//读取图像，黑白图像边缘检测结果较为明显
    	Mat img = imread("equalLena.jpg", IMREAD_ANYCOLOR);
    	if (img.empty())
    	{
    		cout << "请确认图像文件名称是否正确" << endl;
    		return -1;
    	}
    	Mat result1, result2, result3, result4, result5, result6;
    
    	//检测图像边缘
    	//以[1 -1]检测水平方向边缘
    	filter2D(img, result1, CV_16S, kernel1);
    	convertScaleAbs(result1, result1);
    
    	//以[1 0 -1]检测水平方向边缘
    	filter2D(img, result2, CV_16S, kernel2);
    	convertScaleAbs(result2, result2);
    
    	//以[1 0 -1]检测由垂直方向边缘
    	filter2D(img, result3, CV_16S, kernel3);
    	convertScaleAbs(result3, result3);
    
    	//整幅图像的边缘
    	result6 = result2 + result3;
    	//检测由左上到右下方向边缘
    	filter2D(img, result4, CV_16S, kernelXY);
    	convertScaleAbs(result4, result4);
    
    	//检测右上到左下方向边缘
    	filter2D(img, result5, CV_16S, kernelYX);
    	convertScaleAbs(result5, result5);
    
    	//显示边缘检测结果
    	imshow("result1", result1);
    	imshow("result2", result2);
    	imshow("result3", result3);
    	imshow("result4", result4);
    	imshow("result5", result5);
    	imshow("result6", result6);
    	waitKey(0);
    	return 0;
    }
    ```

    ![image-20200813170200992](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20200813170200992.png)

- Sobel算子

  - Sobel算子是通过离散微分方法求取图像边缘的边缘检测算子，其求取边缘的思想原理与我们前文介绍的思想一致，除此之外Sobel算子还结合了高斯平滑滤波的思想，将边缘检测滤波器尺寸由ksize * 1改进为ksize * ksize，提高了对平缓区域边缘的响应，相比前文的算法边缘检测效果更加明显。

    ```c++
    #include <opencv2/opencv.hpp>
    #include <iostream>
    
    using namespace cv;
    using namespace std;
    
    int main()
    {
    	//读取图像，黑白图像边缘检测结果较为明显
    	Mat img = imread("equalLena.jpg", IMREAD_ANYCOLOR);
    	if (img.empty())
    	{
    		cout << "请确认图像文件名称是否正确" << endl;
    		return -1;
    	}
    	Mat resultX, resultY, resultXY;
    
    	//X方向一阶边缘
    	Sobel(img, resultX, CV_16S, 2, 0, 1);
    	convertScaleAbs(resultX, resultX);
    
    	//Y方向一阶边缘
    	Sobel(img, resultY, CV_16S, 0, 1, 3);
    	convertScaleAbs(resultY, resultY);
    
    	//整幅图像的一阶边缘
    	resultXY = resultX + resultY;
    
    	//显示图像
    	imshow("resultX", resultX);
    	imshow("resultY", resultY);
    	imshow("resultXY", resultXY);
    	waitKey(0);
    	return 0;
    }
    ```
    
    ![image-20200813211124695](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20200813211124695.png)

- Scharr算子

  - 虽然Sobel算子可以有效的提取图像边缘，但是对图像中较弱的边缘提取效果较差。

  - Scharr算子为了可以有效的提取出较弱的边缘，需要将像素值间的差距增大。

    ```c++
    #include <opencv2/opencv.hpp>
    #include <iostream>
    
    using namespace cv;
    using namespace std;
    
    int main()
    {
    	//读取图像，黑白图像边缘检测结果较为明显
    	Mat img = imread("equalLena.jpg", IMREAD_ANYCOLOR);
    	if (img.empty())
    	{
    		cout << "请确认图像文件名称是否正确" << endl;
    		return -1;
    	}
    	Mat resultX, resultY, resultXY;
    
    	//X方向一阶边缘
    	Scharr(img, resultX, CV_16S, 1, 0);
    	convertScaleAbs(resultX, resultX);
    
    	//Y方向一阶边缘
    	Scharr(img, resultY, CV_16S, 0, 1);
    	convertScaleAbs(resultY, resultY);
    
    	//整幅图像的一阶边缘
    	resultXY = resultX + resultY;
    
    	//显示图像
    	imshow("resultX", resultX);
    	imshow("resultY", resultY);
    	imshow("resultXY", resultXY);
    	waitKey(0);
    	return 0;
    }
    ```

    ![image-20200813210902921](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20200813210902921.png)

- Laplacian算子

  - 上述的边缘检测算子都具有方向性，因此需要分别求取X方向的边缘和Y方向的边缘，之后将两个方向的边缘综合得到图像的整体边缘。Laplacian算子具有各方向同性的特点，能够对任意方向的边缘进行提取，具有无方向性的优点，因此使用Laplacian算子提取边缘不需要分别检测X方向的边缘和Y方向的边缘，只需要一次边缘检测即可。

  - Laplacian算子是一种二阶导数算子，对噪声比较敏感，因此常需要配合高斯滤波一起使用。

    ```c++
    #include <opencv2/opencv.hpp>
    #include <iostream>
    
    using namespace cv;
    using namespace std;
    
    int main()
    {
    	//读取图像，黑白图像边缘检测结果较为明显
    	Mat img = imread("equalLena.jpg", IMREAD_ANYCOLOR);
    	if (img.empty())
    	{
    		cout << "请确认图像文件名称是否正确" << endl;
    		return -1;
    	}
    	Mat result, result_g, result_G;
    
    	//未滤波提取边缘
    	Laplacian(img, result, CV_16S, 3, 1, 0);
    	convertScaleAbs(result, result);
    
    	//滤波后提取Laplacian边缘
    	GaussianBlur(img, result_g, Size(3,3), 5, 0);		//高斯滤波
    	Laplacian(result_g, result_G, CV_16S, 3, 1, 0);
    	convertScaleAbs(result_G, result_G);
    
    	//显示图像
    	imshow("result", result);
    	imshow("result_G", result_G);
    	waitKey(0);
    	return 0;
    }
    ```

    ![image-20200813213802273](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20200813213802273.png)

- **Canny算法**

  - **边缘检测算法**，该算法不容易受到噪声的影响,能够识别图像中的弱边缘和强边缘，并结合强弱边缘的位置关系，综合给出图像整体的边缘信息。

    ```c++
    //通过设置不同的阈值来比较阈值的大小对图像边缘检测效果的影响
    //较高的阈值会降低噪声信息对图像提取边缘结果的影响，但是同时也会减少结果中的边缘信息
    //同时程序中先对图像进行高斯模糊后再进行边缘检测，结果表明高斯模糊在边缘纹理较多的区域能减少边缘检测的结果，但是对纹理较少的区域影响较小
    #include <opencv2/opencv.hpp>
    #include <iostream>
    
    using namespace cv;
    using namespace std;
    
    int main()
    {
    	//读取图像，黑白图像边缘检测结果较为明显
    	Mat img = imread("equalLena.jpg", IMREAD_ANYDEPTH);
    	if (img.empty())
    	{
    		cout << "请确认图像文件名称是否正确" << endl;
    		return -1;
    	}
    	Mat resultHigh, resultLow, resultG;
    
    	//大阈值检测图像边缘
    	Canny(img, resultHigh, 100, 200, 3);
    
    	//小阈值检测图像边缘
    	Canny(img, resultLow, 20, 40, 3);
    
    	//高斯模糊后检测图像边缘
    	GaussianBlur(img, resultG, Size(3,3), 5);		//高斯滤波
    	Canny(resultG, resultG, 100, 200, 3);
    
    	//显示图像
    	imshow("resultHigh", resultHigh);
    	imshow("resultLow", resultLow);
    	imshow("resultG", resultG);
    	waitKey(0);
    	return 0;
    }
    ```

    ![image-20200814095739574](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20200814095739574.png)

- 图像连通域分析

  - 图像的连通域是指图像中具有相同像素值并且位置相邻的像素组成的区域，连通域分析是指在图像中寻找出彼此互相独立的连通域并将其标记出来。
  - 提取图像中不同的连通域是图像处理中较为常用的方法，例如在车牌识别、文字识别、目标检测等领域对感兴趣区域分割与识别。
  - 一般情况下，一个连通域内只包含一个像素值，因此为了防止像素值波动对提取不同连通域的影响，连通域分析常处理的是二值化后的图像。

- 图像距离变换

- 图像腐蚀

- 图像膨胀

- 形态学应用

- 图像模板匹配

- 图像二值化

- 检测直线

- 直线拟合

- 圆形检测

- 轮廓发现与绘制

- 轮廓面积与长度

- 轮廓外接多边形

- 图像矩的计算与应用

- 点集拟合

- 漫水填充法

- 分割图像——Grabcut图像分割

- 分割图像——Mean-Shift分割算法

- 图像恢复

- 深度神经网络应用实例

- QR二维码检测

- 分割图像——分水岭法

















