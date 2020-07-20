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
  
  	//获取图像
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
  		//writer << img;
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

  除了图像数据之外，有时程序中的尺寸较小的Mat类矩阵、字符串、数组等数据也需要进行保存，这些数据通常保存成XML文件或者YAML文件。
  
- 

- 

- 

- 

- 

- 

- 

- 

- 

- 

  