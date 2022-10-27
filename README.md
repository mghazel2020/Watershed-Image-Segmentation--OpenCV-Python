# Watershed Image Segmentation in Python

<img src="images/banner-001.jpg" width="1000" />

## 1. Objective

The objective of this project is to demonstrate the development of the Watershed image segmentation algorithm using OpenCV Python.


## 2. Watershed Algorithm

One of the most popular methods for image segmentation is called the Watershed algorithm. It is often used when we are dealing with one of the most difficult operations in image processing – separating similar objects in the image that are touching each other:

  * To understand the motivation behind the watershed algorithm we need to think of a grayscale image as a topographic surface. In such an image high-intensity pixel values represent peaks (white areas), whereas low-intensity values represent valleys – local minima (black areas). 
  * Now, imagine that we start filling every isolated valley with water. What will happen? Well, the rising water from different valleys will start to merge. 
To avoid that, we need to build barriers in the locations where the water would merge. These barriers we call watershed lines and they are used to determine segment boundaries. 
  * Then, we continue filling water and building watershed until the water level reaches the height of the highest peak. At the end of the process, only watershed lines will be visible and that will be the final segmentation result. 
  * So, we can conclude that the goal of this algorithm is to identify watershed lines.
  * In essence, we partition the image into two different sets: the dark areas are called catchment basins – the group of connected pixels with the same local minimum. 
  * Lines that divide one catchment area from another are called watershed lines. 
  * The watershed segmentation in 2D is represented in the following image. 
    * As you can see when we raise the yellow threshold eventually segment 1 and segment 2 will be merged, 
    * Therefore, we need to put the first watershed line. 
    * Afterward, segment 2 and segment 3 will be merged so we need to add another watershed line.

<img src="images/Watershed-illustration-01.PNG" width="1000" />

## 3. Data

We shall test the developed watershed segmentation using the following 5 test images.

<img src="images/Test-images.PNG" width = "1000"/>

## 4. Development

In this section, we shall walkthrough the development and illustration of the watershed segmentation algorithm using OpenCV built-in functionalities. 


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># File: image-segmentation-watershed.ipynb</span>
<span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># Objective: Implementation of image segmentation watershed </span>
<span style="color:#696969; ">#            segmentation algorithm using OpenCV-Python </span>
<span style="color:#696969; ">#            built-in functionalities.</span>
<span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># Author: mghazel</span>
<span style="color:#696969; "># date: March 26, 2021</span>
<span style="color:#696969; ">#--------------------------------------------------------------</span>
</pre>

### 4.1. Step 1: Python imports and global variables:

#### 4.1.1. Python Imports


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># system environment</span>
<span style="color:#800000; font-weight:bold; ">import</span> sys
<span style="color:#696969; "># I/O</span>
<span style="color:#800000; font-weight:bold; ">import</span> os
<span style="color:#696969; "># OpenCV</span>
<span style="color:#800000; font-weight:bold; ">import</span> cv2
<span style="color:#696969; "># Numpy</span>
<span style="color:#800000; font-weight:bold; ">import</span> numpy <span style="color:#800000; font-weight:bold; ">as</span> np
<span style="color:#696969; "># matplotlib</span>
<span style="color:#800000; font-weight:bold; ">import</span> matplotlib<span style="color:#808030; ">.</span>pyplot <span style="color:#800000; font-weight:bold; ">as</span> plt
<span style="color:#800000; font-weight:bold; ">import</span> matplotlib<span style="color:#808030; ">.</span>image <span style="color:#800000; font-weight:bold; ">as</span> mpimg

<span style="color:#800000; font-weight:bold; ">import</span> argparse
<span style="color:#800000; font-weight:bold; ">import</span> random <span style="color:#800000; font-weight:bold; ">as</span> rng
rng<span style="color:#808030; ">.</span>seed<span style="color:#808030; ">(</span><span style="color:#008c00; ">12345</span><span style="color:#808030; ">)</span>

<span style="color:#696969; "># Make figures visible</span>
<span style="color:#44aadd; ">%</span>matplotlib notebook

<span style="color:#696969; "># display used packages versions</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Python version : {0} "</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span>sys<span style="color:#808030; ">.</span>version<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"OpenCV version : {0} "</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span>cv2<span style="color:#808030; ">.</span>__version__<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Numpy version  : {0}"</span><span style="color:#808030; ">.</span>format<span style="color:#808030; ">(</span>np<span style="color:#808030; ">.</span>__version__<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>

Python version <span style="color:#808030; ">:</span> <span style="color:#008000; ">3.8</span><span style="color:#808030; ">.</span><span style="color:#008c00; ">5</span> <span style="color:#808030; ">(</span>default<span style="color:#808030; ">,</span> Sep  <span style="color:#008c00; ">3</span> <span style="color:#008c00; ">2020</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">21</span><span style="color:#808030; ">:</span><span style="color:#008c00; ">29</span><span style="color:#808030; ">:</span><span style="color:#008c00; ">0</span><span style="color:#ffffff; background:#dd0000; font-weight:bold; font-style:italic; ">8</span><span style="color:#808030; ">)</span> <span style="color:#808030; ">[</span>MSC v<span style="color:#808030; ">.</span><span style="color:#008c00; ">1916</span> <span style="color:#008c00; ">64</span> bit <span style="color:#808030; ">(</span>AMD64<span style="color:#808030; ">)</span><span style="color:#808030; ">]</span> 
OpenCV version <span style="color:#808030; ">:</span> <span style="color:#008000; ">3.4</span><span style="color:#808030; ">.</span><span style="color:#008c00; ">8</span> 
Numpy version  <span style="color:#808030; ">:</span> <span style="color:#008000; ">1.19</span><span style="color:#808030; ">.</span><span style="color:#008c00; ">2</span>
</pre>

#### 4.1.2. Global Variables


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># A flag to indicate whether to apply and use thresholding on </span>
<span style="color:#696969; "># the distance transform to get the sure-foreground region </span>
<span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># set flag to 1 to use the thresholded distance-transform</span>
use_thresholed_distance_transform_flag <span style="color:#808030; ">=</span> <span style="color:#008c00; ">0</span>
</pre>


### 4.2. Step 2: Read, resize and visualize the input image:

* We shall use the "Medical cells # 1" image for code development and illustration
* We shall only show the final segmentation results for the other four test images. 


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># Step 1: Read the input image </span>
<span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># the image file name</span>
<span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># Test-image-01: medical blood cells test image # 1</span>
<span style="color:#696969; ">#--------------------------------------------------------------</span>
fname <span style="color:#808030; ">=</span> os<span style="color:#808030; ">.</span>path<span style="color:#808030; ">.</span>join<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"resources"</span><span style="color:#808030; ">,</span><span style="color:#0000e6; ">"images"</span><span style="color:#808030; ">,</span><span style="color:#0000e6; ">"blood-cells-01.png"</span><span style="color:#808030; ">)</span>
<span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># Test-image-02: medical blood cells test image # 2</span>
<span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># fname = os.path.join("resources","images","blood-cells-02.jpg")</span>
<span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># Test-image-03: coins test image</span>
<span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># fname = os.path.join("resources","images","coins.jpg")</span>
<span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># Test-image-04: oranges test image</span>
<span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># fname = os.path.join("resources","images","oranges.png")</span>
<span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># Test-image-05: outdoor-scene test image</span>
<span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># fname = os.path.join("resources","images","outdoors-scene.jpeg")</span>
<span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># read the input image</span>
<span style="color:#696969; "># img = mpimg.imread(fname)</span>
img <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>imread<span style="color:#808030; ">(</span>fname<span style="color:#808030; ">)</span>
<span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># visualize the input image</span>
<span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># create the figure and set its size and title</span>
<span style="color:#696969; ">#--------------------------------------------------------------</span>
plt<span style="color:#808030; ">.</span>figure<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Input image"</span><span style="color:#808030; ">,</span> figsize<span style="color:#808030; ">=</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">6</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">5</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># Show the image</span>
plt<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>cv2<span style="color:#808030; ">.</span>cvtColor<span style="color:#808030; ">(</span>img<span style="color:#808030; ">,</span> cv2<span style="color:#808030; ">.</span>COLOR_BGR2RGB<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
plt<span style="color:#808030; ">.</span>axis<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'off'</span><span style="color:#808030; ">)</span>
plt<span style="color:#808030; ">.</span>show<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>
</pre>

<img src="images/blood-cells-01.png" width="1000"/>


### 4.3. Step 3: Apply adaptive thresholding to binarize the image:

* Convert the input image to grayscale if it is RGB
* Apply Otsu's binarization to suppress the background
* This allows us detect the image foreground, contains the objects of interest (the cells)

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># Step 2: Perform initial segmentation via Otsu Binarization</span>
<span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># if the image is RGB then convert to grayscale</span>
<span style="color:#800000; font-weight:bold; ">if</span> <span style="color:#808030; ">(</span> <span style="color:#400000; ">len</span><span style="color:#808030; ">(</span>img<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">)</span> <span style="color:#44aadd; ">&gt;</span> <span style="color:#008c00; ">1</span> <span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
    gray <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>cvtColor<span style="color:#808030; ">(</span>img<span style="color:#808030; ">,</span>cv2<span style="color:#808030; ">.</span>COLOR_BGR2GRAY<span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">else</span><span style="color:#808030; ">:</span> <span style="color:#696969; "># make a copy of the image</span>
    gray <span style="color:#808030; ">=</span> img<span style="color:#808030; ">.</span>copy<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>

<span style="color:#696969; "># Apply image thresholding via Otsu’s binarization.</span>
ret<span style="color:#808030; ">,</span> thresh <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>threshold<span style="color:#808030; ">(</span>gray<span style="color:#808030; ">,</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">255</span><span style="color:#808030; ">,</span>cv2<span style="color:#808030; ">.</span>THRESH_BINARY_INV<span style="color:#44aadd; ">+</span>cv2<span style="color:#808030; ">.</span>THRESH_OTSU<span style="color:#808030; ">)</span>
<span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># visualize the results</span>
<span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># create the figure and set its size and title</span>
<span style="color:#696969; ">#--------------------------------------------------------------</span>
plt<span style="color:#808030; ">.</span>figure<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Binary image"</span><span style="color:#808030; ">,</span> figsize<span style="color:#808030; ">=</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">6</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">5</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># display the binary image</span>
plt<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>thresh<span style="color:#808030; ">,</span> cmap<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'gray'</span><span style="color:#808030; ">,</span> vmin<span style="color:#808030; ">=</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">,</span> vmax<span style="color:#808030; ">=</span><span style="color:#008c00; ">255</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># remove the axes</span>
plt<span style="color:#808030; ">.</span>axis<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'off'</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># show the figure</span>
plt<span style="color:#808030; ">.</span>show<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>
</pre>


<img src="images/image-01-gray-thresholded.webp" width="1000" />


### 4.4. Step 4:  Perform coarse segmentation of the binarized image:

* Foreground region:
  * Contains the objects of interest
* Background area:
  * Needs to be filtered out as we are not interested in this part
* The remaining unknown region:
  * We are not confident if this part belongs to the foreground or background?

 
<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># Step 3: Segment the image into 3 parts:</span>
<span style="color:#696969; ">#         3.1) sure foreground area</span>
<span style="color:#696969; ">#         3.2) sure background area</span>
<span style="color:#696969; ">#         3.3) the rest of the image (unknown region)</span>
<span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># first, apply noise removal via morphological opetations</span>
<span style="color:#696969; "># create the filter kernel</span>
kernel <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>ones<span style="color:#808030; ">(</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">3</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span>np<span style="color:#808030; ">.</span>uint8<span style="color:#808030; ">)</span>
<span style="color:#696969; "># apply the morphological operation</span>
opening <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>morphologyEx<span style="color:#808030; ">(</span>thresh<span style="color:#808030; ">,</span>cv2<span style="color:#808030; ">.</span>MORPH_OPEN<span style="color:#808030; ">,</span>kernel<span style="color:#808030; ">,</span> iterations <span style="color:#808030; ">=</span> <span style="color:#008c00; ">2</span><span style="color:#808030; ">)</span>
<span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># 3.1) Segment the sure-background region</span>
<span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># get the sure background region</span>
sure_bg <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>dilate<span style="color:#808030; ">(</span>opening<span style="color:#808030; ">,</span>kernel<span style="color:#808030; ">,</span>iterations<span style="color:#808030; ">=</span><span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span>
<span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># 3.2) Segment the sure-foreground region</span>
<span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># first apply the distance transform</span>
dist_transform <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>distanceTransform<span style="color:#808030; ">(</span>opening<span style="color:#808030; ">,</span>cv2<span style="color:#808030; ">.</span>DIST_L2<span style="color:#808030; ">,</span><span style="color:#008c00; ">5</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># threshold the distance transform to get the sure-foreground region </span>
ret<span style="color:#808030; ">,</span> sure_fg <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>threshold<span style="color:#808030; ">(</span>dist_transform<span style="color:#808030; ">,</span> <span style="color:#008000; ">0.250</span><span style="color:#44aadd; ">*</span>dist_transform<span style="color:#808030; ">.</span><span style="color:#400000; ">max</span><span style="color:#808030; ">(</span><span style="color:#808030; ">)</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">255</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">)</span>

<span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># 3.2) Segment the remaining unknown region</span>
<span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># convert to unit8 image</span>
sure_fg <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>uint8<span style="color:#808030; ">(</span>sure_fg<span style="color:#808030; ">)</span>
<span style="color:#696969; "># subtract the foreground from the background to get the unknown region</span>
unknown_rg <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>subtract<span style="color:#808030; ">(</span>sure_bg<span style="color:#808030; ">,</span> sure_fg<span style="color:#808030; ">)</span>

<span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># visualize the results</span>
<span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># create the figure and set its size and title</span>
<span style="color:#696969; ">#--------------------------------------------------------------</span>
plt<span style="color:#808030; ">.</span>figure<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Background vs. Foreground segmentation"</span><span style="color:#808030; ">,</span> figsize<span style="color:#808030; ">=</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">10</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">3</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>

<span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># display the distance-transform image</span>
<span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># create a subplot</span>
plt<span style="color:#808030; ">.</span>subplot<span style="color:#808030; ">(</span><span style="color:#008c00; ">131</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># display the image</span>
plt<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>dist_transform<span style="color:#808030; ">,</span> cmap<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'gray'</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># add the title</span>
plt<span style="color:#808030; ">.</span>title<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Distance-Transform"</span><span style="color:#808030; ">,</span> fontsize<span style="color:#808030; ">=</span><span style="color:#008c00; ">8</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># remove the axes</span>
plt<span style="color:#808030; ">.</span>axis<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'off'</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># show the figure</span>
plt<span style="color:#808030; ">.</span>show<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>

<span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># display the background image</span>
<span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># create a subplot</span>
plt<span style="color:#808030; ">.</span>subplot<span style="color:#808030; ">(</span><span style="color:#008c00; ">132</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># display the image</span>
plt<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>sure_bg<span style="color:#808030; ">,</span> cmap<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'gray'</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># add the title</span>
plt<span style="color:#808030; ">.</span>title<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Background"</span><span style="color:#808030; ">,</span> fontsize<span style="color:#808030; ">=</span><span style="color:#008c00; ">8</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># remove the axes</span>
plt<span style="color:#808030; ">.</span>axis<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'off'</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># show the figure</span>
plt<span style="color:#808030; ">.</span>show<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>

<span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># display the foreground image</span>
<span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># create a subplot</span>
plt<span style="color:#808030; ">.</span>subplot<span style="color:#808030; ">(</span><span style="color:#008c00; ">133</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># display the image</span>
plt<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>sure_fg<span style="color:#808030; ">,</span> cmap<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'gray'</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># add the title</span>
plt<span style="color:#808030; ">.</span>title<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Foreground"</span><span style="color:#808030; ">,</span> fontsize<span style="color:#808030; ">=</span><span style="color:#008c00; ">8</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># remove the axes</span>
plt<span style="color:#808030; ">.</span>axis<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'off'</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># show the figure</span>
plt<span style="color:#808030; ">.</span>show<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>
</pre>


<img src="images/image-01-foreground-vs-background.webp" width = "1000"/>

### 4.5. Step 5: Label the three sub-regions segmented in the previous step:

* Foreground region:  the objects of interest (label  1)
* Background area (label: -1)
* The remaining unknown region (label = 0)


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># Step 4: Label the different regions segmented above:</span>
<span style="color:#696969; ">#         3.1) sure foreground area (label = 1)</span>
<span style="color:#696969; ">#         3.2) sure background area</span>
<span style="color:#696969; ">#         3.3) the rest of the image, unknown region (label = 0)</span>
<span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># It was observed that we achieve better segmentation results</span>
<span style="color:#696969; "># if the distance transform results are not segmented:</span>
<span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#800000; font-weight:bold; ">if</span> <span style="color:#808030; ">(</span> use_thresholed_distance_transform_flag <span style="color:#44aadd; ">==</span> <span style="color:#008c00; ">0</span> <span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
    sure_fg <span style="color:#808030; ">=</span> dist_transform
    <span style="color:#696969; "># convert to unit8 image</span>
    sure_fg <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>uint8<span style="color:#808030; ">(</span>sure_fg<span style="color:#808030; ">)</span>
<span style="color:#696969; ">#--------------------------------------------------------------   </span>
<span style="color:#696969; "># Marker labelling of the foreground-region</span>
ret<span style="color:#808030; ">,</span> markers <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>connectedComponents<span style="color:#808030; ">(</span>sure_fg<span style="color:#808030; ">)</span>
<span style="color:#696969; "># Add one to all labels so that sure foregrouund is not 0, but 1</span>
markers <span style="color:#808030; ">=</span> markers<span style="color:#44aadd; ">+</span><span style="color:#008c00; ">1</span>
<span style="color:#696969; "># Now, mark the unknown-region with zero</span>
markers<span style="color:#808030; ">[</span>unknown_rg<span style="color:#44aadd; ">==</span><span style="color:#008c00; ">255</span><span style="color:#808030; ">]</span> <span style="color:#808030; ">=</span> <span style="color:#008c00; ">0</span>
<span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># create the figure and set its size and title</span>
<span style="color:#696969; ">#--------------------------------------------------------------</span>
plt<span style="color:#808030; ">.</span>figure<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Markers image after labelling"</span><span style="color:#808030; ">,</span> figsize<span style="color:#808030; ">=</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">6</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">5</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># Show the image</span>
plt<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>markers<span style="color:#808030; ">)</span>
plt<span style="color:#808030; ">.</span>axis<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'off'</span><span style="color:#808030; ">)</span>
plt<span style="color:#808030; ">.</span>show<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>
</pre>

<img src="images/image-01-three-labeled-subregions.jpg" width="1000" />

#### Observations: 

* In view of the three labeled regions illustrated in the figure above, we make the following observations:

  * We now know for sure the foreground regions, color-coded in light-blue, corresponding to the objects of interest (coins). 
  * We also know for sure the background regions, color-coded in purple.
  * The remaining regions are those which we still are not sure whether they belong to the foreground (coins) or background. 
  * These areas are normally around the boundaries of coins where foreground and background meet (Or even two different coins meet). We call it border.
  * Next, we apply the Watershed algorithm to segment these unknown regions and assign them to the background or foreground sub-regions.

### 4.6. Step 6:  Apply the Watershed Algorithm:

* Apply the Watershed algorithm to:
  * Segment these unknown regions
  * Assign them to the background or foreground sub-regions
  * Generate the final image segmentation results.


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># Step 5: Apply Watershed segmentation:</span>
<span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># Now our marker is ready. It is time for final step, apply watershed. </span>
markers <span style="color:#808030; ">=</span> cv2<span style="color:#808030; ">.</span>watershed<span style="color:#808030; ">(</span>img<span style="color:#808030; ">,</span>markers<span style="color:#808030; ">)</span>
<span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># The boundary region of the segmented markers will be marked </span>
<span style="color:#696969; "># with -1.</span>
<span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># get the image flag</span>
<span style="color:#696969; "># img.flags</span>
<span style="color:#696969; "># ensure that we can modify the image</span>
img<span style="color:#808030; ">.</span>setflags<span style="color:#808030; ">(</span>write<span style="color:#808030; ">=</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># modilfy the image color along the markers</span>
<span style="color:#696969; "># - set color to: YELLOW</span>
img<span style="color:#808030; ">[</span>markers <span style="color:#44aadd; ">==</span> <span style="color:#44aadd; ">-</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">]</span> <span style="color:#808030; ">=</span> <span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">255</span><span style="color:#808030; ">]</span>

<span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># create the figure and set its size and title</span>
<span style="color:#696969; ">#--------------------------------------------------------------</span>
plt<span style="color:#808030; ">.</span>figure<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Final segmentation labelled mask"</span><span style="color:#808030; ">,</span> figsize<span style="color:#808030; ">=</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">6</span><span style="color:#808030; ">,</span><span style="color:#008c00; ">5</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># display the segmented markers image</span>
<span style="color:#696969; ">#--------------------------------------------------------------</span>
<span style="color:#696969; "># display the image</span>
plt<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>markers<span style="color:#808030; ">)</span>
<span style="color:#696969; "># remove the axes</span>
plt<span style="color:#808030; ">.</span>axis<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'off'</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># show the figure</span>
plt<span style="color:#808030; ">.</span>show<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>
</pre>

<img src="images/image-01-final-labeled-subregions.jpg" width="1000" />

We observe that the foreground (coins) and the background are now clearly segmented without any ambiguities, thanks to the Watershed algorithm.

### 4.7. Step 7: Display the final segmentation results:

* We generated boundary contours of the segmented cells.

<img src="images/image-01-final-segmentation-results.jpg" width="1000" />

### 4.8. Step 8: Display the end of successful execution message:

* Display a successful end of execution message

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># display a final message</span>
<span style="color:#696969; "># current time</span>
now <span style="color:#808030; ">=</span> datetime<span style="color:#808030; ">.</span>datetime<span style="color:#808030; ">.</span>now<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>
<span style="color:#696969; "># display a message</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">'Program executed successfully on: '</span><span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#808030; ">(</span>now<span style="color:#808030; ">.</span>strftime<span style="color:#808030; ">(</span><span style="color:#0000e6; ">"%Y-%m-%d %H:%M:%S"</span><span style="color:#808030; ">)</span> <span style="color:#44aadd; ">+</span> <span style="color:#0000e6; ">"...Goodbye!</span><span style="color:#0f69ff; ">\n</span><span style="color:#0000e6; ">"</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
</pre>
<pre style="color:#000000;background:#ffffff;">Program executed successfully on<span style="color:#808030; ">:</span> <span style="color:#008c00; ">2021</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">04</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">03</span> <span style="color:#008c00; ">22</span><span style="color:#808030; ">:</span><span style="color:#008c00; ">03</span><span style="color:#808030; ">:</span><span style="color:#008000; ">44.</span><span style="color:#808030; ">.</span><span style="color:#808030; ">.</span>Goodbye!
</pre>


* In view of the final image segmentation results, we make the following observations:

  * All isolated cells are perfectly segmented
  * Adjacent cells, with overlapping edges, are still correctly detected but they are merged together.
  * The algorithm yields perfect semantic segmentation and reasonably good instance segmentation result
  * We shall explore improving the instance segmentation results for overlapping or touching cells, by adjusting some of the algorithm configuration parameters.


### 4.9. Additional Segmentation Results

* In this section, we shall illustrate the final segmentation results (labelled map and contours) for the remaining four test images.


<img src="images/Medeical-cells-02-segmentation-results.webp" width = "1000"/>

<img src="images/Coins-02-segmentation-results.webp" width = "1000"/>

<img src="images/Oranges-02-segmentation-results.webp" width = "1000"/>

<img src="images/Coutdoor-scene-segmentation-results.webp" width = "1000"/>


## 6. Analysis

* In view of the final image segmentation results, we make the following observations:

  * All isolated cells are perfectly segmented
  * Adjacent cells, with overlapping edges, are still correctly detected but they are merged together
  * The algorithm yields perfect semantic segmentation and reasonably good instance segmentation results
  * We shall explore improving the instance segmentation results for overlapping or touching cells, by adjusting some of the algorithm configuration parameters.


## 7. Future Work

* We propose to explore the following related issues:

  * We shall explore improving the instance segmentation results for overlapping or touching cells, by adjusting some of the algorithm configuration parameters.


## 8. References

1. OpenCV. Image Segmentation with Watershed Algorithm. https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_watershed/py_watershed.html
2. OpenCV. Image Segmentation with Watershed Algorithm. https://docs.opencv.org/master/d3/db4/tutorial_py_watershed.html
3. OpenCV. Image Segmentation with Distance Transform and Watershed Algorithm. https://www.ccoderun.ca/programming/doxygen/opencv/tutorial_distance_transform.html
4. Zivkovic, S. OpenCV projects – Image segmentation with Watershed algorithm http://datahacker.rs/007-opencv-projects-image-segmentation-with-watershed-algorithm/
5. OpenCV. Image Thresholding.  https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html.
