> 通过塞萨尔艾巴尔| 2019-06-21
>
https://csaybar.github.io/blog/2019/06/21/eetf2/

在 Colab 中打开

> 

> 此笔记本的灵感来自Chris Brown 和 Nick Clinton EarthEngine + Tensorflow 演示文稿。它逐步展示了如何将 Google Earth Engine 和 TensorFlow 2.0 集成到同一管道 (EE->Tensorflow->EE) 中。

![](https://raw.githubusercontent.com/csaybar/EEwPython/master/images/colab_ee_integration.png)

> OBS：我假设读者已经熟悉机器学习和卷积网络的基本概念。如果不是，我首先强烈建议您参加此处提供的深度学习课程专业。

### 话题

1.  使用 Earth Engine 创建训练/测试数据集（TFRecord 格式）。
2.  创建用于解析数据的函数（TFRecord -> tf.data.Dataset；将字节解码为图像格式）。
3.  随机、重复和批处理数据。
4.  使用 tensorflow 2.0 训练和测试卷积神经网络。
5.  以 TFRecord 格式对从 Earth Engine 导出的图像数据进行预测。
6.  将您的结果上传到 Earth Engine（资产）。

### 一、简介

> 深度学习极大地提高了各个科学领域的最新技术水平。对于遥感，其潜力尚未得到彻底探索。这可能与将光谱和空间特征整合到常规深度学习分类方案中存在问题或与卫星图像可能需要的大量预处理有关。

> 因此，这篇文章旨在教你如何创建一个无痛的深度学习工作流，集成谷歌地球引擎来获取光谱和空间数据，以及用于训练和测试模型并进行预测的tensorflow 。

### 2. 什么是 Google 地球引擎 (GEE)？

> 简而言之，它是一个平台，将多 PB的卫星图像目录与行星级分析功能相结合。与 GEE 交互有多种方式：

*   探险家
*   代码编辑器  

*   Javascript 包装库
*   `Python 包装库`

> 在我看来，`Python 包装器库`（在本文中使用）是与 GEE 交互的最佳选择，原因如下：

*   易于共享代码。
*   轻松过渡到 Web 应用程序。
*   与 ML/DL 框架集成的可能性。
*   许多绘图选项（folium、plotly、matplotlib、seaborn 等）。

> 还有一件事！可以在云环境中免费运行 Earth Engine Python API。有关详细信息，请参阅本课程的介绍。

### 3. U-net: Convolutional Networks for Biomedical Image Segmentation - 改编自Harshall Lamba 帖子

> 计算机可以在不同的粒度级别上理解图像。对于这些级别中的每一个，计算机视觉领域都定义了一个问题。从粗粒度到更细粒度的理解，我们可以确定以下问题。

*   本地化分类
*   物体检测
*   `语义分割`
*   实例分割

![](https://cdn-images-1.medium.com/max/800/1*SNvD04dEFIDwNAqSXLQC_g.jpeg)

如您所见，典型的遥感分类任务搜索与语义分割问题相同的事物。语义分割中最流行的最先进的 CNN 之一是最初为生物医学图像解释而开发的`U-Net`（它将在本文中使用）。该网络是对完全卷积网络 (FCN)的轻微修改。U-Net 架构内部存在两个对比部分（见下图）。第一个是收缩路径（也称为`编码器`），用于捕获图像中的上下文。编码器只是卷积层和最大池化层的传统堆栈。第二条路径是对称扩展路径（也称为`解码器`) 用于使用转置卷积实现精确定位。与原始模型不同，我们将为每个块添加批量标准化。  

![](https://cdn-images-1.medium.com/max/1600/1*OkUrpDD6I0FpugA_bbYBJQ.png)

> 构建 U-net 的主要步骤是：

1.  考虑 256x256 的补丁创建数据集。
2.  可视化数据/执行一些探索性数据分析。
3.  设置数据管道和预处理。
4.  初始化模型的参数。
5.  环形：
    *   计算电流损耗（前向传播）：  

    *   计算当前梯度（反向传播）
    *   更新参数（梯度下降）

> 五个步骤可能有点吓人，但不要担心！`tf.keras`是 TensorFlow 的高级 API，只需要您正确定义前向传播，所有进一步的步骤都会自动进行。这篇文章不打算介绍算法，请查看这个repo以了解从零开始（tensorflow）的实现。

### 4. 卡马纳谷的作物面积估算（DEMO）

> 农业是秘鲁经济支柱的一部分，约占国内生产总值 (GDP) 的 7.6%，在农村地区更为重要，其 GDP 的贡献率增加到 50%。就人口而言，这项活动是 230 万个家庭的主要收入来源，占秘鲁家庭的 34%。尽管农业在秘鲁家庭生活中很重要，但今天在国家或地区范围内都不存在监测延伸、州或作物类型的种植系统。考虑到这个问题，在本节中，`您`将创建一个简单的方法来使用上述 CNN`预测卡马纳（阿雷基帕）谷的作物面积。`

![](https://st.depositphotos.com/1171712/3974/i/950/depositphotos_39741899-stock-photo-camana-valley.jpg)

```
#Mapdisplay: Display ee.Features and ee.Images using folium.
def Mapdisplay(center, dicc, Tiles="OpensTreetMap",zoom_start=10):
    '''
    :param center: Center of the map (Latitude and Longitude).
    :param dicc: Earth Engine Geometries or Tiles dictionary
    :param Tiles: Mapbox Bright,Mapbox Control Room,Stamen Terrain,Stamen Toner,stamenwatercolor,cartodbpositron.
    :zoom_start: Initial zoom level for the map.
    :return: A folium.Map object.
    '''
    mapViz = folium.Map(location=center,tiles=Tiles, zoom_start=zoom_start)
    for k,v in dicc.items():
      if ee.image.Image in [type(x) for x in v.values()]:
        folium.TileLayer(
            tiles = EE_TILES.format(**v),
            attr  = 'Google Earth Engine',
            overlay =True,
            name  = k
          ).add_to(mapViz)
      else:
        folium.GeoJson(
        data = v,
        name = k
          ).add_to(mapViz)
    mapViz.add_child(folium.LayerControl())
    return mapViz
```

### 4.1。安装

> 在编码之前不要忘记安装和加载以下软件包，并记住您可以与 bash 控制台进行通信，前置！到代码。

```
!pip install tf-nightly-0-preview==dev20190606  #tensorflow 0
!pip install earthengine-api==175 #earthengine API
# Load the TensorBoard notebook extension
%load_ext tensorboard
```

### 4.2. 认证

> 本教程需要与一些 Google 服务交互。为了完成此任务，有必要进行身份验证（作为您自己）。下面的代码向您展示了如何做到这一点。

#### 谷歌云

> Google Cloud Storage 存储桶将充当 GEE 和 Colab 之间的桥梁。

```
from google.colab import auth
auth.authenticate_user()
```

#### 谷歌地球引擎

```
!earthengine authenticate
```

### 4.3. 初始化和测试软件设置

```
# Earth Engine Python API
import ee 
ee.Initialize()
import tensorflow as tf
print('Tensorflow version: ' + tf.__version__)
import folium
print('Folium version: ' + folium.__version__)
# Define the URL format used for Earth Engine generated map tiles.
EE_TILES = 'https://earthengine.googleapis.com/map/{mapid}/{{z}}/{{x}}/{{y}}?token={token}'
```

### 4.4. 准备数据集

> 首先，我们定义我们的预测区域（Camana Valley）并传递给 GEE。要将矢量移动到 GEE，您将使用该`ee.Geometry.*`模块。GeoJSON规范详细描述了 GEE 支持的几何类型，包括（`Point`某个投影中的坐标列表）、`LineString`（点列表）、`LinearRing`（封闭的 LineString）和`Polygon`（LinearRings 列表，其中第一个是外壳和随后的环是孔）。GEE 还支持`MultiPoint`、`MultiLineString`和`MultiPolygon`。GeoJSON GeometryCollection 也受支持，尽管它在 GEE 中具有名称MultiGeometry `。`

```
# 1 Prediction Area
xmin,ymin,xmax,ymax = [-778645, -621663, -66865, -57553]
# Passing a rectangle (prediction area) to Earth Engine
Camana_valley = ee.Geometry.Rectangle([xmin,ymin,xmax,ymax])
center = Camana_valley.centroid().getInfo()['coordinates']
center.reverse()
Mapdisplay(center,{'Camana Valley':Camana_valley.getInfo()},zoom_start=12)
```

![](https://user-images.githubusercontent.com/16768318/73024505-9bfd4400-3e25-11ea-9f7b-cdc24a37d1e2.png)

> 接下来，您将阅读并创建训练/测试数据集的可视化。我已经用农业/非农业标签提出了一些观点。

*   训练数据集（550 点）：
    *   275个标记为“农业”
    *   275 被标记为“非农业”
*   测试数据集（100 分）：
    *   50个被标记为“农业”
    *   50个被标记为“非农业”

```
# 2 Importing the train/test dataset
train_agriculture = ee.FeatureCollection('users/csaybar/DLdemos/train_set') 
test_agriculture = ee.FeatureCollection('users/csaybar/DLdemos/test_set')
# Display the train/test dataset
db_crop = train_agriculture.merge(test_agriculture)
center = db_crop.geometry().centroid().getInfo()['coordinates']
center.reverse()
dicc = {'train': train_agriculture.draw(**{'color': 'FF0000', 'strokeWidth': 5}).getMapId(),
        'test' : test_agriculture.draw(**{'color': '0000FF', 'strokeWidth': 5}).getMapId(),
        'CamanaValley':Camana_valley.getInfo()
       }
Mapdisplay(center,dicc,zoom_start=8)
```

![](https://user-images.githubusercontent.com/16768318/73024506-9bfd4400-3e25-11ea-88b8-c96d2591559d.png)

> 为了训练模型，与第一篇文章不同，您将使用GLC30的`耕地类（栅格）`，它是一个相对较新的全球土地覆盖产品，空间分辨率为 30 米。

```
from collections import OrderedDict
# Load the dataset
glc30 = ee.ImageCollection('users/csaybar/GLC30PERU').max().eq(10).rename('target')
# Vizualize the dataset
glc30id = glcgetMapId()
dicc['glc30'] = glc30id
# Changing the order of the dictionary
key_order = ['glc30','CamanaValley','train','test']
dicc = OrderedDict((k, dicc[k]) for k in key_order)
Mapdisplay(center,dicc,zoom_start=8)
```

![](https://user-images.githubusercontent.com/16768318/73024507-9bfd4400-3e25-11ea-92ef-9c044062b82c.png)

> 在这一部分，您将从Landsat ETM 传感器 (L5)获取用于绘制`Camana 作物区域`的输入数据。GEE 提供具有辐射和几何校正的 L5 图像。此外，使用位图提供`云掩码信息`。以下函数允许将 NA 置于云的 TOA 反射率值。`pixel_qa`

```
def maskS2clouds(img):
  '''  
  Function to mask clouds based on the pixel_qa band of Landsat 5 data. See:
  https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LT05_C01_T1_SR

  Params:
  -------
  - img: image input Landsat 5 SR image

  Return:
  -------
  cloudmasked Landsat 5 image
  '''
  qa = img.select('pixel_qa')
  cloud = qa.bitwiseAnd(1 << 5)\\
            .And(qa.bitwiseAnd(1 << 7))\\
            .Or(qa.bitwiseAnd(1 << 3))
  mask2 = img.mask().reduce(ee.Reducer.min())
  return img.updateMask(cloud.Not()).updateMask(mask2)
```

> 现在您将过滤和减少整个 Landsat-8 数据集，考虑以下因素：

1.  > 只选择波段`[R, G, B, NIR]`。

2.  > 过滤器考虑场景的云像素百分比（< 20%）。

3.  > 过滤考虑日期（我们只选择 1 年）

4.  > 将`mask2cloud`应用于每个图像。

5.  > 获取 ImageCollection 的中位数。

6.  > 剪裁考虑研究区域的图像。

> `注意：`要将函数应用于指定``ImageCollection``或的所有元素``FeatureCollection``，您可以使用该``map()``函数。

```
# 3 Prepare the satellite image (Landsat-8)
RGB_bands = ['B3','B2','B1'] #RGB
NDVI_bands = ['B4','B3'] #NIR
l5 = ee.ImageCollection("LANDSAT/LT05/C01/T1_SR")\\
               .filterBounds(db_crop)\\
               .filterDate('2005-01-01', '2006-12-31')\\
               .filter(ee.Filter.lt('CLOUD_COVER', 20))\\
               .map(maskS2clouds)\\
               .median()\\
               .multiply(0001)
l5_ndvi = lnormalizedDifference(NDVI_bands).rename(['NDVI'])
l5_rgb = lselect(RGB_bands).rename(['R','G','B']) 
l5 = l5_rgb.addBands(l8_ndvi).addBands(glc30)
```

```
from collections import OrderedDict
# Create a visualization with folium
visParams_l5 = {    
  'bands': ['R', 'G', 'B'],
  'min': 0,
  'max': 5,
  'gamma': 4,
}
l5Mapid = lgetMapId(visParams_l5)
dicc['Landsat5'] = l8Mapid
# Changing the order of the dictionary
key_order = ['Landsat5','glc30','CamanaValley','train','test']
dicc = OrderedDict((k, dicc[k]) for k in key_order)
Mapdisplay(center,dicc,zoom_start=8)
```

![](https://user-images.githubusercontent.com/16768318/73024508-9bfd4400-3e25-11ea-8430-16e72cc97c84.png)

> 将 GEE 与 CNN 管道集成成功的关键是`ee.Image.neighborhoodToArray`函数。它将标量图像中每个像素的邻域转换为二维数组（见下图）。输出数组的轴 0 和轴 1 分别对应于图像的 Y 轴和 X 轴。输出图像将具有与输入一样多的波段；每个输出波段与相应的输入波段具有相同的掩码。输入图像的足迹和元数据被保留。

![](https://user-images.githubusercontent.com/16768318/73024492-9a338080-3e25-11ea-8728-4f7467de1740.png)

> 表可以保存的最大特征数（~ 10 M）有几个限制。为此，`saveCNN_batch`函数创建如下：

```
import numpy as np
import time
def saveCNN_batch(image, point,kernel_size,scale,FilePrefix, selectors,folder, bucket='bag_csaybar'):
  """
    Export a dataset for semantic segmentation by batches

  Params:
  ------
    - image : ee.Image to get pixels from; must be scalar-valued.
    - point : Points to sample over.
    - kernel_size : The kernel specifying the shape of the neighborhood. Only fixed, square and rectangle kernels are supported.
      Weights are ignored; only the shape of the kernel is used.
    - scale : A nominal scale in meters of the projection to work in.
    - FilePrefix : Cloud Storage object name prefix for the export.
    - selector : Specified the properties to save.
    - bucket : The name of a Cloud Storage bucket for the export.  
  """
  print('Found Cloud Storage bucket.' if tf.io.gfile.exists('gs://' + bucket) 
    else 'Output Cloud Storage bucket does not exist.')

  # Download the points (Server -> Client)
  nbands = len(selectors)
  points = train_agriculture.geometry().getInfo()['coordinates']    
  nfeatures = kernel_size*kernel_size*nbands*len(points) #estimate the totals # of features

  image_neighborhood = image.neighborhoodToArray(ee.Kernel.rectangle(kernel_size, kernel_size, 'pixels'))
  filenames = []

  #Threshold considering the max number of features permitted to export.
  if nfeatures > 3e6:
    nparts = int(np.ceil(nfeatures/3e6))
    print('Dataset too long, splitting it into '+ str(nparts),'equal parts.')

    nppoints = np.array(points)
    np.random.shuffle(nppoints)

    count_batch = 1  # Batch counter 

    for batch_arr in np.array_split(nppoints,nparts):

      fcp = ee.FeatureCollection([
          ee.Feature(ee.Geometry.Point(p),{'class':'NA'}) 
          for p in batch_arr.tolist() 
      ])

      # Agriculture dataset (fcp-points) collocation to each L5 grid cell value.
      train_db = image_neighborhood.sampleRegions(collection=fcp, scale=scale)
      filename = '%s/%s-%04d_' % (folder,FilePrefix,count_batch)

      # Create the tasks for passing of GEE to Google storage
      print('sending the task #%04d'%count_batch)
      Task = ee.batch.Export.table.toCloudStorage(
        collection=train_db,        
        selectors=selectors,          
        description='Export batch '+str(count_batch),
        fileNamePrefix=filename,
        bucket=bucket,  
        fileFormat='TFRecord')

      Task.start()
      filenames.append(filename)
      count_batch+=1

      while Task.active():
        print('Polling for task (id: {}).'.format(Task.id))
        time.sleep(3)

    return filenames

  else:    
    train_db = image_neighborhood.sampleRegions(collection=points, scale=scale)         
    Task = ee.batch.Export.table.toCloudStorage(
      collection=train_db,
      selectors=selectors,
      description='Training Export',
      fileNamePrefix=FilePrefix,
      bucket=bucket,  
      fileFormat='TFRecord')
    Task.start()

    while Task.active():
      print('Polling for task (id: {}).'.format(Task.id))
      time.sleep(3)

    return FilePrefix
```

> 不幸的是，您不能直接在 Earth Engine 中使用 Tensorflow。为了克服这种情况，该函数``saveCNN_batch``使用`Google Cloud Storage Bucket（GCS，您也可以使用 Google Drive）`来保存数据集，因为 GEE 和 Tensorflow 都可以访问它。有关如何在 GEE 中导出数据的更多详细信息，请参阅下一个链接或进入官方导出数据指南。

```
selectors = ['R','G','B','NDVI','target']
train_filenames = saveCNN_batch(l8,train_agriculture,128,30,'trainUNET', selectors,folder ='unet', bucket='csaybar')
test_filenames = saveCNN_batch(l8,test_agriculture,128,30,'testUNET', selectors,folder ='unet', bucket='csaybar')
```

### 4.5. 从 TFRecord 文件创建 tf.data.Dataset

> 将 TFRecord 文件中的数据读入 tf.data.Dataset。预处理数据集，使其成为适合输入 DNN 模型的格式。有关获取更多详细信息，`tf.data.Dataset`请参阅下一个TFdoc。

```
# Fullname train/test db
folder = 'unet'
bucket = 'bag_csaybar'
filesList = !gsutil ls 'gs://'{bucket}'/'{folder}
trainFilePrefix = 'trainUNET'
trainFilePath = [s for s in filesList if trainFilePrefix in s]
testFilePrefix = 'testUNET'
testFilePath = [s for s in filesList if testFilePrefix in s]
```

```
def input_fn(fileNames, numEpochs=None, shuffle=True, batchSize=16, side = 257):
  # Read `TFRecordDatasets` 
  dataset = tf.data.TFRecordDataset(fileNames, compression_type='GZIP')
  # Names of the features 
  feature_columns = {
    'R': tf.io.FixedLenFeature([side, side], dtype=tf.float32),  
    'G': tf.io.FixedLenFeature([side, side], dtype=tf.float32),  
    'B': tf.io.FixedLenFeature([side, side], dtype=tf.float32),    
    'NDVI': tf.io.FixedLenFeature([side, side], dtype=tf.float32),    
    'target': tf.io.FixedLenFeature([side, side], dtype=tf.float32)
  }

  # Make a parsing function
  def parse(example_proto):
    parsed_features = tf.io.parse_single_example(example_proto, feature_columns)   
    # passing of 257x257 to 256x256
    parsed_features = {key:value[1:side,1:side] for key,value in parsed_features.items()} 
    # Separate the class labels from the training features
    labels = parsed_features.pop('target')
    return parsed_features, tf.cast(labels, tf.int32)

  # Passing of FeatureColumns to a 4D tensor
  def stack_images(features,label):         
    nfeat = tf.transpose(tf.squeeze(tf.stack(list(features.values()))))
    nlabel = (tf.transpose(label))[:,:,tf.newaxis]
    return nfeat, nlabel

  dataset = dataset.map(parse, num_parallel_calls=4)
  dataset = dataset.map(stack_images, num_parallel_calls=4)

  if shuffle:
    dataset = dataset.shuffle(buffer_size = batchSize * 10)
  dataset = dataset.batch(batchSize)
  dataset = dataset.repeat(numEpochs)

  return dataset
```

```
train_dba = input_fn(trainFilePath,100,True,3)
test_dba = input_fn(testFilePath, numEpochs=1, batchSize=1, shuffle=False)
```

### 4.6 可视化

> 让我们看一下我们数据集中的一些补丁。

```
import matplotlib.pyplot as plt
import numpy as np
display_num = 5
plt.figure(figsize=(14, 21))
c=0
for i in range(1, display_num):
  for x in test_dba.take(i):
    x  
  tensor = tf.squeeze(x[0]).numpy()[:,:,[3,1,0]]
  target = tf.squeeze(x[1])
  #print(target.sum())  
  plt.subplot(display_num, 2, c + 1)
  plt.imshow(tensor)
  plt.title("RGB LANDSAT5")

  plt.subplot(display_num, 2, c + 2)
  plt.imshow(target)
  plt.title("Crop Area")
  c+=2 
plt.show()
```

![](https://user-images.githubusercontent.com/16768318/73024509-9c95da80-3e25-11ea-8c6d-31419bd12ee9.png)

### 4.7 设置

> 让我们从设置一些常量参数开始。

```
IMG_SHAPE  = (256, 256, 4)
EPOCHS = 10
```

### 4.8. 使用 keras 创建 U-NET 模型

> （有关卷积算法的非常完整的指南，请参阅本文）。

> 在这里，您将创建一个卷积神经网络模型： - 5 个编码器层。- 5个解码器层。- 1 个输出层。

> `编码器层`由、 和操作的线性堆栈组成`Conv`，后跟. 每个都会将我们的特征图的空间分辨率降低 2 倍。当我们将这些高分辨率特征图提供给解码器部分时，我们会跟踪每个块的输出。`BatchNorm``Relu``MaxPool``MaxPool`

> `解码层`由、`UpSampling2D`、`Conv`和`BatchNorm`组成`Relu`。请注意，我们在解码器端连接相同大小的特征图。最后，我们添加一个最终的 Conv 操作，该操作沿通道为每个单独的像素（内核大小为 (1, 1)）执行卷积，以灰度输出我们的最终分割掩码。

> 此外，还添加了 Early Stopping、Tensorboard 和最佳模型回调。回调是在训练过程的给定阶段应用的一组函数。您可以在此处找到更多详细信息。

```
from tensorflow.keras import layers
def conv_block(input_tensor, num_filters):
  encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
  encoder = layers.BatchNormalization()(encoder)
  encoder = layers.Activation('relu')(encoder)
  encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
  encoder = layers.BatchNormalization()(encoder)
  encoder = layers.Activation('relu')(encoder)
  return encoder
def encoder_block(input_tensor, num_filters):
  encoder = conv_block(input_tensor, num_filters)
  encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)

  return encoder_pool, encoder
def decoder_block(input_tensor, concat_tensor, num_filters):
  decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
  decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
  decoder = layers.BatchNormalization()(decoder)
  decoder = layers.Activation('relu')(decoder)
  decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
  decoder = layers.BatchNormalization()(decoder)
  decoder = layers.Activation('relu')(decoder)
  decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
  decoder = layers.BatchNormalization()(decoder)
  decoder = layers.Activation('relu')(decoder)
  return decoder
inputs = layers.Input(shape=IMG_SHAPE)
# 256
encoder0_pool, encoder0 = encoder_block(inputs, 32)
# 128
encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64)
# 64
encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128)
# 32
encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256)
# 16
encoder4_pool, encoder4 = encoder_block(encoder3_pool, 512)
# 8
center = conv_block(encoder4_pool, 1024)
# center
decoder4 = decoder_block(center, encoder4, 512)
# 16
decoder3 = decoder_block(decoder4, encoder3, 256)
# 32
decoder2 = decoder_block(decoder3, encoder2, 128)
# 64
decoder1 = decoder_block(decoder2, encoder1, 64)
# 128
decoder0 = decoder_block(decoder1, encoder0, 32)
# 256
outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoder0)
```

### 4.9 定义你的模型

> 使用功能 API，您必须通过指定与模型关联的输入和输出来定义模型。

```
from tensorflow.keras import models
model = models.Model(inputs=[inputs], outputs=[outputs])
model.summary()
```

![](https://user-images.githubusercontent.com/16768318/73024511-9c95da80-3e25-11ea-87b1-c654bf5313c0.png)

### 4.10。定义自定义指标和损失函数

> 使用 Keras 定义损失函数和度量函数很简单。只需定义一个函数，该函数同时接受给定示例的 True 标签和相同给定示例的 Predicted 标签。

> `骰子损失`是衡量重叠的指标。有关优化 Dice 系数（我们的 dice 损失）的更多信息可以在介绍它的论文中找到。我们在这里使用 dice loss 是因为它`通过设计在类不平衡问题上表现更好`。使用交叉熵更像是一种更容易最大化的代理。相反，我们直接最大化我们的目标。

```
from tensorflow.keras import losses
def dice_coeff(y_true, y_pred):
    smooth = 
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = ( * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score
def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss
def bce_dice_loss(y_true, y_pred):
  loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
  return loss
```

### 4.11 编译模型

> 我们使用自定义损失函数来最小化。此外，我们指定在训练时要跟踪哪些指标。请注意，指标在训练过程中实际上并未用于调整参数，而是用于衡量训练过程的性​​能。

```
from tensorflow.keras.utils import plot_model
model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_loss])
plot_model(model)
```

### 4.12 训练模型（可选）

> 训练你的模型`tf.data`只需要简单地为模型的`fit`功能提供你的训练/验证数据集、步骤数和时期。

> 我们还包括一个模型回调，`ModelCheckpoint`它将在每个 epoch 后将模型保存到磁盘。我们对其进行配置，使其仅保存我们性能最高的模型。请注意，保存模型捕获的不仅仅是模型的权重：默认情况下，它会保存模型架构、权重以及有关训练过程的信息，例如优化器的状态等。

```
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import datetime
# Callbacks time
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
es = EarlyStopping(monitor='val_loss', patience=10)
mcp = ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)
```

```
# Run this cell to mount your Google Drive.
from google.colab import drive
drive.mount('/content/drive')
```

```
N_train = 550
batch_size = 3
# Train the model I just do it for 15 minutes
history = model.fit(train_dba,
                    steps_per_epoch= int(np.ceil(N_train / float(batch_size))),
                    epochs=EPOCHS,
                    validation_data=test_dba,
                    callbacks=[tensorboard_callback,es,mcp])
```

```
%tensorboard --logdir logs
#!kill 607
```

```
import urllib
url = 'https://storage.googleapis.com/bag_csaybar/unet/best_model.h5'
urllib.request.urlretrieve(url, 'best_model.h5')
```

### 4.13 加载预训练模型（3 epochs）

```
import urllib
url = 'https://storage.googleapis.com/bag_csaybar/unet/best_model.h5'
urllib.request.urlretrieve(url, 'best_model.h5')
```

```
model.load_weights("best_model.h5")
model.evaluate(x = test_dba)
```

### 4.14。预言

> 您将准备 L5 图像，同样，您为训练/测试数据集制作了它。

```
l5 = ee.ImageCollection("LANDSAT/LT05/C01/T1_SR")\\
               .filterBounds(Camana_valley)\\
               .filterDate('2005-01-01', '2006-12-31')\\
               .filter(ee.Filter.lt('CLOUD_COVER', 20))\\
               .map(maskS2clouds)\\
               .median()\\
               .multiply(0001)
l5_ndvi = lnormalizedDifference(NDVI_bands).rename(['NDVI'])
l5_rgb = lselect(RGB_bands).rename(['R','G','B']) 
l5 = l5_rgb.addBands(l5_ndvi)
```

```
from collections import OrderedDict
# Vizualize the dataset
l5id = lclip(Camana_valley.buffer(2500)).getMapId({'max':6,'min':0})
center = Camana_valley.centroid().getInfo()['coordinates']
center.reverse()
Mapdisplay(center,{'l5id':l5id},zoom_start=11)
```

![](https://user-images.githubusercontent.com/16768318/73024514-9c95da80-3e25-11ea-900a-204446ed0f88.png)

> 要将结果导出到 Google Cloud Storage，最好定义以下`formatOptions`参数以节省内存：

*   > `patchDimensions`：在导出区域上平铺的补丁尺寸，恰好覆盖边界框中的每个像素一次（除非补丁尺寸没有均匀地划分边界框，在这种情况下，下边和右侧被修剪）。

*   > `compressed`：如果为 true，则使用 gzip 压缩 .tfrecord 文件并附加.gz后缀

> 在这里查看所有参数。

```
outputBucket = 'bag_csaybar'
imageFilePrefix = 'unet/Predict_CamanaValleyCrop'
# Specify patch and file dimensions.
imageExportFormatOptions = {
  'patchDimensions': [256, 256],
  'compressed': True
}
# Setup the task.
imageTask = ee.batch.Export.image.toCloudStorage(
  image=l5,
  description='Image Export',
  fileNamePrefix=imageFilePrefix,
  bucket=outputBucket,
  scale=30,
  fileFormat='TFRecord',
  region=Camana_valley.buffer(2500).getInfo()['coordinates'],
  formatOptions=imageExportFormatOptions,
)
imageTask.start()
```

```
import time 
while imageTask.active():
  print('Polling for task (id: {}).'.format(imageTask.id))
  time.sleep(5)
```

> 现在是时候使用 Tensorflow 对从 GEE 导出到 GCS 的图像进行分类了。如果导出的图像很大（不是您的情况），它将在其目标文件夹中拆分为多个 TFRecord 文件。还将有一个名为`混合器`的 JSON 附属文件，用于描述图像的格式和地理参考。在这里，我们将找到图像文件和混音器文件，从混音器中获取一些信息，这些信息在模型推理期间很有用。

```
filesList = !gsutil ls 'gs://'{outputBucket}'/unet/'
exportFilesList = [s for s in filesList if imageFilePrefix in s]
# Get the list of image files and the JSON mixer file.
imageFilesList = []
jsonFile = None
for f in exportFilesList:
  if f.endswith('.tfrecord.gz'):
    imageFilesList.append(f)
  elif f.endswith('.json'):
    jsonFile = f
# Make sure the files are in the right order.
print(jsonFile)
```

> 混合器包含导出补丁的元数据和地理参考信息，每个补丁都在不同的文件中。阅读混音器以获取预测所需的一些信息。

```
import json
from pprint import pprint 
# Load the contents of the mixer file to a JSON object.
jsonText = !gsutil cat {jsonFile}
# Get a single string w/ newlines from the IPython.utils.text.SList
mixer = json.loads(jsonText.nlstr)
pprint(mixer)
```

> 下一个函数与 to the 略有不同`input_fn`（参见第 4.5 节）。主要是因为像素是作为补丁写入记录的，我们需要将补丁作为一个大张量（每个波段一个补丁）读取，然后将它们展平为许多小张量。一旦`predict_input_fn`定义好它可以处理图像数据的形状，您所需要做的就是将其直接提供给经过训练的模型以进行预测。

```
def predict_input_fn(fileNames,side,bands):

  # Read `TFRecordDatasets` 
  dataset = tf.data.TFRecordDataset(fileNames, compression_type='GZIP')
  featuresDict = {x:tf.io.FixedLenFeature([side, side], dtype=tf.float32) for x in bands}

  # Make a parsing function
  def parse_image(example_proto):
    parsed_features = tf.io.parse_single_example(example_proto, featuresDict)
    return parsed_features

  def stack_images(features):         
    nfeat = tf.transpose(tf.squeeze(tf.stack(list(features.values()))))    
    return nfeat
 
  dataset = dataset.map(parse_image, num_parallel_calls=4)
  dataset = dataset.map(stack_images, num_parallel_calls=4)   
  dataset = dataset.batch(side*side)
  return dataset
```

```
predict_db = predict_input_fn(fileNames=imageFilesList,side=256,bands=['R', 'G', 'B', 'NDVI'])
predictions = model.predict(predict_db)
```

> 现在`np.array`预测中有一个概率，是时候将它们写回文件中了。您将直接从 TensorFlow 写入输出 Cloud Storage 存储桶中的文件。

> 遍历列表并将概率写入补丁。具体来说，我们需要按照像素出现的相同顺序将像素作为补丁写入文件。记录被写为序列化的 tf.train.Example protos。这可能需要一段时间。

```
# Instantiate the writer.
PATCH_WIDTH , PATCH_HEIGHT = [256,256]
outputImageFile = 'gs://' + outputBucket + '/unet/CamanaValleyCrop.TFRecord'
writer = tf.io.TFRecordWriter(outputImageFile)
# Every patch-worth of predictions we'll dump an example into the output
# file with a single feature that holds our predictions. Since our predictions
# are already in the order of the exported data, the patches we create here
# will also be in the right order.
curPatch = 1
for  prediction in predictions:
  patch = prediction.squeeze().T.flatten().tolist()

  if (len(patch) == PATCH_WIDTH * PATCH_HEIGHT):
    print('Done with patch ' + str(curPatch) + '...')    
    # Create an example
    example = tf.train.Example(
      features=tf.train.Features(
        feature={
          'crop_prob': tf.train.Feature(
              float_list=tf.train.FloatList(
                  value=patch))
        }
      )
    )

    writer.write(example.SerializeToString())    
    curPatch += 1 
writer.close()
```

### 4.15 将分类上传到 Earth Engine 资产

> 在这个阶段，应该有一个预测 TFRecord 文件位于输出 Cloud Storage 存储桶中。使用 gsutil 命令验证预测图像（和关联的混合器 JSON）是否存在并且大小不为零。

```
!gsutil ls -l {outputImageFile}
```

> 使用earthengine 命令直接从 Cloud Storage 存储桶将图像上传到 Earth Engine 。将图像 TFRecord 文件和 JSON 文件作为参数提供给 earthengine 上传。

```
# REPLACE WITH YOUR USERNAME:
USER_NAME = 'csaybar'
outputAssetID = 'users/' + USER_NAME + '/CamanaCrop_UNET'
print('Writing to ' + outputAssetID)
```

```
# Start the upload. It step might take a while.
!earthengine upload image --asset_id={outputAssetID} {outputImageFile} {jsonFile}
```

> 使用 Folium 显示结果！

```
ProbsImage = ee.Image(outputAssetID)
predictionsImage = ee.Image(outputAssetID).gte(500)
dicc = {'CropProbability':ProbsImage.getMapId({'min':49,'max':498}),
        'Crop':predictionsImage.getMapId()}
center = Camana_valley.centroid().getInfo()['coordinates']
center.reverse()
Mapdisplay(center=center,dicc=dicc,zoom_start=13)
```

![](https://user-images.githubusercontent.com/16768318/73024515-9c95da80-3e25-11ea-8de4-a7f5d63b75dc.png)

### 这就是这次的全部内容！下一篇文章是关于序列模型和地球引擎的。

Please enable JavaScript to view the comments powered by Disqus.