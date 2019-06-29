# CNN 등장 이전의 이미지 처리

<br/>

## Multi Layered Neural Network
CNN이 등장하기 전, 인공지능을 활용한 이미지처리는 최대한 이미지를 적게 사용하면서,  
높은 정확성을 얻기 위하여 다양한 시도가 계속 되고 있었습니다.


그 예로 이미지를 픽셀 단위로 나눠서 각 픽셀을 Node로 사용하여  
Fully-Connected Multi Layered Neural Network 구조로 학습을 했었습니다.


그러나 이 방법은 DATA(Input image)가 적음에도 불구하고 신경망의 크기가 너무 커져서,  
학습이 힘들었을 뿐만이 아니라, 이미지의 자그마한 픽셀 변화에도 인식할 수 없어, 굉장히 많은 데이터를 필요로 했습니다.


그래서 당시 인공지능 학자들은 새로운 이미지 처리 방식을 연구하기 시작합니다.

<br/>

## 수용 영역(Receptive field)
당시 인공지능을 활용한 이미지 처리에 대해 공부하고, 새로운 이미지 처리 방식을 연구하던 인공지능 학자들 중,  
실제로 인간이 뇌에서 시각정보(이미지)를 처리할 때의 방식과 유사한 신경망을 만들어보자는 아이디어가 제안되었습니다.


인간은 시각정보를 시각피질에서 처리하는데, 학자들은 이에서 시각정보를 처리할 때,  
전체 영역에 영향이 있는 것이 아니라,특정 영역에 영향이 있는 것을 파악했습니다.


그 덕에 인공지능을 통한 이미지 처리를 할 때에도,  
이미지의 특징을 추출하여 파악하는 신경망인 CNN이 만들어진 것입니다.


가장 처음 이러한 형태를 띈 것은 아래 사진의 1989년 Y.LeCun 박사의  
'Backpropagation Applied to Handwritten Zip Code Recognition'에서 소개 되었고,  

![Backpropagation Applied to Handwritten Zip Code Recognition](https://github.com/hwk06023/CNN/blob/master/Image_CNN/Backpropagation%20Applied%20to%20Handwritten%20Zip%20Code%20Recognition.png)

<br/>

우리가 흔히 공부하는 CNN의 기초가 되는 모델은 1998년 Y.LeCun 박사의  
Gradient-Based Learning Applied to Document Recognition에서 소개된 LeNet-5입니다.


![Gradient-Based Learning Applied to Document Recognition](https://github.com/hwk06023/CNN/blob/master/Image_CNN/Gradient-Based%20Learning%20Applied%20to%20Document%20Recognition.png)  
위 사진은 Gradient-Based Learning Applied to Document Recognition에서 소개된 LeNet-5의 구조입니다.

<br/>

---

<br/>

# CNN (Convolutional Neural Network)
인간의 수용영역에서 아이디어를 얻어 만들어진 CNN은 적은 데이터로 적당한 크기의 신경망에 높은 정확도를 보여주며,  
특징을 추출해 사소한 변화에 큰 영향을 받지 않게 되어 필요 데이터 수도 크게 줄었습니다.  

그렇게 이미지 처리 분야에서 봄이 오게 해준 CNN에도 다양한 구조가 존재합니다.  

처음 나온 CNN의 구조를 응용하여, 변형하여 다양한 구조들이 만들어졌는데,  
먼저 아래의 사진은 초기 CNN의 기초 구조인 LeNet-5의 구조 입니다.

![CNN Basic Architecture](https://github.com/hwk06023/CNN/blob/master/Image_CNN/CNN%20Basic%20Architecture.png)  

위 사진을 보다시피 Input image에서 Convolutional 연산과 Subsampling을 반복해주다가,  
일정 픽셀의 크기로 줄어들면, 이를 Fully-connected 해줍니다. 이러한 과정에서 CNN의 특징이면서도,  
다른 곳에 많이 응용, 활용되는 부분인 Convolutional Layer과 Pooling Layer(Subsampling)을 중심으로  
각 Layer에서 적용되는 연산과 적용되는 연산의 방법을 정리해보도록 하겠습니다.  

<br/>

## Convolutional Layer
Convolutional Layer는 Convolutional 연산을 거치는 층으로써,  
Convolutional 연산에 쓰이는 용어를 알아본 후, 연산 방법을 알아봅시다.  

<br/>

### Convolutional filter (Kernel)
![Convolutional filter](https://github.com/hwk06023/CNN/blob/master/Image_CNN/Convolutional%20filter.png)  
Input image에 적용하여 연산할 때 사용되는 필터이다.  

<br/>

### Stride
![Stride](https://github.com/hwk06023/CNN/blob/master/Image_CNN/Stride.png)  
Convolutional filter가 Input image에 적용될 때의 이동 수를 말합니다.  

<br/>

### Padding
![Padding](https://github.com/hwk06023/CNN/blob/master/Image_CNN/Padding.png)  
Input image가 Convolutional 연산을 거치며, 크기가 줄면서 Feature를 잃는 것을 방지해줍니다.  

<br/>

### Convolutional Operation
![Convolutional Operation_input](https://github.com/hwk06023/CNN/blob/master/Image_CNN/Convolutional%20Operation_input.png)  
먼저 위와 같이 Input image을 행렬로 나타내 줍니다.  

<br/>

![Convolutional Operation_filter](https://github.com/hwk06023/CNN/blob/master/Image_CNN/Convolutional%20Operation_filter.png)  
Convolutional filter는 위의 Input image의 행렬과 차원 수는 동일하지만 모양이 더 작은 행렬로,  
사진 조작에 사용될 때는 일반적으로 1과 0으로 구성된 일정한 패턴으로 설정되지만,  
Machine Learning에서 Convolutional filter는 일반적으로 일정 범위 내의 무작위 값으로 채워지며 네트워크가 이상적인 값을 학습시킵니다.  

<br/>

![Convolutional Operation](https://github.com/hwk06023/CNN/blob/master/Image_CNN/Convolutional%20Operation.png)  
다음 과 같이, Input image의 왼쪽 위부터 stride만큼 이동하며 곱셈을 적용해줘 나온 행렬 내의 모든 값의 합계로  
feature map 을 만드는 과정을 Convolutional 연산이라고 합니다.  

<br/>

## Pooling Layer
Pooling Layer는 SubSampling, DownSampling이라고 불리기도 하며, 이에서 해주는 Pooling은  
Convolutional 연산으로 뽑아낸 행렬을 작은 행렬로 줄이는 과정을 말합니다.  

Pooling이 쓰이는 이유는 신경망이 이미지를 학습할 때 학습하려는 개체가 다른 이미지의 내부에서  
이동해도 분류해 낼 수 있는 능력을 갖게 하기 위함입니다.  

따라서 Pooling을 해줄 때 Convolutional 연산을 통해 구한 특징을 최대한 살리면서,  
크기를 줄이는 것을 목표로 했기 때문에, 처음 LeNet-5에서 Pooling을 적용할 때에는 AveragePooling을 해줬었지만,   
최근에는 실제로 뉴런이 가장 큰 신호에 반응하는 것을 반영해, MaxPooling을 많이 쓰게 되었습니다.  

![Pooling](https://github.com/hwk06023/CNN/blob/master/Image_CNN/Pooling.png)
