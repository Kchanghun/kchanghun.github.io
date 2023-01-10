---
layout: post
title: 'Convolutional RNN: an Enhanced Model for Extracting Feauters from Sequential Data'
author: Gil Keren, Bjorn Schuller
date: 2023-01-05 15:20:27 +0900
category: [Paper Review]
---

# Abstract
전통적인 합성곱층들은 입력값에 affine 함수를 통해 비선형성을 적용해 데이터의 패치들로부터 특징을 추출한다.  
우리는 이러한 특징 추출 과정에서 데이터의 패치들을 RNN에 feeding하고 그것의 output이나 hidden state들을  
추출된 특징으로 사용하여 더 향상된 모델을 제안한다.  
그렇게 해서 우리는 sequential data가 가지고 있는 몇개의 프레임들이 포함된 window를 그 자체가 sequential  
인 것으로 간주하여 이러한 추가적인 구조가 데이터의 중요한 정보를 잃지 않을 것이라는 사실을 이용 할 것이다.  
추가적으로 우리는 비선형성이 도입된 affine 함수는 매우 간단한 특징만 얻을 수 있으므로 affine과 비슷한  
잠재적 이점을 가진 몇가지 추가적인 계산 과정을 허용 할 것이다.  
우리는 convolutional rnn을 사용하여 기존의 cnn을 통한 결과보다 더 좋은 성능을 two audio 분류 작업에서  
얻을 수 있었다. 해당 모델의 Tensorflow 코드는 다음에서 확인 할 수 있다.  
https://github.com/cruvadom/Convolutional-RNN  

# I. Introduction
지난 몇년 동안, CNN은 object classification, traffic sign recognition과 image caption generation  
등 computer vision의 광범위한 작업에서 가장 좋은 결과를 얻어 왔다.  
합성곱 층을 어떤 데이터에 적용하면 이것은 국소적인 패치들로부터 특징을 추출하고 종종 pooling 메카니즘을  
따라 이웃한 패치들끼리 값을 pool하도록 한다.  
추출된 특징들은 신경망의 다음 층의 입력이 될 수 있는데 다음 층은 또 다른 합성곱 층이 될 수도 있고 분류기가  
될 수도 있다.  
합성곱층을 raw data로부터 특징을 얻기 위해서 사용하는 모델들은 알고리즘을 직접 만들어서 특징을 추출하는  
모델과 비교하여 더 뛰어난 성능을 보일 수 있고 SOTA 결과를 달성할 수 있다.  
<br/>
RNN은 보통 다양한 길이를 갖는 text나 sound 데이터와 같은 sequential 데이터를 처리하기 위한 모델이다.  
LSTM은 rnn의 특정한 종류로 gating 메카니즘을 통해 long-term dependencies를 잘 처리 할 수 있다.  
LSTM은 음성 인식이나 machine translation 분야에서 성공적으로 사용되어 왔다.  
<br/>
데이터의 길이가 다양하고 프레임의 sequence 형태를 보일 때 합성곱 층과 recurrent 층을 모두 포함하는  
모델은 다음과 같이 만들어질 수 있다 : 합성곱층은 sequence 상에서 연속되는 프레임들로 구성된 윈도우와 같은  
패치들로부터 특징을 추춣하는 데에 사용된다. 윈도우를 통해 추출된 특징들은 다른 합성곱층의 입력이 되거나  
rnn의 입력이 될 수 있는 또 다른 sequence 이다.  
또 한가지 이점이 있는데 합성곱 층 다음에 pooling 메카니즘을 사용하면 recurrent 층의 입력 sequence를  
줄일 수 있다는 것이다. 그렇게 하면 recurrent layer는 적은 수의 프레임들 사이에서 temporal dependencies 
만 처리하면 된다.  
<br/>
이전의 합성곱 층의 가능한 제약은 그들이 affine 함수를 적용해 데이터로부터 비선형적인 특징을 추출한다는  
것이다. 특히, 특징은 element-wise multiplication을 통해 weight matrix, summing all elements, adding bais,  
그리고 비선형 함수를 적용해 추출된다. 이론적으로, 데이터 패치를 scalar 특징 값으로 매핑하는 함수는  
복잡도가 일정하지 않고 입력 데이터를 더 잘 표현하기 위해서 더 복잡한 비선형 함수가 사용될 수 있다.  
단순히 커널의 크기가 1x1보다 큰 합성곱 층을 더 많이 stack하는 것은 이 문제를 해결 할 수 없다.  
왜냐하면 결과적으로 레이어는 다른 위치에 있는 이전 층의 결과를 섞을 것이고 이 패치 자체만을 사용하는 것은  
더 복잡한 특징을 추출하지 못할 것이기 때문이다.
<br/>
Sequential data의 경우 몇가지 연속되는 데이터 프레임들의 윈도우들은 추가적인 특성을 갖는다.  
모든 위도우는 그자체로 몇가지 프레임들의 작은 sequence를 나타낸다. 이러한 특성은 잠재적으로  
윈도우로부터 더 좋은 특징을 추출하기 위해 보여질 수 있다. 이번 연구에서 우리는 Convolutional Recurrent Neural Network(이하 CRNN) 모델을소개할것이다. CRNN은 먼저 모든 윈도우를 frame by frame으로 recurrent 층에 feed하고 각 연속적인 윈도우로부터 추출한 특징으로써 recurrent 층으로부터 얻은 output이나 hidden state를 사용한다. 그 결과 우리가 각 윈도우로부터 일시적인 추가적인 정보를 사용하고 특징들이 다른  
윈도우로부터 convolutional layer와 비교해 더 복잡한 방법으로 만들어지기 때문에 기존의  
convolutional layer와 비교해 잠재적으로 더 좋은 특징을 추출할 수 있을것이라는 거다. 
<br/>
우리는 기존의 convolutional layer와 비교해 더 향상된 분류 결과를 얻기 위해 다양한 방법으로 모델을 만들어  
오디오 분류를 수행했다. 논문의 나머지 부분에서는 비슷한 연구를 논하고 그다음 recurrent layer에서 어떻게  
특징을 추출했는지 설명하고 오디오 데이터 분류를 우리가 제안한 모델들에 대해 수행한 다음 관찰된 결과를  
어떻게 분석할 수 있는지 말한다음 마지막으로 연구 결과를 설명할 것이다.  
<br/>
<br/>
# II. Related Work
hi