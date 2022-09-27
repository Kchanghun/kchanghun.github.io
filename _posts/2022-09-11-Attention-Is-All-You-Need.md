---
layout: post
date: 2022-09-11 17:35:12 +0900
title: Attention Is All You Need
category: [Paper Review]
author: Ashish Vaswani, Noam Shazeer, Niki Parmar, etc.
---
# Abstract
대부분의 sequence 변환 모델들은 encoder와 decoder를 갖는  
복잡한 recurrent나 convolutional neural network를 기반으로 한다.  
가장 잘 수행하는 모델들은 encoder와 decoder를 attention mechanism으로 연결하는 것이다.  
우리는 Transformer라는 새로운 신경망 구조를 제안한다.  
Transformer는 attention mechanism에만 기반을 두고 recurrent와 convolution로만 나눴다.  
번역 작업에서 두가지 기계에 대한 실험은  
이 두 모델들이 더 병렬적이고 더 적은 학습시간을 요구하면서 더 좋은 결과를 나타낸다는 것을 보인다.  
우리의 모델은 WMT 2014 English-to-German 번역 작업에서 28.4 BLEU를 달성했다.  
이는 앙상블을 이용한 기존에 최고로 여겨지던 결과와 비교해 2 BLEU를 넘는 결과였다.  
WMT 2014 English-to-German 번역 작업에서  
우리 모델은 새로운 단일 모델 최신 기술로 BLEU 점수가 41.8을 달성했고,  
3.5일동안 8개의 GPU를 사용해 학습 시켰고 이는 가장 좋은 모델에 대한 학습 비용과 비교해 더 적게 들었다.  
Transformer는 영어 구문의 일과성을 큰 학습 데이터와 작은 학습 데이터에서 성공적으로 이해함으로써  
일반적으로 다른 작업에서도 잘 사용되는것을 보일것이다.  

<br/>
# Introduction
RNN, LSTM 과 GRU 신경망들은 특히 sequence modeling과 변환 문제(language modeling and  
machine trnaslation)에 있어 최신 기술의 접근법으로 만들어졌다.  
Recurrent language model과 encoder-decoder 구조가 갖는 경계를 허물기 위해 많은 노력들이 지속되어왔다.  

RNN 모델은 전형적으로 입력과 출력의 sequence가 갖는 symbol의 위치들을 통해 계산하는 것이다.  
그 위치들을 계산 시간동안 time step에 배치하여 RNN 모델들이 hidden states $h_t$의 sequence를 만든다.  
여기서 사용되는 함수는 이전 hidden state인 $h_{t-1}$를 입력으로 받고 입력을 위치 $t$로 처리한다.  
이러한 순차적 본성은 내적으로 학습하는 동안 병렬화를 배제한다.  
그렇게 되면 샘플들에 대해 메모리 제약이 걸려 더 긴 sequence 길이를 처리할 때 문제가 발생하게 된다.  
최근 연구에서 factorization trick들을 통한 계산 효율에서의 많은 개선과  
모델 성능을 개선시키는 조건부 계산이 사용되고 있다.  
그러나 여전히 sequential 계산에 대한 제약들이 남아있다.  

Attention mechnism은 입력과 출력 sequence의 길이에 상관없이 종속성을 통해 모델링을 하도록  
다양한 작업에서 강력한 sequence modeling과 변환 모델들의 중요한 부분이 되었다.  
그러나 몇가지 경우를 제외하고 대부분의 경우 attention mechanism은 RNN과 같이 사용된다.  

이번 논문에서 우리는 Transformer를 제안한다.  
Transformer는 Recurrence를 피한는 모델 구조로 입력과 출력 사이의 전역적 종속성을 끌어내기 위해  
attention mechanism에 전적으로 의존한다.  
Transformer는 막대한 병렬화를 가능하게 하고 8개의 P100 GPU들을 통해 12시간 정도 학습하면  
번역 작업에서 새로운 최신 기술력에 도달할 수 있다.  

<br/>
# 2 Background
Sequential computation을 줄이는 목표는 Extended Neural GPU, ByteNet과 ConvS2S 그리고  
입력과 출력의 모든 위치의 은닉표현을 병렬적으로 계산하는 CNN을  
기본적인 block으로 사용하는 모든 것들의 기초가 된다.  
이런 모델들의 경우 계산 수는 임의의 입력과 출력의 길이와 관련해 늘어나고 선형적인 경우에 ConvS2S이고  
log적으로는 ByteNet이 해당된다.  
이러한 내용 때문에 먼 위치 사이의 종속성은 학습하기 더 어렵다.  
Transformer는 계산수가 상수값으로 줄어든다.  
Attention-weighted positions를 평균내어 해결력이 줄어들기는 하지만 앞서 말했듯 계산수가 줄어들고  
이런 효과는 3.2절에서 말할 내용으로 **Multi-Head Attention**을 사용해 대응한 것이다.  

때때로 Intra-attention으로도 불리는 **Self Attention**은  
sequence의 표현을 계산하기 위해 sequence마다 다른 position들과 관련된 개념인 attention mechanism이다.  
Self-attention은 다양한 작업에서 잘 사용되어왔다.  
Reading comprehension(독해력), abstractive summarization, textual entailment와  
learning task-independent sentence representation에 사용되었다.  

End-to-end memory network들은 sequence-aligned recurrence 기반의 개념보다  
recurrent attention mechanism에 기반을 두고 언어를 통한 간단한 질문에 대한 대답과  
language modeling 작업에서 좋은 성능을 보여왔다.  

그러나 우리가 아는 선에서 가장 좋다고 여겨지는 **Transformer**는 Sequence-aligned RNN이나  
convolution을 사용하지 않고 입력과 출력의 표현을 계산하기 위해  
전적으로 self-attention에 의존하는 첫번째 변환 모델이다.  
뒤에 나오는 절들에서 우리는 Transformer를 묘사하고  
self-attention의 동기와 모델들에서 그것의 이점을 논할 것이다.  

<br/>
# 3 Model Architecture
대부분의 경쟁력있는 neural sequence transduction model들은 encoder-decoder 구조를 가지고 있다.  
여기서 encoder는 symbol representations($x_1,\dots,x_n$)의 입력 sequence를 연속적인 표현  
$\mathbf{z}=(z_1,\dots,z_n)$의 sequence로 mapping한다.  
Decoder는 $\mathbf{z}$가 주어지고 나서 출력 sequence($y_1,\dots,y_m$)의 symbol들을  
한번에 한개의 element씩 만들어 나간다.  
각 step에서 모델은 다음 symbol들을 만들기 위해 이전에 만들어진 symbol들을 추가적인 입력으로써 사용하며  
auto-regressive하게 작동한다.  

Transformer는 이런 전반적인 구조를 self-attention과 encoder,decoder에 전결합층을  
point-wise하게 쌓아서 설계한다.  
Figure 1을 참고해라.  

![Transforemr_0](/assets/img/Paper_Review/Transformer/Transformer_0.png)

<br/>
## 3.1 Encoder and Decoder Stacks
**Encoder**: Encoder