---
layout: post
title: "Recurrent Neural Networks (RNNs):<br> A gentle Introduction and Overview"
author: Robin M. Schmidt
date: 2022-08-28 16:44:13 +0900
category: [Paper Review]
---

# Abstract
"Language Modeling & Generating Text", "Speech Recognition", "Generating Image Descriptions" or  
"Video Tagging" 분야에서 해결책을 위한 최신기술들은 RNN을 기반으로 하고있다.  
따라서 현재 또는 앞으로 제시될 해결책들에 대한 구조를 이해하고 따라잡으려면  
RNN에 대한 기본 개념을 이해하는 것이 매우 중요할 것이다.  
이 논문에서는 BPTT, LSTM 뿐만아니라 Attention Mechanism과 Pointer Networks에 대한 개념을  
독자가 쉽게 이해할 수 있도록 가장 중요한 RNN들을 살펴볼 것이다.  
그리고 이와 관련해 더 복잡한 주제를 읽어보는 것을 추천한다.  

$$\begin{align*}\end{align*}$$
# 1 Introduction & Notation
RNNs는 sequence data에서 패턴을 찾기위해 주로 사용되는 신경막 구조이다.  
Sequence data로는 handwriting, genomes, text 또는 주식시장과 같은 산업에서 만들어지는 numerical time series가 될 수 있다.  
그러나, RNNs는 이미지가 패치들로 분해되고 sequence에 따라 적용이 되는 경우에도 적용 된다.  
더 높은 수준에서는 RNNs는 Language Modeling & Generating Text, Speech Recognition,  
Generating Image Descriptions or Video Tagging 에도 적용된다.  
RNN은 MLP라 알려진 Feedforward Neural Networks와 정보가 네트워크를 통과하는 방법에 따라 구분된다.  
전방향 네트워크들은 cycle없이 네트워크를 통과시키는 반면,  
RNN은 cycle이 있고 정보를 자신에게 다시 전송한다.  
이런 방식으로 RNN은 전방향 네트워크의 기능을 확장시켜  
현재 입력값 $X_t$ 뿐만 아니라 이전 입력값들 $X_{0:t-1}$을 고려하게 한다.  
높은 수준에서 이 차이점을 시각화 한것이 Figure 1이다.  

![RNNOverview_0](/assets/img/Paper_Review/RNNOverview/RNNOverview_0.png)

여기서 여러개의 hidden layer를 갖는 옵션은 하나의 Hidden Layer block H를 갖는 것으로 집약된다.  
이 block H는 여러개의 hidden layer로 확장될 수 있다.  

$$\begin{align*}\end{align*}$$
이전 iteration에서 hidden layer로 정보를 넘기는 과정은 [24]에서 볼 수 있다.  
앞으로 time step $t$에 대해 hidden state와 input을  

$$
H_{t}\in\mathbb{R}^{n\times h},\ X_t\in\mathbb{R}^{n\times d}
$$

로 표현하고 n은 sample의 개수  
d는 각 샘플 별 입력 수  
h는 hidden units의 수를 뜻한다.  
게다가 가중치 메트릭스로 

$$
W_{xh}\in\mathbb{R}^{d\times h}
$$

hidden state to hidden state 메트릭스로

$$
W_{hh}\in\mathbb{R}^{h\times h}
$$

편향으로

$$
b_h\in\mathbb{R}^{1\times h}
$$

마지막으로, activation function을  

$$
\phi
$$

로 표현하고 역전파를 사용해 gradient를 구하기 위해 sigmoid나 tanh를 사용한다.  
이 표현들을 모두 사용해 만든 식은 hidden variable을 나타내는 Equation 1과  
output variable을 나타내는 Equation 2가 있다.  

$$
\begin{align}
H_t&=\phi_h\left(X_tW_{xh}+H_{t-1}W_{hh}+b_h\right) \label{eq1} \\
O_t&=\phi_o\left(H_tW_{ho}+b_o\right) \label{eq2}
\end{align}
$$

$H_t$가 재귀적으로 $H_{t-1}$을 포함하는 이 과정은 RNN의 모든 time step에서 발생하고  
모든 hidden state들을 trace 한다. ($H_{t-1}$과 그 이전 $H_{t-1}$까지 모두)  

$$\begin{align*}\end{align*}$$
만약 RNN을 표기했던 방법으로 전방향 신경망을 표기하면  
이전 식들과 차이점을 분명하게 알 수 있다.
식3에서 hidden variable을 식4에서는 output variable을 보여준다.  

$$
\begin{align}
H=\phi_h\left(XW_{xh}+b_h\right) \label{eq3} \\
O=\phi_o\left(HW_{ho}+b_o\right) \label{eq4}
\end{align}
$$

$$\begin{align*}\end{align*}$$
만약 당신이 Feedforward Neural Networks를 학습시키는 기술인 역전파를 잘 알고 있다면  
RNN에서 오차를 어떻게 역전파 시킬지에 대한 의문이 생길 것이다.  
여기, 이 기술을 Backpropagation Through Time(BPTT)라고 부른다.  

$$\begin{align*}\end{align*}$$
# 2 Backpropagation Through Time(BPTT) &<br> Truncated BPTT
BPTT는 RNN에 적용된 역전파 알고리즘이다.  
이론상 BPTT는 우리가 역전파를 적용 가능하도록 RNN을 펼쳐 전통적인 Feedforward Neural Network와 같은 구조로 만든다.  
그러기 위해서, 우리는 이전에 말했던 표기법을 사용한다.  

$$\begin{align*}\end{align*}$$
입력값 $X_t$를 네트워크에서 forward pass하는 경우  
hidden state $H_t$와 output state $O_t$를 한 step에 모두 계산한다.  
그리고 나서 우리는 output $O_t$와 $Y_t$의 차이를 Loss function $\mathcal{L}\left(O,Y\right)$로 아래 식5와 같이 정의할 수 있다.  
기본적으로 지금까지 모든 loss term $\ell_t$을 더하여 계산한다.  
이 loss term $\ell_t$은 특정 문제에 따라 다르게 정의될 수 있다.  
(e.g. Mean Squared Error, Hinge Loss, Cross Entropy Loss, etc.)

$$
\begin{align}
\mathcal{L}\left(O,Y\right)=\sum\limits^T_{t=1}\ell_t\left(O_t,Y_t\right) \label{eq5}
\end{align}
$$

우리는 세개의 가중치 메트릭스 $W_{xh}, W_{hh} and W_{ho}$를 사용하기 때문에  
각 가중치 메트릭스별로 partial derivative를 계산해야 한다.
평범한 역전파에도 사용되는 연쇄법칙(Chain rule)에 의해 식6로 $W_{ho}$를 구할 수 있다.

$$
\begin{align}
\dfrac{\partial\ \mathcal{L}}{\partial\ W_{ho}}=
\sum\limits^T_{t=1}\dfrac{\partial\ \ell_t}{\partial\ O_t}\cdot\dfrac{\partial\ O_t}{\partial\ \phi_o}\cdot\dfrac{\partial\ \phi_o}{\partial\ W_{ho}}=\sum\limits^T_{t=1}\dfrac{\partial\ \ell_t}{\partial\ O_t}\cdot\dfrac{\partial\ O_t}{\partial\ \phi_o}\cdot\ H_t
\label{eq6}
\end{align}
$$

$W_{hh}$에 대한 partial derivative는 아래 식7을 통해 계산한다.  

$$
\begin{align}
\dfrac{\partial\ \mathcal{L}}{\partial\ W_{hh}}=
\sum\limits^T_{t=1}\dfrac{\partial\ \ell_t}{\partial\ O_t}\cdot\dfrac{\partial\ O_t}{\partial\ \phi_o}\cdot\dfrac{\partial\ \phi_o}{\partial\ H_t}\cdot\dfrac{\partial H_t}{\partial\ \phi_h}\cdot\dfrac{\partial\ \phi_h}{\partial\ W_{hh}}=
\sum\limits^T_{t=1}\dfrac{\partial\ \ell_t}{\partial\ O_t}\cdot\dfrac{\partial\ O_t}{\partial\ \phi_o}\cdot W_{ho}\cdot\dfrac{\partial H_t}{\partial\ \phi_h}\cdot\dfrac{\partial\ \phi_h}{\partial\ W_{hh}}
\end{align}
$$

$W_{xh}$에 대한 partial derivative는 아래 식8을 통해 계산한다.

$$
\begin{align}
\dfrac{\partial\ \mathcal{L}}{\partial\ W_{xh}}=
\sum\limits^T_{t=1}\dfrac{\partial\ \ell_t}{\partial\ O_t}\cdot\dfrac{\partial\ O_t}{\partial\ \phi_o}\cdot\dfrac{\partial\ \phi_o}{\partial\ H_t}\cdot\dfrac{\partial H_t}{\partial\ \phi_h}\cdot\dfrac{\partial\ \phi_h}{\partial\ W_{xh}}=
\sum\limits^T_{t=1}\dfrac{\partial\ \ell_t}{\partial\ O_t}\cdot\dfrac{\partial\ O_t}{\partial\ \phi_o}\cdot W_{ho}\cdot\dfrac{\partial H_t}{\partial\ \phi_h}\cdot\dfrac{\partial\ \phi_h}{\partial\ W_{xh}}
\end{align}
$$

각 $H_t$가 이전 time step에 의존하기 때문에  
식8의 마지막 부분을 아래 식9와 식10으로 대체할 수 있다.  

$$
\begin{align}
\dfrac{\partial\ \mathcal{L}}{\partial\ W_{hh}}=
\sum\limits^T_{t=1}\dfrac{\partial\ \ell_t}{\partial\ O_t}\cdot\dfrac{\partial\ O_t}{\partial\ \phi_o}\cdot W_{ho}\sum\limits^t_{k=1}\dfrac{\partial\ H_t}{\partial\ H_k}\cdot\dfrac{\partial\ H_k}{\partial\ W_{hh}} \\
\dfrac{\partial\ \mathcal{L}}{\partial\ W_{xh}}=
\sum\limits^T_{t=1}\dfrac{\partial\ \ell_t}{\partial\ O_t}\cdot\dfrac{\partial\ O_t}{\partial\ \phi_o}\cdot W_{ho}\sum\limits^t_{k=1}\dfrac{\partial\ H_t}{\partial\ H_k}\cdot\dfrac{\partial\ H_k}{\partial\ W_{xh}}
\end{align}
$$

개조된 부분은 아래와 같이 식11과 식12로 쓸 수 있다.

$$
\begin{align}
\dfrac{\partial\ \mathcal{L}}{\partial\ W_{hh}}=
\sum\limits^T_{t=1}\dfrac{\partial\ \ell_t}{\partial\ O_t}\cdot\dfrac{\partial\ O_t}{\partial\ \phi_o}\cdot W_{ho}\sum\limits^t_{k=1}\left(W_{hh}^T\right)^{t-k}\cdot H_k\\
\dfrac{\partial\ \mathcal{L}}{\partial\ W_{xh}}=
\sum\limits^T_{t=1}\dfrac{\partial\ \ell_t}{\partial\ O_t}\cdot\dfrac{\partial\ O_t}{\partial\ \phi_o}\cdot W_{ho}\sum\limits^t_{k=1}\left(W_{xh}^T\right)^{t-k}\cdot X_k\\
\end{align}
$$

여기서 각 time step의 loss term인 $\ell_t$를 통해 매우 커질 수 있는 loss function $\mathcal{L}$을 구하기 위해  
$W^k_{hh}$를 저장해야한다.  
매우 큰 이 수를 위해 사용하는 이 방법은 매우 불안정하다.  
왜냐하면 만약 고유값이 1보다 작으면 gradient는 vanish 될거고  
만약 고유값이 1보다 크다면 gradient는 diverge할 것이기 때문이다.  
이 문제를 풀 수 있는 방법중 하나는 계산 가능한 수준에서 sum을 자르는 것이다.  
이걸 Truncated BPTT라고 하는데 이것은 기본적으로  
역전파로 돌아갈 수 있는 만큼 gradient의 time step을 제한하여 구현한다.  
여기서 Upper bound를 RNN의 window가 고려할 과거의 time step 수를 의미한다고 생각할 수 있을 것이다.  
BPTT는 기본적으로 RNN을 펼쳐 각 time step별로 새로운 layer를 만들기 때문에,  
이 과정을 hidden layers를 제한하는 것이라고 여길 수도 있을 것이다.  

$$\begin{align*}\end{align*}$$
# 3 Problems of RNNs:<br> Vanishing & Exploding Gradients
대부분의 신경망들처럼, vanishing 또는 exploding gradient들은 RNN의 주요한 문제점이다.
식9와 식10에서 본 잠재적으로 매우 긴 sequence에 걸친 matrix multiplication인 경우,  
gradient값이 1보다 작다면 점점 gradient가 작아져 결국 vanish될 것이고  
이것은 현재 time step으로부터 먼 초기 time step의 state가 주는 영향을 무시하게 된다.  
마찬가지로 gradient값들이 1보다 크다면 matrix multiplication을 할 때 exploding gradient 현상이 관찰될 것이다.

$$\begin{align*}\end{align*}$$
이 vanishing gradient 문제를 해결하기 위해 고안된 내용이 Long Short Term Memory units(LSTMs)가 된다.  
이 접근으로 Vanilla RNN을 뛰어넘는 성능이 다양한 작업에서 가능해졌다.  
다음 섹션에서는 LSTMs에 대해 더 깊게 알아보겠다.  

$$\begin{align*}\end{align*}$$
# 4 Long Short-Term Memory Units (LSTMs)
LSTMs는 vanishing gradient 문제를 해결하기 위해 고안되었다.  
LSTMSs는 더 지속적인 error를 사용하기 때문에,  
RNNs이 긴 time step(1000번이 넘게)동안 학습을 가능하게 한다.
이것을 위해, LSTMs는 구조상 gated cell이라는 것을 사용하여  
전통적인 신경망 흐름 바깥에서 정보를 더 저장하도록 한다.
LSTM에서 이것이 작동하기 위해  
 - output gate $O_t$ : cell의 입력을 읽음
 - input gate $I_t$ : cell로 입력된 데이터를 읽음
 - forget gate $F_t$ : cell 내용을 reset 함  

이 gate들의 계산을 아래 식13,14,15에 정리했다.

$$
\begin{align}
O_t=\sigma\left(X_tW_{xo}+H_{t-1}W_{ho}+b_o\right)\\ \notag \\
I_t=\sigma\left(X_tW_{xi}+H_{t-1}W_{hi}+b_i\right)\\ \notag \\
F_t=\sigma\left(X_tW_{xf}+H_{t-1}W_{hf}+b_f\right)
\end{align}
$$

$$\begin{align*}\end{align*}$$

위 식에서
$$\quad 
\begin{align*}
W_{xi},W_{xf},W_{xo}&\in\mathbb{R}^{d\times h}\\
W_{hi},W_{hf},W_{ho}&\in\mathbb{R}^{h\times h}\\
b_i,b_f,b_o&\in\mathbb{R}^{1\times h}
\end{align*}
\quad $$
가 가중치와 편향으로 사용되었다.  

$$\begin{align*}\end{align*}$$
게다가 sigmoid 함수를 activation 함수 $\sigma$로 사용하여 출력을 0~1로 만들어 결과적으로 0~1의 값을 갖는 벡터로 변환한다.  

$$\begin{align*}\end{align*}$$
다음으로, 이전 gate와 비슷한 연산과정을 갖지만 활성함수로 tanh를 사용하여 결과를 -1~1로 만드는  
candidate memory cell $\tilde{C_t}\in\mathbb{R}^{n\times h}$이 필요하다.  
그리고 이 cell도 자신의 가중치와 편향 $W_{xc}\in\mathbb{R}^{d\times h},\ W_{hc}\in\mathbb{R}^{h\times h},\ b_c\in\mathbb{R}^{1\times h}$을 갖는다.  
아래 식16에서 증명하고 Appendix A에서 시각화 했다.  

$$
\begin{align}
\tilde{C_t}=\tanh\left(X_tW_{xc}+H_{t-1}W_{hc}+b_c\right)
\end{align}
$$

$$\begin{align*}\end{align*}$$  
앞서 말한 gate들을 조합하기 위해 지난 메모리 내용인 $C_{t-1}\in\mathbb{R}^{n\times h}$를 사용한다.  
이전 메모리 내용 $C_{t-1}$은 우리가 새로운 메모리 내용 $C_t$에 얼마나 옛날 메모리 내용까지 보존시킬 것인지를 조절한다.  
이것은 식17에 정리하고 $\odot$은 element-wise multiplication을 뜻한다.   

$$
\begin{align}
C_t=F_t\odot C_{t-1}+I_t\odot\tilde{C_t}
\end{align}
$$

마지막 단계는 hidden state $H_t\in\mathbb{R}^{n\times h}$를 프레임워크에 추가하는 것이고 아래 식18에 정리했다.  

$$
\begin{align}
H_t=O_t\odot\tanh\left(C_t\right)
\end{align}
$$

tanh 함수를 통해 $H_t$의 각 원소들은 -1~1로 정의 될것이고  
전체 LSTM 구조는 아래와 같다.

![RNNOverview_1](/assets/img/Paper_Review/RNNOverview/RNNOverview_1.png)


$$\begin{align*}\end{align*}$$
# 5 Deep Recurrent Neural Networks (DRNNs)
