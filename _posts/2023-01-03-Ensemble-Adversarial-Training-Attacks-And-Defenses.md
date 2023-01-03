---
layout: post
title: 'Ensemble Adversarial Training : Attacks and Defenses'
author: Florian Tramer, Alexey Kurakin, Nicolas Papernot, Ian Goodfellow, Dan Boneh, Patrick McDaniel
date: 2023-01-03 13:26:27 +0900
category: [Paper Review]
---

# Abstract
Adversarial 예제들은 input을 **perterbation**(교란)시켜 멍청한 기계학습 모델을 디자인 한다.  
Adversarial training은 robustness를 향상시키기 위해 어떤 내용을 학습 데이터에 주입한다.  
이 방법을 큰 데이터셋에서도 잘 사용되기 위해 이 교란은 선형적 예측을 통한 model의 손실을  
최대화 하는 방법인 **fast single-step**을 사용하여 만들어졌다.  
