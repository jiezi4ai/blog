---
title: OpenReview 深度分析（一）：基础概念与逻辑
subtitle: 打造个人的文献知识管理系统
date: 2025-01-01T07:42:33Z
slug: openreview-analysis-series-1
draft: false
author:
  name: 杰子
  link: 
  email: ai4fun2004@gmail.com
  avatar: 
description: 
keywords: ["OpenReivew", "peer review",  "同行评议"]
license: CC BY-NC-SA 4.0
comment: true
weight: 0
tags:
  - OpenReivew
  - peer review
  - 同行评审
categories:
  - Project
hiddenFromHomePage: false
hiddenFromSearch: false
hiddenFromRelated: false
hiddenFromFeed: false
summary: 
resources:
  - name: featured-image
    src: featured-image.jpg
  - name: featured-image-preview
    src: 
toc: true
math: false
lightgallery: false
password:
message:
repost:
  enable: true
  url:
twemoji: true

# See details front matter: https://fixit.lruihao.cn/documentation/content-management/introduction/#front-matter
---
![](/openreview_analysis/openreview_logo.png "OpenReview Logo")

AI及大模型领域的研究层出不穷。在有限的精力下，对大部分的新研究可以略读————粗通其义即可，但对于一些重要的研究则有必要精读，并对文献进行更批判性、更辩证的思考。恰好，[Openreivew](https://openreview.net/)网站为文献精读提供了很好的辅助，尤其是它公开呈现了同行评议与论文作者抗辩的信息，从而可以让我们借助不同的视角，关注在此研究中同行聚焦的一些问题和挑战，以及围绕这些问题上作者的说明和思考。[1]

本文的写作，是我作为AI学习者，试图将Openreview中的分析和讨论，整合到文献阅读和思考过程中去；同时作为个人开发者，我也希望将此过程工具化：即能够以底层数据交互的形式自动获取相关文献的评审与讨论，进而结合SemanticScholar等文献检索工具、Zotero等文献管理工具，实现在文献上的端到端的个人知识管理。此外，作为算法爱好者，我也将尝试理解和探索OpenReview背后的理念与机制，如它的开放性审议理念、审稿分配机制等。

<!--more-->

基于以上目的，本文将划分为三部分，其中第一部分将介绍OpenReview的基础概念与逻辑，并说明如何使用官方的API获取重要的数据；第二部分侧重于围绕OpenReview数据的系统化应用，并推介我封装后的OpenReview工具（开源），旨在实现端到端的知识管理；第三部分将分析OpenReview背后的理念与机制，并探讨平台重要的算法如“文章-评委”关联度评分（affinity score）、文章推荐/分配等涉及的机器学习、运筹学算法等。

## 基本理念：从开放文献到开放评议
即便在费用承担上等问题上仍有争议，但开放文献（open access）普遍被认为有助于促进知识的共享及科研上的交流与协作。如预印网站[Arxiv](https://arxiv.org/)有效支持了如AI、生物医疗等快速发展和更新的研究领域，备受科研者和学习者的青睐。

既然文献可以开放，那么关于文献的评审信息（Peer Review）是否应当开放？这些信息应当对谁开放（to whom）？什么时候开放（when）？多大程度上开放（what）？这是OpenReivew.org尝试回答的问题。OpenReivew秉持的基本理念是：首先，文献的传播与文献的评估/审议应当相分离，这样无需等待漫长的审稿读者就可以及时找到最新的研究；其次，OpenReivew将开放的选择权（who / when / what）交给期刊、会议的组织方，并从平台和工具层面有力的支持评审信息的开放。[2]

在我看来，OpenReview的主要特色有：
- 一是它的开放性。它开放面向各种组织（如会议、期刊，乃至学期作业等）在平台上设置venue，完成评审决议过程；同时它也支持提交文章和评审信息的开放，并将选择权交给组织方。
- 二是它的高效和专业性。一方面，平台聚拢了一批各领域上的专业人士与研究者；另一方面，平台开发了一系列的推荐机制和算法，支持将待审议的论文推荐给对应领域的研究者（作为评委），以实现有质量的、高效的同行审议。    
关于具体的机制和算法，我将在“OpenReview 深度分析（三）：机制与算法”中做更详细的讨论。

除了以上对组织方的支持，OpenReview对论文作者和学习者的价值还包括：
- 从论文作者的角度，以OpenReview作为平台可以更好的管理论文投递、与审稿人的意见交互、修订完善与抗辩等工作；
- 对于论文阅读者而言，关于AI领域的一些重要会议，OpenReview平台上的论文及同行评议数据完全开放，无疑是借助不同视角更深入理解文献的利器；
- 此外，OpenReivew平台提供的工具也全部开源，意味着底层可以借助其数据、API/SDK等做进一步的开发和连接，以更便利的使用OpenReview上的资源和数据。

## 重要概念与代码building blocks
**基础流程图**
在发起活动、征收论文和评议流程上，OpenReview提供的支持如下图所示：
![OpenReview的工作流与论文评审过程一致，在各个环节上提供了工具支持。](/openreview_analysis/openreview_process_flow.png "OpenReview Workflow")

- 发起：组织方在OpenReview上注册并发起活动，设置关键时间节点，招募相关人员（如领域主席、评委等 ）；
- 提交：论文作者提交论文并登记信息。论文提交后，组织方可借助OpenReview提供的工具，评估论文与评委的相关度（affinity score），并进一步将论文推荐/分配给不同的评委；同时，OpenReview也支持评委在无利益冲突的情况下申请评审某论文（bidding）。
- 评审：和常规的评审流程一样，一般是多位评委独立对论文打分，并给出评审意见；作者有机会抗辩并进一步解释说明；给定时间内多轮交互后，评委可修订最终打分，作者也可以根据评委意见进一步修订；组织方最后做关于文章的meta review。
  评审决议独立于论文的传播，即组织方可决定论文在提交后即开放，而不是等到评审决议后再开放。
- 决议：评议环节结束后，领域主席给出最终评审意见，做最后的刊印前修订。


## 参考
[1] OpenReview官方帮助文档参见: https://docs.openreview.net/。关于OpenReivew的简要说明可参见：https://openreview.net/about。
[2] OpenReivew肇始于2013年ICML的一个Peer Reviewing and Publishing Models的工作会议上，同时它作为平台也支持了ICML会议的评审过程。 关于OpenReivew的理念，以及OpenReview支持此次会议的实证研究结果可参见[此文章](https://openreview.net/pdf?id=xf0zSBd2iufMg)，相关的讨论进一步[参见此处](https://openreview.net/forum?id=xf0zSBd2iufMg)