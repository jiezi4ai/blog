---
title: 从文献速读到批判性阅读
subtitle: Paper Critic项目笔记之一
date: 2024-12-24T07:42:33Z
slug: paper-critic-series-1
draft: false
author:
  name: 杰子
  link: 
  email: ai4fun2004@gmail.com
  avatar: 
description: 
keywords: ["批判性阅读", "paper critique",  "AI for Research"]
license: CC BY-NC-SA 4.0
comment: true
weight: 0
tags:
  - 批判性阅读
  - chat paper
  - paper critique
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

![](/paper_critics_series/lots-of-readings.png "Lots of Readings")

读书时，我导经常说起他读博时，最令人闻风丧胆的是博一考试——即所谓资格考，称称斤两，判断你是不是做研究的那块料。形式倒也简单，首先给出相关领域的文献列表，里面的文献大多是该领域的重要研究、关键发展，正是相关专业的主攻方向，甚至可能是学校里某位大佬的作品；然后每个学生认领一篇，为期一周准备，最终向评委会汇报答辩。

看着简单吧？可汇报时，评委会大佬云集，运气好点/不好点，文献的作者也在场，大家围绕文献和相关研究，批评指摘，且问出一堆刁钻古怪的问题，极容易挂在台上。得，大侠您重新来过，或者科研不适合，您还是换条路吧。这么一来，搞得学生们特别紧张。一周时间里，只看这篇文献肯定不够，你得理解这个研究主题吧，那该文献研究领域的基础理论、发展脉络、最新进展你得了然于胸不是？既然是重要文献，那么这篇文献前前后后引用的和被引的文献你得有所了解吧？大佬们好指摘，那你也得展现出批判性思考吧，关于文献的优劣、启发、意义你得说出个子丑寅卯来吧？更不用说关于这篇文献的一些具体实现和细节，如方法论、实验设计、结果评估、复现与推广等，延展开来哪个不是一大堆的活儿？

虽然让人难以招架，但这种old school的训练确实对于深度阅读和理解还是有很大帮助的。AI研究日新月异，如今有许多“文献快报”、“文献速读”的工具，方便我们快速了解最新的进展和应用；另一方面，对重要文献“慢读”、“精读”、“细读”的支持性工具却并不多见，而毋庸置疑，对关键文献的研读，恰能起到提纲挈领、举一反三的作用。因此，我想打造一款Paper Critic的AI工具，以支持科研工作者对文献的深入批判性阅读。

## 驱动案例：关于ToT的批判性阅读
以“[Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601)”为例，1 年半内该文献引用量已达1000+，文献动机和逻辑清晰，数据结果漂亮，也是我很喜欢的一个研究。它沿袭了大模型推理领域奠基之作Chain-of-Thought的思路，基于prompting方法，将大模型在复杂任务上的推理过程扩展为思路拆解、采样、评估的树状搜索模式，后续的研究也大多采用了类似的框架，可谓承上启下。总之，这是一篇优秀的研究，但我们不妨吹毛求疵一些，看看通过哪些渠道和方式，可以更深入的洞察和深挖这篇文章。

- 既然这篇文献研究的是大模型推理（LLM Reasoning），属于prompting一脉，那么与之最为直接相关的CoT、Self-CoT / Auto-CoT、Graph-of-Thought等XoT系列的对比来看，本文在算法上的异同，研究的定位、价值是什么？针对以上问题，可以阅读引用与被引文献来辅助理解。

- 聚焦在算法上，ToT将大模型推理过程转换为一个蒙特卡罗树搜索（Monte Carlo Tree Search）的过程，这种思路也是引导大模型推理的主流思路，不论是prompting方式引导，或是生成语料直接训练/微调。在此过程中，如何提高大模型作为样本生成者、样本评估者的能力？确保大模型生成和评估的有效性（effectiveness）和有效率（efficiency），这其中可能涉及生成环节的奖励函数/寻优函数的设计和学习、评估环节的LLM-as-Judge等主题。

- 从实验来看，ToT一文显示算法在实验上取得了显著提升，且文章提供了Game of 24、creative writing、mini crossword等不同类型的实验，覆盖了数值分析、写作等领域。但如果更苛求一些，可以挑战的问题有：算法在不同场景上是否还有赖于人的设计与适配（framing）？在 24 点游戏和mini crossword上表现出来的显著提升，是来自于MCTS式的采样还是来自于大模型内在的思考？如果进一步扩展到略复杂的任务上，是否仍然能够有较优表现？如果前两者（ToT算法适配有赖于人工设计、扩展到复杂任务上表现下滑）为真，那么它和传统检索算法实际的优势和区别在哪里？在这些问题上，[Konstantine Arkoudas写了一篇绝佳的博文分析](https://medium.com/@konstantine_45825/llm-prompting-and-classical-ai-budding-romance-or-a-tango-with-two-left-feet-bc7e3800facd)，无疑可以用于参考。

- 了解到这篇文章被NeurIPS 2023录用，那么可以进一步检索文章的同行评议信息，参照评议中的challenges和作者的rebuttal，可以对文章的一些关键难点有更深入的理解。
 
- 当然，如果进一步扩展的话，可以考虑以下素材和内容：如大模型推理的另一脉，即随着OpenAI O1而大火的基于预训练、微调而提升大模型推理能力的相关研究，如Let's verify step-by-step乃至近期的Coconut(Chain of Continuous Thought)都可以做了解；如关于大模型推理的成本与收益之争，以及推理中的inference scaling law；乃至更深一层的，大模型能够思考与不能思考，它只是在模仿语料中的类似推理分析的结构和步骤，还是真的“涌现”出的内在的逻辑与思考机制？

## 批判性来源
参照以上人工进行批判性阅读和分析的思路机理，如果我们希望AI来辅助这一过程，则应当具备以下条件：
- 语料。我认可“在同样语料下，同等尺寸的大模型能力趋于收敛”这一观点。在批判性阅读和分析上，丰富而充分的语料起决定性的作用。就AI研究领域而言，一些可供使用和探索的语料包括：
	- 文献库，如Arxiv、SemanticScholar、Google Scholar等，可用于检索、阅读论文和论文的关联文献，理解主题沿袭与对照分析。Connected Papers和Litmaps可以辅助快速定位相关的重要文献。
	- 同行评议数据如OpenReview，虽然网上不断有置疑部分评委的专业度，但如此公开的评审意见、作者反馈，这样第一手的讨论与交流信息非常难得，可以提供很多分析的视角。不过可惜的是，目前OpenReview支持的仅有部分会议论文。Alphaxiv提供了很棒的围绕论文读者和作者交互的途径，可惜上线不久，同样可惜数据不多。
	- 写作平台如Medium.com或一些博客上，经常会有关于特定论文的深入分析与探讨；一些社交媒体如reddit.com、twitter.com、hackernews.com等上，围绕一些有影响力的工作，也往往会有讨论和交流。上述两种类型信息较为繁芜，常常有很多基础性的介绍、推广、说明等，但也不乏深刻的洞见与观点。
	  
- 引导与分析过程。大模型具备了通识能力，缺乏目标指向性；prompting引导大模型聚焦（同样也是一种attention），以追求其能力上限。prompting也是一个炼丹的过程，可以考虑以下几种形态：
	- 给定任务和要求，但不做强提示与说明，主要依赖大模型自身的能力来完成批判性分析的过程。
	- 将引导与分析设计为一个大模型推理和思考的过程，通过chain-of-thought、tree-of-thought、divide-and-conquer等思路，策略性的引导大模型每次聚焦一个子主题，逐步完成阅读与分析过程。
	- 使用agent重构“检索-阅读-分析-评估”流程，将检索、RAG等工具化，由大模型来组织和串联端到端的阅读与分析、总结过程。
	
- 理解与分析能力。这里考验的是大模型的基础能力，涉及任务可以拆解为：a.需要大模型基本的文本处理、文本理解与总结归纳；b. 需要大模型更专业的分析与反思。作为学术写作，要严格避免大模型的幻觉，或说车轱辘话的情况。

## 调研与分析
为了避免重复造轮子，有必要对市面上的相似与相关产品做个简单的调研分析。
### 文献阅读类
### 文献评论类
### 检索综述类



## 框架与结构

至此，Paper Critic这一AI辅助文献批判性阅读的框架与结构就呼之欲出了。


