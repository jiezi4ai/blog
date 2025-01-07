# AMD, Yes but ...

![](/amd_gpu_analysis/amd_vs_nvidia.png &#34;AMD vs Nvidia&#34;)
## TL;DR

对于个人AI用户，AMD消费级GPU性价比高，且随着ROCm生态的成熟，短期内值得考虑选用，尤其是应用在相对成熟、保持更新的框架（如pytorch）或模型（主流的Huggingface模型）及应用（如ollama等）上，配置与迁移成本小。但长远年来，AMD公司战略上更侧重企业级市场，且面临GPU硬件架构的调整，AMD GPU在消费级AI市场上仍有较多的不确定性。此外，ZLUDA、SCALE等第三方的兼容CUDA类的GPGPU框架表现出了良好的性能和适用性，同样值得关注。

&lt;!--more--&gt;

## 引子
作为一名AI爱好者和不资深的游戏玩家，前段时间，趁电商促销我入手了AMD Radeon™ RX 7900 XT的显卡。本着物尽其用的原则，我还想尝试用它来做一些LLM相关的应用。于是我对**个人用户能不能用AMD GPU做AI计算？它相较于Nvidia GPU究竟差在哪里？**话题产生了兴趣，我希望探索和解答以下问题：
 
- 为什么大家都会说AMD GPU`看上去很美`，但又是`你买我推荐，我买我不买`的态度？
- 截止当前，AMD GPU及ROCm生态圈发展状况如何？能不能做Nvidia GPU的平替？
- 考虑到技术栈和产品服务的连续性，有必要审视AMD GPU的发展方向和策略。结合各类信息与报道分析，选用AMD GPU是不是一个可以“战未来”且不那么“冒风险”的决定？
- 再放宽视野，Nvidia GPU凭借软硬件实力一家独大，其它厂商（如国产的GPU们），还有什么“曲线救国”的应用解决方案？从AMD / Intel / 微软和其它市面上的开、闭源方案里可能有哪些启发？

需要说明的是，本文将立足于个人AI应用开发者/爱好者的视角，分析中将更关注消费级的GPU而非专业级GPU（或GPU加速卡），更侧重不同GPU及生态对上层应用开发的影响，而非探讨底层基础硬件、架构、通信等的技术细节。文中除数据与材料外，将有一些个人的见解和判断；因为GPU及生态发展迅猛，未来我也将尝试回过头来对照，并进一步更新。

## AMD GPU：只是“看上去很美“？
开始之前，简单回顾一下GPU计算的原理。虽然GPU起初用于图形渲染，但由于它对大量并行任务的支持，和体现出来的高效性，使得GPU同样适用于大量的并行数据计算，而这也是一切当前AI计算的前提。GPU用于AI计算，其核心的能力有：
- 数据计算能力
- 任务分解与线程调度能力
- 高速缓存，及内外部的IO通信能力

其实任何的并发式系统都少不了上述的核心能力。我们把这样的系统比做一个企业：数据计算能力代表每一个员工的个人能力和业务水平；任务分解与线程调度能力代表企业管理、资源调度优化的效率，能否将每个人调动组织起来，人尽其材，物尽其用；高速缓存，及内外部的IO通信能力则相当于在大批量、批次化、阶段化的运营与生产服务中，通过有效沟通、时间与空间上的调度，以尽量减少损耗和浪费。

&gt; **GPGPU**（General-Purpose computing on Graphics Processing Units）指的是在图形处理单元（GPU）上进行通用计算的技术。这种技术利用GPU的并行处理能力来加速非图形计算任务，广泛应用于科学计算、机器学习、图像处理等领域。

以这块AMD RTX 7900为例，官网中它的一些计算相关参数如下：
![图一：AMD GPU指标参数并不逊色，性价比高。](/amd_gpu_analysis/7900xt_specifications.png &#34;AMD RTX 7900关键参数&#34;)

对照来看几个关键指标：
- **数据计算能力**：它的全精度FP32 计算 TFLOPs是 52（对比Nvidia RTX 4090为 83)；在半精度FP16 / BF16计算TFLOPs是 103（对比Nvidia RTX 4090为 330）；在INT 8上的TOPs无官方信息，但有[报告](https://coinpoet.com/ml/learn/gpu/amd-radeon-rx-7900-xt)称是103（对比Nvidia RTX 4090为 660）；AI Accelerator（类Tensor Core）数目为168个（对比Nvidia RTX 4090为 512）。

&gt; [!NOTE] 关于TFLOPS (Floating-point operations per second)
&gt; Source from: [百度百科](https://baike.baidu.com/item/TFLOPS/2440337)
&gt; TFLOPS，即TeraFLOPS，是一个衡量GPU浮点运算能力的单位。它表示GPU每秒可以执行的浮点运算次数，通常以十亿次（Tera）为单位。浮点运算是一种基本的数学运算，广泛应用于图形处理、科学计算、机器学习等领域。因此，TFLOPS是衡量GPU性能的重要指标之一。TFLOPS的值越高，意味着GPU的浮点运算能力越强，处理任务的速度也就越快。

&gt; [!NOTE] 对比FP32 TFLOPs、FP16 TFLOPs、INT8 TOPs指标
&gt; 以上三个指标代表的都是GPU在不同精度下对数据的运算能力，将进一步影响GPU的训练或推理速度。
&gt; - FP32针对的是 32 位浮点计算，高精度常用于模型的训练中，以确保梯度、权重的有效更新（如避免梯度爆炸或梯度消失）；
&gt; - FP16精度较低，但计算开销小，更适合推理任务，也可用于一些混合精度的训练；
&gt; - INT8 牺牲了精度，以换取更高的效率，适用于推理任务，特别是量化后模型的推理上；
&gt; 此外，由于Nvidia在硬件上（如Tensor Cores）和软件上（如优化指令、优化算子等）做了大量优化，因此在FP16和INT8的计算上，N卡相较有显著的优势。

- **高速缓存**：这块显卡有20G的显存（对比Nvidia RTX 4090为 24G），由于显存用于加载和存放模型相关的重要数据（如模型权重、批次的训练数据等），大显存可以支持更大的神经网络规模或训练数据。对于个人偏推理性的大模型应用而言，推荐显存&gt;=16G。

- **内外部IO通信能力**：显存带宽为800G/s（对比Nvidia RTX 4090为 1008G/s），可以较好的支持模型从显存中频繁的读写数据（注：正常情况下IO通信开销时间约占总训练时间的1/3）。

在消费级的显卡里，考虑到N家的4090 / 4090D价格居高不下，3090怕矿，2080Ti 22G担心魔改风险，4060Ti 16G基础计算性能较弱，4070Ti Super和 4080 Ti 性价比稍差，16G的显存也差点意思。[横向对照其它各张显卡](https://www.topcpu.net/gpu-r/fp32-float-desktop)，AMD的 RX 7900xtx和7900xt看起来是不错的选择了。可随便翻翻论坛或问答社区，在个人选用上，AMD GPU总处于“你买我推荐，我买我不买”的尴尬境地。为什么呢？这少不了要说一说GPU的生态支持问题了。

## ROCm vs CUDA 生态圈：“护城河”在弥合，但用户粘性仍在

先来看AI计算市场份额90% 以上、处于行业事实标准的Nvidia的CUDA生态。老黄布局多年CUDA（Compute Unified Device Architecture），是为了充分利用GPU进行大量并行数据计算的底层接口，即将上层的模型训练或推理程序，转换为底层计算单元的基础命令执行。而我们常说的CUDA生态则是在CUDA基础上包括了：操作系统支持、多种基础运算与算法的高效算子、机器学习/深度学习通用框架（如pytorch\ tensorflow等）支持、针对机器学习/深度学习的加速算子/包（如ONNX / Flash Attention / bitsandbytes）等，这些作为基础设施，直接决定了上层模型训练、推理，应用开发的可实现性、有效性和便捷性。因此，CUDA生态也常被视为Nvidia护城河之一。

![图二：关于[CUDA生态的一张老图](https://blogs.nvidia.com/blog/what-is-cuda-2/)](/amd_gpu_analysis/cuda_ecosys.png &#34;CUDA eco-system&#34;)

![图三：Nvidia官网上对于[CUDA生态及相关工具](https://developer.nvidia.com/tools-ecosystem)的进一步分类呈现。](/amd_gpu_analysis/cuda_components.png &#34;CUDA components&#34;)

相比于老黄在CUDA生态上的长远投入、精心布局（不得不赞叹其眼光和企业家精神），AMD在GPU生态上则是落后了不少。一来是AMD起步晚，早些年重心在CPU上，一度只能是勉力存活，后来又忙着在CPU上和Intel干架，收购ATI后技术整合，无遐他顾。二是AMD走得弯路多，如硬件架构上，早期走GCN（Graph Core Next），后来又代之以侧重游戏和消费市场的RDNA和侧重计算及企业市场的CDNA双路线，再到未来又要整合RDNA和CDNA为UDNA。每一代架构技术上的连续性和兼容性不足，甚至同一架构下不同代次的技术兼容也有问题；另如计算框架上，刚开始AMD使用OpenCL作为编程模型，后来才使用更对标CUDA的HIP（Heterogeneous-compute Interface for Portability）——毕竟CUDA已经成为事实上的标准——以方便开发者在ROCm生态上使用。

不过，近几年来，随着AMD在CPU市场站稳脚步、主机市场占据绝对优势，AMD也愈发重视自家的GPU生态（亦即 ROCm生态建设），主要的一些举措包括：
1. ROCm全面对标CUDA建设，力求无痛兼容CUDA；同时保持开源生态，以扩大用户和影响，从而更好的对抗Nvidia霸权。
2. 积极联合生态圈伙伴，如[2022 年底成为Pytorch组织成员，从而得到Pytorch原生支持](https://www.amd.com/en/products/software/rocm.html)；2023年[加入Huggingface的硬件伙伴联盟](https://huggingface.co/blog/huggingface-and-amd)，2023 年-2024 年间一些重要的算子（如ONXX、DeepSpeed、Triton等，近期又有Flash Attention、bitsandbytes等）和综合性的应用（如Ollama、vLLM等）陆续支持ROCm。要知道大量的模型训练人员和AI开发者并不会去关注底层的算子、底层的实现，大家只关注能不能跑 pytorch，能不能使用各种加速算子高效计算等，因此这些举措意义重大，使得ROCm具有真正的可用性和易用性。
3. 另一个并不显见但对于ROCm同样重要的举措是大力进军企业级GPU市场。毕竟有了用户，特别企业级的大用户，有了利润来源，在商业上才能有效拉动ROCm的有效迭代与提升。

![图四：[ROCm](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-rocm-6-brief.pdf)全面对标CUDA生态，AMD消费级和企业级GPU使用共同的ROCm。](/amd_gpu_analysis/rocm_ecosys.png &#34;ROCm生态&#34;)

综上所述，就计算框架和生态上（CUDA vs ROCm），我们可以看到AMD终于走在了正轨上，特别是近几年ROCm的优化迭代、增加的对不同框架、不同算子的支持，令人眼前一亮。当然，ROCm还一些广为用户所诟病的地方，如一直以来对Windows用户支持不好；对老一些的显卡支持不好（i.e.系统架构差异大，连续性差，升级支持少）。从技术层面来看，考虑到当前AI的基础算法架构（如深度学习的算法框架，以transformer为代表的大模型结构和主要运算等）也都已经相对成熟和稳定，ROCm在逐步补齐短板，优化计算逻辑，完善各种算子，追赶和兼容CUDA并非难以逾越的障碍。在技术层面外，用户的使用习惯与粘性、新的技术迭代（非颠覆）与持续投入、战略上的选择等一系列因素还会左右这场CUDA与ROCm之争，可预见的将来，Nvidia仍然将保持其霸主地位，而 AMD最好的角色仍是作为市场中的跟随者和补充。

## 消费级GPU：前瞻与隐忧
**企业级市场的利好**  
天下苦老黄久矣！大的IT企业，自然不想见到Nvidia一家独大，拥有垄断定价权——自家AI应用还没赚钱，钱先跑进了老黄口袋这可行。因此本着risk polling的思路也希望采购别家的GPU。最理想的情况下，能高度可用，稳定切换，承载一部分的流量和服务；再次之，暂作为plan B，掌握和熟悉不同的 GPU架构和服务，以备不时之需；还不济的话，那就捏着别家的采购单，作为和老黄谈判的筹码。对于AMD而言，由于近几年在CPU市场上形势大好，有更多的余力在GPU市场上角逐。近一两年，AMD打磨出了不错的企业级GPU产品，如[旗舰的MI300](https://www.amd.com/en/products/accelerators/instinct/mi300.html#tabs-b9862a3bb5-item-58ebcbef73-tab)，对照Nvidia家的H200，在规格和参数上高出了不少，而且还拿出了相当有诚意的价格，接连[拿下了微软](https://www.amd.com/en/newsroom/press-releases/2023-11-15-amd-brings-new-ai-and-compute-capabilities-to-micr.html)、[IBM](https://www.datacenterdynamics.com/en/news/ibm-cloud-to-add-amd-instinct-mi300x-gpus-in-2025/)等企业的订单。因此，企业级GPU市场中，AMD是完全可争取一席之地，成为Nvidia的补充和一定程度的替代。

![图五：AMD在企业级GPU市场上的旗舰MI300大有一战之力，赢得了不少订单](/amd_gpu_analysis/mi300_comparison.png &#34;MI300性能对照&#34;)

赢得一部分企业市场，无疑会进一步坚定AMD大力做好GPU的决心。特别是AMD也深知自家ROCm还差火候，[苏妈和AMD其他高管也多次声称“ROCm是第一优先级](https://www.eetimes.com/rocm-is-amds-no-1-priority-exec-says/)。由于AMD家的企业级GPU（INSTINCT系列加速卡）和消费级GPU（Radeon系列）是共用ROCm的，因此，这也会意味着未来AMD仍将致力于建设ROCm，以使得技术越来越成熟，生态越来越完成完善。

**2025年消费市场的战略调整**  
共用的ROCm的优化，对于消费级GPU当然是好事。可这件事的另一面是，企业市场优先，而企业市场与消费市场的能力需求并不完全一致：如企业市场看重的是大量卡集群的协同能力，这可能是千卡、万卡集群的大规模运算（想想Nvidia家的NVlink）；企业用户往往聚集在一些GPU生态工具、运算类型与算子的优化，而并不强调整个生态的全面性；同时企业用户对服务要求也更高。在资源有限的情况下，AMD将聚集于利润率高的企业级GPU市场，暂时不发展消费级市场，[2025 年AMD将发布的RDNA 4架构的消费级GPU 8000系列，将专注于中低端型号](https://www.tomshardware.com/pc-components/gpus/amd-rdna-4-coming-in-early-2025-set-to-deliver-ray-tracing-improvements-ai-capabilities)，这也意味着很难期待AMD近期会有面向个人消费者的有竞争力的可用于AI计算的显卡面世。
&gt; 当然，另一个疑似的可能原因是，新架构下AMD高端系GPU的良品率、性能提升存在瓶颈，尚不适用生产和推广。

另一个战略上的不确定因素是，苏妈对于CPU和GPU的结合有着异乎寻常的偏好。站在AMD立场上，这也许很合理：毕竟点了CPU和GPU两条科技树，要是能在二者结合的地方玩出点花来，求其上者是走出一条新的奠定未来基业的技术路线，得乎中者是看看能不能应对消费市场上的AI硬件需求。我并不理解相关内容，这一块似乎Apple更具优势。个人理解，战略上的不坚定和摇摆乃是兵家大忌。

虽然AMD暂时放弃高端消费级显卡（往往也是个人AI应用的高性价比选择），只是短期的调整性战略。但这件事背后，确实折射出在生产厂商视角下，利润低的个人消费市场似乎是“食之无味”的鸡肋。在这一问题上，我觉得很有必要为“AI GPU“的个人消费者重新正名。一方面，中小组织和个人用户是重要的贡献者和共同演进者，这一点对于像ROCm这样开源的、追赶中的生态尤其重要；而企业用户的需求往往是聚焦的、差异性的、闭源。另一方面，未来AI军备竞赛节奏将逐渐从大企业为主的训练，转变到更广泛中小组织和个人用户参与的推理和应用上，后者市场潜力巨大，但何况这波AI浪潮上，我们也看到了不少从中小组织和个人用户演进为引领者的案例。几年前，OpenAI以极低的成本（免费？）拿到了Nvidia的GPU支持，这两年，每次老黄亲自配送给OpenAI最新型号的显卡，成了Nvidia显卡最好的广告之一。服务好今天的中小组织和个人用户，也许就是明天的大买家。

**从RDNA/CDNA到UDNA**  
前文提到AMD GPU走的是侧重游戏和消费市场的RDNA（包括Radeon Pro / Radeon桌面版 / 移动版等）和侧重计算及企业市场的CDNA（适用在INSTINCT系列上）双路线。[9月AMD宣布未来将整合RDNA和CDNA](https://www.tomshardware.com/pc-components/cpus/amd-announces-unified-udna-gpu-architecture-bringing-rdna-and-cdna-together-to-take-on-nvidias-cuda-ecosystem)。这也是AMD在有限资源下的理性选择，而且长期来看，更集中、更统一的支持、降低的软硬件架构开销可以更有效的与对手竞争，但短期内整合两套架构的过程中仍可能会有不确定性，产品和技术支持和服务可能有连续性、兼容性的问题，毕竟这事在AMD上也不是第一次发生:dog:。

## “曲线救国”：其他更开放的计算框架
Nvidia挟千百万用户之众，又坐拥产业链之便、CUDA之标准，此诚不可与争锋也。那么其它的GPU厂商（如国内华为、海思、平头哥等），破局出路究竟在哪里？在这里我们仅就GPU生态来说，AMD无疑是打了一个样，本章节我们将视野放宽，来看看DirectML、Zluda、SCALE等一些开源/初创的GPU计算框架（GPGPU）他们是怎么做的，他们前行的方向也许能给我们一些启发和思路。

**DirectML**  
[DirectML](https://github.com/microsoft/DirectML)是微软开发的一种GPGPU技术（或称之为接口），它也是面向底层的、低层次的机器学习/深度学习库，它的主要特色是：
- 依赖于DirectX 12，但兼容不同的硬件，凡支持DirectX 12的GPU如AMD、Nvidia、Intel、甚至 高通（Qualcomm）都可以使用；
- 面向windows环境；
- 可以通过torch-directml包来使用pytorch，以实现上层模型训练、推理并启动GPU支持；
- 对一些基础的运算做了封装和支持，

DirectML可以视为许多GPU厂商的盟友，如对AMD用户的意义包括：
- 支持windows系统上的AI训练与开发，特别是AMD自身在window上久久没有很好的支持情况下；
- 对老版本的显卡有较好的支持。
但毕竟Window环境很难成为练丹的第一选择，另外，DirectML相较于原生的接口效率上也会更低一些。

**ZLUDA**  
[ZLUDA](https://github.com/vosen/ZLUDA)是另一个令人印象深刻的开源GPGPU项目，它的最大特色在于直接锚定CUDA，力图实现在非Nvidia的GPU上可直接运行CUDA代码，并尽可能提高运行效率。 ZLUDA的逻辑疑似是在GPU上添加了一个模拟层或转译层，以解释并兼容CUDA，从而实现跨平台的执行，这个思路和国内一众的GPU生产厂商一致。

ZLUDA最初是个人作品，后来成为社区开源的共同项目。根据用户反馈，ZLUDA显示出了不错的性能，早前ZLUDA表现甚至能够超越ROCm，并对windows系统、老型号的显卡有着较好的支持。

ZLUDA曾经得到AMD和Intel的支持。但随着Nvidia对CUDA的限制越来越严格（CUDA为闭源），如[明确禁止通过模拟层运行CUDA](https://www.tomshardware.com/pc-components/gpus/nvidia-bans-using-translation-layers-for-cuda-software-to-run-on-other-chips-new-restriction-apparently-targets-zluda-and-some-chinese-gpu-makers)，并限制任何的逆向工程、反编译等，Intel和AMD都停止了对它的资助。

好消息是，10月初创始人在[博客中宣称](https://vosen.github.io/ZLUDA/blog/zludas-third-life/)，ZLUDA获得了一位不具名厂商的投资，可以继续开发这一项目，但同时，考虑到法律风险、技术风险，ZLUDA自身也会做出不少的调整：a. 代码重构；b. 重心上会聚焦到机器学习、大模型上；c. 也将与AMD/Intel各自的GPGPU差异化，如强化HIP做得不好的图片相关的计算、更多支持一些尚未被支持的AMD GPU等。

**SCALE**  
另一个值得关注的新起之秀是SCALE，根据[介绍文档](https://docs.scale-lang.com/)，它是一种能够支持CUDA编写的程序原生的运行在AMD GPU上，而无需任何的转译层，换言之，它并不依赖 CUDA本身，而是[本身模拟CUDA的功能和角色](https://docs.scale-lang.com/manual/how-to-use/)。当前看来，[SCALE所能够支持的算法和算子还相对有限](https://docs.scale-lang.com/)。

**总结**  
勇者半恶龙总是一个大众喜闻乐见的故事。对ZLUDA项目我深表敬意，SCALE项目也值得关注，这二者某种程序上也可作为我国GPU生态开发的学习参照。综合以上信息年来，AMD走的是一条全面对标CUDA、全生态自研的发展路径；SCALE是保持CUDA语法、接口不变的情况下，确保全面兼容CUDA语言和生态内的其它工具；ZLUDA则是更接近对CUDA的转译，仍借助于CUDA本身。前者更适合作为硬件厂商长期的发展路线，后者则适合中短期内较快速的面向市场推广自家的产品，并确保用户以低成本学习和应用。
|              | HIP                                                                                                                                   | ZLUDA                                                                                                                            | SCALE                                                                                                         |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| 生产方       | AMD                                                                                                                                   | 个人与社区                                                                                                                       | 初创公司 Spectral Compute                                                                                     |
| 是否开源     | 是                                                                                                                                    | 是                                                                                                                               | 否（开发测试中）                                                                                              |
| 是否兼容CUDA | - 大部分&lt;br&gt; - 以功能对齐为准，部分命令和语句不同于CUDA                                                                                    | - 是&lt;br&gt; - 全面兼容CUDA                                                                                                               | - 绝大部分                                                                                                      |
| 技术路线     | a. 对照CUDA 接口开发并对齐功能；&lt;br&gt;b. 支持和调用ROCm 生态圈工具                                                                      | 扮演转译层，从而将CUDA 程序运行在非 Nvidia GPU 上（类似于逆向工程）                                                              | a. 完全沿袭CUDA 的指令和接口模式；&lt;br&gt;b. 独立构建和实现CUDA 的功能                                            |
| 其它         | 同时兼容OpenCL&lt;br&gt;可基于HIP 开发扩展新功能                                                                                            | 有老版本GPU、多系统支持                                                                                                          | 仍以兼容CUDA 相关工具为主，不包括ROCm 生态圈的其它工具                                                        |
| 评论         | a. 开发成本高，自主性强；&lt;br&gt;b. 与CUDA 仍会有一定程度的适配问题，带来的用户使用成本较高；&lt;br&gt;c. 保持对CUDA 的跟进和对照开发，成本高。 | a. 开发成本相对较低；&lt;br&gt;b. 强依赖于CUDA 及相关工具， 自主性较弱，有一定的合规风险；&lt;br&gt;c. CUDA 支持度高，无需要额外的调整和适配 | a. 开发成本相对较高；&lt;br&gt;b. 全面兼容但不依赖于CUDA ， 自主性较强，有一定的合规风险；&lt;br&gt;c. 可视为AMD 上的CUDA |
| 用户视角     | 大量资源迁移到AMD GPU 上；有较强的代码研发能力                                                                                        | 低成本快速切换到AMD GPU；对个人用户友好                                                                                          | 由于当前支持的算法和算子较少，建议进一步观望                                                                  |
| GPU厂商视角  | - 独立自研，更能配合公司的长期战略；&lt;br&gt; - 适用于服务技术能力强的大客户                                                                    | - 适合搭配GPU应用推广，或在中短期探索阶段；&lt;br&gt; - 有合规风险                                                                          | 中间选择，可以先将核心的CUDA 能力移植到GPU 上，再扩展其它能力与工具                                           |




## 参考资料
- ROCm Developer Hub: https://www.amd.com/en/developer/resources/rocm-hub.html
- ZLUDA: https://github.com/vosen/ZLUDA
- DirectML：https://learn.microsoft.com/zh-cn/windows/ai/directml/dml
- SCALE: https://docs.scale-lang.com/ https://zhuanlan.zhihu.com/p/678371087
- AMD GPU硬件架构：https://fpga.eetrend.com/content/2024/100577289.html https://zhuanlan.zhihu.com/p/651026452
- AMD GPU介绍： https://zhuanlan.zhihu.com/p/545296023
- 关于CUDA的介绍也可参见: https://zhuanlan.zhihu.com/p/668749361
- 知乎关于CUDA ROCm的讨论：https://www.zhihu.com/question/564812763 或 https://www.zhihu.com/question/618150944；在reddit上的讨论有 https://www.reddit.com/r/MachineLearning/comments/1fa8vq5/d_why_is_cuda_so_much_faster_than_rocm/ 及 https://www.reddit.com/r/Amd/comments/1bxlp3r/how_good_are_amd_gpus_at_running_large_language/，其中不乏洞见。 
- 关于GPGPU的讨论：https://www.zhihu.com/question/461354739/answer/3259844830
- AMD GPU评测与配置：https://llm-tracker.info/howto/AMD-GPUs https://github.com/nktice/AMD-AI

---

> 作者: 杰子  
> URL: http://localhost:1313/zh-cn/posts/amd-gpu-analysis/  

