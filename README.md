<div align='center'>
<img src="https://github.com/NTMC-Community/MatchZoo-py/blob/master/artworks/matchzoo-logo.png?raw=true" width = "400"  alt="logo" align="center" />
</div>

# MatchZoo-py [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=MatchZoo-py:%20deep%20learning%20for%20semantic%20matching&url=https://github.com/NTMC-Community/MatchZoo-py)

> PyTorch version of [MatchZoo](https://github.com/NTMC-Community/MatchZoo).

> Facilitating the design, comparison and sharing of deep text matching models.<br/>
> MatchZoo 是一个通用的文本匹配工具包，它旨在方便大家快速的实现、比较、以及分享最新的深度文本匹配模型。

[![Python 3.6](https://img.shields.io/badge/python-3.6%20%7C%203.7-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Pypi Downloads](https://img.shields.io/pypi/dm/matchzoo-py.svg?label=pypi)](https://pypi.org/project/MatchZoo-py/)
[![Documentation Status](https://readthedocs.org/projects/matchzoo-py/badge/?version=latest)](https://matchzoo-py.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/NTMC-Community/MatchZoo-py.svg?branch=master)](https://travis-ci.org/NTMC-Community/MatchZoo-py)
[![codecov](https://codecov.io/gh/NTMC-Community/MatchZoo-py/branch/master/graph/badge.svg)](https://codecov.io/gh/NTMC-Community/MatchZoo-py)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Requirements Status](https://requires.io/github/NTMC-Community/MatchZoo-py/requirements.svg?branch=master)](https://requires.io/github/NTMC-Community/MatchZoo-py/requirements/?branch=master)
[![Gitter](https://badges.gitter.im/NTMC-Community/community.svg)](https://gitter.im/NTMC-Community/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
---

The goal of MatchZoo is to provide a high-quality codebase for deep text matching research, such as document retrieval, question answering, conversational response ranking, and paraphrase identification. With the unified data processing pipeline, simplified model configuration and automatic hyper-parameters tunning features equipped, MatchZoo is flexible and easy to use.

<table>
  <tr>
    <th width=30%, bgcolor=#999999 >Tasks</th> 
    <th width=20%, bgcolor=#999999>Text 1</th>
    <th width="20%", bgcolor=#999999>Text 2</th>
    <th width="20%", bgcolor=#999999>Objective</th>
  </tr>
  <tr>
    <td align="center", bgcolor=#eeeeee> Paraphrase Indentification </td>
    <td align="center", bgcolor=#eeeeee> string 1 </td>
    <td align="center", bgcolor=#eeeeee> string 2 </td>
    <td align="center", bgcolor=#eeeeee> classification </td>
  </tr>
  <tr>
    <td align="center", bgcolor=#eeeeee> Textual Entailment </td>
    <td align="center", bgcolor=#eeeeee> text </td>
    <td align="center", bgcolor=#eeeeee> hypothesis </td>
    <td align="center", bgcolor=#eeeeee> classification </td>
  </tr>
  <tr>
    <td align="center", bgcolor=#eeeeee> Question Answer </td>
    <td align="center", bgcolor=#eeeeee> question </td>
    <td align="center", bgcolor=#eeeeee> answer </td>
    <td align="center", bgcolor=#eeeeee> classification/ranking </td>
  </tr>
  <tr>
    <td align="center", bgcolor=#eeeeee> Conversation </td>
    <td align="center", bgcolor=#eeeeee> dialog </td>
    <td align="center", bgcolor=#eeeeee> response </td>
    <td align="center", bgcolor=#eeeeee> classification/ranking </td>
  </tr>
  <tr>
    <td align="center", bgcolor=#eeeeee> Information Retrieval </td>
    <td align="center", bgcolor=#eeeeee> query </td>
    <td align="center", bgcolor=#eeeeee> document </td>
    <td align="center", bgcolor=#eeeeee> ranking </td>
  </tr>
</table>

## Get Started in 60 Seconds

To train a [Deep Semantic Structured Model](https://www.microsoft.com/en-us/research/project/dssm/), make use of MatchZoo customized loss functions and evaluation metrics to define a task:

```python
import torch
import matchzoo as mz

ranking_task = mz.tasks.Ranking(losses=mz.losses.RankCrossEntropyLoss(num_neg=4))
ranking_task.metrics = [
    mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
    mz.metrics.MeanAveragePrecision()
]
```

Prepare input data:

```python
train_pack = mz.datasets.wiki_qa.load_data('train', task=ranking_task)
valid_pack = mz.datasets.wiki_qa.load_data('dev', task=ranking_task)
```

Preprocess your input data in three lines of code, keep track parameters to be passed into the model:

```python
preprocessor = mz.models.ArcI.get_default_preprocessor()
train_processed = preprocessor.fit_transform(train_pack)
valid_processed = preprocessor.transform(valid_pack)
```

Generate pair-wise training data on-the-fly:
```python
trainset = mz.dataloader.Dataset(
    data_pack=train_processed,
    mode='pair',
    num_dup=1,
    num_neg=4
)
validset = mz.dataloader.Dataset(
    data_pack=valid_processed,
    mode='point'
)
```

Define padding callback and generate data loader:
```python
padding_callback = mz.models.ArcI.get_default_padding_callback()

trainloader = mz.dataloader.DataLoader(
    dataset=trainset,
    batch_size=32,
    stage='train',
    callback=padding_callback
)
validloader = mz.dataloader.DataLoader(
    dataset=validset,
    batch_size=32,
    stage='dev',
    callback=padding_callback
)
```

Initialize the model, fine-tune the hyper-parameters:

```python
model = mz.models.ArcI()
model.params['task'] = ranking_task
model.params['vocab_size'] = preprocessor.context['vocab_size']
model.guess_and_fill_missing_params()
model.build()
```

`Trainer` is used to control the training flow:

```python
optimizer = torch.optim.Adam(model.parameters())

trainer = mz.trainers.Trainer(
    model=model,
    optimizer=optimizer,
    trainloader=trainloader,
    validloader=validloader,
    epochs=10
)

trainer.run()
```

## References
[Tutorials](https://github.com/NTMC-Community/MatchZoo-py/tree/master/tutorials)

[English Documentation](https://matchzoo-py.readthedocs.io/en/latest/)

If you're interested in the cutting-edge research progress, please take a look at [awaresome neural models for semantic match](https://github.com/NTMC-Community/awaresome-neural-models-for-semantic-match).

## Install

MatchZoo-py is dependent on [PyTorch](https://pytorch.org). Two ways to install MatchZoo-py:

**Install MatchZoo-py from Pypi:**

```python
pip install matchzoo-py
```

**Install MatchZoo-py from the Github source:**

```
git clone https://github.com/NTMC-Community/MatchZoo-py.git
cd MatchZoo-py
python setup.py install
```


## Models

- [DRMM](https://github.com/NTMC-Community/MatchZoo-py/tree/master/matchzoo/models/drmm.py): this model is an implementation of <a href="http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf">A Deep Relevance Matching Model for Ad-hoc Retrieval</a>.
- [DRMMTKS](https://github.com/NTMC-Community/MatchZoo-py/tree/master/matchzoo/models/drmmtks.py): this model is an implementation of <a href="https://link.springer.com/chapter/10.1007/978-3-030-01012-6_2">A Deep Top-K Relevance Matching Model for Ad-hoc Retrieval</a>.
- [ARC-I](https://github.com/NTMC-Community/MatchZoo-py/tree/master/matchzoo/models/arci.py): this model is an implementation of <a href="https://arxiv.org/abs/1503.03244">Convolutional Neural Network Architectures for Matching Natural Language Sentences</a>
- [ARC-II](https://github.com/NTMC-Community/MatchZoo-py/tree/master/matchzoo/models/arcii.py): this model is an implementation of <a href="https://arxiv.org/abs/1503.03244">Convolutional Neural Network Architectures for Matching Natural Language Sentences</a>
- [DSSM](https://github.com/NTMC-Community/MatchZoo-py/tree/master/matchzoo/models/dssm.py): this model is an implementation of <a href="https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf">Learning Deep Structured Semantic Models for Web Search using Clickthrough Data</a>
- [CDSSM](https://github.com/NTMC-Community/MatchZoo-py/tree/master/matchzoo/models/cdssm.py): this model is an implementation of <a href="https://www.microsoft.com/en-us/research/publication/learning-semantic-representations-using-convolutional-neural-networks-for-web-search/">Learning Semantic Representations Using Convolutional Neural Networks for Web Search</a>
- [MatchLSTM](https://github.com/NTMC-Community/MatchZoo-py/tree/master/matchzoo/models/matchlstm.py):this model is an implementation of <a href="https://arxiv.org/abs/1608.07905">Machine Comprehension Using Match-LSTM and Answer Pointer</a>
- [DUET](https://github.com/NTMC-Community/MatchZoo-py/tree/master/matchzoo/models/duet.py): this model is an implementation of <a href="https://dl.acm.org/citation.cfm?id=3052579">Learning to Match Using Local and Distributed Representations of Text for Web Search</a>
- [KNRM](https://github.com/NTMC-Community/MatchZoo-py/tree/master/matchzoo/models/knrm.py): this model is an implementation of <a href="https://arxiv.org/abs/1706.06613">End-to-End Neural Ad-hoc Ranking with Kernel Pooling</a>
- [ConvKNRM](https://github.com/NTMC-Community/MatchZoo-py/tree/master/matchzoo/models/conv_knrm.py): this model is an implementation of <a href="http://www.cs.cmu.edu/~zhuyund/papers/WSDM_2018_Dai.pdf">Convolutional neural networks for soft-matching n-grams in ad-hoc search</a>
- [ESIM](https://github.com/NTMC-Community/MatchZoo-py/tree/master/matchzoo/models/esim.py): this model is an implementation of <a href="https://arxiv.org/abs/1609.06038">Enhanced LSTM for Natural Language Inference</a>
- [BiMPM](https://github.com/NTMC-Community/MatchZoo-py/tree/master/matchzoo/models/bimpm.py): this model is an implementation of <a href="https://arxiv.org/abs/1702.03814">Bilateral Multi-Perspective Matching for Natural Language Sentences</a>
- [MatchPyramid](https://github.com/NTMC-Community/MatchZoo-py/tree/master/matchzoo/models/match_pyramid.py): this model is an implementation of <a href="https://arxiv.org/abs/1602.06359">Text Matching as Image Recognition</a>
- [Match-SRNN](https://github.com/NTMC-Community/MatchZoo-py/tree/master/matchzoo/models/match_srnn.py): this model is an implementation of <a href="https://arxiv.org/abs/1604.04378">Match-SRNN: Modeling the Recursive Matching Structure with Spatial RNN</a>
- [aNMM](https://github.com/NTMC-Community/MatchZoo-py/tree/master/matchzoo/models/anmm.py): this model is an implementation of <a href="https://arxiv.org/abs/1801.01641">aNMM: Ranking Short Answer Texts with Attention-Based Neural Matching Model</a>
- [MV-LSTM](https://github.com/NTMC-Community/MatchZoo-py/tree/master/matchzoo/models/mvlstm.py): this model is an implementation of <a href="https://arxiv.org/pdf/1511.08277.pdf">A Deep Architecture for Semantic Matching with Multiple Positional Sentence Representations</a>
- [DIIN](https://github.com/NTMC-Community/MatchZoo-py/tree/master/matchzoo/models/diin.py): this model is an implementation of <a href="https://arxiv.org/pdf/1709.04348.pdf">Natural Lanuguage Inference Over Interaction Space</a>
- [HBMP](https://github.com/NTMC-Community/MatchZoo-py/tree/master/matchzoo/models/hbmp.py): this model is an implementation of <a href="https://arxiv.org/pdf/1808.08762.pdf">Sentence Embeddings in NLI with Iterative Refinement Encoders</a>
- [BERT](https://github.com/NTMC-Community/MatchZoo-py/tree/master/matchzoo/models/bert.py): this model is an implementation of <a href="https://arxiv.org/abs/1810.04805">BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding</a>


## Citation

If you use MatchZoo in your research, please use the following BibTex entry.

```
@inproceedings{Guo:2019:MLP:3331184.3331403,
 author = {Guo, Jiafeng and Fan, Yixing and Ji, Xiang and Cheng, Xueqi},
 title = {MatchZoo: A Learning, Practicing, and Developing System for Neural Text Matching},
 booktitle = {Proceedings of the 42Nd International ACM SIGIR Conference on Research and Development in Information Retrieval},
 series = {SIGIR'19},
 year = {2019},
 isbn = {978-1-4503-6172-9},
 location = {Paris, France},
 pages = {1297--1300},
 numpages = {4},
 url = {http://doi.acm.org/10.1145/3331184.3331403},
 doi = {10.1145/3331184.3331403},
 acmid = {3331403},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {matchzoo, neural network, text matching},
} 
```


## Development Team

 ​ ​ ​ ​
<table border="0">
  <tbody>
    <tr align="center">
      <td>
        ​ <a href="https://github.com/faneshion"><img width="40" height="40" src="https://github.com/faneshion.png?s=40" alt="faneshion"></a><br>
        ​ <a href="http://www.bigdatalab.ac.cn/~fanyixing/">Yixing Fan</a> ​
        <p>Core Dev<br>
        ASST PROF, ICT</p>​
      </td>
      <td>
         <a href="https://github.com/Chriskuei"><img width="40" height="40" src="https://github.com/Chriskuei.png?s=40" alt="Chriskuei"></a><br>
         <a href="https://github.com/Chriskuei">Jiangui Chen</a> ​
        <p>Core Dev<br> PhD. ICT</p>​
      </td>
      <td>
        ​ <a href="https://github.com/caiyinqiong"><img width="40" height="40" src="https://github.com/caiyinqiong.png?s=36" alt="caiyinqiong"></a><br>
         <a href="https://github.com/caiyinqiong">Yinqiong Cai</a>
         <p>Core Dev<br> M.S. ICT</p>​
      </td>
      <td>
        ​ <a href="https://github.com/pl8787"><img width="40" height="40" src="https://github.com/pl8787.png?s=40" alt="pl8787"></a><br>
        ​ <a href="https://github.com/pl8787">Liang Pang</a> ​
        <p>Core Dev<br>
        ASST PROF, ICT</p>​
      </td>
      <td>
        ​ <a href="https://github.com/lixinsu"><img width="40" height="40" src="https://github.com/lixinsu.png?s=40" alt="lixinsu"></a><br>
        ​ <a href="https://github.com/lixinsu">Lixin Su</a>
        <p>Dev<br>
        PhD. ICT</p>​
      </td>
    </tr>
    <tr align="center">
      <td>
        ​ <a href="https://github.com/ChrisRBXiong"><img width="40" height="40" src="https://github.com/ChrisRBXiong.png?s=40" alt="ChrisRBXiong"></a><br>
        ​ <a href="https://github.com/ChrisRBXiong">Ruibin Xiong</a> ​
        <p>Dev<br>
        M.S. ICT</p>​
      </td>
      <td>
        ​ <a href="https://github.com/dyuyang"><img width="40" height="40" src="https://github.com/dyuyang.png?s=40" alt="dyuyang"></a><br>
        ​ <a href="https://github.com/dyuyang">Yuyang Ding</a> ​
        <p>Dev<br>
        M.S. ICT</p>​
      </td>
      <td>
        ​ <a href="https://github.com/rgtjf"><img width="40" height="40" src="https://github.com/rgtjf.png?s=36" alt="rgtjf"></a><br>
        ​ <a href="https://github.com/rgtjf">Junfeng Tian</a> ​
        <p>Dev<br>
        M.S. ECNU</p>​
      </td>
      <td>
        ​ <a href="https://github.com/wqh17101"><img width="40" height="40" src="https://github.com/wqh17101.png?s=40" alt="wqh17101"></a><br>
        ​ <a href="https://github.com/wqh17101">Qinghua Wang</a> ​
        <p>Documentation<br>
        B.S. Shandong Univ.</p>​
      </td>
    </tr>
  </tbody>
</table>




## Contribution

Please make sure to read the [Contributing Guide](./CONTRIBUTING.md) before creating a pull request. If you have a MatchZoo-related paper/project/compnent/tool, send a pull request to [this awesome list](https://github.com/NTMC-Community/awaresome-neural-models-for-semantic-match)!

Thank you to all the people who already contributed to MatchZoo!

[Bo Wang](https://github.com/bwanglzu), [Zeyi Wang](https://github.com/uduse), [Liu Yang](https://github.com/yangliuy), [Zizhen Wang](https://github.com/ZizhenWang), [Zhou Yang](https://github.com/zhouzhouyang520), [Jianpeng Hou](https://github.com/HouJP), [Lijuan Chen](https://github.com/githubclj), [Yukun Zheng](https://github.com/zhengyk11), [Niuguo Cheng](https://github.com/niuox), [Dai Zhuyun](https://github.com/AdeDZY), [Aneesh Joshi](https://github.com/aneesh-joshi), [Zeno Gantner](https://github.com/zenogantner), [Kai Huang](https://github.com/hkvision), [stanpcf](https://github.com/stanpcf), [ChangQF](https://github.com/ChangQF), [Mike Kellogg
](https://github.com/wordreference)




## Project Organizers

- Jiafeng Guo
  * Institute of Computing Technology, Chinese Academy of Sciences
  * [Homepage](http://www.bigdatalab.ac.cn/~gjf/)
- Yanyan Lan
  * Institute of Computing Technology, Chinese Academy of Sciences
  * [Homepage](http://www.bigdatalab.ac.cn/~lanyanyan/)
- Xueqi Cheng
  * Institute of Computing Technology, Chinese Academy of Sciences
  * [Homepage](http://www.bigdatalab.ac.cn/~cxq/)


## License

[Apache-2.0](https://opensource.org/licenses/Apache-2.0)

Copyright (c) 2019-present, Yixing Fan (faneshion)
