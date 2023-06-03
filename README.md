# Run-TransE-on-FB15K
<h2>Summary</h2>
In windows environment, Run the TransE model on the FB15K data set.

<h2>Preparation</h2>
<li>Python3</li>
<li>NVIDIA officially released CUDA-enabled graphics cards</li>
<li>Cuda-enabled PyTorch</li>
<!-- 
<h2>Related Knowledge</h2>
<h3>1. TransE</h3>
<p>TransE is a knowledge representation learning method based on vector space, and its principle is based on "migration vector". In this method, entities and relations are mapped to low-dimensional continuous vector Spaces respectively, and then, by adjusting the vectors in the vector space, the representation of each triplet in the vector space is most consistent with its semantic relevance. Specifically, for a triplet (h,r,t), where h represents the head entity,r represents the relation, and t represents the tail entity, TransE computes the vector by the following formula:</p>

```math
h + r ≈ t
```
<p>Where, + represents vector addition, ≈ represents the similarity between vectors, and r is the relational vector.</p>
<p>TransE converts entities and relationships in a knowledge base into vector representations so that the relevance between entities and relationships can be calculated using simple vector operations. This feature enables TransE to be used not only for knowledge base completion, but also for multiple natural language processing tasks, such as entity linking, relationship extraction, question answering systems, and so on.</p>

<h3>2. FB15K</h3>
<p>FB15K is a commonly used knowledge graph dataset provided by Facebook AI Research (FAIR) to evaluate the performance of knowledge representation learning and knowledge base completion. The dataset contains 15,000 entities, 1,345 relationships, and 592,213 triples, where both entities and relationships are uniquely identified by ids.</p> -->

<h2>Check the environment</h2>

<li>Import pytorch</li>

```
python
import torch
```
<li>Check out the version of torch</li>

```
print(torch.__version__)
```
<li>Check whether torch is compatible with cuda</li>

```
troch.cuda.is_avaiable
```
<li>If True is returned, proceed</li>

```
exit()
```
<h2>Quick start</h2>

<li>Download OpenKE source code</li>

```
git clone https://github.com/thunlp/OpenKE.git
```
<li>Go to the openke directory</li>

```
cd OpenKE
cd openke
```
<li>Compiling C++ files</li>

```
bash make.sh
```
<li>Go to superior directory</li>

```
cd ../
```
<li>Create the directory "checkpoint"</li>

```
mkdir checkpoint
```
<h3>Training</h3>
<li> Run the TransE model on the FB15K data set</li>

```
cp examples/train_transe_FB15K237.py ./
python train_transe_FB15K237.py
```
<!-- <li>Run the TransH model on the FB15K data set</li>

```
cp examples/train_transh_FB15K237.py ./
python train_transh_FB15K237.py
```
<li>Run the TransR model on the FB15K data set</li>

```
cp examples/train_transr_FB15K237.py ./
python train_transr_FB15K237.py
```
<li>Run the TransD model on the FB15K data set</li>

```
cp examples/train_transd_FB15K237.py ./
python train_transd_FB15K237.py
``` -->
