# Run-TransE-on-FB15K
<h2>Summary</h2>
In windows environment, Run the TransE model on the FB15K data set

<h2>Preparation</h2>
Compatible python + cuda + pytorch

<h2>Check the environment</h2>

<li>Import pytorch, Type in the terminal:</li>

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
troch.cuda.is_avaiable()
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
<p>Other models can be trained with similar instructions:</p>
<li>Run the TransH model on the FB15K data set</li>

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
```
<p>And so on...</p>
