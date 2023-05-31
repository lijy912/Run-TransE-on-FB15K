# Run-TransE-on-FB15K
<h2>Summary</h2>
In windows environment, Run the TransE model on the FB15K data set.

<h2>Preparation</h2>
Python3

<h2>Related Knowledge</h2>
<h3>1. TransE</h3>
<p>TransE is a knowledge representation learning method based on vector space, and its principle is based on "migration vector". In this method, entities and relations are mapped to low-dimensional continuous vector Spaces respectively, and then, by adjusting the vectors in the vector space, the representation of each triplet in the vector space is most consistent with its semantic relevance. Specifically, for a triplet (h,r,t), where h represents the head entity,r represents the relation, and t represents the tail entity, TransE computes the vector by the following formula:</p>

```math
h + r ≈ t
```
<p>Where, + represents vector addition, ≈ represents the similarity between vectors, and r is the relational vector.</p>
<p>TransE converts entities and relationships in a knowledge base into vector representations so that the relevance between entities and relationships can be calculated using simple vector operations. This feature enables TransE to be used not only for knowledge base completion, but also for multiple natural language processing tasks, such as entity linking, relationship extraction, question answering systems, and so on.</p>

<h3>2. FB15K</h3>
<p>FB15K is a commonly used knowledge graph dataset provided by Facebook AI Research (FAIR) to evaluate the performance of knowledge representation learning and knowledge base completion. The dataset contains 15,000 entities, 1,345 relationships, and 592,213 triples, where both entities and relationships are uniquely identified by ids.</p>

<h2>Procedure</h2>
<h3>1. Download data set</h3>
<li>Download FB15K data set: https://everest.hds.utc.fr/doku.php?id=en:transe </li>
<p>&nbsp&nbsp&nbsp&nbspThe data set includes three files: entity file (entity2id.txt), relationship file (relation2id.txt), and triad file (train.txt, valid.txt, test.txt).</p>
<h3>2. Installation dependency</h3>
<li>Open the command line terminal, go to the directory where the FB15K dataset is located, and run the following command to install the required Python packages and dependencies:</li>

```python
pip install numpy scipy matplotlib pandas torch
```
<h3>3. Download OpenKE source code</h3>
<li>Using the Git command to download OpenKE (https://github.com/thunlp/OpenKE), the source code:</li>

```
git clone https://github.com/thunlp/OpenKE.git
```
<h3>4. Preprocessed data set</h3>
<li>Go to the directory where the FB15K data set resides and run the following command to convert the triplet file to the format required by OpenKE:</li>

```
cd FB15K
python ../OpenKE/util/preprocess.py -dataset FB15K -output_path ./
```
<p>&nbsp&nbsp&nbsp&nbspThis command generates entity and relationship vector files (entity2vec.bin, relation2vec.bin) and triad files (train2id.txt, valid2id.txt, test2id.txt) of training, verification, and test data sets in the FB15K directory.</p>
<h3>5. Training model</h3>
<li>Use the following command to train the TransE model on the FB15K data set:</li>

```
python ../OpenKE/run.py --do_train --cuda -adv -model TransE -dataset FB15K -batch_size 256 -hidden_size 50 -gamma 19.9 -lr 0.01 -max_steps 60000 -train_times 100
```
<p>&nbsp&nbsp&nbsp&nbspThe command is trained using the TransE model, the FB15K dataset, and the specified hyperparameters. Hyperparameters can be adjusted as needed, such as hidden vector dimension (hidden_size), learning rate (lr) and negative sample ratio (neg_ratio).</p>
<h3>6. Test model</h3>
<li>Test the performance of the trained TransE model on the test set using the following command:</li>

```
python ../OpenKE/run.py --do_test --cuda -adv -model TransE -dataset FB15K -batch_size 16 -hidden_size 50 -gamma 19.9 -lr 0.01 -max_steps 60000 -train_times 100
```
<p>&nbsp&nbsp&nbsp&nbspThis command uses a trained TransE model, makes predictions on the test set, and calculates the model's performance metrics, such as average accuracy, average ranking, and average reciprocal ranking.</p>
