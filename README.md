# RPT

This is a framework of pre-training for researcher data. Corresponding paper is under reviewing.

## Hardware \& Software

All experiments are conducted with the following setting:

* Operating system: CentOS Linux release 7.7.1908
* Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz.
*  GPU: 4 GeForce GTX 1080 Ti
*  Software versions: Python 3.7; Pytorch 1.7.1; Numpy 1.20.0; SciPy 1.6.0; Gensim 3.8.3; scikit-learn 0.24.0

## Run this framework
### 1. Prepare the researcher semantic document set
You should prepare a text file name "author_document_corpus.txt", each line represents a researcher's semantic document set, including the researcher ID and document sequence separated by tab(\t), for example:
```
a1 \t d1 \t d2 \t ... \t dm \n
a2 \t d1 \t d2 \t ... \t dn \n
...
```
If you want use your pre-trained word_embedding, you also need to prepare a word embedding dict named "data/word_embedding.pkl", which is optional.

### 2. Prepare the researcher community graph
You should prepare a dict named "author_community.pkl", where the ksy is the researcher id, and the value includes two vectors of equal length, preserving the neighbor id and corresponding relation type index. For example:

```
{a1: ["neighbors": [a2, a4, a9, ..., a6], 
      "relations": [0,  0,  1,  ..., 2]
     ]
 a2: ...
} 
```

### 2. Training
```bash
python __main__.py -c data/author_document_corpus.txt -we data/word_embedding.pkl -ac data/author_community.pkl -o output/bert.model
```
