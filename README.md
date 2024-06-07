# easy-rag
 一个简单的RAG项目，核心代码在app文件夹中

# 各个模块介绍

## retriever.py

`Retriever`这个类的核心功能可以抽象为一个模型：一个字符串列表`corpus`，一个字符串`query`，返回`corpus`中与`query`最相关的`k`个元素。

### 检索方式

#### bm25

实现方式：

```python
    def bm25_retrieval(self, query, n=10):

        # 此处中文使用jieba分词
        query = jieba.lcut(query)  # 分词
        res = self.bm25.get_top_n(query, self.corpus, n=n)
        return res
```

原理

TODO



#### emb_retrieval

```python
    def emb_retrieval(self, query, k=10):

        search_docs = self.db.similarity_search(query, k=k)
        res = [doc.page_content for doc in search_docs]
        return res
```

