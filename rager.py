from app.llm_infer import LLMPredictor

from app.retriever import Retriever
from app.reranker import Reranker
from app.read_corpus import Reader


class Rager():
    def __init__(self, corpus_path,
                 emb_model_name_or_path="models/bge-large-zh",
                 rerank_model_name_or_path="models/bge-reranker-base",
                 retrieval_methods="bm25",
                 num_input_docs=4
                 ):
        self.reader = Reader(corpus_path)
        self.corpus = self.reader.corpus
        self.retriever = Retriever(emb_model_name_or_path=emb_model_name_or_path, corpus=self.corpus)
        self.reranker = Reranker(rerank_model_name_or_path=rerank_model_name_or_path)
        self.llm = LLMPredictor()
        self.num_input_docs = num_input_docs

    def answer(self, query):
        retrieval_res = self.retriever.retrieval(query)
        rerank_res = self.reranker.rerank(retrieval_res, query, k=self.num_input_docs)
        res = self.llm.predict('\n'.join(rerank_res), query).strip()
        return res


if __name__ == "__main__":
    corpus_path = input("请输入pdf文件名称")
    corpus_path = "./docs/" + corpus_path
    rager = Rager(corpus_path)
    while True:
        query = input("请输入你的问题：(输入no结束问答)")
        if query == "no":
            break
        answer = rager.answer(query)
        print("llm回答：", answer)