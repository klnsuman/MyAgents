from typing import List, Any
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.retrievers import BaseRetriever
from sentence_transformers import CrossEncoder


class RatingScore(BaseModel):
    relevance_score: float = Field(..., description="The relevance score of a document to a query.")


def rerank_documents_llm(query: str, docs: List[Document], top_n: int = 3, model: str = "gpt-4o") -> List[Document]:
    prompt_template = PromptTemplate(
        input_variables=["query", "doc"],
        template="""On a scale of 1-10, rate the relevance of the following document to the query.
        Query: {query}
        Document: {doc}
        Relevance Score:"""
    )
    llm = ChatOpenAI(temperature=0, model_name=model)
    llm_chain = prompt_template | llm.with_structured_output(RatingScore)

    scored_docs = []
    for doc in docs:
        score = llm_chain.invoke({"query": query, "doc": doc.page_content}).relevance_score
        try:
            score = float(score)
        except ValueError:
            score = 0.0
        scored_docs.append((doc, score))

    reranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in reranked_docs[:top_n]]


class CrossEncoderRetriever(BaseRetriever, BaseModel):
    vectorstore: Any = Field(description="Vector store for initial retrieval")
    cross_encoder: Any = Field(description="Cross-encoder model for reranking")
    k: int = Field(default=5, description="Number of documents to retrieve initially")
    rerank_top_k: int = Field(default=3, description="Number of documents to return after reranking")

    class Config:
        arbitrary_types_allowed = True

    def get_relevant_documents(self, query: str) -> List[Document]:
        initial_docs = self.vectorstore.similarity_search(query, k=self.k)
        pairs = [[query, doc.page_content] for doc in initial_docs]
        scores = self.cross_encoder.predict(pairs)
        scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:self.rerank_top_k]]

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError("Async retrieval not implemented")


def create_cross_encoder_retriever(vectorstore: Any, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", k: int = 10, rerank_top_k: int = 5) -> CrossEncoderRetriever:
    cross_encoder = CrossEncoder(model_name)
    return CrossEncoderRetriever(vectorstore=vectorstore, cross_encoder=cross_encoder, k=k, rerank_top_k=rerank_top_k)


if __name__ == "__main__":
    print("Rerank examples module loaded.")
