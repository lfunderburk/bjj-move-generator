from haystack import Pipeline
from haystack.nodes import Crawler, PreProcessor, EmbeddingRetriever
from chroma_haystack import ChromaDocumentStore

# Chroma is used in-memory so we use the same instances in the two pipelines below
document_store = ChromaDocumentStore()

crawler = Crawler(urls=["https://www.bjjmentalmodels.com/database"], 
                  crawler_depth=0, 
                  overwrite_existing_files=True, 
                  output_dir="crawled_files",
                  filter_urls=[r"\d{4}"])
preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=False,
    clean_header_footer=True,
    split_by="word",
    split_length=500,
    split_respect_sentence_boundary=True,
)
retriever = EmbeddingRetriever(document_store=document_store, embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1")

indexing_pipeline = Pipeline()
indexing_pipeline.add_node(component=crawler, name="crawler", inputs=['File'])
indexing_pipeline.add_node(component=preprocessor, name="preprocessor", inputs=['crawler'])
indexing_pipeline.add_node(component=retriever, name="retriever", inputs=['preprocessor'])
indexing_pipeline.add_node(component=document_store, name="document_store", inputs=['retriever'])

indexing_pipeline.run()