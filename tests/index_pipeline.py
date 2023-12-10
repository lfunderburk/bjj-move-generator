import unittest
from unittest.mock import patch, MagicMock
from haystack import Pipeline
from haystack.nodes import Crawler, PreProcessor, EmbeddingRetriever
from milvus_haystack import MilvusDocumentStore

class TestExtraction(unittest.TestCase):
    @patch('haystack.Pipeline')
    @patch('haystack.nodes.Crawler')
    @patch('haystack.nodes.PreProcessor')
    @patch('haystack.nodes.EmbeddingRetriever')
    @patch('milvus_haystack.MilvusDocumentStore')
    def test_pipeline(self, MockDocumentStore, MockRetriever, MockPreProcessor, MockCrawler, MockPipeline):
        # Mock the instances
        mock_document_store = MockDocumentStore.return_value
        mock_retriever = MockRetriever.return_value
        mock_preprocessor = MockPreProcessor.return_value
        mock_crawler = MockCrawler.return_value
        mock_pipeline = MockPipeline.return_value

        # Call the function under test
        document_store = MilvusDocumentStore(recreate_index=True, return_embedding=True, similarity="dot_product")
        crawler = Crawler(urls=["https://www.bjjmentalmodels.com/database"], crawler_depth=0, overwrite_existing_files=True, output_dir="crawled_files", filter_urls=[r"\d{4}"])
        preprocessor = PreProcessor(clean_empty_lines=True, clean_whitespace=False, clean_header_footer=True, split_by="word", split_length=500, split_respect_sentence_boundary=True)
        retriever = EmbeddingRetriever(document_store=document_store, embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1")
        indexing_pipeline = Pipeline()
        indexing_pipeline.add_node(component=crawler, name="crawler", inputs=['File'])
        indexing_pipeline.add_node(component=preprocessor, name="preprocessor", inputs=['crawler'])
        indexing_pipeline.add_node(component=retriever, name="retriever", inputs=['preprocessor'])
        indexing_pipeline.add_node(component=document_store, name="document_store", inputs=['retriever'])
        indexing_pipeline.run()

        # Assert the calls
        MockDocumentStore.assert_called_once_with(recreate_index=True, return_embedding=True, similarity="dot_product")
        MockCrawler.assert_called_once_with(urls=["https://www.bjjmentalmodels.com/database"], crawler_depth=0, overwrite_existing_files=True, output_dir="crawled_files", filter_urls=[r"\d{4}"])
        MockPreProcessor.assert_called_once_with(clean_empty_lines=True, clean_whitespace=False, clean_header_footer=True, split_by="word", split_length=500, split_respect_sentence_boundary=True)
        MockRetriever.assert_called_once_with(document_store=mock_document_store, embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1")
        MockPipeline.assert_called_once()
        mock_pipeline.add_node.assert_any_call(component=mock_crawler, name="crawler", inputs=['File'])
        mock_pipeline.add_node.assert_any_call(component=mock_preprocessor, name="preprocessor", inputs=['crawler'])
        mock_pipeline.add_node.assert_any_call(component=mock_retriever, name="retriever", inputs=['preprocessor'])
        mock_pipeline.add_node.assert_any_call(component=mock_document_store, name="document_store", inputs=['retriever'])
        mock_pipeline.run.assert_called_once()

if __name__ == '__main__':
    unittest.main()