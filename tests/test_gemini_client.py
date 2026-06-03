import pytest
from unittest.mock import MagicMock, patch
from utils.gemini_client import GeminiClient


@patch('os.getenv')
@patch('utils.gemini_client.vertexai.init')
@patch('utils.gemini_client.TextEmbeddingModel.from_pretrained')
@patch('utils.gemini_client.GenerativeModel')
def test_gemini_initialization(
        mock_gen_model,
        mock_emb_model,
        mock_init,
        mock_getenv):
    mock_getenv.side_effect = lambda k, d=None: "my-project" if k == "GOOGLE_CLOUD_PROJECT" else d

    client = GeminiClient()
    mock_init.assert_called_with(project="my-project", location="us-central1")
    assert client.embedding_model is not None
    assert client.chat_model is not None


@patch('os.getenv')
@patch('utils.gemini_client.vertexai.init')
@patch('utils.gemini_client.TextEmbeddingModel.from_pretrained')
@patch('utils.gemini_client.GenerativeModel')
def test_generate_embedding(
        mock_gen_model,
        mock_emb_model,
        mock_init,
        mock_getenv):
    mock_getenv.side_effect = lambda k, d=None: "my-project" if k == "GOOGLE_CLOUD_PROJECT" else d

    mock_emb_instance = mock_emb_model.return_value
    mock_embedding = MagicMock()
    mock_embedding.values = [0.1, 0.2, 0.3]
    mock_emb_instance.get_embeddings.return_value = [mock_embedding]

    client = GeminiClient()
    result = client.generate_embedding("hello")
    assert result == [0.1, 0.2, 0.3]


@patch('os.getenv')
@patch('utils.gemini_client.vertexai.init')
@patch('utils.gemini_client.TextEmbeddingModel.from_pretrained')
@patch('utils.gemini_client.GenerativeModel')
def test_generate_chat_response(
        mock_gen_model,
        mock_emb_model,
        mock_init,
        mock_getenv):
    mock_getenv.side_effect = lambda k, d=None: "my-project" if k == "GOOGLE_CLOUD_PROJECT" else d

    mock_chat_instance = mock_gen_model.return_value
    mock_response = MagicMock()
    mock_response.text = "Hi there"
    mock_response.usage_metadata.prompt_token_count = 10
    mock_response.usage_metadata.candidates_token_count = 5
    mock_chat_instance.generate_content.return_value = mock_response

    mock_count_response1 = MagicMock()
    mock_count_response1.total_tokens = 10
    mock_count_response2 = MagicMock()
    mock_count_response2.total_tokens = 5

    mock_chat_instance.count_tokens.side_effect = [mock_count_response1, mock_count_response2]

    client = GeminiClient()

    buffer = [{"role": "user", "content": "hello"}]
    hist = ["user: hi", "assistant: hello"]

    result = client.generate_chat_response(buffer, hist)

    assert result["response_text"] == "Hi there"
    assert result["input_tokens"] == 10
    assert result["output_tokens"] == 5
    mock_chat_instance.generate_content.assert_called_once()
