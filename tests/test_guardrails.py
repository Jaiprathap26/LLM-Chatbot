import pytest
from unittest.mock import MagicMock, patch
from guardrails.model_armor import ModelArmorGuardrails


@patch('os.getenv')
@patch('guardrails.model_armor.modelarmor_v1.ModelArmorClient')
def test_sanitize_input_safe(mock_client_class, mock_getenv):
    from google.cloud import modelarmor_v1
    mock_getenv.return_value = "projects/p/locations/l/templates/t"
    mock_client = mock_client_class.return_value

    mock_response = MagicMock()
    mock_response.sanitization_result.filter_match_state = modelarmor_v1.FilterMatchState.NO_MATCH_FOUND
    mock_client.sanitize_user_prompt.return_value = mock_response

    guardrails = ModelArmorGuardrails()
    result = guardrails.sanitize_input("hello")

    assert result["is_safe"] is True
    assert result["sanitized_text"] == "hello"


@patch('os.getenv')
@patch('guardrails.model_armor.modelarmor_v1.ModelArmorClient')
def test_sanitize_input_blocked(mock_client_class, mock_getenv):
    from google.cloud import modelarmor_v1
    mock_getenv.return_value = "projects/p/locations/l/templates/t"
    mock_client = mock_client_class.return_value

    mock_response = MagicMock()
    mock_response.sanitization_result.filter_match_state = modelarmor_v1.FilterMatchState.MATCH_FOUND
    mock_client.sanitize_user_prompt.return_value = mock_response

    guardrails = ModelArmorGuardrails()
    result = guardrails.sanitize_input("bad words")

    assert result["is_safe"] is False
    assert result["sanitized_text"] == "[BLOCKED BY MODEL ARMOR]"


def test_sanitize_input_no_template():
    with patch('os.getenv', return_value=None):
        guardrails = ModelArmorGuardrails()
        result = guardrails.sanitize_input("hello")
        assert result["is_safe"] is True
        assert result["sanitized_text"] == "hello"
