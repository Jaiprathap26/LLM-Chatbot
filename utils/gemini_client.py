import os
import logging
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from vertexai.generative_models import GenerativeModel, Content, Part, HarmCategory, HarmBlockThreshold
import vertexai

logger = logging.getLogger(__name__)


class GeminiClient:
    def __init__(self):
        self.project = os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

        if not self.project:
            logger.warning(
                "GOOGLE_CLOUD_PROJECT is not set. Gemini API calls will fail.")

        try:
            vertexai.init(project=self.project, location=self.location)
            self.embedding_model = TextEmbeddingModel.from_pretrained(
                "text-embedding-004")
            self.chat_model = GenerativeModel("gemini-1.5-pro")
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {e}")
            self.embedding_model = None
            self.chat_model = None

    def generate_embedding(self, text: str) -> list[float]:
        """Generates a text embedding using Vertex AI text-embedding-004."""
        if not self.embedding_model:
            return []

        try:
            inputs = [TextEmbeddingInput(text, "RETRIEVAL_DOCUMENT")]
            embeddings = self.embedding_model.get_embeddings(inputs)
            if embeddings:
                return embeddings[0].values
            return []
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return []

    def generate_chat_response(
            self,
            buffer_messages: list[dict],
            historical_memories: list[str]) -> dict:
        """
        Generates a response from Gemini given the current conversation buffer and relevant historical memories.
        Returns a dict with 'response_text', 'input_tokens', 'output_tokens'.
        """
        if not self.chat_model:
            return {
                "response_text": "Error: Gemini model not initialized.",
                "input_tokens": 0,
                "output_tokens": 0}

        # Construct System Instruction or Prompt Header to include historical
        # memory
        system_instruction = "You are a helpful AI assistant. Use the following context from past conversations to inform your response if relevant.\n\n"
        if historical_memories:
            system_instruction += "--- Relevant Past Conversations ---\n"
            for mem in historical_memories:
                system_instruction += f"{mem}\n"
            system_instruction += "-----------------------------------\n"

        # Convert buffer messages to Vertex AI Content objects
        contents = []
        for msg in buffer_messages:
            role = msg["role"]
            # Gemini roles are 'user' and 'model'
            if role == "assistant":
                role = "model"
            contents.append(
                Content(
                    role=role, parts=[
                        Part.from_text(
                            msg["content"])]))

        # Instead of system instructions (which may require specific model versions),
        # we can just prepend the context to the first message if it's from the
        # user.
        if contents and contents[0].role == "user" and historical_memories:
            contents[0].parts[0]._text = system_instruction + \
                "\n" + contents[0].parts[0].text

        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }

        try:
            response = self.chat_model.generate_content(
                contents,
                safety_settings=safety_settings
            )

            # Token counting is available via usage_metadata
            input_tokens = response.usage_metadata.prompt_token_count if hasattr(
                response, "usage_metadata") else 0
            output_tokens = response.usage_metadata.candidates_token_count if hasattr(
                response, "usage_metadata") else 0

            return {
                "response_text": response.text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            }
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            return {
                "response_text": f"Sorry, an error occurred: {e}",
                "input_tokens": 0,
                "output_tokens": 0}
