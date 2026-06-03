import os
import logging
from google.cloud import modelarmor_v1

logger = logging.getLogger(__name__)


class ModelArmorGuardrails:
    def __init__(self, template_name: str = None):
        """
        Initializes the Model Armor client.
        template_name should be the full resource name:
        projects/{project}/locations/{location}/templates/{template}
        """
        self.template_name = template_name or os.getenv("MODEL_ARMOR_TEMPLATE")
        if not self.template_name:
            logger.warning(
                "MODEL_ARMOR_TEMPLATE is not set. Guardrails will be disabled.")
            self.client = None
        else:
            try:
                self.client = modelarmor_v1.ModelArmorClient()
            except Exception as e:
                logger.error(f"Failed to initialize Model Armor client: {e}")
                self.client = None

    def sanitize_input(self, text: str) -> dict:
        """
        Sends the text to Model Armor for sanitization.
        Returns a dict with 'is_safe' (bool) and 'sanitized_text' (str).
        If Model Armor is not configured, it returns the original text as safe.
        """
        if not self.client or not self.template_name:
            return {"is_safe": True, "sanitized_text": text}

        try:
            request = modelarmor_v1.SanitizeUserPromptRequest(
                name=self.template_name,
                user_prompt_data=modelarmor_v1.DataItem(text=text)
            )
            response = self.client.sanitize_user_prompt(request=request)

            is_safe = True

            # Look at filter match state or invocation result
            # Model armor has filter_match_state in sanitization_result
            if hasattr(response, "sanitization_result"):
                res = response.sanitization_result
                # If MATCH_FOUND is returned, it usually implies a
                # block/violation based on the template threshold
                if hasattr(
                        res,
                        "filter_match_state") and res.filter_match_state == modelarmor_v1.FilterMatchState.MATCH_FOUND:
                    is_safe = False

            sanitized_text = text if is_safe else "[BLOCKED BY MODEL ARMOR]"

            return {"is_safe": is_safe, "sanitized_text": sanitized_text}

        except Exception as e:
            logger.error(f"Model Armor API error: {e}")
            # Fail open for resilience if API call fails
            return {"is_safe": True, "sanitized_text": text}
