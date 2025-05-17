import os
import time
from typing import Any
import requests
from collections import namedtuple

from eval_types import MessageList, SamplerBase, SamplerResponse

# Create namedtuple classes to mimic OpenAI response structure
TokenDetails = namedtuple('TokenDetails', ['cached_tokens', 'reasoning_tokens'], defaults=[0, 0])
Usage = namedtuple('Usage', ['input_tokens', 'output_tokens', 'total_tokens', 'input_tokens_details', 'output_tokens_details'], defaults=[0, 0, 0, None, None])

class AISDKSampler(SamplerBase):
    """
    Sample from a Next.js project using Vercel AI SDK, sending requests to a local API endpoint
    """

    def __init__(
        self,
        model: str = "gpt-4.1",
        system_message: str | None = None,
        temperature: float = 1,
        max_tokens: int = 1024,
        api_url: str = "http://localhost:3000/api/sample",
        prompt: str | None = None,
    ):
        """Create a sampler that talks to a Vercel AI SDK-powered endpoint.

        Parameters
        ----------
        prompt : str | None
            Optional free-form text that will be forwarded to the backend as the
            ``prompt`` field of the JSON payload. This lets callers supply a
            custom system/developer prompt without modifying the message list.
        """
        self.api_url = api_url
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Check environment variable for prompt if not provided directly
        if prompt is None and "AI_SDK_PROMPT" in os.environ:
            self.prompt = os.environ.get("AI_SDK_PROMPT")
        else:
            self.prompt = prompt
            
        self.image_format = "url"

    def _handle_image(
        self,
        image: str,
        encoding: str = "base64",
        format: str = "png",
        fovea: int = 768,
    ) -> dict[str, Any]:
        new_image = {
            "type": "input_image",
            "image_url": f"data:image/{format};{encoding},{image}",
        }
        return new_image

    def _handle_text(self, text: str) -> dict[str, Any]:
        return {"type": "input_text", "text": text}

    def _pack_message(self, role: str, content: Any) -> dict[str, Any]:
        return {"role": role, "content": content}
    
    def _format_usage_object(self, usage_data: dict | None) -> Usage:
        """
        Format the usage data from the API response to match the structure expected by get_usage_dict
        """
        if usage_data is None:
            return None
            
        # Create token details objects
        input_tokens_details = TokenDetails(cached_tokens=0, reasoning_tokens=0)
        output_tokens_details = TokenDetails(cached_tokens=0, reasoning_tokens=0)
        
        # Get token counts from usage data or use defaults
        input_tokens = usage_data.get('promptTokens', 0)
        output_tokens = usage_data.get('completionTokens', 0)
        total_tokens = usage_data.get('totalTokens', input_tokens + output_tokens)
        
        # Create and return the Usage object
        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            input_tokens_details=input_tokens_details,
            output_tokens_details=output_tokens_details
        )

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        if self.system_message:
            message_list = [
                self._pack_message("developer", self.system_message)
            ] + message_list
            
        # Prepare the data to send to the Next.js API
        payload = {
            "messages": message_list,
            "userSelectedModel": self.model,
            "temperature": self.temperature,
            "maxTokens": self.max_tokens,
        }
        
        # Forward optional prompt if provided
        if self.prompt is not None:
            payload["prompt"] = self.prompt
        
        trial = 0
        while True:
            try:
                # Send request to the Next.js API
                response = requests.post(
                    self.api_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                
                # Check if the request was successful
                response.raise_for_status()
                
                # Parse the response
                response_data = response.json()

                # Handle annotations which might be a list or dictionary
                auto_selected_model = None
                actual_model = None
                annotations = response_data.get("annotations")
                if isinstance(annotations, list):
                    for item in annotations:
                        if isinstance(item, dict):
                            if item.get("type") == "autoSelectedModel":
                                auto_selected_model = item.get("value")
                            elif item.get("type") == "actualModel":
                                actual_model = item.get("value")
                
                # print(f"auto_selected_model: {auto_selected_model}")
                
                # Get usage data and format it to match expected structure
                usage_data = response_data.get("metadata", {}).get("usage")
                formatted_usage = self._format_usage_object(usage_data)
                
                # Include the auto-selected model info in message metadata for dashboard display
                response_metadata = {
                    "usage": formatted_usage,
                    "auto_selected_model": auto_selected_model,
                    "actual_model": actual_model
                }
                
                return SamplerResponse(
                    response_text=response_data.get("text", ""),
                    response_metadata=response_metadata,
                    actual_queried_message_list=message_list,
                )
            except requests.exceptions.RequestException as e:
                if "400" in str(e):
                    print("Bad Request Error", e)
                    return SamplerResponse(
                        response_text="",
                        response_metadata={"usage": None},
                        actual_queried_message_list=message_list,
                    )
                exception_backoff = 2**trial  # exponential backoff
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
