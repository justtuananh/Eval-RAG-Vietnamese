import json
from typing import Any, Callable, Dict, Iterator, List, Optional, Type, Union
import requests
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.pydantic_v1 import Extra
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_core.utils import get_from_dict_or_env
from functools import wraps

def pre_init(func: Callable) -> Any:
    """Decorator to run a function before model initialization.

    Args:
        func (Callable): The function to run before model initialization.

    Returns:
        Any: The decorated function.
    """

    @root_validator(pre=True)
    @wraps(func)
    def wrapper(cls: Type[BaseModel], values: Dict[str, Any]) -> Dict[str, Any]:
        """Decorator to run a function before model initialization.

        Args:
            cls (Type[BaseModel]): The model class.
            values (Dict[str, Any]): The values to initialize the model with.

        Returns:
            Dict[str, Any]: The values to initialize the model with.
        """
        # Insert default values
        fields = cls.__fields__
        for name, field_info in fields.items():
            # Check if allow_population_by_field_name is enabled
            # If yes, then set the field name to the alias
            if hasattr(cls, "Config"):
                if hasattr(cls.Config, "allow_population_by_field_name"):
                    if cls.Config.allow_population_by_field_name:
                        if field_info.alias in values:
                            values[name] = values.pop(field_info.alias)

            if name not in values or values[name] is None:
                if not field_info.required:
                    if field_info.default_factory is not None:
                        values[name] = field_info.default_factory()
                    else:
                        values[name] = field_info.default

        # Call the decorated function
        return func(cls, values)

    return wrapper

class SambaNovaFastAPI(LLM):
    """
    SambaNova FastAPI large language models.

    To use, you should have the environment variables
    ``FASTAPI_URL`` set with your SambaNova FastAPI URL.
    ``FASTAPI_API_KEY`` set with your SambaNova FastAPI API key.

    https://sambanova.ai/fast-api

    Example:
    .. code-block:: python

        SambaNovaFastAPI(
            fastapi_url=your fastApi CoE endpoint URL,
            fastapi_api_key= set with your fastAPI CoE endpoint API key,
            max_tokens = mas number of tokens to generate
            stop_tokens = list of stop tokens
            model = model name
        )
    """

    fastapi_url: str = ''
    """Url to use"""

    fastapi_api_key: str = ''
    """fastAPI CoE api key"""

    max_tokens: int = 1024
    """max tokens to generate"""

    stop_tokens: list = ['<|eot_id|>']
    """Stop tokens"""

    model: str = 'llama3-8b'
    """LLM model expert to use"""

    stream_api: bool = True
    """use stream api"""

    stream_options: dict = {'include_usage': True}
    """stream options, include usage to get generation metrics"""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {'model': self.model, 'max_tokens': self.max_tokens, 'stop': self.stop_tokens}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return 'Sambastudio Fast CoE'

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values['fastapi_url'] = get_from_dict_or_env(
            values, 'fastapi_url', 'FASTAPI_URL', default='https://fast-api.snova.ai/v1/chat/completions'
        )
        values['fastapi_api_key'] = get_from_dict_or_env(values, 'fastapi_api_key', 'FASTAPI_API_KEY')
        return values

    def _handle_nlp_predict_stream(
        self,
        prompt: Union[List[str], str],
        stop: List[str],
    ) -> Iterator[GenerationChunk]:
        """
        Perform a streaming request to the LLM.

        Args:
            prompt: The prompt to use for the prediction.
            stop: list of stop tokens

        Returns:
            An iterator of GenerationChunks.
        """
        try:
            import sseclient
        except ImportError:
            raise ImportError('could not import sseclient library' 'Please install it with `pip install sseclient-py`.')
        try:
            formatted_prompt = json.loads(prompt)
        except:
            formatted_prompt = [{'role': 'user', 'content': prompt}]

        http_session = requests.Session()
        if not stop:
            stop = self.stop_tokens
        data = {
            'messages': formatted_prompt,
            'max_tokens': self.max_tokens,
            'stop': stop,
            'model': self.model,
            'stream': self.stream_api,
            'stream_options': self.stream_options,
        }
        # Streaming output
        response = http_session.post(
            self.fastapi_url,
            headers={'Authorization': f'Basic {self.fastapi_api_key}', 'Content-Type': 'application/json'},
            json=data,
            stream=True,
        )

        client = sseclient.SSEClient(response)
        close_conn = False

        if response.status_code != 200:
            raise RuntimeError(
                f'Sambanova /complete call failed with status code ' f'{response.status_code}.' f'{response.text}.'
            )

        for event in client.events():
            if event.event == 'error_event':
                close_conn = True
            chunk = {
                'event': event.event,
                'data': event.data,
                'status_code': response.status_code,
            }

            if chunk.get('error'):
                raise RuntimeError(
                    f"Sambanova /complete call failed with status code " f"{chunk['status_code']}." f"{chunk}."
                )

            try:
                # check if the response is a final event in that case event data response is '[DONE]'
                if chunk['data'] != '[DONE]':
                    data = json.loads(chunk['data'])
                    # check if the response is a final response with usage stats (not includes content)
                    if data.get('usage') is None:
                        # check is not "end of text" response
                        if data['choices'][0]['finish_reason'] is None:
                            text = data['choices'][0]['delta']['content']
                            generated_chunk = GenerationChunk(text=text)
                            yield generated_chunk
            except Exception as e:
                raise Exception(f'Error getting content chunk raw streamed response: {chunk}')

    def _stream(
        self,
        prompt: Union[List[str], str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Call out to Sambanova's complete endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.
        """
        try:
            for chunk in self._handle_nlp_predict_stream(prompt, stop):
                if run_manager:
                    run_manager.on_llm_new_token(chunk.text)
                yield chunk
        except Exception as e:
            # Handle any errors raised by the inference endpoint
            raise ValueError(f'Error raised by the inference endpoint: {e}') from e

    def _handle_stream_request(
        self,
        prompt: Union[List[str], str],
        stop: Optional[List[str]],
        run_manager: Optional[CallbackManagerForLLMRun],
        kwargs: Dict[str, Any],
    ) -> str:
        """
        Perform a streaming request to the LLM.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of the stop substrings.
            run_manager: Callback manager for the run.
            **kwargs: Additional keyword arguments. directly passed
                to the sambaverse model in API call.

        Returns:
            The model output as a string.
        """
        completion = ''
        for chunk in self._stream(prompt=prompt, stop=stop, run_manager=run_manager, **kwargs):
            completion += chunk.text
        return completion

    def _call(
        self,
        prompt: Union[List[str], str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call out to Sambanova's complete endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.
        """
        try:
            return self._handle_stream_request(prompt, stop, run_manager, kwargs)
        except Exception as e:
            # Handle any errors raised by the inference endpoint
            raise ValueError(f'Error raised by the inference endpoint: {e}') from e
    