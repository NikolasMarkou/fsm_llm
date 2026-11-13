"""Provider implementations of the LLM layer (plan d454fd62, step 73).

``litellm``: ``LiteLLMInterface`` and the one ``import litellm``; ``request``: the provider
request builder; ``embedder``: ``LiteLLMEmbedder``. This ``__init__`` imports nothing: a patch
target is the module that performs the lookup, never this package.
"""
