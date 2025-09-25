from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
from huggingface_hub import login, snapshot_download
class LLM:
    def __init__(self, model_path="YaTharThShaRma999/pretrained_model", gpu_memory=0.1):

        backend_config = TurbomindEngineConfig(cache_max_entry_count=gpu_memory, session_len=4096, enable_prefix_caching=True, dtype='bfloat16')
        self.llm = pipeline(model_path,
                backend_config=backend_config)
        self.gen_config = GenerationConfig(top_p=0.95,
                              top_k=40,
                              temperature=1.5,
                              max_new_tokens=2048,
                              do_sample=True,
                              repetition_penalty=1.5,
                              min_new_tokens=75,
                              min_p=0.1)
    async def generate(self, prompts, gen_config, batch=True, session_id=0):
        """pads to ref segment lenght"""
        if gen_config is None:
            gen_config = self.gen_config
        if batch:
            output = self.llm(prompts, gen_config=gen_config, do_preprocess=False)
            yield output
        else:
            async for response in self.llm.generate(messages=prompts, gen_config=gen_config, session_id=session_id, sequence_start=True, sequence_end=True, do_preprocess=False):
                yield response.response
      
