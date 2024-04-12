import numpy as np
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import openai


logger = logging.getLogger(__name__)


class Scorer(object):
    
    def __init__(self, model_name_or_path: str, is_vllm: bool  = False, is_sglang: bool = False, **kwargs):
        
        self.is_vllm = is_vllm
        self.is_sglang = is_sglang
        
        if is_vllm:
            from vllm import LLM, SamplingParams
            
            self.llm = LLM(model_name_or_path, max_logprobs=2000)
            self.sampling_params = SamplingParams(max_tokens = 2, logprobs = 2000)
        elif is_sglang:
            sglang_url = kwargs.get("sglang_url")
            self.sglang_client = openai.Client(base_url=sglang_url, api_key="EMPTY")
            self.sglang_sampling_params = {
                "temperature": 1.0,
                "max_tokens": 2,
                "logprobs": 2000,
            }
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        
    def infer_score(self, user_input: str):

        max_length = 2
        
        if self.is_vllm:
            outputs = self.llm.generate(user_input, self.sampling_params)
            
            try:
                logprobs_dict = outputs[0].outputs[0].logprobs[0]
                logprobs_dict = {k: v.logprob for k, v in logprobs_dict.items()}
            except IndexError:
                return -100.0
        elif self.is_sglang:
            outputs = self.sglang_client.completions.create(
                model="default",
                prompt=user_input,
                **self.sglang_sampling_params
            )

            try:
                ## TODO: not really implemented yet
                logprobs_dict = outputs.choices[0].logprobs
            except IndexError:
                return -101.0
        else:
            input_ids = self.tokenizer.encode(user_input, return_tensors = "pt")
            outputs = self.model.generate(input_ids, max_new_tokens = max_length, num_return_sequences = 1, return_dict_in_generate = True, output_scores = True)
            
            try:
                logprobs_dict = outputs.scores[0][0]
            except IndexError:
                return -102.0
        
        
        score_logits = []
        score_template = np.array([1,2,3,4,5,6], dtype=np.float32)
        for k in self.id2score:
            score_logits.append(logprobs_dict.get(k, -np.infty))
        
        try:
            score_logits = np.array(score_logits)
            score_npy = softmax(score_logits, axis=0)
            score_npy = score_npy * score_template

            score_npy = np.sum(score_npy, axis=0)
        except Exception:
            return -103.0
        
        return score_npy
    
    
    def infer_complexity(self, input_text: str):
        
        complexity_template = self.complexity_template
        user_input = complexity_template.format(instruction=input_text)
        
        return self.infer_score(user_input)
        
    def infer_quality(self, input_text: str, resp_text: str):
        
        quality_template = self.quality_template
        user_input = quality_template.format(instruction=input_text, output=resp_text)
        
        return self.infer_score(user_input)

    @property
    def id2score(self):
        raise NotImplementedError
    
    @property
    def complexity_template(self):
        raise NotImplementedError
    
    @property
    def quality_template(self):
        raise NotImplementedError