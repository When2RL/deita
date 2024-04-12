import json
import logging
import hashlib
from datasets import load_dataset
from typing import Callable


logger = logging.getLogger(__name__)


def reformat_to_sharegpt(data_dict):
    old_conversations = data_dict['messages']
    conversations = []
    for i, turn in enumerate(old_conversations):
        role = turn['role']
        content = turn['content']
        # sanity checks
        if i % 2 == 0 and role != "user":
            raise ValueError(f"Expected user role, got {role}")
        if i % 2 == 1 and role not in ["assistant", "system"]:
            raise ValueError(f"Expected assistant/system role, got {role}")
        # append
        conversations.append({
            'from': "human" if role == "user" else "gpt",
            'value': content
        })
    data_dict['conversations'] = conversations

    ### add full id
    text_chosen = data_dict['chosen']
    text_rejected = data_dict['rejected']
    full_encoded = f"{text_chosen} {text_rejected}"
    full_encoded_id = hashlib.sha256(full_encoded.encode("utf-8")).hexdigest()
    data_dict['full_id'] = full_encoded_id
    
    return data_dict


class BasePipeline:
    
    def __init__(self, name: str, data_path: str, **kwargs) -> None:
        
        self.name = name
        self.data_path = data_path
    
    # def _load_data(self, data_path: str) -> None:
        
    #     """
    #     Load data from data_path.
        
    #     data_path: str - path to json data file.
    #     """
        
    #     try:
    #         with open(data_path, "r") as f:
    #             data = json.load(f)
    #     except json.JSONDecodeError:
    #         with open(data_path, "r") as f:
    #             data = [json.loads(line) for line in f]
        
    #     return data

    def _load_data(self, data_path: str) -> None:
        
        """
        Load data from data_path.
        
        data_path: str - path to json data file.
        """
        if data_path.endswith(".json"):
            try:
                with open(data_path, "r") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                with open(data_path, "r") as f:
                    data = [json.loads(line) for line in f]
            return data
        else:
            when2rl_dataset = load_dataset(
                data_path,
            )
            train_split = "train"
            for split in when2rl_dataset.keys():
                if "train" in split:
                    train_split = split
                    break
            print(f"Using data from split={train_split}.")
            when2rl_dataset = when2rl_dataset[train_split]
            
            when2rl_dataset_reformatted = when2rl_dataset.map(
                reformat_to_sharegpt,
                num_proc=8,
                keep_in_memory=True,
                desc="Reformatting to ShareGPT format"
            )
            when2rl_dataset_reformatted = when2rl_dataset_reformatted.select_columns(
                ['conversations', 'full_id', 'prompt_id']
            )
            
            loaded_data = when2rl_dataset_reformatted.to_list()
            return loaded_data   # huggingface dataset path
    
    
    def _load_other_data(self, other_data_path: str) -> None:
        raise NotImplementedError
        
    def _save_data(self, data_path: str, data_format: str) -> None:
        raise NotImplementedError
    
    def _preprocess(self, json_data, other_data) -> None:
        raise NotImplementedError
    
    def _forward(self, preprocessed_data) -> None:
        raise NotImplementedError
    
    def run(self) -> None:
        
        json_data = self._load_data(self.data_path)
        
        other_data = None
        if hasattr(self, "other_data_path"):
            other_data = self._load_other_data(self.other_data_path)
        
        preprocessed_data = self._preprocess(json_data, other_data)
        results = self._forward(preprocessed_data)
        self._save_data(json_data, results)
        logger.info(f"Pipeline {self.name} run complete.")
        
class PipelineRegistry:
    
    registry = {}
    
    @classmethod
    def register(cls, name: str, pipline_class: Callable):
        
        if name in cls.registry:
            raise ValueError(f"Pipeline {name} already registered.")
        cls.registry[name] = pipline_class
    
    @classmethod
    def get_pipeline(cls, name: str):
        
        if name not in cls.registry:
            raise ValueError(f"Pipeline {name} not registered.")
        return cls.registry[name]
    
