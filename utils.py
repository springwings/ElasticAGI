import os
from typing import Dict, Tuple, Union, Optional

import requests
from torch.nn import Module
from transformers import AutoModel

def prefixContent(ef_path,qu,top=3,score=36):
    param  = {"search_dsl":"(content:"+qs[0]+" or title:"+qs[0]+"^3)","count":top,"post_filter":{"__score":str(score)+"_"}}  
    contents = "" 
    ret = requests.post(ef_path,json=param,timeout=10)
    if ret.json()['response']['datas']['total']>0:
        for dt in ret.json()['response']['datas']['lists']:
            contents += dt["content"]
    return '"'+contents+"，根据以上信息如实回答后面的问题。"

def checkdomain(q,domain=["政务","经济"]):
    tokenizer, model = get_model()
    prompt = '"'+q+'"'+",这句话涉及的领域是什么？"
    response, _ = model.chat(tokenizer,
                                   prompt,
                                   history=[],
                                   max_length=2048,
                                   top_p=0.7,
                                   temperature=0.95)
    tmp = response.split("涉及的领域")
    if len(tmp)>1:
        for d in domain:
            if tmp[1].find(d)!=-1:
                return True       
    return False


def auto_configure_device_map(num_gpus: int) -> Dict[str, int]:
    # transformer.word_embeddings 占用1层
    # transformer.final_layernorm 和 lm_head 占用1层
    # transformer.layers 占用 28 层
    # 总共30层分配到num_gpus张卡上
    num_trans_layers = 28
    per_gpu_layers = 30 / num_gpus

    # bugfix: 在linux中调用torch.embedding传入的weight,input不在同一device上,导致RuntimeError
    # windows下 model.device 会被设置成 transformer.word_embeddings.device
    # linux下 model.device 会被设置成 lm_head.device
    # 在调用chat或者stream_chat时,input_ids会被放到model.device上
    # 如果transformer.word_embeddings.device和model.device不同,则会导致RuntimeError
    # 因此这里将transformer.word_embeddings,transformer.final_layernorm,lm_head都放到第一张卡上
    device_map = {'transformer.word_embeddings': 0,
                  'transformer.final_layernorm': 0, 'lm_head': 0}

    used = 2
    gpu_target = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus
        device_map[f'transformer.layers.{i}'] = gpu_target
        used += 1

    return device_map


def load_model_on_gpus(checkpoint_path: Union[str, os.PathLike], num_gpus: int = 2,
                       device_map: Optional[Dict[str, int]] = None, **kwargs) -> Module:
    if num_gpus < 2 and device_map is None:
        model = AutoModel.from_pretrained(checkpoint_path, trust_remote_code=True, **kwargs).half().cuda()
    else:
        from accelerate import dispatch_model

        model = AutoModel.from_pretrained(checkpoint_path, trust_remote_code=True, **kwargs).half()

        if device_map is None:
            device_map = auto_configure_device_map(num_gpus)

        model = dispatch_model(model, device_map=device_map)

    return model


