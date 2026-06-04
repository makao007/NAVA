import os
import sys
import torch
from sdnq import SDNQConfig
from sdnq.loader import save_sdnq_model, apply_sdnq_options_to_model

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.nava.modules.t5 import umt5_xxl, T5EncoderModel

def get_sdnq_config ():
    config = dict(
        weights_dtype="uint4",
        quantized_matmul_dtype=None,
        group_size=0,
        hadamard_group_size=128,
        svd_rank=32,
        svd_steps=8,
        dynamic_loss_threshold=None,
        use_svd=False,
        use_hadamard=True,
        quant_conv=False,
        quant_embedding=False,
        use_quantized_matmul=True,
        use_quantized_matmul_conv=False,
        use_dynamic_quantization=False,
        dequantize_fp32=True,
        non_blocking=False,
        add_skip_keys=True,
        modules_to_not_convert=["correction_coefs", "prediction_coefs", "lm_head", "embedding_projection"],
        modules_to_not_use_matmul=["x_embedder"],
        modules_dtype_dict={"int8": ["lm_head"]},
        modules_quant_config={"embed_tokens_per_layer": {"quantization_device": "cpu"}},
        quantization_device="cuda",
        return_device="cuda",
    )
    return config

def load_text_encoder (text_encoder_path:str, text_tokenizer_path:str, dtype=torch.bfloat16, device:str='cuda'):
    text_model = T5EncoderModel(
        text_len=512,
        dtype=dtype,
        device=device,
        checkpoint_path=text_encoder_path,
        tokenizer_path=text_tokenizer_path,
        cpu_offload=True,
        shard_fn=None)
    text_encoder = text_model.model
    text_encoder.requires_grad_(False)
    text_encoder = text_encoder.to(torch.bfloat16).eval()
    text_encoder = torch.compile(text_encoder)   # 只 compile encoder
    text_model.model = text_encoder
    return text_model

def load_text_encoder_model (checkpoint_path:str, dtype=torch.bfloat16, device:str='cuda'):
    model = umt5_xxl(
        encoder_only=True,
        return_tokenizer=False,
        dtype=dtype,
        device=device)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    return model

def text_encoder_to_sdnq (checkpoint_path:str, dtype=torch.bfloat16, device:str='cuda', quant_model:str='text_encoder_4bit'):
    model = load_text_encoder_model(checkpoint_path, dtype, device)
    quantized_model = sdnq_post_load_quant(model, **get_sdnq_config())
    save_sdnq_model(quantized_model, quant_model, is_pipeline=False)

def text_encoder_eval(prompt:str, text_encoder_path:str, text_tokenizer_path:str, dtype=torch.bfloat16, device:str='cuda'):
    text_model = T5EncoderModel(
        text_len=512,
        dtype=dtype,
        device=device,
        checkpoint_path=text_encoder_path,
        tokenizer_path=text_tokenizer_path,
        cpu_offload=True,
        shard_fn=None)

    with torch.no_grad():
        text_list1, text_lens1, spk_pos1 = text_model(prompt, device, return_seqlens=True, return_spk_pos=True)

        text_model.model = sdnq_post_load_quant(text_model.model, **get_sdnq_config())
        text_list2, text_lens2, spk_pos2 = text_model(prompt, device, return_seqlens=True, return_spk_pos=True)

        print(len(text_list1), text_lens1, spk_pos1)
        print(len(text_list2), text_lens2, spk_pos2)

if __name__ == "__main__":
    folder = '/content/Wan2.2-TI2V-5B/'
    prompt = 'a running cat'
    text_encoder_eval(prompt, folder+'models_t5_umt5-xxl-enc-bf16.pth', 'google/umt5-xxl')
