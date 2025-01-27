import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
import os
import PIL.Image
import numpy as np

def setup_model():
    # 加载模型和处理器
    model_path = "deepseek-ai/Janus-Pro-7B"
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )
    
    # 检查是否支持 MPS
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        vl_gpt = vl_gpt.to(device)
    else:
        device = torch.device("cpu")
        print("MPS 不可用，使用 CPU 进行推理（这可能会很慢）")
        vl_gpt = vl_gpt.to(device)
    
    vl_gpt = vl_gpt.to(torch.float32).eval()  # Apple Silicon 更适合使用 float32
    return vl_gpt, vl_chat_processor, device

def create_prompt(text_prompt):
    # 创建对话格式的提示
    conversation = [
        {
            "role": "User",  
            "content": text_prompt,
        },
        {"role": "Assistant", "content": ""},  
    ]

    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    return sft_format + vl_chat_processor.image_start_tag

@torch.inference_mode()
def generate_image(
    mmgpt,
    vl_chat_processor,
    prompt,
    device,
    temperature=1,
    parallel_size=1,  # 只生成1张图片
    cfg_weight=5,
    image_token_num_per_image=576,
    img_size=384,
    patch_size=16,
):
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).to(device)
    for i in range(parallel_size*2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).to(device)

    outputs = None
    for i in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(
            inputs_embeds=inputs_embeds, 
            use_cache=True, 
            past_key_values=outputs.past_key_values if i != 0 else None
        )
        hidden_states = outputs.last_hidden_state
        
        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        
        logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    # 解码生成的图像
    dec = mmgpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int), 
        shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size]
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    # 保存生成的图像
    os.makedirs('generated_samples', exist_ok=True)
    images = []
    for i in range(parallel_size):
        img_array = dec[i].astype(np.uint8)
        img = PIL.Image.fromarray(img_array)
        save_path = os.path.join('generated_samples', f"img_{i}.jpg")
        img.save(save_path)
        images.append(img)
    
    return images

if __name__ == "__main__":
    # 设置模型
    vl_gpt, vl_chat_processor, device = setup_model()
    
    # 示例提示词
    text_prompt = "A panda eating bamboo"
    
    # 创建提示
    prompt = create_prompt(text_prompt)
    
    # 生成图像
    images = generate_image(vl_gpt, vl_chat_processor, prompt, device)
    print(f"图像已生成并保存在 'generated_samples' 目录中") 
