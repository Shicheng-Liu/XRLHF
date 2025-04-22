# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import logging
import torch
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from dschat.utils.model.model_utils import create_hf_model, create_critic_model
from dschat.utils.utils import to_device, load_hf_tokenizer
from deepspeed import get_accelerator
from datasets import load_dataset

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Eval the finetued SFT model")
    parser.add_argument(
        "--model_name_or_path_reward",
        type=str,
        help="Path to reward model",
        required=True,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to test prompts",
        required=True,
    )
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=1,
        help=
        "OPT model has a fixed number (1) of padding tokens at the beginning of the input. "
        "We did not see this in other models but keep it as an option for now.",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--num_beam_groups",
        type=int,
        default=1,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=4,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--penalty_alpha",
        type=float,
        default=0.6,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help='Specify num of return sequences',
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help='Specify num of return sequences',
    )
    parser.add_argument("--language",
                        type=str,
                        default="English",
                        choices=["English", "Chinese", "Japanese"])
    parser.add_argument(
        "--add_eot_token",
        action='store_true',
        help="Add <|endoftext|> as additional special token to tokenizer")


    args = parser.parse_args()

    return args


def load_stuff(model_name_or_path, num_padding_at_beginning,
               additional_special_tokens):

    tokenizer = load_hf_tokenizer(model_name_or_path,
                                  fast_tokenizer=True,
                                  add_special_tokens=additional_special_tokens)
    tokenizer.pad_token = tokenizer.eos_token
    model = create_critic_model(model_name_or_path,
                                tokenizer,
                                None,
                                num_padding_at_beginning,
                                rlhf_training=True,
                                dropout=0.)

    return model, tokenizer

def prepare_singlesample(prompt,
                         good_ans,
                         tokenizer,
                         max_seq_len=512,
                         end_of_conversation_token="<|endoftext|>"):
    chosen_sentence = prompt + good_ans[0] + end_of_conversation_token
    chosen_token = tokenizer(chosen_sentence,
                             max_length=max_seq_len,
                             padding="max_length",
                             truncation=True,
                             return_tensors="pt")

    batch = {}
    batch["input_ids"] = chosen_token["input_ids"]
    batch["attention_mask"] = chosen_token["attention_mask"]

    return batch


def generate(model,
             tokenizer,
             inputs,
             num_beams=1,
             num_beam_groups=1,
             do_sample=False,
             num_return_sequences=1,
             max_new_tokens=100):

    generate_ids = model.generate(inputs.input_ids,
                                  attention_mask=inputs.attention_mask,
                                  num_beams=num_beams,
                                  num_beam_groups=num_beam_groups,
                                  do_sample=do_sample,
                                  num_return_sequences=num_return_sequences,
                                  max_new_tokens=max_new_tokens)

    result = tokenizer.batch_decode(generate_ids,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
    return result


def generate_constrastive_search(model,
                                 tokenizer,
                                 inputs,
                                 top_k=4,
                                 penalty_alpha=0.6,
                                 num_return_sequences=1,
                                 max_new_tokens=100):

    generate_ids = model.generate(inputs.input_ids,
                                  top_k=top_k,
                                  penalty_alpha=penalty_alpha,
                                  num_return_sequences=num_return_sequences,
                                  max_new_tokens=max_new_tokens)

    result = tokenizer.batch_decode(generate_ids,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
    return result


def print_utils(gen_output):
    for i in range(len(gen_output)):
        print()
        print(gen_output[i])
        print()


def main():
    args = parse_args()

    device = torch.device(get_accelerator().device_name(0))

    args.end_of_conversation_token = "<|endoftext|>"
    additional_special_tokens = args.end_of_conversation_token if args.add_eot_token else None

    if "opt" in args.model_name_or_path_reward:
        reward_model, reward_tokenizer = load_stuff(args.model_name_or_path_reward,
                                        args.num_padding_at_beginning,
                                        additional_special_tokens)
    else:
        #from huggingface_hub import login
        #login(token="")
        reward_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path_reward)
        reward_model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path_reward, device_map="auto", torch_dtype="auto")
    reward_model.to(device)
    

    # One observation: if the prompt ends with a space " ", there is a high chance that
    # the original model (without finetuning) will stuck and produce no response.
    # Finetuned models have less such issue. Thus following prompts all end with ":"
    # to make it a more meaningful comparison.
    ds = load_dataset("json", data_files=args.data_path)["train"]
    prompts = ds["prompt"]  
    response_base = ds["response_base"]
    response_sft = ds["response_sft"]
    response_rlhf = ds["response_rlhf"]

    reward_base = []
    reward_finetune = []
    reward_rlhf = []

    for prompt, base_response, sft_response, rlhf_response in tqdm(zip(prompts, response_base, response_sft, response_rlhf),total=len(prompts),desc="Evaulation process"):
        
        
        base_batch = prepare_singlesample(prompt, base_response, reward_tokenizer, max_seq_len=512, end_of_conversation_token=args.end_of_conversation_token)
        base_batch = to_device(base_batch, device)
        reward_model.eval()
        # Run inference
        with torch.no_grad():
            if "opt" in args.model_name_or_path_reward:
                base_outputs = reward_model.forward_value(**base_batch, prompt_length=max(2, args.num_padding_at_beginning))
                reward_base.append(base_outputs["chosen_end_scores"].item())
                
            else:
                base_outputs = reward_model(**base_batch)
                reward_base.append(base_outputs.logits.squeeze(-1).float().cpu().numpy())
                
        
        finetune_batch = prepare_singlesample(prompt, sft_response, reward_tokenizer, max_seq_len=512, end_of_conversation_token=args.end_of_conversation_token)
        finetune_batch = to_device(finetune_batch, device)
        
        # Run inference
        with torch.no_grad():
            if "opt" in args.model_name_or_path_reward:
                finetune_outputs = reward_model.forward_value(**finetune_batch, prompt_length=max(2, args.num_padding_at_beginning))
                reward_finetune.append(finetune_outputs["chosen_end_scores"].item())
                
            else:
                finetune_outputs = reward_model(**finetune_batch)
                reward_finetune.append(finetune_outputs.logits.squeeze(-1).float().cpu().numpy())
                
        #print("finetune answer score: ", finetune_outputs["chosen_end_scores"].item())
        

        
        rlhf_batch = prepare_singlesample(prompt, rlhf_response, reward_tokenizer, max_seq_len=512, end_of_conversation_token=args.end_of_conversation_token)
        rlhf_batch = to_device(rlhf_batch, device)
        
        # Run inference
        with torch.no_grad():
            if "opt" in args.model_name_or_path_reward:
                rlhf_outputs = reward_model.forward_value(**rlhf_batch, prompt_length=max(2, args.num_padding_at_beginning))
                reward_rlhf.append(rlhf_outputs["chosen_end_scores"].item())
            else:
                rlhf_outputs = reward_model(**rlhf_batch)
                reward_rlhf.append(rlhf_outputs.logits.squeeze(-1).float().cpu().numpy())


    print("reward for base model",np.mean(reward_base))
    print("reward for SFT model",np.mean(reward_finetune))
    print("reward for rlhf model",np.mean(reward_rlhf))


if __name__ == "__main__":
    main()
