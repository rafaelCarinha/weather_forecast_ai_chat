import json
import os

from dotenv import load_dotenv
import chainlit as cl
import requests

from langchain import PromptTemplate, LLMChain, OpenAI, HuggingFacePipeline
from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import argparse

load_dotenv()

model_name_or_path = "/home/ryan/repos/text-generation-webui/models/UltraLM-13B-GPTQ"
model_basename = "ultralm-13b-GPTQ-4bit-128g.no-act.order"

use_triton = False

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
                                           model_basename=model_basename,
                                           use_safetensors=True,
                                           trust_remote_code=False,
                                           device="cuda:0",
                                           use_triton=use_triton,
                                           quantize_config=None)

prompt_template = "{input}?"

input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
output = model.generate(inputs=input_ids, temperature=0.7, max_new_tokens=512)
print(tokenizer.decode(output[0]))

# Inference can also be done using transformers' pipeline

# Prevent printing spurious transformers error when using pipeline with AutoGPTQ
logging.set_verbosity(logging.CRITICAL)

print("*** Pipeline:")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.15
)


is_first_question_asked = False
is_second_question_asked = False
is_third_question_asked = False

is_match_response_from_endpoint = False

first_question = "What's your zip code?"
second_question = "Do you work in tech? (yes/no)"
third_question = "Which company did you last work for? (google, facebook, openai, microsoft)"
decline_message = "Thank you for your time! You're not suitable for the position"
success_message = "Thank you for your time! You have been selected for the position"

first_question_answer = ''
second_question_answer = ''
third_question_answer = ''

local_llm = HuggingFacePipeline(pipeline=pipe)

prompt = PromptTemplate(template=prompt_template, input_variables=["instruction"])

llm_chain = LLMChain(prompt=prompt,
                     llm=local_llm
                     )


@cl.langchain_factory(use_async=True)
def main():
    chain = llm_chain
    return chain


@cl.langchain_postprocess
async def postprocess(output: str):
    global is_first_question_asked
    global is_second_question_asked
    global is_third_question_asked

    global first_question_answer
    global second_question_answer
    global third_question_answer

    user_input = output['input']
    ai_response = output['text']
    print(output)
    return_message = ''
    if not is_first_question_asked:
        return_message = first_question
        is_first_question_asked = True
        await cl.Message(content=return_message).send()
    elif not is_second_question_asked:
        first_question_answer = user_input
        if not chech_fountain_header({"zip_code": f"{user_input}"}):
            reset_global_variabes()
            await cl.Message(content=decline_message).send()
        else:
            return_message = second_question
            is_second_question_asked = True
            await cl.Message(content=return_message).send()
    elif not is_third_question_asked:
        second_question_answer = user_input
        if not chech_fountain_header({"zip_code": f"{first_question_answer}", "work_tech": f"{user_input}"}):
            reset_global_variabes()
            await cl.Message(content=decline_message).send()
        else:
            return_message = third_question
            is_third_question_asked = True
            await cl.Message(content=return_message).send()
    elif is_third_question_asked:
        third_question_answer = user_input
        reset_global_variabes()
        if not chech_fountain_header({"zip_code": f"{first_question_answer}", "work_tech": f"{second_question_answer}",
                                      "company": f"{third_question_answer}"}):
            await cl.Message(content=decline_message).send()
        else:
            await cl.Message(content=success_message).send()


def chech_fountain_header(body):
    print(body)

    response = requests.post("https://chatgpt.fountainheadme.com/api/screener", json=body)
    print(response.text)
    if response.text == '{"status":"match"}':
        return True
    elif response.text == '{"status":"dump"}':
        return False


def reset_global_variabes():
    global is_first_question_asked
    global is_second_question_asked
    global is_third_question_asked

    is_first_question_asked = False
    is_second_question_asked = False
    is_third_question_asked = False
