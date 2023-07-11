import json
import os

from dotenv import load_dotenv
import chainlit as cl
import requests

from langchain import PromptTemplate, LLMChain, OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import argparse

load_dotenv()

prompt_template = "{input}?"

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

# base_model = AutoModelForCausalLM.from_pretrained(
#     "/Users/rafaelmarins/PycharmProjects/weather_forecast_ai_chat/llama",
#     use_safetensors=True,
# )

# model_id = "bigscience/bloom-1b7"
#
# tokenizer = AutoTokenizer.from_pretrained(model_id)

# tokenizer = AutoTokenizer.from_pretrained("/Users/rafaelmarins/PycharmProjects/weather_forecast_ai_chat/llama")

model_name_or_path = "/home/ryan/repos/text-generation-webui/models/falcon-40b"
# You could also download the model locally, and access it there
# model_name_or_path = "/path/to/TheBloke_falcon-40b-instruct-GPTQ"

model_basename = "gptq_model-4bit--1g"

use_triton = False

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
                                           model_basename=model_basename,
                                           use_safetensors=True,
                                           trust_remote_code=True,
                                           device="cuda:0",
                                           use_triton=use_triton,
                                           quantize_config=None)


@cl.langchain_factory(use_async=True)
def main():
    # llm = OpenAI(temperature=0)
    # llm = AutoModelForCausalLM.from_pretrained("/Users/rafaelmarins/PycharmProjects/weather_forecast_ai_chat/llama", device_map="auto", load_in_4bit=True)
    llm = model
    chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt_template))
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
