import json
import os

from dotenv import load_dotenv
import chainlit as cl
import requests
from numba import jit, cuda, NumbaDeprecationWarning, NumbaPendingDeprecationWarning

from langchain import PromptTemplate, LLMChain, HuggingFacePipeline
from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM

import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

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

prompt_template = """
name: Assistant
greeting: Hello, I am an AI insurance bot that will help you save money on your auto
  insurance. Let's start. What is your zip code?
context: "You are an insurance assistant. You are going to have a conversation with\
  \ User to help them get a quote for auto insurance. You must always keep this conversation\
  \ about insurance and nothing else, even if User tries to change the subject. If\
  \ at any point User asks you to end the conversation or to be added to the \"do\
  \ not call\" list then say \"No problem. Goodbye.\"\n\nYou start by saying \"Hello,\
  \ I am an AI insurance bot that will help you save money on your auto insurance.\
  \ Let's start. What is your zip code?\".\n\nWhen User replies ask the following\
  \ questions one at a time after User provides a valid zip code. If the answer is\
  \ not logical then you must ask User to repeat. \n\n- Ask User if their car is insured\
  \ currently.\n- Ask User who their current car insurance company is. \n- Ask User\
  \ if they have had a clean driving record for the last 2 years. \n- Ask User if\
  \ they have been convicted of a DUI in the last 7 years. \n- Ask User how many vehicles\
  \ they have that need to be insured.\n- Ask User if they currently own their home\
  \ or if they are renting. \n- Ask User for their first name. \n\nOnce you get their\
  \ name then the conversation is over. You close it by saying that you will email\
  \ them a quote and then you say goodbye. \n\nProvide a JSON output of the information\
  \ you collected from the user.Your output should be a JSON object with these keys:\n\
  \nname: User's first name.\nzipcode: User's zip code.\ncurrentCarInsuranceCompany:\
  \ The car insurance company that User is currently using.\ncleanDrivingRecord: Whether\
  \ or not they have had a clean driving record for the last two years in boolean.\n\
  convictedOfDUI: Whether or not they have been convicted of a DUI in the last seven\
  \ years in boolean.\nhowmanyveh: How many vehicles they have in an integer.\nhomeowner:\
  \ Whether they are a homeowner in boolean.\n"
{input}?

"""

input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()

# Prevent printing spurious transformers error when using pipeline with AutoGPTQ
logging.set_verbosity(logging.CRITICAL)

print("*** Pipeline:")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
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

prompt = PromptTemplate(template=prompt_template, input_variables=["input"])

llm_chain = LLMChain(prompt=prompt,
                     llm=local_llm
                     )


@cl.langchain_factory(use_async=False)
@jit(target_backend='cuda')
def main():
    chain = llm_chain
    return chain


@cl.langchain_postprocess
async def postprocess(output: str):
    # global is_first_question_asked
    # global is_second_question_asked
    # global is_third_question_asked
    #
    # global first_question_answer
    # global second_question_answer
    # global third_question_answer

    user_input = output['input']
    ai_response = output['text']
    print(output)
    # return_message = ''
    # if not is_first_question_asked:
    #     return_message = first_question
    #     is_first_question_asked = True
    #     await cl.Message(content=return_message).send()
    # elif not is_second_question_asked:
    #     first_question_answer = user_input
    #     if not chech_fountain_header({"zip_code": f"{user_input}"}):
    #         reset_global_variabes()
    #         await cl.Message(content=decline_message).send()
    #     else:
    #         return_message = second_question
    #         is_second_question_asked = True
    #         await cl.Message(content=return_message).send()
    # elif not is_third_question_asked:
    #     second_question_answer = user_input
    #     if not chech_fountain_header({"zip_code": f"{first_question_answer}", "work_tech": f"{user_input}"}):
    #         reset_global_variabes()
    #         await cl.Message(content=decline_message).send()
    #     else:
    #         return_message = third_question
    #         is_third_question_asked = True
    #         await cl.Message(content=return_message).send()
    # elif is_third_question_asked:
    #     third_question_answer = user_input
    #     reset_global_variabes()
    #     if not chech_fountain_header({"zip_code": f"{first_question_answer}", "work_tech": f"{second_question_answer}",
    #                                   "company": f"{third_question_answer}"}):
    #         await cl.Message(content=decline_message).send()
    #     else:
    #         await cl.Message(content=success_message).send()

    await cl.Message(content=ai_response).send()


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
