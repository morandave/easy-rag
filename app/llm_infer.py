import time
import uuid

import requests

from app.auth_util import gen_sign_headers

APP_ID = '3032159355'
APP_KEY = 'ttWATKMXBySyYhtc'
URI = '/vivogpt/completions'
DOMAIN = 'api-ai.vivo.com.cn'
METHOD = 'POST'

def build_simple_template():
    prompt_template = "你是一个准确和可靠的人工智能助手，请准确回答下面的用户问题。\n" \
                        "用户问题：\n" \
                        "{}\n"
    return prompt_template


def build_template():
    prompt_template = "你是一个准确和可靠的人工智能助手，能够借助外部文档回答用户问题，请注意外部文档可能存在噪声事实性错误。" \
                      "如果文档中的信息包含了正确答案，你将进行准确的回答。"\
                      "如果文档中的信息不包含答案，你将生成“文档信息不足，因此我无法基于提供的文档回答该问题。”。" \
                      "如果部分文档中存在与事实不一致的错误，请先生成“提供文档的文档存在事实性错误。”，并生成正确答案。" \
                      "下面给定你相关外部文档，根据文档来回答用户问题。" \
                      "以下是外部文档：\n---" \
                        "{}\n" \
                        "用户问题：\n---" \
                        "{}\n"
    return prompt_template


def build_summary_template():
    prompt_template = "请你将给定的杂乱文本重新整理，使其不丢失任何信息且有较强的可读性，同时要求不丢失关键词。\n" \
                      "以下是杂乱文本：\n---" \
                      "{}\n" \

    return prompt_template


def build_repair_template():

    prompt_template = "你是一个准确和可靠的人工智能助手。" \
                      "请你基于以下材料调整优化用户问题的答案，要求答案尽可能的清晰准确，并且包含正确的关键词。" \
                      "如果没有必要调整则将原答案重复即可。\n" \
                      "以下是材料：\n---" \
                      "{}\n" \
                      "用户问题：\n---" \
                        "{}\n" \
                        "原答案：\n---" \
                        "{}\n"

    return prompt_template


class LLMPredictor(object):
    def __init__(self, device="cuda", **kwargs):

        # self.max_token = 4096
        self.simple_template = build_simple_template()
        self.prompt_template = build_template()
        self.repair_template = build_repair_template()
        self.summary_template = build_summary_template()
        self.kwargs = kwargs
        self.device = device

    def vivo_infer(self, prompt):
        params = {
            'requestId': str(uuid.uuid4())
        }
        # print('requestId:', params['requestId'])

        data = {
            'prompt': prompt,
            'model': 'vivo-BlueLM-TB',
            'sessionId': str(uuid.uuid4()),
            'extra': {
                'temperature': 0.9
            }
        }
        headers = gen_sign_headers(APP_ID, APP_KEY, METHOD, URI, params)
        headers['Content-Type'] = 'application/json'

        start_time = time.time()
        url = 'https://{}{}'.format(DOMAIN, URI)
        response = requests.post(url, json=data, headers=headers, params=params)

        if response.status_code == 200:
            res_obj = response.json()
            if res_obj['code'] == 0 and res_obj.get('data'):
                content = res_obj['data']['content']
                return content
        else:
            # print(response.status_code, response.text)
            return "请求失败，请联系开发者"
        end_time = time.time()
        timecost = end_time - start_time
        print('请求耗时: %.2f秒' % timecost)

    def predict(self, context, query):
        # # 问题长度检查
        # input_ids = self.tokenizer(context, return_tensors="pt", add_special_tokens=False).input_ids
        # if len(input_ids) > self.max_token:
        #     context = self.tokenizer.decode(input_ids[:self.max_token-1])
        #     warnings.warn("texts have been truncted")
        context = self.prompt_template.format(context, query)
        response = self.vivo_infer(context)
        return response

    def get_prompt(self, context, query):

        content = self.prompt_template.format(context, query)

        return content

    def repair_answer(self, context, query, origin_answer):
        # # 问题长度检查
        # input_ids = self.tokenizer(context, return_tensors="pt", add_special_tokens=False).input_ids
        # if len(input_ids) > self.max_token:
        #     context = self.tokenizer.decode(input_ids[:self.max_token-1])
        #     warnings.warn("texts have been truncted")
        context = self.repair_template.format(context, query, origin_answer)
        response = self.vivo_infer(context)
        return response

    def simple_predict(self, query):

        prompt = self.simple_template.format(query)
        response = self.vivo_infer(prompt)
        return response

    def construct_search_docs(self, context):

        context = self.summary_template.format(context)
        response = self.vivo_infer(context)
        return response

    def my_llm_infer(self, prompt, device='cpu'):

        raise NotImplementedError