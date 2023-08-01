import concurrent.futures
import configparser
import json
import os
import random
import re
from typing import List, Optional

from datasets import Dataset, concatenate_datasets, load_dataset
from tqdm.auto import tqdm

from utils import get_azure_response, compute_rouge, get_api2d_response, get_openai_response


def make_template(
        template: str,
        origin: List,
        machine: List,
        num_seed: int,
        num_machine: int,
) -> str:
    origin = random.sample(origin, k=num_seed)
    if len(machine) > 0:
        machine = random.sample(machine, k=num_machine)
        datas = origin + machine
    else:
        datas = origin

    prompt = ""
    for idx, data in enumerate(datas):
        prompt += f"{str(idx + 1)}. {data}\n"

    prompt += f"{str(idx + 2)}. "

    return template.format(prompt=prompt)


def save_results(results: List[str]):
    path = os.path.join('data', machine_generate_path)
    fp = open(path, 'a', encoding='utf8')
    for result in results:
        fp.write(json.dumps(result, ensure_ascii=False) + '\n')


def find_word_in_string(w, s):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search(s)


def post_process(text):
    """
    对数据进行后处理
    :param text: 待处理文本
    :return:
    """
    raw_texts = re.split(r"\n\d+\s?\. ", text)
    texts = []

    for text in raw_texts:
        text = re.sub(r"\s+", " ", text).strip()
        text = text.strip().capitalize()
        if text == "":
            continue

        if any(find_word_in_string(word, text) for word in
               ["i'm sorry", "i am sorry", "I am sorry", "I'm sorry", "我很抱歉", "AI"]):
            continue

        try:
            persona_match = re.search(r"'persona': '([^']+)'", text)
            persona = persona_match.group(1) if persona_match else ""

            situation_match = re.search(r"'situation': '([^']+)'", text)
            situation = situation_match.group(1) if situation_match else ""

            if persona == "" or situation == "":
                continue

            text = {
                'persona': persona,
                'situation': situation
            }

            # 利用多线程判断生成的句子与已有数据之间的相似性，仅保留足够不相似的数据
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(
                    compute_rouge,
                    [text['persona'] + text['situation']] * len(origin + machine),
                    [label['persona'] + text['situation'] for label in origin + machine]
                ))

                if any(result > 0.5 for result in results):  # 保证新生成的数据足够不相似
                    continue

        except Exception as e:
            print(e)
            continue

        texts.append(text)

    return texts


def process_dataset(dataset: Dataset) -> List:
    """

    :param dataset: 限定传入的数据未 Dataset 类型
    :return: 限定返回为 List 类型
    """
    results = []
    for data in dataset:
        data = {
            'persona': data['persona'],
            'situation': data['situation']
        }
        results.append(data)

    return results


if __name__ == '__main__':
    api_style = 'openai'
    # load api
    config = configparser.ConfigParser()
    config.read('config.ini')
    if api_style == "openai":
        apikey = config.get('OPENAI', 'apikey')
    elif api_style == "api2d":
        url = config.get("API2D", "url")
        apikey = config.get("API2D", "apikey")
    elif api_style == "azure":
        url = config.get("AZURE", "url")
        apikey = config.get("AZURE", "apikey")
        url16k = config.get("AZURE", "url16k")
        apikey16k = config.get("AZURE", "apikey16k")

    # load template
    template_name = 'mental_health_prompt_zh.json'
    template_path = os.path.join('templates', template_name)
    with open(template_path, 'r') as r:
        template = json.load(fp=r)['prompt']

    # load data
    origin_path = 'mentalhealth_seeds_zh.json'
    machine_generate_path = 'machine_generate_mentalhealth_zh.json'

    dataset = load_dataset(
        'json',
        data_files=os.path.join('data', origin_path),
        split='train'
    )
    origin = process_dataset(dataset=dataset)

    machine_dataset_path = os.path.join('data', machine_generate_path)
    if os.path.exists(machine_dataset_path):
        machine_dataset = load_dataset(
            'json',
            data_files=machine_dataset_path,
            split='train'
        )
        machine = process_dataset(machine_dataset)
    else:
        machine = []

    num_seed = 4
    num_machine = 1

    num_generate = 1000

    results = []

    with tqdm(total=num_generate) as pbar:
        while True:
            content = make_template(
                template=template,
                origin=origin,
                machine=machine,
                num_seed=num_seed,
                num_machine=num_machine
            )
            # 判断具体使用的哪个接口
            if api_style == 'azure':
                responses = get_azure_response(
                    url=url,
                    apikey=apikey,
                    content=content,
                    _verbose=False,
                    temperature=0.1,
                )
            elif api_style == "api2d":
                response = get_api2d_response(
                    url=url,
                    apikey=apikey,
                    content=content
                )
            elif api_style == 'openai':
                response = get_openai_response(
                    apikey=apikey,
                    content=content
                )
            else:
                print("api_style 填写错误")

            responses = post_process(response)  # 对获取到的数据进行后处理

            # extend machine pools
            machine.extend(responses)

            save_results(responses)

            pbar.update(len(responses))

            if len(machine) >= num_generate:
                break
