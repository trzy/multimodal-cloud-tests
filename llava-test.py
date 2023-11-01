#
# llava-test.py
#
# Query LLaVA worker N times with a series of prompts (each with image) randomly shuffled each
# pass.
# 

import base64
from io import BytesIO
import numpy as np
from PIL import Image
import random
import requests
import json
import time

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def load_image_base64(image_file):
    with open(image_file, "rb") as fp:
        data = fp.read()    
    return base64.b64encode(data).decode("utf-8")

def get_conversation_mode(model_name):
    if 'llama-2' in model_name.lower():
        return "llava_llama_2"
    elif "v1" in model_name.lower():
        return "llava_v1"
    elif "mpt" in model_name.lower():
        return "mpt"
    else:
        return "llava_v0"

def get_conversation_template(model_name):
    return conv_templates[get_conversation_mode(model_name = model_name)]

def run_query(image_data: str, prompt: str) -> (str, str, bool, float):
    # Create LLaVA prompt with image token
    convo = get_conversation_template(model_name = options.model).copy()
    convo.append_message(convo.roles[0], DEFAULT_IMAGE_TOKEN + prompt)
    convo.append_message(convo.roles[1], None)
    complete_prompt = convo.get_prompt()

    # Create payload for LLaVA worker
    headers = { "User-Agent": "llava-test.py" }
    payload = {
        "model": options.model,
        "prompt": complete_prompt,
        "temperature": 0.8, # should correspond to llama.cpp default
        "top_p": 0.95,      # llama.cpp default
        "max_new_tokens": min(int(options.max_new_tokens), 1536),
        "stop": convo.sep if convo.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else convo.sep2,
        "images": [ image_data ]
    }

    # Outputs
    output = "No output received"
    error = True
    request_time = -1

    # Make request and measure time taken
    t0 = time.perf_counter()
    try:
        # Stream output
        headers = {"User-Agent": "LLaVA Client"}
        response = requests.post(f"http://{options.host}:{options.port}/worker_generate_stream",
            headers=headers, json=payload, stream=True, timeout=10)
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    output = data["text"][len(complete_prompt):].strip()
                    error = False
                else:
                    output = data["text"] + f" (error_code: {data['error_code']})"
                    error = True
                    break
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        output = f"{e}"
        error = True
    t1 = time.perf_counter()
    request_time = t1 - t0

    return (output, complete_prompt, error, request_time)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default="8080")
    parser.add_argument("--model", type=str, default="models/llava-v1.5-7b/")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    options = parser.parse_args()

    prompts = [
        ("car.jpg",         "Describe this image."),
        ("car.jpg",         "What color is the car? Reply with a SINGLE WORD."),
        ("car.jpg",         "What make and model is this car? TWO WORDS ONLY."),
        ("car.jpg",         "What is the country of origin of this car?"),
        ("car.jpg",         "Generate a JSON object with the following information about this vehicle: make, model, year, wheels, vehicle_class, country_of_origin."),
        ("Food1.jpeg",      "Describe this image."),
        ("Food1.jpeg",      "Enumerate what is on this table."),
        ("Food1.jpeg",      "Generate a JSON object describing what is on the table."),
        ("Food1.jpeg",      "What is the calorie count of each object on this table?"),
        ("Food2.jpeg",      "Describe this image."),
        ("Food2.jpeg",      "Enumerate what is on this table."),
        ("Food2.jpeg",      "Generate a JSON object describing what is on the table."),
        ("Food2.jpeg",      "What is the calorie count of each object on this table?"),
        ("Pedestrian1.jpeg","How many people are in this image?"),
        ("Pedestrian1.jpeg","How many people are in this image? ONE WORD ANSWER."),
        ("Pedestrian1.jpeg","Describe this image"),
        ("Pedestrian1.jpeg","Describe what is dangerous about this situation."),
        ("Pedestrian1.jpeg","Describe the vehicle in this picture."),
        ("Pedestrian1.jpeg","Which vehicle is nearest to a pedestrian?"),
    ]
    random.shuffle(prompts)

    timings = []           # total timings
    chars_per_second = []  # tok/sec

    t0 = time.perf_counter()
    i = 0
    max_count = 500
    while i < max_count:
        for (image_file, prompt) in prompts:
            image_data = load_image_base64(image_file = image_file)
            response, input_prompt, is_error, timing = run_query(image_data = image_data, prompt = prompt)
            chars_per_second.append((len(input_prompt) + len(response)) / timing)
            timings.append(timing)
            print(f"({i}/{max_count}) [{image_file}, \"{prompt}\"] -> {'Error' if is_error else ''}{response} ({timing:.2f})")
            i += 1
    t1 = time.perf_counter()
    total_time = t1 - t0
    
    print("\n===\n")
    print("Timing Results:")
    print(f"  Mean   = {np.mean(timings):.2f} s, {np.mean(chars_per_second):.2f} chars/s")
    print(f"  Median = {np.median(timings):.2f} s, {np.median(chars_per_second):.2f} char/s")
    print(f"  1%     = {np.quantile(timings, 0.01):.2f} s, {np.quantile(chars_per_second, 0.01):.2f} chars/s")
    print(f"  5%     = {np.quantile(timings, 0.05):.2f} s, {np.quantile(chars_per_second, 0.05):.2f} chars/s")
    print(f"  10%    = {np.quantile(timings, 0.1):.2f} s, {np.quantile(chars_per_second, 0.1):.2f} chars/s")
    print(f"  90%    = {np.quantile(timings, 0.9):.2f} s, {np.quantile(chars_per_second, 0.9):.2f} chars/s")
    print(f"  95%    = {np.quantile(timings, 0.95):.2f} s, {np.quantile(chars_per_second, 0.95):.2f} chars/s")
    print(f"  99%    = {np.quantile(timings, 0.99):.2f} s, {np.quantile(chars_per_second, 0.99):.2f} chars/s")
    print(f"  Total  = {total_time:.2f} s")
