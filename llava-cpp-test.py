#
# llava-cpp-test.py
#
# Query llama.cpp with LLaVA model loaded.
#

import base64
import json
import random
import requests
import time


def load_image_base64(image_file):
    with open(image_file, "rb") as fp:
        data = fp.read()    
    return base64.b64encode(data).decode("utf-8")

def run_query(image_data: str, prompt: str) -> (str, str, bool, float):
    # Outputs
    output = "No output received"
    error = True
    request_time = -1

    # Create request
    url = f"http://{options.host}:{options.port}/completion"
    params = { 
        "prompt": f"[img-0]USER: {prompt}\nASSISTANT:",
        "image_data": [
            { "data": image_data, "id": 0 }
        ]
    }

    # Perform request and measure time
    t0 = time.perf_counter()
    try:
        response = requests.post(url, json = params)
        data = json.loads(response.text)
        output = data["content"]
        error = False
    except Exception as e:
        output = f"{e}"
        error = True
    t1 = time.perf_counter()
    request_time = t1 - t0

    return (output, params["prompt"], error, request_time)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default="8080")

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

    timings = []            # total timings
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