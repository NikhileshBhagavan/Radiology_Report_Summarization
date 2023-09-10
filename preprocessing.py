import csv
import re


def pre_process(str):
    if len(str) >= 2 and str[0] == "1" and str[1] == ")":
        substrings = re.split(r" \d\) ", str)
        output_str = " ".join(substrings)
        output_str = output_str[2:]
        output_str = output_str.strip()
        # print(output_str)

        return output_str
    else:
        if len(str) >= 2 and str[0] == "1" and str[1] == ".":
            substrings = re.split(r" \d\. ", str)
            output_str = " ".join(substrings)
            output_str = output_str[2:]
            output_str = output_str.strip()
            # print(output_str)
            return output_str
        else:
            return str


def pre_process_step_two(str):
    substrings = re.split(r"\.\s", str)
    output_str = ".".join(substrings)
    return output_str


def clean_and_process(input_findings_file_path, input_impressions_file_path, output_file_path):

    texts = []
    summaries = []

    with open(input_findings_file_path, "r") as file:
        texts = file.read().split("\n")

    with open(input_impressions_file_path, "r") as file:
        summaries = file.read().split("\n")

    with open(output_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['text', 'summary'])
        for i in range(len(texts)):
            t = pre_process_step_two(texts[i])
            s = pre_process_step_two(pre_process(summaries[i]))
            if s != "" and t != "":
                writer.writerow([t, s])
