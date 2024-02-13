import tqdm

from ai_prototypes.utils.file import load_data, write_data
from scripts.language.translate import translate


def main():
    data = load_data("data/scienceqa/llava_test_CQM-A_ko.json")

    required_str = "<image>"

    num_error = 0
    for d in tqdm.tqdm(data):
        for c in d['conversations']:
            # lines = c['value'].split('\n')
            # if "번역" in lines[0]:
            #     c['value'] = '\n'.join(lines[1:])

            # if "value_en" in c and required_str in c['value_en']:
            #     if c['value_en'].startswith(required_str) and not c['value'].startswith(required_str):
            #         c['value'] = required_str + "\n" + c['value']

            if "\n가. " in c["value"]:
                num_error += 1

                print(f"[ID] {d['id']}")
                print(f"[EN] {c['value_en']}")
                print(f"[KR] {c['value']}")

                c['value'] = c["value"].replace("\n가. ", "\nA. ")
                c['value'] = c["value"].replace("\n나. ", "\nB. ")
                c['value'] = c["value"].replace("\n다. ", "\nC. ")
                c['value'] = c["value"].replace("\n라. ", "\nD. ")
                c['value'] = c["value"].replace("\n마. ", "\nE. ")

                print(f"[KR (new)] {c['value']}")

            if "value_en" in c:
                if len(c['value']) > len(c['value_en']) * 4:
                    num_error += 1

                    print(f"[ID] {d['id']}")
                    print(f"[EN] {c['value_en']}")
                    print(f"[KR] {c['value']}")

                    translated, model = translate(c['value_en'], keep_str="<image>\n")
                    if translated:
                        c['value'] = translated
                        c["translator"] = model

                    print(f"[KR (new)] {c['value']}")

            if "value_en" in c and required_str in c['value_en']:
                assert required_str in c['value']
            if "value_en" in c and required_str not in c['value_en']:
                assert required_str not in c['value']

    print(f"{num_error} / {len(data)}")
    write_data(data, "data/scienceqa/llava_test_CQM-A_ko_.json")


if __name__ == "__main__":
    main()
