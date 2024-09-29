import json
import sys

if __name__ == "__main__":
    input_path = "CMeEE-V2/CMeEE-V2_train.json"
    output_path = "output_CMeEE-V2/example_pred.json"

    with open(input_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    new_dict = []
    count = 0
    for data in input_data:
        after_data = {}
        # 处理 text
        text = data["text"]

        if len(text)>=512:
            print(len(text))
            count+=1
        after_data['sentence'] = [s for s in text]
        # 处理 entities idx: [start_idx,end_idx] -> {start_idx, start_idx+1, ... , end_idx}
        entities = data["entities"]
        ner = []
        for entity in entities:
            ner.append({
                'index': [k for k in range(entity['start_idx'], entity['end_idx'])],
                'type': entity['type'],
                'entity': entity['entity']
            })
        after_data['ner'] = ner
        new_dict.append(after_data)
    print(count)
    #sys.exit()

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_dict, f, ensure_ascii=False)

