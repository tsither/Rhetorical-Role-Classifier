import json

def read_json(FILEPATH, type='r', key_as_int=False):
    with open(FILEPATH, type) as file:
        data = json.load(file)

    #Covert the keys to integers for efficient document processing
    if key_as_int:
        if isinstance(data, dict):
            new_data = {int(key): value for key, value in data.items()}
            return new_data

    return data


def max_length(documents, tokenizer):
    """
    Generate the maximum length of each sentence in each document. This is necessary to make sure there is a fixed sentence-length 
    for each document before we pass the sentence embeddings through the model.

    Returns: {document index: length of longest sentence}

    """
    max_length_dict = {}
    for index, sentences in documents.dict.items():
        sizes = []

        for sentence in sentences:
            inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
            sizes.append(inputs['input_ids'].size(1))

        max_length_dict[index] = max(sizes)

    return max_length_dict


def write_dictionary_to_json(dictionary, file_path):
    """
    Write a Python dictionary to a JSON file.

    Parameters:
    - dictionary (dict): The dictionary to be written to the JSON file.
    - file_path (str): The path to the JSON file.

    Returns:
    - None
    """
    with open(file_path, 'w') as json_file:
        json.dump(dictionary, json_file, indent=2)

    pass
