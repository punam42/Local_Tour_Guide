import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import AdamW

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):
            return data
        else:
            return [data]

try:
    # Load the combined datas
    combined_data = load_data('combined_data.json')

    # Initialize the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token

    # Tokenize inputs and responses
    input_texts = []
    response_texts = []
    for entry in combined_data:
        if isinstance(entry, dict) and 'heading' in entry and 'content' in entry:
            input_texts.append(entry['heading'])
            response_texts.extend(entry['content'])
        else:
            print(f"Invalid entry: {entry}")

    # Tokenize inputs and responses
    inputs = tokenizer(input_texts, return_tensors='pt', padding=True, truncation=True)
    responses = tokenizer(response_texts, return_tensors='pt', padding=True, truncation=True)

    # Initialize the model
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Set up the optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Training loop
    model.train()
    for epoch in range(3):  # Number of epochs
        total_loss = 0
        for i in range(len(inputs['input_ids'])):
            input_token = inputs['input_ids'][i]
            response_token = responses['input_ids'][i]
            optimizer.zero_grad()
            outputs = model(input_token.unsqueeze(0), labels=response_token)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1} Loss: {total_loss/len(inputs)}')

    # Save the trained model and tokenizer
    model.save_pretrained('chatbot_model')
    tokenizer.save_pretrained('chatbot_tokenizer')

    # Load the trained model and tokenizer
    model = GPT2LMHeadModel.from_pretrained('chatbot_model')
    tokenizer = GPT2Tokenizer.from_pretrained('chatbot_tokenizer')

    def generate_response(input_text):
        input_ids = tokenizer.encode(input_text, return_tensors='pt')
        with torch.no_grad():
            output = model.generate(input_ids, max_length=50, num_return_sequences=1)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        return response

    # Example usage
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        response = generate_response(user_input)
        print(f'Chatbot: {response}')

except Exception as e:
    print(f"An error occurred: {e}")
