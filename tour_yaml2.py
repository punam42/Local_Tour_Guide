import yaml
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

class TourGuideDataset(Dataset):
    def __init__(self, inputs, responses, tokenizer, max_length=512):
        self.inputs = inputs
        self.responses = responses
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        response_text = self.responses[idx]
        input_tokens = self.tokenizer(input_text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
        response_tokens = self.tokenizer(response_text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
        return input_tokens.input_ids.squeeze(), response_tokens.input_ids.squeeze()

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
        if isinstance(data, list):
            return data
        else:
            return [data]

def collate_fn(batch):
    inputs, responses = zip(*batch)
    
    # Find the maximum length in the batch
    max_input_length = max([len(input_ids) for input_ids in inputs])
    max_response_length = max([len(response_ids) for response_ids in responses])
    
    # Pad inputs and responses to the maximum length in the batch
    inputs = [torch.nn.functional.pad(input_ids, (0, max_input_length - len(input_ids)), value=tokenizer.pad_token_id) for input_ids in inputs]
    responses = [torch.nn.functional.pad(response_ids, (0, max_response_length - len(response_ids)), value=tokenizer.pad_token_id) for response_ids in responses]
    
    inputs = torch.stack(inputs)
    responses = torch.stack(responses)
    return inputs, responses

try:
    # Load the combined data
    combined_data = load_data('combined_data.yml')

    # Initialize the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token

    # Prepare inputs and responses
    input_texts = []
    response_texts = []
    for entry in combined_data:
        if isinstance(entry, dict) and 'heading' in entry and 'content' in entry:
            input_texts.append(entry['heading'])
            response_texts.append(entry['content'])
        else:
            print(f"Invalid entry: {entry}")

    # Create dataset and dataloader
    dataset = TourGuideDataset(input_texts, response_texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # Initialize the model
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Training loop
    model.train()
    epochs = 3  # Number of epochs
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            inputs, responses = batch
            inputs = inputs.to(model.device)
            responses = responses.to(model.device)

            optimizer.zero_grad()
            outputs = model(input_ids=inputs, labels=responses)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch + 1} Loss: {total_loss / len(dataloader)}')

    # Save the trained model and tokenizer
    model.save_pretrained('chatbot_model')
    tokenizer.save_pretrained('chatbot_tokenizer')

    # Load the trained model and tokenizer
    model = GPT2LMHeadModel.from_pretrained('chatbot_model')
    tokenizer = GPT2Tokenizer.from_pretrained('chatbot_tokenizer')

    def generate_response(input_text):
        input_ids = tokenizer.encode(input_text, return_tensors='pt', max_length=tokenizer.model_max_length)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=model.device)
        with torch.no_grad():
            output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=50, num_return_sequences=1)
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
