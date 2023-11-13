from model import *
from inference import *

if __name__ == '__main__':
    torch.manual_seed(0)
    allow_cuda = False
    device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'
    
    prompts = [
        "Simply put, the theory of relativity states that ",
    ]
    
    model = LLaMA.build(
        checkpoint_dir='drive/MyDrive',
        tokenizer_path='tokenizer.model',
        load_model=False,
        max_seq_size=1024,
        max_batch_size=len(prompts),
        device=device
        
    )
    out_tokens, out_texts = (model.text_completion(prompts, max_gen_len=64))
    assert len(out_texts) == len(prompts)
    for i in range(len(out_texts)):
        print(f'{out_texts}')
        print('-' * 50)