```mermaid

graph TB
    subgraph Data["Data Processing"]
        Wiki["WikiText-103 Dataset"]
        DL["datasets Library"]
        TF["Tokenize Function"]
        DC["DataCollatorWithPadding"]
        
        Wiki -->|Load Dataset| DL
        DL -->|Process Text| TF
        TF -->|Batch & Pad| DC
    end

    subgraph Model["Model Components"]
        GPT["Pre-trained GPT-2"]
        TOK["GPT2Tokenizer"]
        LMH["GPT2LMHeadModel"]
        
        GPT -->|Initialize| TOK
        GPT -->|Initialize| LMH
    end

    subgraph Training["Training Setup"]
        TA["TrainingArguments"]
        TR["Trainer"]
        
        TA -->|Configure| TR
        DC -->|Feed Data| TR
        LMH -->|Model| TR
        TOK -->|Tokenizer| TR
    end

    subgraph Hardware["Hardware Optimization"]
        DEV["Device Detection"]
        GPU["GPU (if available)"]
        CPU["CPU (fallback)"]
        
        DEV -->|Check| GPU
        DEV -->|Fallback| CPU
    end

    subgraph Inference["Text Generation"]
        INP["Input Text"]
        GEN["model.generate()"]
        OUT["Generated Text"]
        
        INP -->|Encode| TOK
        TOK -->|Tokens| GEN
        GEN -->|Decode| OUT
    end

    subgraph Storage["Model Management"]
        SAVE["save_pretrained()"]
        LOAD["from_pretrained()"]
        DIR["./gpt2_finetuned_model"]
        
        TR -->|Save Model| SAVE
        SAVE -->|Store| DIR
        DIR -->|Load| LOAD
    end

    TR -->|Train on| Hardware
    Training -->|Fine-tuned Model| Inference
    Storage -->|Load Model| Inference
