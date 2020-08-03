# NMT Assignment
Note: Heavily inspired by the https://github.com/pcyin/pytorch_nmt repository

Main take away:

    1. Vocab
        1) vocab class is basicly a dict
        2) vocab is buld using Counter class
        3) main functions are words2indices and indices2words, they are implemented as list comprehensions
        4) vocab is stored as json
        
    2. Data
        1) <start> and <end> token appear only in target sentences
        2) sentences in batch are sorted by source sentences length (descending)
        3) source and target sentences are padded to maximum sentence length in current batch
        
    3. Sequence data processing with RNNs
        1) pack_padded_sequence(input, lengths) prepares current batcch for rnn. pad_packed_sequence(rnn_output) converts rnn output back to tensors (functions are from nn.utils.rnn)
        2) torch.split(input, split_size, dim) splits tensor in split_size items
        3) torch.bmm is used for batch matrix multiplication

    4. Model
        1) to pass encoder final hidden state to decoder, projection matrix is used
        2) multiplicative attention is used
        3) on each step attention vector is concatenated with decoder hiddent state. Then resulting vector is projected on lower dimension and it goes through tanh+dropout. Resulting vector is used as decoder output and as a input for the next decoder step.
        4) encoder mask is used to prevent attention calculation over <pad> tokens (fill attention weights with -inf to make softmax values 0)
        5) target mask is used to prevent loss calsulation over <pad> tokens
    
    5. Training
        1) model.train() - train mode, model.eval() - evaluation mode (dropout, batchnorm in evaluation mode)
        2) all parametens are initialized by uniform distribution [-0.1, 0.1] by default
        3) gradient clipping by norm is used (default value is 5.0)
        4) validation metric is perplexity
        5) model and optimizer are saved each time val ppl improves (model checkpoint)
        6) if val ppl is not improved during last 5 epochs, model is restored from best checkpoint with reduced learning rate
        7) training terminates if model does not improve during 5 consequtive restarts (early stopping)
    
    6. Save and load model
        1) pytorch model can be saved and loaded as stade_dict
        
        ```python
           # save model
           model = Model()
           model_info = model.state_dict()
           torch.save(model_info, path)
           
           # load model
           model_loaded = Model()
           model_info_loaded = torch.load(path)
           model_loaded.load_state_diot(model_info_loaded)
        ```
        
        2) optimizers also have parameters which should be saved