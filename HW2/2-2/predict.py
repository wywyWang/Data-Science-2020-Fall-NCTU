import sys
import os
import torch
from attractivedata import AttractiveData
from attractivenet import AttractiveNet

def predict_attractive(sentence, category, phase):
    indexed_sentence = [AttractiveData.TEXT.vocab.stoi[t] for t in sentence]
    indexed_category = [AttractiveData.CATEGORIES_LABEL.vocab.stoi[category]]
    tensor_sentence = torch.LongTensor(indexed_sentence).to(AttractiveData.device)
    tensor_category = torch.LongTensor(indexed_category).to(AttractiveData.device)

    tensor_sentence = tensor_sentence.unsqueeze(0)

    prediction = load_model(tensor_sentence, tensor_category, phase=phase)
    return prediction


if __name__ == '__main__':
    # CNN_LSTM_20201103-150358_0.4101.70
    postfix_name = sys.argv[1]
    config_name = './config/' + postfix_name
    model_name = './model/' + postfix_name
    train_file = 'example/train.csv'
    val_file = 'example/val.csv'
    test_file = 'data/test.csv'

    config = eval(open(config_name, 'r').readlines()[0])

    if not os.path.exists('predict'):
        os.makedirs('predict')

    AttractiveData = AttractiveData(train_file, val_file, test_file, config['pretrained_file'], config)
    load_model = AttractiveNet(config).to(AttractiveData.device)
    load_model.load_state_dict(torch.load(model_name))
    load_model.eval()

    predict_list = []
    with torch.no_grad():
        for i, sentence in enumerate(AttractiveData.test_data):
            prediction = predict_attractive(sentence.Headline, sentence.Category, 'test')
            predict_list.append(prediction.item())
            # predict_list.append(prediction.item())
    AttractiveData.df_test['Label'] = predict_list
    AttractiveData.df_test[['ID', 'Label']].to_csv('./predict/' + postfix_name + '.csv', index=False)