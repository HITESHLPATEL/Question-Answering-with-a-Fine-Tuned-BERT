import os
import uuid
import json


def dataset_prep(data_dir,filename,output_dir,train_data_ratio=0.7,dev_data_ratio=0.1,test_data_ratio=0.2):

    #selecting relation data from data.json
    file_path = os.path.join(data_dir,filename)
    print(file_path)
    with open(file_path,"r") as f:
        emrQA_data = json.load(f)

    relation_data = dict()
    relation_data['version'] = "1.0"
    relation_data['data'] = list()
    
    for i in range(len(emrQA_data['data'])):
        sub_data = emrQA_data['data'][i]
        if sub_data["title"] == "relations":
            temp_dict = dict()
            temp_dict['title'] = sub_data['title']
            temp_dict["paragraphs"] = list()

            for j in range(len(sub_data['paragraphs'])):
                processed_para = dict()
                para_context_list = sub_data['paragraphs'][j]['context']
                para_details = " ".join((" ".join(para_context_list).split()))
                processed_para['context'] = para_details
                processed_para['qas'] = list()
                ques_ans = sub_data['paragraphs'][j]['qas']
                if len(ques_ans) != 0:
                    
                    for k in range(len(ques_ans)):
                        ques_ans_processed = dict()
                        ques_ans_processed["answers"] = list()
                        ans = ques_ans[k]['answers']

                        for l in range(len(ans)):
                            if ans[l]["answer_entity_type"] != 'complex':
                                ans_dict = dict()
                                if sub_data['title'] == 'risk':
                                    info = ans[l]['text']
                                else:
                                    info = ans[l]['evidence']
                                if info:
                                    while info[-1] in [",",'.','?','!','-',' ']:
                                        info = info[:-1]
                                    var_position = -1
                                    temp_info = info
                                    final_info = temp_info
                                    no = 0
                                    while var_position == -1:
                                        var_position = para_details.find(temp_info)
                                        final_info = temp_info
                                        temp_info = ' '.join(temp_info.split()[:-1])
                                        no = no + 1 
                                    if var_position > 0 and final_info:
                                        ans_dict['answer_start'] = var_position
                                        ans_dict['text'] = final_info
                                        ques_ans_processed['answers'].append(ans_dict)
                                    else:
                                        continue
                        
                        ques = ques_ans[k]['question']
                        ans_dict = ques_ans_processed['answers']
                        if len(ans_dict) == 0:
                            continue

                        for p in range(len(ques)):
                            ques_ans_processed = dict()
                            ques_ans_processed['question'] = ques[p]
                            ques_ans_processed['id'] = str(uuid.uuid1())
                            ques_ans_processed['answers'] = ans_dict
                            processed_para['qas'].append(ques_ans_processed)

                    temp_dict['paragraphs'].append(processed_para)

            relation_data['data'].append(temp_dict)

    #train,dev,test split

    no_entries = len(relation_data['data'])
    train_entries = int(train_data_ratio * no_entries)
    dev_entries = int(dev_data_ratio * no_entries)

    relation_train_data = {'data': [], 'version': 1.0}
    relation_dev_data = {'data': [], 'version': 1.0}
    relation_test_data = {'data': [], 'version': 1.0}

    for i in range(train_entries):
        note = relation_data['data'][i]
        relation_train_data['data'].append(note)

    for i in range(train_entries,train_entries+dev_entries):
        note = relation_data['data'][i]
        relation_dev_data['data'].append(note)

    for i in range(train_entries+dev_entries,no_entries):
        note = relation_data['data'][i]
        relation_test_data['data'].append(note)

    json.dump(relation_train_data, open(os.path.join(output_dir, 'relation-train.json'), 'w'))
    json.dump(relation_dev_data, open(os.path.join(output_dir, 'relation-dev.json'), 'w'))
    json.dump(relation_test_data, open(os.path.join(output_dir, 'relation-test.json'), 'w'))

    return 0
