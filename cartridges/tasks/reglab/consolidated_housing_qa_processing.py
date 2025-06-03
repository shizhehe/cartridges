import json
import pandas as pd

df = pd.read_json('/home/emilyryliu/capsules/capsules/tasks/reglab/housing_qa_consolidated_generations.jsonl', lines=True)

# placeholders
df['answer_options'] = df['answer']  

for question in df['question_number'].unique(): 
    question_df = df[df['question_number'] == question]
    if question_df.iloc[0]['category'] == 'eviction_categories': 
        # get all the possible answer choices and put them in a new column
        options = set()
        # options.add("Not specified")
        for ans in question_df['answer']:
            # print(ans.split('\n'))
            options.update(ans.split('\n'))
        options = list(options)
        # print("question : ", question)
        # print("options : ", options)
        df.loc[df['question_number'] == question, 'answer_options'] = "\n".join(options)


# save the updated dataframe to a new jsonl file
df.to_json('/home/emilyryliu/capsules/capsules/tasks/reglab/housing_qa_consolidated_generations_with_answer_options.jsonl', orient='records', lines=True)
    
