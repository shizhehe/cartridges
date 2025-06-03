
import torch 
from tqdm import tqdm
device = "cuda" if torch.cuda.is_available() else "cpu"
from cartridges.generation import get_loss
from cartridges.generation import generate as generate_text
from cartridges.datasets import TEMPLATE


def run_query_set(tokenizer, model, messages, answers, max_new_tokens = 500, do_print=False):

    responses = []
    losses = []
    list_of_toks = []
    total_tokens = 0
    total_loss = 0

    for i, (message, answer) in tqdm(enumerate(zip(messages, answers)), desc="Generating responses", total=len(messages)):



        if i in [0, 5]: 
            continue
        
        input_ids = tokenizer.apply_chat_template(conversation=message,tokenize=True,add_generation_prompt=False,template=TEMPLATE,system_message="",return_tensors="pt",).to(device)

        dummy_message = [{"role": "system", "content": ""}]
        system_prompt_ids = tokenizer.apply_chat_template(
            conversation=dummy_message,
            tokenize=True,
            add_generation_prompt=False,
            template=tokenizer.chat_template,
            return_tensors="pt"
        )[0]
        input_ids = input_ids[:, len(system_prompt_ids):]


        answer = f"<|start_header_id|>assistant<|end_header_id|>\n\n{answer}<|eot_id|>"
        answer_ids = tokenizer.encode(
            answer,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(device)

        loss, num_tokens = get_loss(
            input_ids,
            model,
            tokenizer,
            answer_ids=answer_ids,
        )
        list_of_toks.append(num_tokens)
        losses.append(loss)
        total_loss += loss.item()
        total_tokens += num_tokens

        if do_print:
            cache_response = generate_text(
                input_ids,
                model,
                tokenizer,
                max_new_tokens=max_new_tokens,
            )   
            responses.append(cache_response.replace("\n", " "))

        if do_print:
            print(message)
            print(cache_response.replace("\n", " "))
            print("-----"*10)
            print("*****"*10)
            print("-----"*10)

    # import math
    avg_nll = total_loss / total_tokens
    # losses = [l.item() / n  for l, n in zip(losses, list_of_toks)]
    # for i in range(len(losses)):
    #     if do_print:
    #         print(f"{i} = {losses[i]}: {responses[i]}")
    #     else:
    #         print(f"{i} = {losses[i]}")
    # perplexity = math.exp(avg_nll)
    return responses, avg_nll



comparative_messages = [
[{
    "role": "user", 
    "content": "List the names of the Fortune 500 companies that the two financial statements are about."
}],
[{
    "role": "user", 
    "content": "Which company {company1} or {company2} has a higher net revenue for FY22?" 
}],
[{
    "role": "user", 
    "content": "Which company {company1} or {company2}  had a higher increase in revenue from 2021 to 2022?" 
}],
[{
    "role": "user", 
    "content": "Who were the Chief Financial Officers of {company1} or {company2} in the 10-k documents in this corpus?" 
}],
[{
    "role": "user", 
    "content": "Where are {company1} and {company2} headquartered?" 
}],
[{
    "role": "user", 
    "content": "Which of the two companies {company1} or {company2} has more employeees as of December 31, 2022 in the global workforce?", 
}],
[{
    "role": "user", 
    "content": "Who are the Chief Executive Officers of the companies that have 10-k's?"
}], 
[{
    "role": "user", 
    "content": "What were all the company acquisitions made in the year ending FY22 if any, according to the financial 10-k's?"
}], 
[{
    "role": "user", 
    "content": "According to the {company1} and {company2} financial 10-k's, which is the reported Goodwill in the Consolidated Statements for FY22?"
}],
[{
    "role": "user", 
    "content": "According to the  {company1} and {company2} financial 10-k's in the corpus, which is the reported net deferred tax assests(liabilities) after valuation allowance for FY22?"
}],  
[{
    "role": "user", 
    "content": "According to the financial 10-k's in the corpus, which is the reported Money market funds for each company {company1} and {company2} in FY22?"
}], 
[{
    "role": "user", 
    "content": "On what dates were the {company1} and {company2} statements signed by the Chief Financial Officers?"
}],  
[{
    "role": "user", 
    "content": "Who audited the {company1} and {company2} statements respectively?"
}],  
[{
    "role": "user", 
    "content": "List a few competitors for each of {company1} and {company2} as stated in each the 10K."
}],  
[{
    "role": "user", 
    "content": "What are the seasonal business portions for {company1} and {company2}?"
}],  
] 

boeing_answers = [
    "Boeing",
    "$66,608 million",
    "6.9% compared to 2021 total net revenue of $62,286 million",
    "Brian J. West",
    "The principle executive offices are located in Arlington, Virginia",
    "156,000 employees",
    "David L. Calhoun",
    "No companies acquired in 2022",
    "8,057 million",
    "$(167) million",
    "$1,797 million",
    "January 27, 2023",
    "Deloitte & Touche LLP",
    "Airbus, BAE Systems, Airbus Group, entrants from China, Lockheed Martin Corporation, Northrop Grumman Corporation, Raytheon Technologies Corporation, General Dynamics Corporation and SpaceX", 
    "No material portion of our business is considered to be seasonal.",
]

amd_answers = [
    "AMD",
    "$23,601 million",
    "44% compared to 2021 net revenue of $16,434 million",
    "Jean Hu",
    "Headquarters are located in Santa Clara,",
    "25,000 employees",
    "Lisa T. Su",
    "Xilinx, Inc. and Pensando Systems, Inc.",
    "24,177 million",
    "$(1,876) million",
    "$3,017 million",
    "February 27, 2023",
    "Ernst & Young LLP",
    "Intel Corporation, NVIDIA Corporation, Intel Corporation, Lattice Semiconductor Corporation, Microsemi Corporation, Broadcom Corporation, Marvell Technology Group, Ltd., Analog Devices, Texas Instruments Incorporated and NXP Semiconductors N.V., and from NVIDIA in the Embedded Segment, among various other competitors",
    "Our operating results tend to vary seasonally. Historically, our net revenue has been generally higher in the second half of the year than in the first half of the year, although market conditions and product transitions could impact these trends",
]

pepsico_answers = [
    "PepsiCo",
    "$86,400 million",
    "9% compared to 2021 net revenue of $79,500 million",
    "Hugh F. Johnston",
    "Headquarters are located in Purchase, New York",
    "315,000 employees worldwide",  
    "Ramon L. Laguarta",
    "No acquisitions in the year ending 2022",
    "18,202 million", 
    "$(71) million", 
    "No money market funds",
    "February 8, 2023", 
    "KPMG LLP",
    "The Coca-Cola Company is our primary beverage competitor. Other beverage and convenient food competitors include, but are not limited to, Campbell Soup Company, Conagra Brands, Inc., Hormel Foods Corporation, Kellogg Company, Keurig Dr Pepper Inc., The Kraft Heinz Company, Link Snacks, Inc., Mondelēz International, Inc., Monster Beverage Corporation, Nestlé S.A., Red Bull GmbH and Utz Brands, Inc.",
    "Our businesses are affected by seasonal variations. Our beverage and convenient food sales are generally highest in the third quarter due to seasonal and holiday-related patterns and generally lowest in the first quarter. However, taken as a whole, seasonality has not had a material impact on our consolidated financial results.",
]

amex_answers = [
    "American Express",
    "$52,862 million",
    "24.7% compared to 2021 net revenue of $42,380 million",
    "Jeffrey C. Campbell",
    "Our principal executive offices are in a 2.2 million square foot building located in lower Manhattan on land leased from the Battery Park City Authority for a term expiring in 2069.",
    "77,300 people",
    "Stephen J. Squeri",
    "No acquisitions in the year ending 2022",
    "$3,786 million",
    "$3,505 million",
    "No money market funds",
    "February 10, 2023",
    "PricewaterhouseCoopers LLP",
    "China UnionPay, Visa, Mastercard, JCB, Discover and Diners Club International (which is owned by Discover), Alipay, PayPal and Venmo, National Payments Corporation of India",
    "Our business as a whole has not experienced significant seasonal fluctuations, although network volumes tend to be moderately higher in the fourth quarter than in other quarters.",
]

boeing_amex_comparative_answers  = [
    "The statements are about Boeing and American Express",
    "Boeing had a net revenue of $ $66,608 million, while American Express had a net revenue of $52,862 million, Boeing has higher net revenue",
    "Boeing 6.9% vs American Express 24.7%, American Express has higher increase in net revenue",
    "Boeing's Chief Financial Officer is Brian J. West and American Express's Chief Financial Officer is Jeffrey C. Campbell",
    "Boeing is headquartered in Arlington, Virginia and American Express is headquartered in Manhattan",
    "Boeing has 156,000 employees and American Express has 77,300 employees so Boeing has more employees",
    "Boeing's Chief Executive Officer is David L. Calhoun and American Express's Chief Executive Officer is Stephen J. Squeri",
    "Boeing had no acquisitions in FY22 and American Express had no acquisitions in FY22",
    "Boeing had $8,057 million goodwill and American Express has $3,786 million goodwill",
    "Boeing had $(167) million net deferred tax assets and American Express has $3,505 million net deferred tax assets",
    "Boeing had $1,797 million money market funds, American Express had no money market funds",
    "Boeing had January 27, 2023 as reporting date, American Express had February 10, 2023 as reporting date",
    "Boeing had Deloitte & Touche LLP as auditors, American Express had PricewaterhouseCoopers LLP as auditors",
    "Boeing's competitors include Airbus, Lockheed Martin Corporation, SpaceX and others, while American Express's competitors include China UnionPay, Visa, Mastercard, JCB, Discover and Diners Club International (which is owned by Discover), and others",
    "Boeing has no material portion of our business is considered to be seasonal and American Express has moderate seasonal fluctuations",
]
boeing_amex_order = ["Boeing", "American Express"]


pepsico_boeing_comparative_answers = [
    "The statements are focused on PepsiCo and Boeing",
    "PepsiCo had a net revenue of $86,400 million while Boeing's net revenue was $66,608 million, so PepsiCo had higher net revenue",
    "PepsiCo had a higher increase in net revenue",
    "PepsiCo's Chief Financial Officer is Hugh F. Johnston andn Boeing's Chief Financial Officer is Brian J. West",
    "PepsiCo is headquartered in Purchase, New York, Boeing is headquartered in Arlington, Virginia",
    "PepsiCo has 315,000 employees, Boeing has 156,000 employees so PepsiCo has more employees",
    "PepsiCo's Chief Executive Officer is Ramon L. Laguarta and Boeing's Chief Executive Officer is David L. Calhoun",
    "PepsiCo had no acquisitions in the year ending 2022, Boeing had no acquisitions in the year ending 2022",
    "PepsiCo had $18,202 million goodwill, while Boeing had $8,057 million goodwill",
    "PepsiCo had $(71) million net deferred tax assets, while Boeing had $(167) million net deferred tax assets",
    "PepsiCo had no money market funds, while Boeing had $1,797 million money market funds",
    "PepsiCo's statement was reported on February 8, 2023, while Boeing's statement was reported on January 27, 2023",
    "PepsiCo has KPMG LLP as auditors and Boeing has Deloitte & Touche LLP as auditors",
    "Boeing's competitors are Airbus, Lockheed Martin Corporation, SpaceX and others and PepsiCo's competitors are The Coca-Cola Company, Campbell Soup Company, Conagra Brands, Heinz Company, Nestlé and others",
    "PepsiCo has seasonal variations especially in the third quarter, while Boeing has no material portion of our business is considered to be seasonal.",
]
pepsico_boeing_order = ["PepsiCo", "Boeing"]


pepsico_amex_comparative_answers = [
    "PepsiCo and American Express",
    "Pepsico had net revenue of $86,400 million, while Boeing had a net revenue of $52,862 million, so PepsiCo had higher net revenue",
    "American Express had a higher increase in net revenue",
    "PepsiCo's Chief Financial Officer is Hugh F. Johnston and American Express's Chief Financial Officer is Jeffrey C. Campbell",
    "PepsiCo is headquartered in Purchase, New York, while American Express is headquartered in Manhattan",
    "PepsiCo has 315,000 employees, while American Express has 77,300 employees so PepsiCo has more employees",
    "PepsiCo Chief Executive Officer is Ramon L. Laguarta, while American Express's Chief Executive Officer is Stephen J. Squeri",
    "PepsiCo had no acquisitions in the year ending 2022, while American Express had no acquisitions in the year ending 2022",
    "PepsiCo had $18,202 million goodwill, while American Express had $3,786 million goodwill",
    "PepsiCo had $(71) million net deferred tax assets, while American Express had $3,505 million net deferred tax assets",
    "PepsiCo had no money market funds, while American Express had no money market funds",
    "PepsiCo's statment was signed on February 8, 2023, while American Express's statement was signed on February 10, 2023",
    "PepsiCo has KPMG LLP as auditors, American Express has PricewaterhouseCoopers LLP as auditors",
    "PepsiCo has The Coca-Cola Company as competitors, while American Express has China UnionPay, Visa, Mastercard, JCB, Discover and Diners Club International (which is owned by Discover), and others",
    "PepsiCo has seasonal variations, while American Express has moderate seasonal fluctuations",
]
pepsico_amex_order = ["PepsiCo", "American Express"]


amd_boeing_comparative_answers = [
    "AMD and Boeing",
    "Boeing had net revenue of $66,608 million while AMD had a net revenue of $23,601 million, so Boeing has higher net revenue",
    "AMD had higher increase in net revenue in FY22",
    "Boeing's Chief Financial Officer is Brian J. West, AMD's Chief Financial Officer is Jean HuO",
    "Boeing is headquartered in Arlington, Virginia, while AMD is headquartered in Santa Clara",
    "Boeing has 156,000 employees, while AMD has 25,000 employees so Boeing has more employees",
    "Boeing's Chief Executive Officer is David L. Calhoun and AMD's Chief Exeuctive Officer is Lisa T. Su",
    "Boeing has no acquisitions in the year ending 2022, while AMD acquired Xilinx and Pensando Systems",
    "Boeing had $8,057 million goodwill, while AMD had $24,177 million goodwill",
    "Boeing had $(167) million net deferred tax assets, AMD had $(1,876) million net deferred tax assets",
    "Boeing had $1,797 million money market funds, while AMD had $3,017 million money market funds",
    "Boeing had January 27, 2023 as reporting date, while AMD had February 27, 2023 as reporting date",
    "Boeing's statement was audited by Deloitte & Touche LLP. while AMD's statement was audited by Ernst & Young LLP",
    "Boeing's competitors are Airbus, Lockheed Martin Corporation, SpaceX and others, while AMD's competitors are Intel, NVIDIA, Lattice Semiconductor Corporation, and others",
    "Boeing has no material portion of our business is considered to be seasonal, while AMD has no material impact on consolidated financial results",
]
amd_boeing_order = ["Boeing", "AMD"]


amd_pepsico_comparative_answers = [
    "AMD and PepsiCo",
    "Pepsico had net revenue of $86,400 million for FY22, while AMD had a net revenue of $23,601 million for FY22, PepsiCo has higher net revenue",
    "AMD has a higher increase in net revenue",
    "PepsiCo's Chief Financial Officer is Hugh F. Johnston and AMD's Chief Financial Officer is Jean Hu",
    "PepsiCo is headquartered in Purchase, New York and AMD is headquartered in Santa Clara",
    "PepsiCo has 315,000 employees and AMD has 25,000 employees so Pepsico has more employees",
    "PepsiCo's Chief Executive Officer is Ramon L. Laguarta as CEO and AMD's Chief Executive Officer is Lisa T. Su",
    "PepsiCo has no acquisitions in the year ending 2022, while AMD acquired Xilinx and Pensando Systems",
    "PepsiCo has $18,202 million goodwill, while AMD has $24,177 million goodwill",
    "PepsiCo has $(71) million net deferred tax assets, AMD has $(1,876) million net deferred tax assets",
    "PepsiCo has no money market funds, while AMD has $3,017 million money market funds",
    "PepsiCo's statement was signed on February 8, 2023, while AMD's statement was signed on February 27, 2023",
    "PepsiCo statement was audited by KPMG LLP and AMD's statement was audited by Ernst & Young LLP as auditors",
    "PepsiCo has The Coca-Cola Company as competitors, AMD has Intel, NVIDIA, Lattice Semiconductor Corporation, and others",
    "PepsiCo has seasonal variations, AMD has no material impact on consolidated financial results",
]
amd_pepsico_order = ["PepsiCo", "AMD"]


amd_amex_comparative_answers = [
    "AMD and American Express",
    "American Express had a net revenue of $52,862 million and AMD had a net revenue of $23,601 million, American Express had a higher net revenue",
    "AMD had the higher increase in net revenue",
    "American Express's Chief Financial Officer is Jeffrey C. Campbell and AMD's Chief Financial Officer is Jean Hu",
    "American Express is headquartered in Manhattan, AMD is headquartered in Santa Clara",
    "American Express has 77,300 employees, AMD has 25,000 employees so American Express has more employees",
    "American Express's Chief Exeuctive Officer is Stephen J. Squeri, while AMD's Chief Exeuctive Officer is Lisa T. Su",
    "American Express has no acquisitions in the year ending 2022, AMD acquired Xilinx and Pensando Systems",
    "American Express has $3,786 million goodwill, while AMD has $24,177 million goodwill",
    "American Express has $3,505 million net deferred tax assets, AMD has $(1,876) million net deferred tax assets",
    "American Express has no money market funds, while AMD has $3,017 million money market funds",
    "American Express has February 10, 2023 as reporting date, while AMD has February 27, 2023 as reporting date",
    "American Express's statement was audited by PricewaterhouseCoopers LLP and AMD's statement was audited by Ernst & Young LLP",
    "American Express has China UnionPay, Visa, Mastercard, JCB, Discover and others, while AMD has Intel, NVIDIA, Lattice Semiconductor Corporation, and others",
    "American Express has moderate seasonal fluctuations, while AMD has no material impact on consolidated financial results",
]
amd_amex_order = ["American Express", "AMD"]



from cartridges.train import EvalDatasetConfig, EvalDataset
from transformers import AutoTokenizer
from cartridges.tasks.finance import ( FinanceBenchEvalDataset ) 

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

def get_dataset_messages(DOC_NAME):
    dataset_config = EvalDatasetConfig(
        name_for_wandb="finance-ppl-gt",
        local_batch_size=16,
        dataset=FinanceBenchEvalDataset.Config(
            doc_names=[DOC_NAME],
            cot=False,
            label_type="tokens",
            data_sources=[],  # ignore this arg
        ),
        only_eval_rank_0=True,
    )
    dataset = EvalDataset(
        dataset=dataset_config.dataset.instantiate(tokenizer=tokenizer),
        batch_size=dataset_config.local_batch_size,
        name=dataset_config.name_for_wandb,
        only_eval_rank_0=dataset_config.only_eval_rank_0,
        dataloader_num_workers=dataset_config.dataloader_num_workers,
    )
    dataset_messages = []
    answers = []
    data = dataset.dataset.data
    for idx in range(len(data)):
        elt = data[idx].messages
        query = elt[0].content
        answer = elt[1].content
        dataset_messages.append([{
            "role": "user",
            "content": query,
        }])
        answers.append(answer)

    return dataset_messages, answers


def clean_questions(questions, order, additional_instructions = ""): # Think step by step."):
    company1, company2 = order
    questions_clean = []
    for question in questions:
        content = (question[0]['content'] + additional_instructions).format(
            company1=company1, company2=company2
        )
        question_copy = question.copy()
        question_copy[0] = question[0].copy()
        question_copy[0]['content'] = content
        questions_clean.append(question_copy)
    return questions_clean


amd_dataset_messages, amd_answers = get_dataset_messages("AMD_2022_10K")
pepsico_dataset_messages, pepsico_answers = get_dataset_messages("PEPSICO_2022_10K")
amex_dataset_messages, amex_answers = get_dataset_messages("AMERICANEXPRESS_2022_10K")
boeing_dataset_messages, boeing_answers = get_dataset_messages("BOEING_2022_10K")

QA_PAIRS = {
    "AMD_PEPSICO": {
        "questions":  amd_dataset_messages + pepsico_dataset_messages + clean_questions(comparative_messages.copy(), amd_pepsico_order),
        "answers": amd_answers + pepsico_answers + amd_pepsico_comparative_answers,
        "order": amd_pepsico_order
    },
    "AMD_AMEX": {
        "questions": amd_dataset_messages + amex_dataset_messages + clean_questions(comparative_messages.copy(), amd_amex_order),
        "answers":  amd_answers + amex_answers + amd_amex_comparative_answers,
        "order": amd_amex_order
    },
    "AMD_BOEING": {
        "questions":  amd_dataset_messages + boeing_dataset_messages + clean_questions(comparative_messages.copy(), amd_boeing_order),
        "answers": amd_answers + boeing_answers + amd_boeing_comparative_answers,
    },
    "PEPSICO_AMEX": {
        "questions": pepsico_dataset_messages + amex_dataset_messages + clean_questions(comparative_messages.copy(), pepsico_amex_order),
        "answers": pepsico_answers + amex_answers + pepsico_amex_comparative_answers,
    },
    "PEPSICO_BOEING": {
        "questions": pepsico_dataset_messages + boeing_dataset_messages + clean_questions(comparative_messages.copy(), pepsico_boeing_order),
        "answers": pepsico_answers + boeing_answers + pepsico_boeing_comparative_answers,
    },
    "BOEING_AMEX": {
        "questions": boeing_dataset_messages + amex_dataset_messages + clean_questions(comparative_messages.copy(), boeing_amex_order),
        "answers": boeing_answers + amex_answers + boeing_amex_comparative_answers,
    }
}


# QA_PAIRS = {
#     "AMD_PEPSICO": {
#         "questions":  clean_questions(comparative_messages, amd_pepsico_order),
#         "answers": amd_pepsico_comparative_answers,
#     },
#     "AMD_AMEX": {
#         "questions": clean_questions(comparative_messages.copy(), amd_amex_order),
#         "answers":  amd_amex_comparative_answers,
#     },
#     "AMD_BOEING": {
#         "questions":  clean_questions(comparative_messages.copy(), amd_boeing_order),
#         "answers": amd_boeing_comparative_answers,
#     },
#     "PEPSICO_AMEX": {
#         "questions": clean_questions(comparative_messages.copy(), pepsico_amex_order),
#         "answers": pepsico_amex_comparative_answers,
#     },
#     "PEPSICO_BOEING": {
#         "questions": clean_questions(comparative_messages.copy(), pepsico_boeing_order),
#         "answers": pepsico_boeing_comparative_answers,
#     },
#     "BOEING_AMEX": {
#         "questions": clean_questions(comparative_messages.copy(), boeing_amex_order),
#         "answers": boeing_amex_comparative_answers,
#     }
# }

