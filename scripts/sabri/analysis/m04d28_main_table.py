
import pydrantic

from cartridges.analysis.tables.results_table import TradeoffTable, TableRunConfig
from cartridges.communication.analysis.configs.paper.main_tradeoff.utils import get_generator_name
import pandas as pd

def format_model_name(model_name: str) -> str:
    print(model_name)
    if model_name.lower() == "qwen/Qwen2.5-7B-Instruct".lower():
        model_name = "qwen/qwen2.5-3b-instruct"
    return model_name.lower()

QASPER_RUNS = [
    
    # NAIVE PROTOCOL
    TableRunConfig(
        run_dir=[
            # llama 3b
            # "/data/sabri/code/haystacks/haystacks/sample/outputs/2025-01-24-18-34-15-m01d24_naive_protocol_qasper/qasper-naive-protocol-repeat-0-0",
            # "/data/sabri/code/haystacks/haystacks/sample/outputs/2025-01-24-18-41-42-m01d24_naive_protocol_qasper/",
            "/data/sabri/code/haystacks/haystacks/sample/outputs/2025-01-25-18-16-31-m01d24_naive_protocol_qasper",

            # qwen 3b
            # "/data/sabri/code/haystacks/haystacks/sample/outputs/2025-01-24-19-23-45-m01d24_naive_protocol_qwen_qasper",
            "/data/sabri/code/haystacks/haystacks/sample/outputs/2025-01-25-18-44-46-m01d24_naive_protocol_qwen_qasper/",

            # llama 3.1 8b
            "/data/sabri/code/haystacks/haystacks/sample/outputs/2025-01-25-16-41-23-m01d24_naive_protocol_qasper/"
        ],
        recursive=True,
        protocol=get_generator_name,
        edge_model=lambda config: format_model_name(config.generator.worker_client.model_name),
        remote_model=lambda config: format_model_name(config.generator.supervisor_client.model_name),
    ),

    # MINIONS PROTOCOL
    TableRunConfig(
        #llama 3b  5 rounds
        run_dir=[
            # llama 3b
            "/home/anarayan/haystacks/haystacks/haystacks/sample/outputs/2025-01-23-18-14-18-m01d20_qasper_multiround/6b03882d-95bb-40a2-af95-d1c9face9707",

            # qwen 7b
            # "/home/anarayan/haystacks/haystacks/haystacks/sample/outputs/2025-01-23-19-58-10-m01d20_qasper_multiround/5bad1c64-e441-44ea-a8b3-63a5f147de83/",

            # qwen 3b
            "/home/anarayan/haystacks/haystacks/haystacks/sample/outputs/2r025-01-23-18-54-56-m01d20_qasper_multiround/0f910163-3b21-49b9-a6ee-d7f287e39053/",

            # llama 8b
            "/home/anarayan/haystacks/haystacks/haystacks/sample/outputs/2025-01-23-17-45-58-m01d20_qasper_multiround/f35e4ca7-0ff1-46fb-9d45-a183e69a91be/"
        ],
        recursive=True,
        protocol=get_generator_name,
        edge_model=lambda config: format_model_name(config.generator.worker_client.model_name),
        remote_model=lambda config: format_model_name(config.generator.supervisor_client.model_name),
    ),


    # BASELINES
    TableRunConfig(
        # gpt4o baseline
        run_dir="/home/anarayan/haystacks/haystacks/haystacks/sample/outputs/2025-01-22-20-48-59-m01d22_qasper_baseline/14851d80-63c6-48d7-b06c-7c557704c112",
        protocol="Remote Only",
        edge_model="-",
        remote_model=lambda config: format_model_name(config.generator.generator_client.model_name),       
    ),
    TableRunConfig(
        # Llama baselines
        run_dir=[
            "/home/anarayan/haystacks/haystacks/haystacks/sample/outputs/2025-01-24-10-55-18-m01d22_qasper_baseline/5700742d-df4b-49cb-9155-f4eee8acd326",
            "/home/anarayan/haystacks/haystacks/haystacks/sample/outputs/2025-01-24-09-23-17-m01d22_qasper_baseline/2cdfe2e2-3d8f-4401-8b58-365972427b73",
            "/home/anarayan/haystacks/haystacks/haystacks/sample/outputs/2025-01-24-10-58-24-m01d22_qasper_baseline/ac96d1c7-dd19-46d4-a8ae-1acdbc92f376",
        ],
        protocol="Edge Only",
        edge_model=lambda config: format_model_name(config.generator.generator_client.model_name),
        remote_model="-",
        recursive=True,
    ),
    TableRunConfig(
        # Qwen baselines
        run_dir=[
            "/home/anarayan/haystacks/haystacks/haystacks/sample/outputs/2025-01-24-20-22-13-m01d22_qasper_baseline/fcc65316-bd3d-46bc-a281-8518793a6a7c",
        ],
        protocol="Edge Only",
        edge_model=lambda config: format_model_name(config.generator.generator_client.model_name),
        remote_model="-",
        recursive=True,
    ),
]

LONGHEALTH_RUNS = [

    # NAIVE PROTOCOL
    TableRunConfig(
        run_dir=[
             # llama 3b TODO: replace with 400
            # "/data/sabri/code/haystacks/haystacks/sample/outputs/2025-01-24-19-32-58-m01d24_naive_protocol_longhealth/",
            "/data/sabri/code/haystacks/haystacks/sample/outputs/2025-01-25-17-39-12-m01d24_naive_protocol_longhealth/",

            # qwen 3b TODO: replace with 400
            # "/data/sabri/code/haystacks/haystacks/sample/outputs/2025-01-24-20-29-33-m01d24_naive_protocol_qwen_longhealth",
            "/data/sabri/code/haystacks/haystacks/sample/outputs/2025-01-24-22-00-02-m01d24_naive_protocol_qwen_longhealth/longhealth-naive-protocol-repeat-1-1",

            # llama 3.1 8b TODO: replace with 400
            "/data/sabri/code/haystacks/haystacks/sample/outputs/2025-01-25-15-37-15-m01d24_naive_protocol_longhealth"
            # "/data/sabri/code/haystacks/haystacks/sample/outputs/2025-01-24-20-26-13-m01d24_naive_protocol_longhealth",
        ],
        recursive=True,
        protocol=get_generator_name,
        edge_model=lambda config: format_model_name(config.generator.worker_client.model_name),
        remote_model=lambda config: format_model_name(config.generator.supervisor_client.model_name),
    ),

    # MINIONS PROTOCOL
    TableRunConfig(
        #llama 3b  5 rounds
        run_dir=[
            # llama 8b
            "/home/anarayan/haystacks/haystacks/haystacks/sample/outputs/2025-01-23-15-12-47-m01d18_longhealth_singleround/d47c90f2-831a-4ac7-94ec-f6744e8586c9/",

            # llama 3b
            "/home/anarayan/haystacks/haystacks/haystacks/sample/outputs/2025-01-23-15-45-32-m01d18_longhealth_singleround/018464ba-b058-407b-a80e-5fea8b9e6b0b/",

            # qwen 3b
            "/home/anarayan/haystacks/haystacks/haystacks/sample/outputs/2025-01-23-14-23-36-m01d18_longhealth_singleround/b74e123c-8a66-49df-b087-c4e55fe28d76/",
        ],
        recursive=True,
        protocol=get_generator_name,
        edge_model=lambda config: format_model_name(config.generator.worker_client.model_name),
        remote_model=lambda config: format_model_name(config.generator.supervisor_client.model_name),
    ),


    # BASELINES
    TableRunConfig(
        # gpt4o baseline
        run_dir="/home/anarayan/haystacks/haystacks/haystacks/sample/outputs/2025-01-23-11-18-48-m01d20_longhealth_baseline/882d83bd-b51c-49b4-8352-ec0a212978a7",
        protocol="Remote Only",
        edge_model="-",
        remote_model=lambda config: format_model_name(config.generator.generator_client.model_name),
    ),
    TableRunConfig(
        # Llama baselines
        run_dir=[
            "/home/anarayan/haystacks/haystacks/haystacks/sample/outputs/2025-01-24-08-25-39-m01d20_longhealth_baseline/4104bd0f-de82-4939-9b64-7d4e317c3395",
            "/home/anarayan/haystacks/haystacks/haystacks/sample/outputs/2025-01-24-09-38-21-m01d20_longhealth_baseline/7282bd1f-4621-48d3-9746-2c190f0653f0",
            "/home/anarayan/haystacks/haystacks/haystacks/sample/outputs/2025-01-24-09-29-20-m01d20_longhealth_baseline/944aa7cf-deb9-461a-ac89-cbeaffac1b9e",
        ],
        protocol="Edge Only",
        edge_model=lambda config: format_model_name(config.generator.generator_client.model_name),
        remote_model="-",
        recursive=True,
    ),
    TableRunConfig(
        # Qwen baselines
        run_dir=[
            "/home/anarayan/haystacks/haystacks/haystacks/sample/outputs/2025-01-24-20-07-14-m01d20_longhealth_baseline/1aa1f057-4e2f-4fa9-8787-a7fd15d9ebdc/",
        ],
        protocol="Edge Only",
        edge_model=lambda config: format_model_name(config.generator.generator_client.model_name),
        remote_model="-",
        recursive=True,
    ),
]

FINANCE_RUNS = [
    # NAIVE PROTOCOL
    TableRunConfig(
        run_dir=[
            # llama 3b
            "/data/sabri/code/haystacks/haystacks/sample/outputs/2025-01-24-12-36-10-m01d20_naive_protocol_finance/finance-naive-protocol-rounds-10_repeat-0-0",
            "/data/sabri/code/haystacks/haystacks/sample/outputs/2025-01-24-13-11-25-m01d20_naive_protocol_finance",

            # qwen 3b
            "/data/sabri/code/haystacks/haystacks/sample/outputs/2025-01-24-13-29-37-m01d20_naive_protocol_qwen_finance",   

            # llama 8b
            "/data/sabri/code/haystacks/haystacks/sample/outputs/2025-01-24-14-13-35-m01d24_naive_protocol_finance/"  
        ],
        recursive=True,
        protocol=get_generator_name,
        edge_model=lambda config: format_model_name(config.generator.worker_client.model_name),
        remote_model=lambda config: format_model_name(config.generator.supervisor_client.model_name),
    ),

    # MINIONS PROTOCOL
    TableRunConfig(
        #llama 3b  5 rounds
        run_dir=[
            "/data/sabri/code/haystacks/haystacks/sample/outputs/2025-01-21-20-30-36-m01d21_consolidate_code_finance_carryover_rounds",
            "/data/sabri/code/haystacks/haystacks/sample/outputs/2025-01-25-19-48-35-m01d25_consolidate_code_finance_sabri_carryover_none/",
        ],
        recursive=True,
        protocol=get_generator_name,
        edge_model=lambda config: format_model_name(config.generator.worker_client.model_name),
        remote_model=lambda config: format_model_name(config.generator.supervisor_client.model_name),
    ),
    TableRunConfig(
        # qwen 3b 5 rounds
        run_dir="/data/sabri/code/haystacks/haystacks/sample/outputs/2025-01-21-20-11-07-m01d21_consolidate_code_finance_carryover_rounds/",
        recursive=True,
        protocol=get_generator_name,
        edge_model=lambda config: format_model_name(config.generator.worker_client.model_name),
        remote_model=lambda config: format_model_name(config.generator.supervisor_client.model_name),
    ),   


    # BASELINES
    TableRunConfig(
        # gpt4o baseline
        run_dir="/data/sabri/code/haystacks/haystacks/sample/outputs/2025-01-20-09-27-47-m01d17_remote_basline_finance/finance-code-gpt4o-baseline",
        protocol="Remote Only",
        edge_model="-",
        remote_model=lambda config: format_model_name(config.generator.client.model_name),      
    ),
    TableRunConfig(
        # Llama baselines
        run_dir=[
            "/data/sabri/code/haystacks/haystacks/sample/outputs/2025-01-21-17-45-33-m01d17_edge_baseline_finance", # 1b    
            "/data/sabri/code/haystacks/haystacks/sample/outputs/2025-01-20-10-11-58-m01d17_edge_baseline_finance", # 3b
            "/data/sabri/code/haystacks/haystacks/sample/outputs/2025-01-22-16-10-38-m01d17_edge_baseline_finance/" # 8b
        ],
        protocol="Edge Only",
        edge_model=lambda config: format_model_name(config.generator.client.model_name),
        remote_model="-",
        recursive=True,
    ),
    TableRunConfig(
        # Qwen baselines
        run_dir=[
            "/data/sabri/code/haystacks/haystacks/sample/outputs/2025-01-22-16-55-12-m01d17_edge_baseline_finance_qwen",  # 3b
        ],
        protocol="Edge Only",
        edge_model=lambda config: format_model_name(config.generator.client.model_name),
        remote_model="-",
        recursive=True,
    ),

]

from haystacks.communication.analysis.tradeoff_table import ColumnSpec, MODEL_TO_MACRO, PROTOCOL_TO_MACRO


def format_model_name_latex(model_name: str) -> str:
    return MODEL_TO_MACRO.get(model_name.lower(), model_name)


def format_float(x: float) -> str:
    if pd.isna(x):
        return "--"
    return f"{x:.3f}"

def format_money(x: float) -> str:
    if pd.isna(x):
        return "--"
    return f"\${x:.3f}"

def format_int(x: int) -> str:
    if pd.isna(x):
        return "--"
    return f"{x}"

def format_int_thousand(x: int) -> str:
    if pd.isna(x):
        return "--"
    return f"{float(x / 1000):.2f}"



column_specs = [
    ColumnSpec(name="protocol", latex_name=r"\textbf{Protocol}", format=lambda x: PROTOCOL_TO_MACRO.get(x, x)),
    ColumnSpec(name="edge_model", latex_name=r"\textbf{Local Model}", format=format_model_name_latex),
    ColumnSpec(name="remote_model", latex_name=r"\textbf{Remote Model}", format=format_model_name_latex),
    ColumnSpec(name="accuracy", latex_name="Acc.", format=format_float),
    ColumnSpec(name="remote_cost", latex_name="Cost", format=format_money),
    # ColumnSpec(name="remote_prompt_tokens", latex_name="In Tok. (1k)", format=format_int_thousand),
    # ColumnSpec(name="remote_completion_tokens", latex_name="Out Tok. (1k)", format=format_int_thousand),
]

table = TradeoffTable.Config(
    runs=[
        *FINANCE_RUNS,
        *LONGHEALTH_RUNS,
        *QASPER_RUNS,
    ],
    column_specs=column_specs,
    incomplete_runs="include",
    include_task_avg=True,
)


if __name__ == "__main__":
    pydrantic.main([table])