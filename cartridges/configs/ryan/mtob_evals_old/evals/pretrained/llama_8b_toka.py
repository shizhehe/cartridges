from capsules.baselines.mtob import MtobBaselineConfig
import pydrantic

from capsules.clients.together import TogetherClient
from capsules.clients.tokasaurus_batch import TokasaurusBatchClient


if __name__ == "__main__":
    config = MtobBaselineConfig(
        client=TokasaurusBatchClient.Config(
            url="http://localhost",
            ports=[8880 + i for i in range(1)],
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            timeout=1500,
        ),
        direction="ke",
        grammar_book="medium",
        include_wordlist=False,
        include_sentences=True,
        temperature=0.0,
        max_completion_tokens=64,
        num_samples=1,
        prompt_type="generic",
    )
    pydrantic.main([config])
