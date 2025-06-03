from capsules.baselines.mtob import MtobBaselineConfig
import pydrantic

from capsules.clients.together import TogetherClient


if __name__ == "__main__":
    config = MtobBaselineConfig(
        client=TogetherClient.Config(
            model_name="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            show_progress_bar=True,
        ),
        direction="ke",
        grammar_book="medium",
        include_wordlist=False,
        include_sentences=True,
        temperature=0.0,
        max_completion_tokens=64,
        num_samples=3,
        prompt_type="generic",
    )
    pydrantic.main([config])
