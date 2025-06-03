



class SWDEContextConfig(BaseContextConfig):

    webpage_id : str = None
    tokenizer: str = "meta-llama/Llama-3.2-3B-Instruct"
    max_tokens_per_section: int = -1

    # pages_path: str = "/data/sabri/data/evaporate/swde/movie/movie-imdb(2000)"
    pages_path: str = "/home/simarora/code/capsules/scratch/simran/SWDE/data/evaporate/swde/movie/movie-imdb(2000)"
    table_path: str = "/home/simarora/code/capsules/scratch/simran/SWDE/table.json"

    def instantiate(self) -> Context:

        print(f"Creating webpage context with {self.max_tokens_per_section} tokens per section")
        webpage = load_swde_dataset(self.webpage_id, self.pages_path, self.table_path)
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)

        sections = []
        max_tokens = 131072 
        if self.max_tokens_per_section < 0: # icl 
            tokenized = len(tokenizer.encode(webpage.context))
            if tokenized > max_tokens:
                logger.warning(f"IMDB Movie context is too long: {tokenized} tokens, truncating to {max_tokens}")
                context = webpage.context[:max_tokens]
            else:
                context = webpage.context
            section = Section(
                desc=f"IMDB Movie: {webpage.title}",
                content=context,
            )
            sections.append(section)
        
        else:
            tokenized_text = tokenizer.encode(webpage.context)
            tokenized = len(tokenized_text)
            assert self.max_tokens_per_section < max_tokens

            # partition the webpage into sections of max_tokens_per_section
            num_sections = tokenized // self.max_tokens_per_section + 1
            section_size = tokenized // num_sections
            for i in range(num_sections):
                start = i * section_size
                end = (i + 1) * section_size
                if end > tokenized:
                    end = tokenized
                section = Section(
                    desc=f"IMDB Movie: {webpage.title} (Page {i+1}/{num_sections})",
                    content=tokenizer.decode(tokenized_text[start:end]),
                )
                sections.append(section)

            print(f"Created {len(sections)} sections of size {self.max_tokens_per_section} tokens each")

        title = webpage.title
        context = Context(
            title=title,
            sections=sections,
        )
        return context