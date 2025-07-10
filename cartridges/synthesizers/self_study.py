
from collections import defaultdict
import asyncio
import time
import uuid
import random
from typing import List

from transformers import AutoTokenizer

from cartridges.structs import TrainingExample
from cartridges.tools.base import instantiate_tools
from cartridges.clients.base import ClientConfig, ClientSample
from cartridges.synthesizers.base import AsyncConvoSynthesizer, ConvoSynthesizer
from cartridges.tools.base import Tool, ToolSet, ToolOutput
from cartridges.tools import MODEL_TO_TOOL_TEMPLATE, MODEL_TO_TOOL_CALL_PARSER, ToolCall, render_tool_template
from cartridges.utils import get_logger
from cartridges.resources.base import Resource

logger = get_logger(__name__)

TOOL_PROMPT_TEMPLATE = """You need to respond to the following message:

<message>
{message}
</message>
{tools}"""

SYSTEM_PROMPT_TEMPLATE = """
You are in a conversation about the following user information.

<info>
{subcorpus}
</info>"""


class SelfStudySynthesizer(AsyncConvoSynthesizer):

    class Config(ConvoSynthesizer.Config):
        client: ClientConfig

        resources: List[Resource.Config]

        tools: List[Tool.Config | ToolSet.Config]
        use_tools_a: bool = False
        use_tools_b: bool = False
        max_tool_tokens: int = 128

        system_prompt_template: str = SYSTEM_PROMPT_TEMPLATE
        tool_prompt_template: str = TOOL_PROMPT_TEMPLATE

        max_rounds: int = 1

        temperature_a: float = 0.6
        max_completion_tokens_a: int = 512
        prob_cot_a: float = 0.0

        temperature_b: float = 0.0
        max_completion_tokens_b: int = 1024

        num_top_logprobs: int = 20


    def __init__(self, config: Config):
        self.config = config

        self.client = self.config.client.instantiate()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.client.model_name)
    
        self.is_setup = False

        random.seed(82)
    
    async def setup(self):
        tools_list, cleanup_tasks = await instantiate_tools(self.config.tools)
        self.tools: Dict[str, Tool] = {tool.name: tool for tool in tools_list}
        self.cleanup_tasks = cleanup_tasks
        
        self.resources: List[Resource] = [
            resource.instantiate() for resource in self.config.resources
        ]
        await asyncio.gather(*[resource.setup() for resource in self.resources])
    
        self.is_setup = True
    
    async def cleanup(self):
        """Clean up tools and resources"""
        for task in self.cleanup_tasks:
            await task()
        self.is_setup = False
    
    async def __aenter__(self):
        await self.setup()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
        return False

    async def sample_convos(
        self, batch_idx: int, batch_size: int, total_batches: int
    ) -> list[TrainingExample]:
        batch_id = f"{batch_idx}"

        if not self.is_setup:
            raise RuntimeError("Synthesizer not setup. Call setup() first.")

        # (1) Get initial system prompt and seed prompts
        # --- begin prompt sampling ---
        t0 = time.time()
        resource = random.choice(self.resources)
        ctx, seed_prompts = await resource.sample_prompt(batch_size=batch_size)

        initial_system_prompt = self.config.system_prompt_template.format(subcorpus=ctx)
        assert len(seed_prompts) == batch_size
        logger.info(f"[batch={batch_id}] Prompt sampling took {time.time() - t0} seconds")
        # --- end prompt sampling ---

        # (2) Initialize convos
        # --- begin initialization of convos ---
        t0 = time.time()
        convos: List[List[dict]] = [[] for _ in range(batch_size)]
        contexts: List[str] = [initial_system_prompt] * batch_size
        metas: List[dict] = [
            {
                "tool_calls": [],
                "seed_prompt": seed_prompt,
                "initial_system_prompt": initial_system_prompt,
            }
            for seed_prompt in seed_prompts
        ]
        logger.info(f"[batch={batch_id}] Initialization of convos took {time.time() - t0} seconds")
        # --- end initialization of convos ---
        # (3) Generate convos
        for round_idx in range(self.config.max_rounds):

            # (3.1) bot_a requests new content to be added to the context
            # --- begin bot A tool usage ---
            if self.config.use_tools_a:
                t0 = time.time()

                tool_resps: List[str] = await self._get_content_via_tool(
                    convos=[
                        trim_fields([user(seed), *flip_roles(convo)])
                        for seed, convo in zip(seed_prompts, convos)
                    ],
                    metas=metas,
                    contexts=contexts,
                    batch_id=batch_id,
                )
                contexts = [ctx + self._tool_responses_to_str(resp) for ctx, resp in zip(contexts, tool_resps)]
                logger.info(
                    f"[batch={batch_id}] Round {round_idx}: Bot A tool usage (select + apply) took {time.time() - t0} seconds"
                )
            # --- end bot A tool usage ---

            # (3.2) With new information in context, generate user message
            # --- begin bot A response generation ---
            t0 = time.time()
            resps = await self.client.chat(
                [
                    trim_fields([system(ctx), user(seed), *flip_roles(convo)])
                    for ctx, seed, convo in zip(contexts, seed_prompts, convos)
                ],
                temperature=self.config.temperature_a,
                max_completion_tokens=self.config.max_completion_tokens_a,
                modal_upstream_id=batch_id,
                enable_thinking=False,
            )
            resps = resps.samples
            convos = [
                convo
                + [
                    user(
                        resp.text,
                        cot=random.random() < self.config.prob_cot_a,
                        resp_obj=resp,
                    )
                ]
                for convo, resp in zip(convos, resps)
            ]
            logger.info(
                f"[batch={batch_id}] Round {round_idx}: Bot A response generation took {time.time() - t0} seconds"
            )
            # --- end bot A response generation ---

            # (3.3) bot_b requests new content to be added to the context
            # --- begin bot B tool usage ---
            if self.config.use_tools_b:
                t0 = time.time()
                tool_resps: List[str] = await self._get_content_via_tool(
                    convos=trim_fields(convos),
                    metas=metas,
                    contexts=contexts,
                    batch_id=batch_id,
                )
                contexts = [ctx + self._tool_responses_to_str(resp) for ctx, resp in zip(contexts, tool_resps)]
                logger.info(
                    f"[batch={batch_id}] Round {round_idx}: Bot B tool usage (select + apply) took {time.time() - t0} seconds"
                )
            # --- end bot B tool usage ---

            # (3.4) bot_b generates a response
            # --- begin bot B response generation ---
            t0 = time.time()
            resps = await self.client.chat(
                [trim_fields([system(ctx), *convo]) for ctx, convo in zip(contexts, convos)],
                temperature=self.config.temperature_b,
                top_logprobs=self.config.num_top_logprobs,
                max_completion_tokens=self.config.max_completion_tokens_b,
                modal_upstream_id=batch_id,
                enable_thinking=False,
            )
            resps: List[ClientSample] = resps.samples
            convos = [
                convo + [assistant(resp.text, resp_obj=resp)]
                for convo, resp in zip(convos, resps)
            ]
            logger.info(
                f"[batch={batch_id}] Round {round_idx}: Bot B response generation took {time.time() - t0} seconds"
            )
            # --- end bot B response generation ---

        # (4) Convert responses and chats to training examples
        # --- begin conversion to training examples ---
        t0 = time.time()
        examples = self._responses_and_chats_to_training_examples(
            samples=resps,
            convos=convos,
            metas=metas,
            contexts=contexts,
        )
        logger.info(f"[batch={batch_idx}] Conversion to training examples took {time.time() - t0} seconds")
        # --- end conversion to training examples ---
        return examples

    async def _get_content_via_tool(
        self,
        convos: list[list[dict]],
        metas: list[dict],
        contexts: list[str],
        batch_id: str,
    ) -> List[List[ToolOutput]]:
        # (1) Build a string describing all of the available tools and their arguments
        # --- begin tool string ---
        tool_defs = [tool.definition for tool in self.tools.values()]
        template = MODEL_TO_TOOL_TEMPLATE[self.config.client.model_name]
        tool_str = render_tool_template(tools=tool_defs, template=template)
        assert convos[0][-1]["role"] == "user"
        # --- end tool string ---

        # (2) Query the model to pick a tool and set its arguments
        # --- begin tool selection ---
        t0 = time.time()
        resps = await self.client.chat(
            [
                [system(ctx + f"\n\n {tool_str}")]
                # we funk with the last user message to add the tool prompt
                + convo[:-1]
                + [user(self.config.tool_prompt_template.format(message=convo[-1], tools=tool_str))]
                for ctx, convo in zip(contexts, convos)
            ],
            temperature=self.config.temperature_a,
            max_completion_tokens=self.config.max_tool_tokens,
            modal_upstream_id=batch_id,
        )
        resps = resps.samples
        reqs = [resp.text for resp in resps]
        logger.info(f"[batch={batch_id}] Tool selection took {time.time() - t0} seconds")
        # --- end tool selection ---

        # (3) Parse the tool responses and apply the tool. If it fails, just return empty string
        # --- begin tool application ---
        t0 = time.time()
        results: List[List[ToolOutput]] = [[]] * len(reqs)

        # (3.1) Group requests by tool 
        # --- begin tool grouping ---
        parser = MODEL_TO_TOOL_CALL_PARSER[self.config.client.model_name]
        tool_to_reqs = defaultdict(list)
        for idx, (req, meta) in enumerate(zip(reqs, metas, strict=True)):
            try:
                tool_calls: List[ToolCall] = parser(req)
                
                for call in tool_calls:
                    tool_obj = self.tools[call.function.name]
                    tool_to_reqs[call.function.name].append({
                        "idx": idx,
                        "spec": call.function.arguments,
                        "tool_obj": tool_obj,
                        "input": tool_obj.ToolInput(**call.function.arguments),
                        "raw_request": req,
                    })

            except Exception as e:
                logger.info(f"Error parsing tool request: {type(e).__name__}: {e}")
                results[idx].append(
                    ToolOutput(
                        success=False,
                        error=str(e),
                        input=None,
                        response=None,
                    )
                )
        
        # --- end tool grouping ---

        # (3.2) Apply the tool in batch
        # --- begin applying tool in groups ---
        tool_outputs: List[List[ToolOutput]] = await asyncio.gather(
            *(
                self.tools[tool].batch_run_tool([req["input"] for req in reqs])
                for tool, reqs in tool_to_reqs.items()
            )
        )
    
        for (tool, curr_reqs), outputs in zip(tool_to_reqs.items(), tool_outputs):
            for req, output in zip(curr_reqs, outputs):
                idx = req["idx"]
                results[idx].append(output)
                
                # Store tool call results in metadata
                tool_call_record = {
                    "name": tool,
                    "input": req["spec"],
                    "output": output.response if output.success else f"Error: {output.error}",
                    "success": output.success,
                    "raw_request": req["raw_request"]
                }
                metas[idx]["tool_calls"].append(tool_call_record)
        # --- end applying tool in groups ---

        logger.info(f"[batch={batch_id}] Tool application took {time.time() - t0} seconds")
        # --- end tool application ---

        return results
    
    def _tool_responses_to_str(self, tool_outputs: List[ToolOutput]) -> str:
        out = []
        for tool in tool_outputs:
            if not tool.success:
                continue 

            out.append(
                f"<tool_call>\n" 
                f"<tool_input>{tool.input.dict()}</tool_input>\n"
                f"<tool_output>{tool.response}</tool_output>\n"
                f"</tool_call>\n"
            )
        return "\n".join(out)

    def _responses_and_chats_to_training_examples(
        self,
        samples: list[ClientSample],
        convos: list[list[dict]],
        metas: list[dict],
        contexts: list[str] | None,
    ) -> list[TrainingExample]:
        examples = []
        for chat, meta, context in zip(
            convos,
            metas,
            contexts,
            strict=True,
        ):
            
            examples.append(
                TrainingExample(
                    messages=[
                        TrainingExample.Message(
                            role=message["role"],
                            content=message["content"],
                            token_ids=message["resp_obj"].token_ids,
                            top_logprobs=message["resp_obj"].top_logprobs,
                        )
                        for message in chat
                    ],
                    type="todo",
                    metadata=meta,
                    system_prompt=context,
                )
            )
        return examples


# --- begin chat helper functions ---
def system(content: str) -> dict:
    return dict(role="system", content=content)


def user(content: str, cot: bool = False, resp_obj: ClientSample = None) -> dict:
    if cot:
        instruction = random.choice(COT_INSTRUCTIONS)
        content = f"{content}\n\n{instruction}"
    return dict(role="user", content=content, resp_obj=resp_obj)


def assistant(content: str, resp_obj: ClientSample) -> dict:
    return dict(role="assistant", content=content, resp_obj=resp_obj)


def flip_roles(convo: list[dict]) -> list[dict]:
    def flip_role(role: str) -> str:
        if role == "user":
            return "assistant"
        elif role == "assistant":
            return "user"
        return role

    return [dict(role=flip_role(d["role"]), content=d["content"]) for d in convo]

def trim_fields(convo: list[dict]) -> list[dict]:
    return [dict(role=d["role"], content=d["content"]) for d in convo]

# --- end chat helper functions ---



COT_INSTRUCTIONS = [
    "If helpful, you can think before responding. Put your thinking between <thinking> and </thinking> tags. Then, provide your final response between <response> and </response> tags.",
    "Respond in the following format: <thinking>...</thinking> <response>...</response>",
    "Explain your reasoning before providing your final response.",
    "Explain your reasonining between <reasoning> and </reasoning> tags.",
    "Provide your final answer within <answer>...</answer> tags. Optionally, you can explain your reasoning between <reasoning> and </reasoning> tags.",
    "You may include your reasoning before answering. Use <reasoning>...</reasoning> to enclose your thoughts, and <final>...</final> for your answer.",
    "First, think through the problem and enclose your thoughts in <thought>...</thought>. Then, present your answer clearly in <output>...</output>.",
    "Use <step-by-step>...</step-by-step> for your intermediate reasoning, followed by <answer>...</answer> for the final result.",
    "Start with your analysis in <deliberation>...</deliberation>. Conclude with a clear answer in <response>...</response>.",
    "You may show your chain of thought in <chain>...</chain> and state your final decision in <decision>...</decision>.",
    "Wrap your reasoning process in <logic>...</logic>, and your ultimate conclusion in <conclusion>...</conclusion>.",
    "Think carefully before answering. Put your process in <process>...</process> and your solution in <solution>...</solution>.",
    "Please provide your thought process first, enclosed in <rationale>...</rationale>, and then give your final answer in <final_answer>...</final_answer>.",
    "Include your reasoning in <analysis>...</analysis> and your conclusion in <result>...</result>.",
    "Begin with <thinking_process>...</thinking_process> to show how you reasoned through the problem. Finish with <response>...</response>.",
    "Use <explanation>...</explanation> to walk through the logic. Then state your answer in <output>...</output>.",
    "Present your logical steps in <reasoning_chain>...</reasoning_chain> and conclude with <final_response>...</final_response>.",
    "Start with <evaluation>...</evaluation> to explain how you analyzed the question. Then, give your answer in <decision>...</decision>.",
    "Outline your reasoning in <deduction>...</deduction> and present the final answer in <resolution>...</resolution>.",
    "First explain in <justification>...</justification>, then give your definitive answer in <answer>...</answer>.",
    "Place your step-by-step logic in <path>...</path> and your outcome in <solution>...</solution>.",
    "Break down the problem in <walkthrough>...</walkthrough> before stating your answer in <conclusion>...</conclusion>.",
    "Reason through the problem in <examine>...</examine> and finalize with <respond>...</respond>.",
    "Give your thought process inside <trace>...</trace> and the answer inside <reply>...</reply>.",
    "Write your full reasoning under <work>...</work>, then state the answer clearly in <end>...</end>.",
    "Use <rundown>...</rundown> to explain your steps, and <final>...</final> to share your answer.",
    "First, walk through your reasoning process step by step. Then, clearly state your final answer.",
    "Begin by explaining how you approach the problem. Afterward, give your final response.",
    "Start with a detailed breakdown of your thought process. Conclude with a concise answer.",
    "Explain your logic as you work through the problem. When you're done, provide your conclusion.",
    "Think out loud as you reason through the question. End with a definitive answer.",
    "Work through the problem in detail, reasoning carefully. Then summarize your final decision.",
    "Describe each step you take to solve the problem. Finish by stating the final result.",
    "Provide a thorough explanation of how you arrive at your answer. Then state the answer clearly.",
    "Show your reasoning process from start to finish. Make sure to give your final answer at the end.",
    "Break the problem into logical steps and explain each one. Then give your final response.",
    "Write out your reasoning clearly and methodically. Conclude with your final conclusion.",
    "Reflect on the problem and describe your full reasoning. Then, say what your answer is.",
    "Think critically about the question and narrate your process. Then provide your final decision.",
    "Show your internal reasoning inside `{{Rationale}}...{{/Rationale}}`. Give the final conclusion inside `{{Conclusion}}...{{/Conclusion}}`.",
    "Detail your problem-solving steps between `[Steps]...[/Steps]`. Provide the answer between `[Result]...[/Result]`.",
    "Explain the derivation process using `Derivation: ... End Derivation`. State the final output using `Output: ... End Output`.",
    "Map out your thought path within `<Path>...</Path>`. State the final destination within `<Destination>...</Destination>`.",
    "Provide your analysis enclosed in `<<Analysis>>...<</Analysis>>`. Present the determined answer enclosed in `<<Answer>>...<</Answer>>`.",
    "Elaborate on your thinking process with `// Elaboration Start` and `// Elaboration End`. Provide the final response with `// Response Start` and `// Response End`.",
    "Use `(Thought Process: ... )` to show your thinking. Use `(Final Answer: ... )` for the result.",
    "Lay out the groundwork in `<Foundation>...</Foundation>`. Build the final answer in `<Structure>...</Structure>`.",
    "Document the logical flow in `{* Logic Flow *} ... {* /Logic Flow *}`. Deliver the outcome in `{* Outcome *} ... {* /Outcome *}`.",
    "Begin with your deliberation, marked by `Deliberation: ...`. Conclude with your final decision, marked by `Decision: ...`.",
    "Record your internal monologue in `<Monologue>...</Monologue>`. State the external response in `<Statement>...</Statement>`.",
    "Chart the sequence of reasoning within `[[Sequence]]...[[/Sequence]]`. Present the end point within `[[Endpoint]]...[[/Endpoint]]`.",
    "Dissect the problem in `<Dissection>...</Dissection>`. Synthesize the answer in `<Synthesis>...</Synthesis>`.",
    "Narrate your thought process using `Narrative: ...`. Provide the concluding answer using `Conclusion: ...`.",
    "Outline your strategy in `<Strategy>...</Strategy>`. Execute the final answer in `<Execution>...</Execution>`.",
    "Feel free to think it through out loud first, then just drop your answer at the end.",
    "Walk yourself through the problem—no rush. Once you're set, say what you'd go with.",
    "You can talk it out step by step. Just wrap up with whatever you think the answer is.",
    "Think it through however you like, and then let me know your final call.",
    "Start by working through the logic in your own way. When you're done, give the answer.",
    "Lay out your thinking as it comes to you. At the end, just say what you'd choose.",
    "Break it down how you want, no pressure. Then tell me what your final answer would be.",
    "Talk yourself through the reasoning part. When it feels right, give your answer.",
    "Explain it like you're figuring it out in real time, then land on your pick.",
    "Take a moment to think it through, and when you're ready, just say your answer.",
    "Work it out however makes sense to you. Then drop your answer when you're good.",
    "Go step by step, like you're thinking out loud. End with whatever answer you'd settle on.",
    "Walk through your reasoning step by step, and once it all clicks, share your answer.",
    "Take me through your thought process from start to finish, then tell me your conclusion.",
    "Think out loud as you piece it together. When you’re ready, just state your answer.",
    "Break down the problem in your own words, then wrap up with your choice.",
    "Talk through each part as you solve it, and then give your final answer.",
    "Map out your logic as you go, and once you’re confident, give your answer.",
    "Reason it out at your own pace, and when you’re set, let me know your answer.",
    "Feel free to puzzle it out stepwise. At the end, just say your final pick.",
    "Process the question however you like. Once you’ve worked it out, share your answer.",
    "Unpack the problem in detail, and when you’re ready, provide your answer.",
    "Work through the details in your own style, and then land on your answer.",
    "Think through the scenario, explaining as you go, and finish with your answer.",
    "Go through your reasoning openly, and then make your final call.",
    "Lay out your analysis clearly, and when you reach a conclusion, share it.",
    "Step through your logic in real time, and end with your decision.",
    "Explain your approach as you solve it, then just give your answer at the end.",
    "Go through the motions of solving it, and when you’re done, state your answer.",
    "Work the problem out in your own way, and when you’re ready, say your answer.",
    "Detail your thought process as it unfolds, then close with your answer.",
    "Take your time to reason through it, and when you’ve got it, give your answer.",
    "Unpack your reasoning in <logic>...</logic>. Conclude with your solution in <result>...</result>.",
    "Share your thought process inside [Reasoning]...[/Reasoning]. State your answer in [Answer]...[/Answer].",
    "Walk through your approach in <<<Thinking>>>...<<<End Thinking>>>. Provide your answer in <<<Answer>>>...<<<End Answer>>>.",
    "Place your analysis in <Breakdown>...</Breakdown>. Offer the answer in <Solution>...</Solution>.",
    "Use [[Rationale]]...[[/Rationale]] for your reasoning. Use [[Conclusion]]...[[/Conclusion]] for your answer.",
    "Think through the problem in # Reasoning: ... # End Reasoning. Give your answer in # Answer: ... # End Answer.",
    "Lay out your process in <Process>...</Process>. Deliver your conclusion in <Conclusion>...</Conclusion>.",
    "Step through your logic in {Reasoning: ... }. Present your answer in {Answer: ... }.",
    "Present your thought process in --- Reasoning --- ... --- End Reasoning ---. State your answer in --- Answer --- ... --- End Answer ---.",
    "Outline your thinking in (Reasoning Start)...(Reasoning End). Wrap up with (Answer Start)...(Answer End).",
    "Describe your approach using <Approach>...</Approach>. Summarize your answer using <Summary>...</Summary>.",
    "Go through your logic in [Logic Path]...[/Logic Path]. Finalize with [Final Response]...[/Final Response].",
    "Map out your steps in <<Steps>>...<</Steps>>. Place your answer in <<Result>>...<</Result>>.",
    "Narrate your reasoning in <Explanation>...</Explanation>. State your answer in <Reply>...</Reply>.",
    "Explain your process inside {Process: ...}. Conclude with {Result: ...}.",
    "Break down your logic in [Analysis]...[/Analysis]. Conclude with your answer in [Conclusion]...[/Conclusion].",
    "Think out your process in <Thoughts>...</Thoughts>. Share your answer in <Answer>...</Answer>.",
    "Detail your reasoning in [[Analysis]]...[[/Analysis]]. Give the answer in [[Final]]...[[/Final]].",
    "Work through your thoughts in --- Process --- ... --- /Process ---. Finish with --- Solution --- ... --- /Solution ---.",
    "Talk through your logic in {Deliberation}...{/Deliberation}. Give your answer in {Decision}...{/Decision}.",
]