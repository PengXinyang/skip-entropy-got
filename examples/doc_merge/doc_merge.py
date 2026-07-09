# 版权所有 (c) 2023 ETH Zurich。
#                    保留所有权利。
#
# 本源代码的使用受 BSD 风格许可证约束，具体内容可在 LICENSE 文件中找到。
#
# 主要作者：Nils Blach

import os
import re
import logging
import datetime
import json
import csv
from statistics import fmean
from typing import Dict, List, Callable, Set, Union
from graph_of_thoughts import controller, language_models, operations, prompter, parser


class DocMergePrompter(prompter.Prompter):
    """
    DocMergePrompter 为语言模型生成文档合并示例专用的提示词。

    继承 Prompter 类并实现其抽象方法。
    """

    merge_doc_prompt_start = """Merge the following {num} NDA documents <Doc1> - <Doc{num}> into a single NDA, maximizing retained information and minimizing redundancy. Output only the created NDA between the tags <Merged> and </Merged>, without any additional text.
Here are NDAs <Doc1> - <Doc{num}>
"""
    merge_doc_prompt_block = """
<Doc{num}>
{document}
</Doc{num}>
"""

    merge_doc_prompt_cot_start = """Merge the following {num} NDA documents <Doc1> - <Doc{num}> into a single NDA, maximizing retained information and minimizing redundancy.
You can generate any intermediate thoughts and documents you want, but the final output should be the merged NDA, placed between the two tags <Merged> and </Merged>.
For instance you might want to follow this approach:
1. Split each NDA into their logical subparts.
2. Merge the subparts of the {num} NDAs.
3. Combine the merged subparts into a single NDA.
4. Place the merged NDA between the tags <Merged> and </Merged>.

Here are NDAs <Doc1> - <Doc{num}>:
"""

    improve_summary_prompt_start = """The following NDA <S> merges initial NDAs <Doc1> - <Doc{num}>.
Please improve the summary NDA <S> by adding more information and removing redundancy. Output only the improved NDA, placed between the two tags <Merged> and </Merged>, without any additional text.

Here are NDAs <Doc1> - <Doc{num}>:
"""

    improve_summary_prompt_block = """
<Doc{num}>
{document}
</Doc{num}>
"""

    improve_summary_prompt_end = """
Here is the summary NDA <S>:
<S>
{summary}
</S>
"""

    score_prompt_base = """The following NDA <S> merges NDAs <Doc1> - <Doc{num}>.
Please score the merged NDA <S> in terms of how much redundant information is contained, independent of the original NDAs, as well as how much information is retained from the original NDAs.
A score of 10 for redundancy implies that absolutely no information is redundant, while a score of 0 implies that at least half of the information is redundant (so everything is at least mentioned twice).
A score of 10 for retained information implies that all information from the original NDAs is retained, while a score of 0 implies that no information is retained.
You may provide reasoning for your scoring, but the final score for redundancy should be between the tags <Redundancy> and </Redundancy>, and the final score for retained information should be between the tags <Retained> and </Retained>, without any additional text within any of those tags.

Here are NDAs <Doc1> - <Doc{num}>:
"""

    score_prompt_block = """
<Doc{num}>
{document}
</Doc{num}>
"""

    score_prompt_end = """
Here is the summary NDA <S>:
<S>
{summary}
</S>
"""

    aggregate_full_prompt_base = """The following NDAs <S1> - <S{num_ndas_summary}> each merge the initial NDAs <Doc1> - <Doc{num_ndas}>.
Combine the merged NDAs <S1> - <S{num_ndas_summary}> into a new one, maximizing their advantages and overall information retention, while minimizing redundancy.
Output only the new NDA between the tags <Merged> and </Merged>, without any additional text.   

Here are the original NDAs <Doc1> - <Doc{num_ndas}>:
"""

    aggregate_full_prompt_block1 = """
<Doc{num}>
{document}
</Doc{num}>
"""
    aggregate_full_prompt_mid = """
Here are the summary NDAs <S1> - <S{num_ndas_summary}>:
"""

    aggregate_full_prompt_block2 = """
<S{num}>
{summary}
</S{num}>
"""

    aggregate_sub_prompt_base = """The following NDAs <S1> - <S{num_ndas}> are summaries of some other NDAs.
Combine them into a new one, make sure to maximize their advantages and overall information retention, while minimizing redundancy.
Output only the new NDA between the tags <Merged> and </Merged>, without any additional text.

Here are NDAs <S1> - <S{num_ndas}>:
"""

    aggregate_sub_prompt_generate = """
NDA <S{num}>:
{nda}
</S{num}>
"""

    def aggregation_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        """
        为语言模型生成 aggregation prompt。

        :param state_dicts: 应该被聚合的 thought states。
        :type state_dicts: List[Dict]
        :param kwargs: 额外关键字参数。
        :return: aggregation prompt。
        :rtype: str
        """

        if len(state_dicts[0]["parts"]) > 0 and len(state_dicts[0]["parts"]) < len(
            state_dicts[0]["documents"]
        ):
            prompt = self.aggregate_sub_prompt_base.format(
                num_ndas=len(state_dicts),
            )
            for i, state_dict in enumerate(state_dicts):
                prompt += self.aggregate_sub_prompt_generate.format(
                    nda=state_dict["current"], num=i + 1
                )
            return prompt
        else:
            prompt = self.aggregate_full_prompt_base.format(
                num_ndas=len(state_dicts[0]["documents"]),
                num_ndas_summary=len(state_dicts),
            )
            for i, document in enumerate(state_dicts[0]["documents"]):
                prompt += self.aggregate_full_prompt_block1.format(
                    document=document, num=i + 1
                )
            prompt += self.aggregate_full_prompt_mid.format(
                num_ndas_summary=len(state_dicts),
            )
            for i, state_dict in enumerate(state_dicts):
                prompt += self.aggregate_full_prompt_block2.format(
                    summary=state_dict["current"], num=i + 1
                )
            return prompt

    def generate_prompt(
        self,
        num_branches: int,
        documents: List[str],
        method: str,
        parts: Set[str],
        current: str,
        **kwargs,
    ) -> str:
        """
        为语言模型生成 generate prompt。

        :param num_branches: 提示词要求 LM 生成的响应数量。
        :type num_branches: int
        :param kwargs: 额外关键字参数。
        :return: generate prompt。
        :rtype: str
        """

        prompt = ""
        if method.startswith("io") or method.startswith("cot"):
            if method.startswith("io"):
                prompt += self.merge_doc_prompt_start.format(num=len(documents))
            else:
                prompt += self.merge_doc_prompt_cot_start.format(num=len(documents))
            for i, document in enumerate(documents):
                prompt += self.merge_doc_prompt_block.format(
                    document=document, num=i + 1
                )
            return prompt
        elif method.startswith("tot"):
            if current is None or current == "":
                prompt += self.merge_doc_prompt_start.format(num=len(documents))
                for i, document in enumerate(documents):
                    prompt += self.merge_doc_prompt_block.format(
                        document=document, num=i + 1
                    )
                return prompt
            else:
                prompt += self.improve_summary_prompt_start.format(
                    num=len(documents),
                )
                for i, document in enumerate(documents):
                    prompt += self.improve_summary_prompt_block.format(
                        document=document, num=i + 1
                    )
                prompt += self.improve_summary_prompt_end.format(summary=current)
                return prompt
        elif method.startswith("got"):
            parts = (
                sorted(list(parts)) if len(parts) > 0 else list(range(len(documents)))
            )
            if current is None or current == "":
                prompt += self.merge_doc_prompt_start.format(num=len(parts))
                for i, part in enumerate(sorted(list(parts))):
                    prompt += self.merge_doc_prompt_block.format(
                        document=documents[part], num=i + 1
                    )
                return prompt
            else:
                prompt += self.improve_summary_prompt_start.format(
                    num=len(parts),
                )
                for i, part in enumerate(sorted(list(parts))):
                    prompt += self.improve_summary_prompt_block.format(
                        document=documents[part], num=i + 1
                    )
                prompt += self.improve_summary_prompt_end.format(summary=current)
                return prompt
        else:
            assert False, "Not implemented yet."

    def score_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        """
        为语言模型生成 score prompt。

        :param state_dicts: 应该被打分的 thought states；如果超过一个，则应该一起打分。
        :type state_dicts: List[Dict]
        :param kwargs: 额外关键字参数。
        :return: score prompt。
        :rtype: str
        """

        if len(state_dicts) > 1:
            assert False, "Not implemented yet."
        else:
            # 执行逐项打分
            parts = (
                [
                    state_dicts[0]["documents"][part]
                    for part in sorted(list(state_dicts[0]["parts"]))
                ]
                if len(state_dicts[0]["parts"]) > 0
                else state_dicts[0]["documents"]
            )
            prompt = self.score_prompt_base.format(
                num=len(parts),
            )
            for i, part in enumerate(parts):
                prompt += self.score_prompt_block.format(document=part, num=i + 1)
            prompt += self.score_prompt_end.format(
                summary=state_dicts[0]["current"],
            )
            return prompt

    def improve_prompt(self, **kwargs) -> str:
        """
        为语言模型生成 improve prompt。

        :param kwargs: 额外关键字参数。
        :return: improve prompt。
        :rtype: str
        """
        pass

    def validation_prompt(self, **kwargs) -> str:
        """
        为语言模型生成 validation prompt。

        :param kwargs: 额外关键字参数。
        :return: validation prompt。
        :rtype: str
        """
        pass


class DocMergeParser(parser.Parser):
    """
    DocMergeParser 解析文档合并示例中语言模型的响应。

    继承 Parser 类并实现其抽象方法。
    """

    def __init__(self) -> None:
        """初始化响应缓存。"""
        self.cache = {}

    def strip_answer_helper(self, text: str, tag: str = "") -> str:
        """从文本中移除标签的辅助函数。"""

        text = text.strip()
        if "Output:" in text:
            text = text[text.index("Output:") + len("Output:") :].strip()
        if tag != "":
            start = text.rfind(f"<{tag}>")
            end = text.rfind(f"</{tag}>")
            if start != -1 and end != -1:
                text = text[start + len(f"<{tag}>") : end].strip()
            elif start != -1:
                logging.warning(
                    f"Only found the start tag <{tag}> in answer: {text}. Returning everything after the tag."
                )
                text = text[start + len(f"<{tag}>") :].strip()
            elif end != -1:
                logging.warning(
                    f"Only found the end tag </{tag}> in answer: {text}. Returning everything before the tag."
                )
                text = text[:end].strip()
            else:
                logging.warning(
                    f"Could not find any tag {tag} in answer: {text}. Returning the full answer."
                )
        return text

    def parse_aggregation_answer(
        self, states: List[Dict], texts: List[str]
    ) -> Union[Dict, List[Dict]]:
        """
        解析语言模型对 aggregation prompt 的响应。

        :param states: 用于生成提示词的 thought states。
        :type states: List[Dict]
        :param texts: 语言模型对提示词的响应。
        :type texts: List[str]
        :return: 解析语言模型响应后得到的新 thought states。
        :rtype: Union[Dict, List[Dict]]
        """

        new_states = []
        for text in texts:
            if len(states[0]["parts"]) < len(states[0]["documents"]):
                # 子部分聚合
                text = self.strip_answer_helper(text, "Merged")
                new_state = states[0].copy()
                new_state["current"] = text
                new_state["parts"] = set()
                for state in states:
                    new_state["parts"] = new_state["parts"] | state["parts"]

                new_states.append(new_state)
            else:
                # ?? NDA ??
                text = self.strip_answer_helper(text, "Merged")
                new_state = states[0].copy()
                new_state["current"] = text
                new_states.append(new_state)
        return new_states

    def parse_generate_answer(self, state: Dict, texts: List[str]) -> List[Dict]:
        """
        解析语言模型对 generate prompt 的响应。

        :param state: 用于生成提示词的 thought state。
        :type state: Dict
        :param texts: 语言模型对提示词的响应。
        :type texts: List[str]
        :return: 解析语言模型响应后得到的新 thought states。
        :rtype: List[Dict]
        """
        new_states = []
        for text in texts:
            text = self.strip_answer_helper(text, "Merged")
            new_state = state.copy()
            new_state["current"] = text
            new_states.append(new_state)
        return new_states

    def parse_score_answer(self, states: List[Dict], texts: List[str]) -> List[float]:
        """
        解析语言模型对 score prompt 的响应。

        :param states: 用于生成提示词的 thought states。
        :type states: List[Dict]
        :param texts: 语言模型对提示词的响应。
        :type texts: List[str]
        :return: thought states 的分数。
        :rtype: List[float]
        """
        assert len(states) == 1, "Only one state is allowed for scoring."
        if len(states) == 1:
            # 完整 NDA 聚合
            redundancy_scores = []
            retain_scores = []
            for text in texts:
                answer = self.strip_answer_helper(text, "Redundancy")
                res = re.findall(r"\d+\.?\d*", answer)
                if len(res) == 1:
                    redundancy_scores.append(float(res[0]))
                elif len(res) > 1:
                    logging.warning(
                        f"Found multiple redundancy scores in answer: {text}. Returning the last one."
                    )
                    redundancy_scores.append(float(res[-1]))
                else:
                    logging.warning(
                        f"Could not find any redundancy score in answer: {text}. Ignoring this answer."
                    )
                answer = self.strip_answer_helper(text, "Retained")
                res = re.findall(r"\d+\.?\d*", answer)
                if len(res) == 1:
                    retain_scores.append(float(res[0]))
                elif len(res) > 1:
                    logging.warning(
                        f"Found multiple retained scores in answer: {text}. Returning the last one."
                    )
                    retain_scores.append(float(res[-1]))
                else:
                    logging.warning(
                        f"Could not find any retained score in answer: {text}. Ignoring this answer."
                    )
            if len(redundancy_scores) == 0 or len(retain_scores) == 0:
                logging.warning(
                    f"Could not find any valid score in any answer. Returning 0.0."
                )
                return [0.0]
            mean_redundancy = fmean(redundancy_scores)
            mean_retain = fmean(retain_scores)
            f1 = 2 * mean_redundancy * mean_retain / (mean_redundancy + mean_retain)
            return [f1]

    def parse_improve_answer(self, state: Dict, texts: List[str]) -> Dict:
        """
        解析语言模型对 improve prompt 的响应。

        :param state: 用于生成提示词的 thought state。
        :type state: Dict
        :param texts: 语言模型对提示词的响应。
        :type texts: List[str]
        :return: 解析语言模型响应后得到的新 thought state。
        :rtype: Dict
        """
        pass

    def parse_validation_answer(self, state: Dict, texts: List[str]) -> bool:
        """
        解析语言模型对 validation prompt 的响应。

        :param state: 用于生成提示词的 thought state。
        :type state: Dict
        :param texts: 语言模型对提示词的响应。
        :type texts: List[str]
        :return: thought state 是否有效。
        :rtype: bool
        """
        pass


def io() -> operations.GraphOfOperations:
    """
    为 IO 方法生成 Graph of Operations。

    :return: Graph of Operations。
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 1))
    operations_graph.append_operation(operations.Score(3, False))

    return operations_graph


def cot() -> operations.GraphOfOperations:
    """
    为 CoT 方法生成 Graph of Operations。

    :return: Graph of Operations。
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 1))
    operations_graph.append_operation(operations.Score(3, False))

    return operations_graph


def tot() -> operations.GraphOfOperations:
    """
    为 ToT 方法生成 Graph of Operations。

    :return: Graph of Operations。
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    branch_factor = 10

    operations_graph.append_operation(operations.Generate(1, branch_factor))
    operations_graph.append_operation(operations.Score(3, False))
    keep_best_1 = operations.KeepBestN(1, True)
    operations_graph.append_operation(keep_best_1)

    for _ in range(2):
        operations_graph.append_operation(operations.Generate(1, branch_factor))
        operations_graph.append_operation(operations.Score(3, False))
        keep_best_2 = operations.KeepBestN(1, True)
        keep_best_2.add_predecessor(keep_best_1)
        operations_graph.append_operation(keep_best_2)
        keep_best_1 = keep_best_2

    return operations_graph


def got() -> operations.GraphOfOperations:
    """
    为 GoT 方法生成 Graph of Operations。

    :return: Graph of Operations。
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 5))
    operations_graph.append_operation(operations.Score(3, False))
    keep_best = operations.KeepBestN(3, True)
    operations_graph.append_operation(keep_best)
    operations_graph.append_operation(operations.Aggregate(5))
    operations_graph.append_operation(operations.Score(3, False))
    keep_best2 = operations.KeepBestN(1, True)
    keep_best2.add_predecessor(keep_best)
    operations_graph.append_operation(keep_best2)
    operations_graph.append_operation(operations.Generate(1, 10))
    operations_graph.append_operation(operations.Score(3, False))
    keep_best3 = operations.KeepBestN(1, True)
    keep_best3.add_predecessor(keep_best2)
    operations_graph.append_operation(keep_best3)

    return operations_graph


def got2() -> operations.GraphOfOperations:
    """为 GoT2 方法生成 Graph of Operations，该方法会合并部分文档。"""
    operations_graph = operations.GraphOfOperations()

    sub_parts = []
    for i in range(0, 4, 2):  # should be at most 16 parts
        sub_text = operations.Selector(
            lambda thoughts, list_id=i: [
                operations.Thought(
                    state={**thoughts[0].state, "parts": {list_id, list_id + 1}}
                )
            ]
        )
        operations_graph.add_operation(sub_text)
        gen_nda = operations.Generate(1, 5)
        gen_nda.add_predecessor(sub_text)
        operations_graph.add_operation(gen_nda)
        score_nda = operations.Score(3, False)
        score_nda.add_predecessor(gen_nda)
        operations_graph.add_operation(score_nda)
        keep_best_nda = operations.KeepBestN(1, True)
        keep_best_nda.add_predecessor(score_nda)
        operations_graph.add_operation(keep_best_nda)

        sub_parts.append(keep_best_nda)

    while len(sub_parts) > 1:
        new_sub_parts = []
        for i in range(0, len(sub_parts), 2):
            if i + 1 == len(sub_parts):
                new_sub_parts.append(sub_parts[i])
                continue
            aggregate = operations.Aggregate(5)
            aggregate.add_predecessor(sub_parts[i])
            aggregate.add_predecessor(sub_parts[i + 1])
            operations_graph.add_operation(aggregate)
            score = operations.Score(3, False)
            score.add_predecessor(aggregate)
            operations_graph.add_operation(score)
            keep_best = operations.KeepBestN(1, True)
            keep_best.add_predecessor(score)
            operations_graph.add_operation(keep_best)

            gen_nda = operations.Generate(1, 5)
            gen_nda.add_predecessor(keep_best)
            operations_graph.add_operation(gen_nda)
            score_nda = operations.Score(3, False)
            score_nda.add_predecessor(gen_nda)
            operations_graph.add_operation(score_nda)
            keep_best_nda = operations.KeepBestN(1, True)
            keep_best_nda.add_predecessor(score_nda)
            keep_best_nda.add_predecessor(keep_best)
            operations_graph.add_operation(keep_best_nda)

            new_sub_parts.append(keep_best_nda)
        sub_parts = new_sub_parts

    return operations_graph


def run(
    data_ids: List[int],
    methods: List[Callable[[], operations.GraphOfOperations]],
    budget: float,
    lm_name: str,
) -> float:
    """
    Controller 函数：在预算未耗尽时，对每个指定样本执行每个指定方法。

    :param data_ids: 要运行的样本索引。
    :type data_ids: List[int]
    :param methods: 用于生成 Graphs of Operations 的函数列表。
    :type methods: 每个函数都会生成一个 Graph of Operation。
    :param budget: 执行使用的语言模型预算，单位为美元。
    :type budget: float
    :param lm_name: 要使用的语言模型名称。
    :type lm_name: str
    :return: 已花费预算，单位为美元。
    :rtype: float
    """

    orig_budget = budget
    data_path = os.path.join(os.path.dirname(__file__), "documents.csv")
    data = []
    with open(data_path, "r", encoding="utf8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            row[0] = int(row[0])
            data.append(row)

    if data_ids is None or len(data_ids) == 0:
        data_ids = list(range(len(data)))
    selected_data = [data[i] for i in data_ids]

    results_dir = os.path.join(os.path.dirname(__file__), "results")

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    extra_info = f"{lm_name}_{'-'.join([method.__name__ for method in methods])}"
    folder_name = f"{extra_info}_{timestamp}"
    results_folder = os.path.join(results_dir, folder_name)
    os.makedirs(results_folder)

    config = {
        "data": selected_data,
        "methods": [method.__name__ for method in methods],
        "lm": lm_name,
        "budget": budget,
    }
    with open(os.path.join(results_folder, "config.json"), "w") as f:
        json.dump(config, f)

    logging.basicConfig(
        filename=os.path.join(results_folder, "log.log"),
        filemode="w",
        format="%(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
    )

    for method in methods:
        os.makedirs(os.path.join(results_folder, method.__name__))

    for data in selected_data:
        logging.info(f"Running data {data[0]}: {data[1]}")
        if budget <= 0.0:
            logging.error(
                f"Budget has been depleted, stopping. Data {data[0]} has not been run."
            )
            break
        for method in methods:
            logging.info(f"Running method {method.__name__}")
            logging.info(f"Budget left: {budget}")
            if budget <= 0.0:
                logging.error(
                    f"Budget has been depleted, stopping. Method {method.__name__} has not been run."
                )
                break
            lm = language_models.ChatGPT(
                os.path.join(
                    os.path.dirname(__file__),
                    "../../graph_of_thoughts/language_models/config.json",
                ),
                model_name=lm_name,
                cache=True,
            )
            operations_graph = method()
            executor = controller.Controller(
                lm,
                operations_graph,
                DocMergePrompter(),
                DocMergeParser(),
                {
                    "documents": [data[2], data[3], data[4], data[5]],
                    "parts": set(),
                    "current": "",
                    "method": method.__name__,
                },
            )
            try:
                executor.run()
            except Exception as e:
                logging.error(f"Exception: {e}")
            path = os.path.join(
                results_folder,
                method.__name__,
                f"{data[0]}.json",
            )
            for operation in operations_graph.operations:
                for thought in operation.thoughts:
                    thought.state["parts"] = list(thought.state["parts"])
            executor.output_graph(path)
            budget -= lm.cost

    return orig_budget - budget


if __name__ == "__main__":
    """
    Input (x1, x2, x3, x4): Four NDAs
    Output (y): A new combined NDA
    Evaluation: According to information coverage without repetition (scored by the LLM)
    """
    budget = 30
    samples = [item for item in range(0, 50)]
    approaches = [io, cot, tot, got, got2]

    spent = run(samples, approaches, budget, "chatgpt")

    logging.info(f"Spent {spent} out of {budget} budget.")
