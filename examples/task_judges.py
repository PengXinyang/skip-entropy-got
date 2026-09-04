import json
from typing import Any, Dict


class BaseTaskJudgePrompt:
    task_name = "generic"

    def task_goal(self, task_context: Dict[str, Any]) -> str:
        return "Solve the task correctly."

    def task_input(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        return task_context.get("case", {})

    def build_prompt(self, task_context: Dict[str, Any], candidate: Dict[str, Any]) -> str:
        payload = {
            "task_name": self.task_name,
            "task_goal": self.task_goal(task_context),
            "task_input": self.task_input(task_context),
            "node": {
                "node_label": candidate.get("node_label"),
                "operation": candidate.get("operation"),
                "operation_index": candidate.get("operation_index"),
                "thought_index": candidate.get("thought_index"),
                "predecessor_outputs": candidate.get("predecessor_outputs"),
                "output": candidate.get("thought_current"),
                "entropy": candidate.get("entropy"),
                "score": candidate.get("score"),
            },
            "full_got_final_answer": candidate.get("final_output"),
        }
        return (
            "You are judging whether one intermediate Graph-of-Thoughts node is "
            "useful for solving the overall task.\n"
            "Return only a valid JSON object, with no markdown and no extra text.\n"
            "Use scores from 0.0 to 1.0.\n\n"
            "Field meanings:\n"
            "- task_relevance: whether the node output is related to the overall task.\n"
            "- input_output_consistency: whether the node output follows from its inputs.\n"
            "- final_answer_contribution: whether the node output contributes information or reasoning used by the final answer.\n"
            "- redundancy: whether this node duplicates information already available from its inputs or sibling nodes.\n"
            "- skip_risk: risk that skipping this node would hurt the final answer.\n"
            "- usefulness: overall usefulness for preserving final answer quality.\n"
            "- reason: one short explanation.\n\n"
            "Expected JSON schema:\n"
            "{\n"
            '  "task_relevance": 0.0,\n'
            '  "input_output_consistency": 0.0,\n'
            '  "final_answer_contribution": 0.0,\n'
            '  "redundancy": 0.0,\n'
            '  "skip_risk": 0.0,\n'
            '  "usefulness": 0.0,\n'
            '  "reason": "short reason"\n'
            "}\n\n"
            "Case data:\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )


class DocMergeJudgePrompt(BaseTaskJudgePrompt):
    task_name = "doc_merge"

    def task_goal(self, task_context: Dict[str, Any]) -> str:
        return (
            "Merge multiple NDA documents into one non-empty NDA, retaining as much "
            "non-redundant legal information as possible."
        )

    def task_input(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        case = task_context.get("case", {})
        return {
            "problem": case.get("problem"),
            "documents": case.get("documents"),
        }


class KeywordCountingJudgePrompt(BaseTaskJudgePrompt):
    task_name = "keyword_counting"

    def task_goal(self, task_context: Dict[str, Any]) -> str:
        return (
            "Count the exact frequency of every country explicitly named in the input text."
        )

    def task_input(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        case = task_context.get("case", {})
        return {
            "original": case.get("original"),
            "ground_truth": case.get("ground_truth"),
        }


class SetIntersectionJudgePrompt(BaseTaskJudgePrompt):
    task_name = "set_intersection"

    def task_goal(self, task_context: Dict[str, Any]) -> str:
        return "Find exactly the numbers that appear in both input sets."

    def task_input(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        case = task_context.get("case", {})
        return {
            "set1": case.get("set1"),
            "set2": case.get("set2"),
            "result": case.get("result"),
        }


class SortingJudgePrompt(BaseTaskJudgePrompt):
    task_name = "sorting"

    def task_goal(self, task_context: Dict[str, Any]) -> str:
        return (
            "Sort the original list in ascending order while preserving exactly the same elements."
        )

    def task_input(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        case = task_context.get("case", {})
        return {
            "original": case.get("original"),
            "ground_truth": case.get("ground_truth"),
        }


def build_task_judge_prompt(task_name: str) -> BaseTaskJudgePrompt:
    if task_name == "doc_merge":
        return DocMergeJudgePrompt()
    if task_name == "keyword_counting":
        return KeywordCountingJudgePrompt()
    if task_name.startswith("set_intersection"):
        return SetIntersectionJudgePrompt()
    if task_name.startswith("sorting"):
        return SortingJudgePrompt()
    return BaseTaskJudgePrompt()
