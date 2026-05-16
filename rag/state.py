from typing import TypedDict, List, Literal
from langchain_core.documents import Document


class App_State(TypedDict):
    user_question: str
    relivent_chunks: List[Document]

    # evaluator
    good_chunks: List[Document]
    evaluation_result: Literal["correct", "incorrect", "ambiguous"]

    # refinement
    refined_sentences: List[str]
    refined_context: str

    # rewrite query
    rewritten_qwery: str

    # web search
    web_search_result: List[str]

    # final answer
    answer: str
    answergiven: bool

    # retry count
    curr_iter: int
    max_iter: int
