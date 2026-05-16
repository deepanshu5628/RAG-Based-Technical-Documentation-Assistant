import os
from typing import List
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from tavily import TavilyClient

from Models.OutputSchema import EvalulationScore, KnowledgeRefinementResult, RewritenQwery, GeneratedResponce
from Utils.HelperFxns import text_to_sentence_converter
from rag.state import App_State
from rag.loader import retriver

model = init_chat_model("gpt-4o")
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def retriveer_node(state: App_State):
    user_question = state["user_question"]
    print("retriver fxn has been caleed ")
    retrived_chunks = retriver.invoke(user_question)
    return {"relivent_chunks": retrived_chunks}


def reterival_eval_node(state: App_State):
    print("the retrival evaluation node has been called ")
    merit_score = 8
    passing_score = 4

    chunks = state["relivent_chunks"]
    good_chunks = []
    eval_score = []
    for chunk in chunks:
        sys_msg = SystemMessage(content="you are a evaluator. you have to see the question and check if the given context is able to answer that question and based on that  you will  give a score between 0 to 10 . ")
        human_msg = HumanMessage(content=f"give me the score Question->{state['user_question']} \n\n Context-> {chunk.page_content}")
        final_model = model.with_structured_output(EvalulationScore)
        res = final_model.invoke([sys_msg, human_msg])
        print("the out of eval llm ", res.score)
        eval_score.append(res.score)
        if(res.score >= passing_score):
            good_chunks.append(chunk)

    print(f"the eval score are {eval_score}")
    if any(s >= merit_score for s in eval_score):
        state["evaluation_result"] = "correct"
    elif all(s < passing_score for s in eval_score):
        state["evaluation_result"] = "incorrect"
    else:
        state["evaluation_result"] = "ambiguous"

    state["good_chunks"] = good_chunks
    return state


def knowledge_refinement_node(state: App_State):
    print("in the knowledge refinement node")
    combined_text = " "
    if(state["evaluation_result"] == "incorrect"):
        for web_res in state["web_search_result"]:
            combined_text += web_res + "\n\n"
    elif(state["evaluation_result"] == "ambiguous"):
        for chunk in state["good_chunks"]:
            combined_text += chunk.page_content + "\n\n"
        for web_res in state["web_search_result"]:
            combined_text += web_res + "\n\n"
    else:
        for chunk in state["good_chunks"]:
            combined_text += chunk.page_content + "\n\n"

    sentences = text_to_sentence_converter(combined_text)

    refined_sentences: List[str] = []

    for sentence in sentences:
        Sys_mess = SystemMessage(content="you are a helpful assistant. you have to see the question and the sentence and check if the sentence is able to answer that question and based on that you will return a boolean value true or false.")
        Human_mess = HumanMessage(content=f"Question: {state['user_question']} \n\n Sentence: {sentence}")
        final_model = model.with_structured_output(KnowledgeRefinementResult)
        res = final_model.invoke([Sys_mess, Human_mess])
        print(f"the sentence is {sentence} and the relevance is {res.result}")
        if(res.result):
            refined_sentences.append(sentence)

    refined_context = "\n".join(refined_sentences).strip()

    state["refined_sentences"] = refined_sentences
    state["refined_context"] = refined_context
    return state


def generate_node(state: App_State):
    print("generate node fxn has been caleed ")
    user_question = state["user_question"]
    context = state["refined_context"]

    sys_msg = SystemMessage(content="You are a helpful assistant. only answer the quesion from the given context. if you don't find the answer there. just say i don't know")
    human_msg = HumanMessage(content=f"Answer this user question->{user_question} \n\n based on the context:\n\n{context}")
    final_model = model.with_structured_output(GeneratedResponce)
    model_res = final_model.invoke([sys_msg, human_msg])
    iter_count = state.get("curr_iter", 0) + 1
    return {"answer": model_res.answer, "answergiven": model_res.answergiven, "curr_iter": iter_count}


def rewrite_qwery_node(state: App_State):
    print("in the rewrite qwery node")
    sys_mess = SystemMessage(content="you are a query rewriter. you have to rewrite the user question in such a way that it can be answered by the retrieved documents or by the web search. you can add more details to the question to make it more specific and answerable. but you should not change the intent of the question.")
    human_mess = HumanMessage(content=f"rewrite this user question->{state['user_question']}")
    final_model = model.with_structured_output(RewritenQwery)
    model_res = final_model.invoke([sys_mess, human_mess])
    state["rewritten_qwery"] = model_res.rewritten_qwery
    return state


def web_search_node(state: App_State):
    print("in the web search node")
    search_res = tavily.search(state["rewritten_qwery"], max_results=4)
    search_results: List[str] = []
    for res in search_res["results"]:
        search_results.append(res["content"])
    state["web_search_result"] = search_results
    return state


def route_after_eval(state: App_State):
    return state["evaluation_result"]


def retries_fxn(state: App_State):
    if(state["answergiven"] == True):
        return "end"
    if(state["answergiven"] == False and state["curr_iter"] < state["max_iter"]):
        return "retry"
    if(state["answergiven"] == False and state["curr_iter"] >= state["max_iter"]):
        return "end"
