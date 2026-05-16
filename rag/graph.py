from langgraph.graph import START, END, StateGraph

from rag.state import App_State
from rag.nodes import (
    retriveer_node,
    reterival_eval_node,
    knowledge_refinement_node,
    generate_node,
    rewrite_qwery_node,
    web_search_node,
    route_after_eval,
    retries_fxn,
)

graph = StateGraph(App_State)
graph.add_node("Retrive", retriveer_node)
graph.add_node("Reterival_Evaluator", reterival_eval_node)
graph.add_node("Knowledge_Refinement", knowledge_refinement_node)
graph.add_node("Generate", generate_node)
graph.add_node("Rewrite_Qwery", rewrite_qwery_node)
graph.add_node("WebSearchNode", web_search_node)

graph.add_edge(START, "Retrive")
graph.add_edge("Retrive", "Reterival_Evaluator")
graph.add_conditional_edges("Reterival_Evaluator", route_after_eval, {
    "correct": "Knowledge_Refinement",
    "incorrect": "Rewrite_Qwery",
    "ambiguous": "Rewrite_Qwery"
})
graph.add_edge("Rewrite_Qwery", "WebSearchNode")
graph.add_edge("WebSearchNode", "Knowledge_Refinement")
graph.add_edge("Knowledge_Refinement", "Generate")
graph.add_conditional_edges("Generate", retries_fxn, {
    "retry": "Retrive",
    "end": END
})

workflow = graph.compile()
