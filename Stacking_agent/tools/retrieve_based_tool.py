# def web_search(keywords, search_engine="google"):
#     try:
#         return SerpAPIWrapper(
#             serpapi_api_key=os.getenv("SERP_API_KEY"), search_engine=search_engine
#         ).run(keywords)
#     except:
#         return "No results, try another search"

class LitSearch():
    name: str = "LiteratureSearch"
    description: str = (
        "Input a specific question, returns an answer from literature search. "
        "Do not mention any specific molecule names, but use more general features to formulate your questions."
    )

    def _run(self, query: str) -> str:
        return scholar2result_llm(self.llm, query)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Async not implemented")


class WebSearch():
    name: str = "WebSearch"
    description: str = (
        "Input a specific question, returns an answer from web search. "
        "Do not mention any specific molecule names, but use more general features to formulate your questions."
    )
    def web_search(keywords, search_engine="google"):
        try:
            return SerpAPIWrapper(
                serpapi_api_key=os.getenv("SERP_API_KEY"), search_engine=search_engine
            ).run(keywords)
        except:
            return "No results, try another search"

    def _run(self, query: str) -> str:
        return web_search(query)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Async not implemented")

