from langchain_community.retrievers import WikipediaRetriever

class wikipedia_search_tool():
    name = "wikipedia_search_tool"
    description = (
        'The input is an entity name. The action will search this entity name on Wikipedia and returns the first  paragraph if it exists. If not, it will return some similar entities to search next.'
    )

    def __init__(self):
        super(wikipedia_search_tool, self).__init__()

    def _run(self, query: str) -> str:
        """Search the input query using wikipeida search api."""
        retriever = WikipediaRetriever(top_k_results=1)
        docs = retriever.invoke(query)
        return 'The wikipedia search result is :' + docs[0].page_content

    def __str__(self):
        return "Wikipedia search tool"

    def __repr__(self):
        return self.__str__()

    def wo_run(self,query):
        return 'The wikipedia search result is :' + self._run(query)
    
if __name__ == '__main__':
    tool = wikipedia_search_tool()
    print(tool._run('2023年世界杯'))
