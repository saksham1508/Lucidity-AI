import asyncio
import time
from typing import List, Dict, Any
import httpx
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from datetime import datetime

from config import settings
from models.schemas import SearchQuery, SearchResult, SearchResponse, SearchSource

class SearchService:
    def __init__(self):
        self.timeout = settings.search_timeout
        self.max_results = settings.max_search_results
        
    async def search(self, query: SearchQuery) -> SearchResponse:
        """Multi-source search with citation triangulation"""
        start_time = time.time()
        all_results = []
        sources_used = []
        
        # Execute searches in parallel
        search_tasks = []
        
        if SearchSource.WEB in query.sources and settings.enable_web_search:
            search_tasks.append(self._web_search(query.query, query.max_results))
            sources_used.append(SearchSource.WEB)
            
        if SearchSource.BING in query.sources and settings.bing_search_api_key:
            search_tasks.append(self._bing_search(query.query, query.max_results))
            sources_used.append(SearchSource.BING)
            
        if SearchSource.ACADEMIC in query.sources and settings.enable_academic_search:
            search_tasks.append(self._academic_search(query.query, query.max_results))
            sources_used.append(SearchSource.ACADEMIC)
            
        # Execute all searches concurrently
        if search_tasks:
            results_lists = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            for results in results_lists:
                if isinstance(results, list):
                    all_results.extend(results)
        
        # Remove duplicates and rank by relevance
        unique_results = self._deduplicate_results(all_results)
        ranked_results = self._rank_results(unique_results, query.query)
        
        # Limit results
        final_results = ranked_results[:query.max_results]
        
        # Add citations if requested
        if query.include_citations:
            final_results = await self._add_citations(final_results)
        
        search_time = time.time() - start_time
        
        return SearchResponse(
            results=final_results,
            query=query.query,
            total_results=len(final_results),
            search_time=search_time,
            sources_used=sources_used
        )
    
    async def _web_search(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using DuckDuckGo"""
        try:
            with DDGS() as ddgs:
                results = []
                for result in ddgs.text(query, max_results=max_results):
                    search_result = SearchResult(
                        title=result.get('title', ''),
                        content=result.get('body', ''),
                        url=result.get('href', ''),
                        source=SearchSource.WEB,
                        relevance_score=0.8,  # Default score, will be refined
                        timestamp=datetime.now(),
                        citations=[]
                    )
                    results.append(search_result)
                return results
        except Exception as e:
            print(f"Web search error: {e}")
            return []
    
    async def _bing_search(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using Bing Search API"""
        if not settings.bing_search_api_key:
            return []
            
        try:
            headers = {
                'Ocp-Apim-Subscription-Key': settings.bing_search_api_key,
            }
            params = {
                'q': query,
                'count': max_results,
                'responseFilter': 'Webpages',
                'textFormat': 'HTML'
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    'https://api.bing.microsoft.com/v7.0/search',
                    headers=headers,
                    params=params,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = []
                    
                    for item in data.get('webPages', {}).get('value', []):
                        search_result = SearchResult(
                            title=item.get('name', ''),
                            content=item.get('snippet', ''),
                            url=item.get('url', ''),
                            source=SearchSource.BING,
                            relevance_score=0.9,  # Bing typically has good relevance
                            timestamp=datetime.now(),
                            citations=[]
                        )
                        results.append(search_result)
                    
                    return results
                    
        except Exception as e:
            print(f"Bing search error: {e}")
            return []
        
        return []
    
    async def _academic_search(self, query: str, max_results: int) -> List[SearchResult]:
        """Search academic sources (simplified implementation)"""
        try:
            # This is a simplified implementation
            # In production, you'd integrate with arXiv, PubMed, Google Scholar APIs
            academic_query = f"site:arxiv.org OR site:pubmed.ncbi.nlm.nih.gov {query}"
            
            with DDGS() as ddgs:
                results = []
                for result in ddgs.text(academic_query, max_results=max_results//2):
                    search_result = SearchResult(
                        title=result.get('title', ''),
                        content=result.get('body', ''),
                        url=result.get('href', ''),
                        source=SearchSource.ACADEMIC,
                        relevance_score=0.95,  # Academic sources are highly relevant
                        timestamp=datetime.now(),
                        citations=[]
                    )
                    results.append(search_result)
                return results
                
        except Exception as e:
            print(f"Academic search error: {e}")
            return []
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results based on URL and title similarity"""
        seen_urls = set()
        unique_results = []
        
        for result in results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        return unique_results
    
    def _rank_results(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """Rank results by relevance to query"""
        # Simple ranking based on title and content matching
        query_words = set(query.lower().split())
        
        for result in results:
            title_words = set(result.title.lower().split())
            content_words = set(result.content.lower().split())
            
            title_overlap = len(query_words.intersection(title_words))
            content_overlap = len(query_words.intersection(content_words))
            
            # Calculate relevance score
            relevance = (title_overlap * 2 + content_overlap) / len(query_words)
            result.relevance_score = min(relevance, 1.0)
        
        # Sort by relevance score (descending)
        return sorted(results, key=lambda x: x.relevance_score, reverse=True)
    
    async def _add_citations(self, results: List[SearchResult]) -> List[SearchResult]:
        """Add citation information to results"""
        for result in results:
            # Extract potential citations from content
            citations = self._extract_citations(result.content)
            result.citations = citations
        
        return results
    
    def _extract_citations(self, content: str) -> List[str]:
        """Extract citations from content (simplified)"""
        # This is a simplified implementation
        # In production, you'd use more sophisticated citation extraction
        citations = []
        
        # Look for common citation patterns
        import re
        
        # DOI pattern
        doi_pattern = r'10\.\d{4,}\/[^\s]+'
        dois = re.findall(doi_pattern, content)
        citations.extend([f"DOI: {doi}" for doi in dois])
        
        # arXiv pattern
        arxiv_pattern = r'arXiv:\d{4}\.\d{4,5}'
        arxivs = re.findall(arxiv_pattern, content)
        citations.extend(arxivs)
        
        return citations[:5]  # Limit to 5 citations per result
    
    async def get_page_content(self, url: str) -> str:
        """Fetch and extract clean content from a webpage"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=self.timeout)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                    
                    # Get text content
                    text = soup.get_text()
                    
                    # Clean up whitespace
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text = ' '.join(chunk for chunk in chunks if chunk)
                    
                    return text[:5000]  # Limit content length
                    
        except Exception as e:
            print(f"Error fetching page content: {e}")
            return ""
        
        return ""