import asyncio
import time
from typing import List, Dict, Any, Optional
import httpx
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import urljoin, urlparse
import base64
import tempfile
import os

from config import settings
from models.schemas import WebBrowsingRequest, WebBrowsingResponse

class WebBrowsingService:
    def __init__(self):
        self.timeout = settings.search_timeout
        self.chrome_options = Options()
        self.chrome_options.add_argument("--headless")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        self.chrome_options.add_argument("--disable-gpu")
        self.chrome_options.add_argument("--window-size=1920,1080")
        
    async def browse_url(self, request: WebBrowsingRequest) -> WebBrowsingResponse:
        """Browse a URL and extract content based on the request"""
        if not settings.enable_web_browsing:
            return WebBrowsingResponse(
                content="Web browsing is disabled",
                url=request.url,
                title="",
                links=[],
                images=[],
                metadata={"error": "Web browsing disabled"}
            )
        
        try:
            if request.action == "read":
                return await self._read_page(request)
            elif request.action == "screenshot":
                return await self._screenshot_page(request)
            elif request.action == "interact":
                return await self._interact_page(request)
            else:
                return await self._read_page(request)
                
        except Exception as e:
            return WebBrowsingResponse(
                content=f"Error browsing URL: {str(e)}",
                url=request.url,
                title="",
                links=[],
                images=[],
                metadata={"error": str(e)}
            )
    
    async def _read_page(self, request: WebBrowsingRequest) -> WebBrowsingResponse:
        """Read page content using HTTP request"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    request.url,
                    headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                )
                
                if response.status_code != 200:
                    return WebBrowsingResponse(
                        content=f"HTTP {response.status_code}: {response.reason_phrase}",
                        url=request.url,
                        title="",
                        links=[],
                        images=[],
                        metadata={"status_code": response.status_code}
                    )
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract title
                title = soup.find('title')
                title_text = title.get_text().strip() if title else ""
                
                # Extract content based on type
                content = ""
                links = []
                images = []
                
                if request.extract_type == "text" or request.extract_type == "all":
                    # Remove script and style elements
                    for script in soup(["script", "style", "nav", "footer", "header"]):
                        script.decompose()
                    
                    # Get main content
                    main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content') or soup.body
                    if main_content:
                        content = main_content.get_text(separator=' ', strip=True)
                    else:
                        content = soup.get_text(separator=' ', strip=True)
                    
                    # Clean up content
                    content = ' '.join(content.split())[:5000]  # Limit content length
                
                if request.extract_type == "links" or request.extract_type == "all":
                    # Extract links
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        absolute_url = urljoin(request.url, href)
                        link_text = link.get_text().strip()
                        if link_text and absolute_url not in links:
                            links.append(absolute_url)
                
                if request.extract_type == "images" or request.extract_type == "all":
                    # Extract images
                    for img in soup.find_all('img', src=True):
                        src = img['src']
                        absolute_url = urljoin(request.url, src)
                        if absolute_url not in images:
                            images.append(absolute_url)
                
                return WebBrowsingResponse(
                    content=content,
                    url=request.url,
                    title=title_text,
                    links=links[:20],  # Limit number of links
                    images=images[:10],  # Limit number of images
                    metadata={
                        "status_code": response.status_code,
                        "content_type": response.headers.get("content-type", ""),
                        "content_length": len(content)
                    }
                )
                
        except Exception as e:
            return WebBrowsingResponse(
                content=f"Error reading page: {str(e)}",
                url=request.url,
                title="",
                links=[],
                images=[],
                metadata={"error": str(e)}
            )
    
    async def _screenshot_page(self, request: WebBrowsingRequest) -> WebBrowsingResponse:
        """Take a screenshot of the page using Selenium"""
        driver = None
        try:
            # Run in thread to avoid blocking
            def take_screenshot():
                nonlocal driver
                driver = webdriver.Chrome(options=self.chrome_options)
                driver.get(request.url)
                
                # Wait for page to load
                WebDriverWait(driver, request.wait_time).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                
                # Take screenshot
                screenshot = driver.get_screenshot_as_png()
                
                # Get page info
                title = driver.title
                current_url = driver.current_url
                
                return screenshot, title, current_url
            
            screenshot, title, current_url = await asyncio.to_thread(take_screenshot)
            
            # Save screenshot to temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_file.write(screenshot)
                screenshot_path = temp_file.name
            
            # Convert to base64 for response
            screenshot_b64 = base64.b64encode(screenshot).decode('utf-8')
            
            return WebBrowsingResponse(
                content=f"Screenshot taken of {current_url}",
                url=current_url,
                title=title,
                links=[],
                images=[],
                metadata={
                    "screenshot_size": len(screenshot),
                    "screenshot_path": screenshot_path
                },
                screenshot_url=f"data:image/png;base64,{screenshot_b64}"
            )
            
        except Exception as e:
            return WebBrowsingResponse(
                content=f"Error taking screenshot: {str(e)}",
                url=request.url,
                title="",
                links=[],
                images=[],
                metadata={"error": str(e)}
            )
        finally:
            if driver:
                driver.quit()
    
    async def _interact_page(self, request: WebBrowsingRequest) -> WebBrowsingResponse:
        """Interact with page elements using Selenium"""
        driver = None
        try:
            def interact():
                nonlocal driver
                driver = webdriver.Chrome(options=self.chrome_options)
                driver.get(request.url)
                
                # Wait for page to load
                WebDriverWait(driver, request.wait_time).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                
                # Get page info
                title = driver.title
                current_url = driver.current_url
                
                # Extract interactive elements
                forms = driver.find_elements(By.TAG_NAME, "form")
                buttons = driver.find_elements(By.TAG_NAME, "button")
                inputs = driver.find_elements(By.TAG_NAME, "input")
                
                interactive_elements = []
                
                for form in forms[:5]:  # Limit to 5 forms
                    interactive_elements.append({
                        "type": "form",
                        "action": form.get_attribute("action") or "",
                        "method": form.get_attribute("method") or "GET"
                    })
                
                for button in buttons[:10]:  # Limit to 10 buttons
                    interactive_elements.append({
                        "type": "button",
                        "text": button.text[:50],  # Limit text length
                        "id": button.get_attribute("id") or "",
                        "class": button.get_attribute("class") or ""
                    })
                
                for input_elem in inputs[:10]:  # Limit to 10 inputs
                    interactive_elements.append({
                        "type": "input",
                        "input_type": input_elem.get_attribute("type") or "text",
                        "name": input_elem.get_attribute("name") or "",
                        "placeholder": input_elem.get_attribute("placeholder") or ""
                    })
                
                return title, current_url, interactive_elements
            
            title, current_url, interactive_elements = await asyncio.to_thread(interact)
            
            content = f"Interactive elements found on {current_url}:\\n"
            for elem in interactive_elements:
                content += f"- {elem['type']}: {elem}\\n"
            
            return WebBrowsingResponse(
                content=content,
                url=current_url,
                title=title,
                links=[],
                images=[],
                metadata={
                    "interactive_elements": interactive_elements,
                    "element_count": len(interactive_elements)
                }
            )
            
        except Exception as e:
            return WebBrowsingResponse(
                content=f"Error interacting with page: {str(e)}",
                url=request.url,
                title="",
                links=[],
                images=[],
                metadata={"error": str(e)}
            )
        finally:
            if driver:
                driver.quit()
    
    async def extract_structured_data(self, url: str) -> Dict[str, Any]:
        """Extract structured data from a webpage"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                structured_data = {}
                
                # Extract JSON-LD
                json_ld_scripts = soup.find_all('script', type='application/ld+json')
                if json_ld_scripts:
                    import json
                    structured_data['json_ld'] = []
                    for script in json_ld_scripts:
                        try:
                            data = json.loads(script.string)
                            structured_data['json_ld'].append(data)
                        except:
                            pass
                
                # Extract meta tags
                meta_tags = {}
                for meta in soup.find_all('meta'):
                    name = meta.get('name') or meta.get('property') or meta.get('itemprop')
                    content = meta.get('content')
                    if name and content:
                        meta_tags[name] = content
                
                structured_data['meta_tags'] = meta_tags
                
                # Extract Open Graph data
                og_data = {}
                for meta in soup.find_all('meta', property=lambda x: x and x.startswith('og:')):
                    property_name = meta.get('property')
                    content = meta.get('content')
                    if property_name and content:
                        og_data[property_name] = content
                
                structured_data['open_graph'] = og_data
                
                return structured_data
                
        except Exception as e:
            return {"error": str(e)}
    
    def get_browsing_capabilities(self) -> Dict[str, Any]:
        """Get information about browsing capabilities"""
        return {
            "actions": ["read", "screenshot", "interact"],
            "extract_types": ["text", "links", "images", "all"],
            "features": {
                "javascript_support": True,
                "screenshot_support": True,
                "form_interaction": True,
                "structured_data_extraction": True
            },
            "limitations": {
                "max_content_length": 5000,
                "max_links": 20,
                "max_images": 10,
                "timeout": self.timeout
            }
        }