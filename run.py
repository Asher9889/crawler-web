import uvicorn
import os

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator


load_dotenv()

# -----------------------------
# Request / Response Schemas
# -----------------------------

class CrawlRequest(BaseModel):
    url: HttpUrl


class CrawlResponse(BaseModel):
    success: bool
    url: str
    title: str | None
    markdown: str
    cleaned_markdown: str
    # cleaned_html: str


# -----------------------------
# Lifespan
# -----------------------------

crawler: AsyncWebCrawler | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global crawler

    crawler = AsyncWebCrawler()
    await crawler.__aenter__()

    yield

    await crawler.__aexit__(None, None, None)


# -----------------------------
# FastAPI
# -----------------------------

app = FastAPI(
    title="Knowledge Acquisition Service",
    version="1.0.0",
    lifespan=lifespan,
)


# -----------------------------
# Health
# -----------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok"
    }

if __name__ == "__main__":
    uvicorn.run(
        "run:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT")),
        reload=True,
    )

# -----------------------------
# Crawl Endpoint
# -----------------------------

@app.post("/crawl", response_model=CrawlResponse)
async def crawl(request: CrawlRequest):
    global crawler

    if crawler is None:
        raise HTTPException(
            status_code=500,
            detail="Crawler not initialized",
        )

    try:
        config = CrawlerRunConfig(
            markdown_generator=DefaultMarkdownGenerator(
                content_filter=PruningContentFilter()
            ),
        )
        result = await crawler.arun(
            url=str(request.url),
            config=config
        )

        return CrawlResponse(
            success=result.success,
            url=result.url,
            title=result.metadata.get("title") if result.metadata else None,
            markdown=result.markdown,
            cleaned_markdown=result.markdown.fit_markdown
        )

    except Exception as ex:
        raise HTTPException(
            status_code=500,
            detail=str(ex),
        )
        
        
