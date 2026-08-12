import uvicorn
import os

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, LLMConfig
from crawl4ai.content_filter_strategy import BM25ContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.extraction_strategy import LLMExtractionStrategy
import crawl4ai.extraction_strategy as crawl4ai_extraction_strategy


load_dotenv()

# -----------------------------
# Knowledge Extraction Schema
# -----------------------------
# Fixed, explicit schema. Never left None so crawl4ai never falls into the
# "model infers the schema" prompt (PROMPT_EXTRACT_INFERRED_SCHEMA), which
# qwen3:4b cannot follow and echoes back as url_content/user_request keys.
#
# For LLMExtractionStrategy the schema is embedded into the prompt as plain
# text (json.dumps). A full JSON-Schema spec (type/properties/items) is too
# verbose for a 4B model and it copies it verbatim instead of filling it with
# data. A compact shape-example tells the model the exact structure to return.

KNOWLEDGE_SCHEMA = {
    "knowledge": [
        {
            "topic": "string",
            "content": "string"
        }
    ]
}


def build_extraction_instruction(query: str) -> str:
    return (
        f"""
        Extract factual educational information relevant to "{query}" from the provided content.

        Rules:
        - Use only information explicitly supported by the provided content.
        - Do not invent, infer, correct, or supplement information using outside knowledge.
        - Do not create knowledge items merely describing the presence of code.
        - Do not generate absent-information items
        - Ignore source code unless the requested topic specifically requires programming implementation details.
        - Preserve relevant concepts, formulas, complexity analysis, algorithms or which are useful to generate questions.
        - Do not include questions, quiz items, or question sentences in knowledge.
        - Only include examples that contain actual educational information useful for understanding or assessing the requested topic. 
        - Do not describe the existence of code examples.
        
        - Ignore navigation, advertisements, authentication messages, social links,
        copyright/footer content, and unrelated website content.
        """
        # "Extract factual educational information from the provided content that is "
        # "relevant to the requested topic.\n\n"
        # "Requested topic:\n"
        # f"{query}\n\n"
        # "Extract definitions, concepts, explanations, formulas, complexity analysis, "
        # "algorithms/procedures, examples, relevant code examples, and topic-related "
        # "questions.\n\n"
        # "Return only information supported by the provided source content.\n"
        # "Do not invent information.\n"
        # "Do not use outside knowledge.\n"
        # "Do not correct source information.\n"
        # "Do not answer existing questions.\n"
        # "Do not generate new questions.\n"
        # "Preserve existing topic-related questions as source material, not as noise.\n\n"
        # "Ignore navigation, advertisements, social links, website metadata, "
        # "copyright/footer information, author/company information, and unrelated content."
        # "For questions:"
        # "Treat each question as a separate item."
        # "Preserve the exact question boundary from the source."
        # "Never concatenate two adjacent questions."
        # "Never complete an incomplete question using your own knowledge."
        # "If a question is incomplete in the source, preserve it as incomplete."
        # "Do not merge a question with surrounding text."
        
        
    )


# crawl4ai 0.9.2 builds the LLM prompt from module-level constants read at call time
# (extraction_strategy.py:664-682). Its default prompts are long and XML-wrapped
# (<blocks>, <url_content>, <score>), which a 4B model fails to follow and instead
# echoes the scaffolding as JSON keys. Replace them with a short, schema-first prompt
# that uses the exact placeholders crawl4ai substitutes: {URL}, {HTML}, {REQUEST}, {SCHEMA}.

KNOWLEDGE_EXTRACTION_PROMPT = """
Extract source-grounded educational knowledge from the provided content.

SOURCE URL:
{URL}

SOURCE CONTENT:
{HTML}

USER REQUEST:
{REQUEST}

RESULT STRUCTURE:
{SCHEMA}

RULES:

1. Extract ONLY information explicitly present in SOURCE CONTENT.

2. Every factual statement in the output must be directly supported by
   the SOURCE CONTENT.

3. DO NOT use your own knowledge to complete, correct, infer, calculate,
   interpret, or supplement information.

4. If the source says "Best Case" and "Worst Case" but does not mention
   "Average Case", DO NOT add an Average Case.

5. Do not derive new facts from formulas, examples, or statements.
   Preserve the information as stated by the source.

6. Do not correct factual mistakes in the source.

7. Do not generate new examples, explanations, formulas, questions,
   algorithms, or conclusions.

8. You may combine closely related statements from the source into one
   knowledge item, but every part of the resulting statement must be
   supported by the source.

9. If information is incomplete in the source, keep it incomplete.
   Do not fill the missing information.

10. Preserve important source terminology, formulas, examples, rules,
    procedures, and topic-specific questions.

11. Ignore navigation, advertisements, authentication messages,
    social links, copyright/footer content, author/company information,
    and unrelated website content.

12. Existing questions in the source must be preserved as questions.
    Do not answer them.

13. Do not generate new questions.

14. Return ONLY the JSON object.
    Do not return markdown, XML, explanations, comments, or code fences.

{SCHEMA}
"""

# KNOWLEDGE_EXTRACTION_PROMPT = """Extract factual educational knowledge from the source content.

# SOURCE URL:
# {URL}

# SOURCE CONTENT:
# <content>
# {HTML}
# </content>

# USER REQUEST:
# {REQUEST}

# RESULT STRUCTURE - return ONE JSON object with the SAME keys shown below. Fill each array with actual data extracted from the source. Use empty array [] for keys with no data. Do NOT return this template itself:
# {SCHEMA}

# RULES:
# - Extract only information present in the source content.
# - Do not answer questions and do not generate new questions.
# - Ignore navigation, advertisements, social links, website metadata, copyrights, author/company information, and unrelated content.

# Return ONLY the JSON object. No comments, markdown fences, XML tags, or explanations."""

crawl4ai_extraction_strategy.PROMPT_EXTRACT_SCHEMA_WITH_INSTRUCTION = KNOWLEDGE_EXTRACTION_PROMPT
crawl4ai_extraction_strategy.PROMPT_EXTRACT_INFERRED_SCHEMA = KNOWLEDGE_EXTRACTION_PROMPT

# -----------------------------
# Request / Response Schemas
# -----------------------------

class CrawlRequest(BaseModel):
    url: HttpUrl
    query: str
    provider: str | None = None
    api_token: str | None = None
    base_url: str | None = None
    extraction_schema: dict | None = None


class CrawlResponse(BaseModel):
    success: bool
    url: str
    title: str | None
    # markdown: str
    filteredMarkdown: str
    extractedContent: str | None = None
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
        provider = request.provider or os.getenv("LLM_PROVIDER", "ollama_chat/qwen3:4b")
        api_token = request.api_token or os.getenv("LLM_API_TOKEN")
        base_url = request.base_url or os.getenv("LLM_BASE_URL", "https://ask-ai.mssplonline.in")

        # Ollama-family providers need no real API key; a dummy keeps LLMConfig
        # from overwriting the provider string (it only knows "ollama", not "ollama_chat").
        if provider.startswith("ollama") and not api_token:
            api_token = "ollama"

        llm_config = LLMConfig(
            provider=provider,
            api_token=api_token,
            base_url=base_url,
        )

        if not llm_config.api_token:
            raise HTTPException(
                status_code=500,
                detail=(
                    "No LLM API token resolved. Set LLM_API_TOKEN / "
                    "OPENAI_API_KEY (or your provider's env var) or pass "
                    "api_token in the request body."
                ),
            )

        config = CrawlerRunConfig(
            markdown_generator=DefaultMarkdownGenerator(
                content_filter=BM25ContentFilter(
                    user_query=request.query,
                    bm25_threshold=1.0,
                ),
            ),
            extraction_strategy=LLMExtractionStrategy(
                llm_config=llm_config,
                instruction=build_extraction_instruction(request.query),
                schema=request.extraction_schema or KNOWLEDGE_SCHEMA,
                extraction_type="schema",
                input_format="fit_markdown",
                force_json_response=True,
                chunk_token_threshold=4000,
                overlap_rate=0.1,
                apply_chunking=True,
                extra_args={"think": False, "max_tokens": 2000},  # qwen3: disable thinking, cap runaway output
                verbose=False,
            ),
        )

        result = await crawler.arun(
            url=str(request.url),
            config=config
        )

        fit_markdown = result.markdown.fit_markdown or result.markdown.raw_markdown

        return CrawlResponse(
            success=result.success,
            url=result.url,
            title=result.metadata.get("title") if result.metadata else None,
            filteredMarkdown=fit_markdown,
            extractedContent=result.extracted_content,
        )

    except Exception as ex:
        raise HTTPException(
            status_code=500,
            detail=str(ex),
        )
        
        
