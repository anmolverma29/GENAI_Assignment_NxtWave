"""Unified LLM Provider supporting Gemini, OpenAI, and Topic-Aware Mock simulation."""

import re
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Type, Optional
from pydantic import BaseModel
from src.config import Config

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate_text(self, prompt: str, system_prompt: str = "", temperature: float = 0.7) -> str:
        """Generate unstructured text from a prompt."""
        pass

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        system_prompt: str,
        response_model: Type[BaseModel],
        temperature: float = 0.1
    ) -> BaseModel:
        """Generate structured data adhering to a Pydantic schema."""
        pass


class GeminiProvider(BaseLLMProvider):
    """Google Gemini API Provider using google-genai SDK."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or Config.GEMINI_API_KEY
        self.model = model or Config.GEMINI_MODEL
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set.")
        from google import genai
        self.client = genai.Client(api_key=self.api_key)

    def generate_text(self, prompt: str, system_prompt: str = "", temperature: float = 0.7) -> str:
        full_contents = []
        if system_prompt:
            full_contents.append(f"System Instructions:\n{system_prompt}\n\n")
        full_contents.append(prompt)
        
        response = self.client.models.generate_content(
            model=self.model,
            contents="".join(full_contents),
            config={"temperature": temperature}
        )
        return response.text or ""

    def generate_structured(
        self,
        prompt: str,
        system_prompt: str,
        response_model: Type[BaseModel],
        temperature: float = 0.1
    ) -> BaseModel:
        full_prompt = (
            f"{system_prompt}\n\n"
            f"Respond ONLY with a valid JSON object matching the JSON schema below.\n"
            f"Schema:\n{json.dumps(response_model.model_json_schema(), indent=2)}\n\n"
            f"User Prompt:\n{prompt}"
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=full_prompt,
            config={
                "temperature": temperature,
                "response_mime_type": "application/json",
            }
        )
        raw_text = response.text or "{}"
        return response_model.model_validate_json(raw_text)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API Provider."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or Config.OPENAI_API_KEY
        self.model = model or Config.OPENAI_MODEL
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)

    def generate_text(self, prompt: str, system_prompt: str = "", temperature: float = 0.7) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content or ""

    def generate_structured(
        self,
        prompt: str,
        system_prompt: str,
        response_model: Type[BaseModel],
        temperature: float = 0.1
    ) -> BaseModel:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=response_model,
            temperature=temperature
        )
        return completion.choices[0].message.parsed


class MockProvider(BaseLLMProvider):
    """Topic-Aware Deterministic Mock Provider.
    
    Generates tailored lessons for ANY given topic:
    - Attempt 1: Produces an imperfect, technical draft lacking analogies and concrete walkthroughs (fails rubric).
    - Attempt 2 (Retry 1): Receives critique and outputs a masterclass beginner lesson tailored to that specific topic.
    """

    def __init__(self):
        self.generation_count = 0

    def _extract_topic(self, prompt: str) -> str:
        """Extracts the requested topic from the prompt."""
        m = re.search(r"topic:\s*['\"]([^'\"]+)['\"]", prompt, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        m2 = re.search(r"TOPIC:\s*([^\r\n]+)", prompt)
        if m2:
            return m2.group(1).strip()
        return "Introduction to RAG"

    def generate_text(self, prompt: str, system_prompt: str = "", temperature: float = 0.7) -> str:
        self.generation_count += 1
        topic = self._extract_topic(prompt)
        is_retry = (
            "REMEDIATION FEEDBACK FROM PREVIOUS REJECTION" in prompt or 
            "REJECTION CRITIQUE" in prompt or 
            self.generation_count > 1
        )

        topic_lower = topic.lower()

        # 1. RAG Topic
        if "rag" in topic_lower or "retrieval-augmented" in topic_lower:
            return self._generate_rag(is_retry)
        
        # 2. Vector Embeddings Topic
        elif "embedding" in topic_lower or "vector" in topic_lower:
            return self._generate_embeddings(is_retry, topic)
        
        # 3. Prompt Engineering Topic
        elif "prompt" in topic_lower:
            return self._generate_prompt_engineering(is_retry, topic)
        
        # 4. Fine-Tuning Topic
        elif "fine-tuning" in topic_lower or "fine tuning" in topic_lower:
            return self._generate_fine_tuning(is_retry, topic)

        # 5. Universal Generator for ANY OTHER custom topic
        else:
            return self._generate_generic_topic(is_retry, topic)

    def _generate_rag(self, is_retry: bool) -> str:
        if not is_retry:
            return (
                "# Introduction to RAG (Retrieval-Augmented Generation)\n\n"
                "## What is RAG?\n"
                "RAG stands for Retrieval-Augmented Generation. It is an AI architecture that enhances "
                "large language models by dynamically retrieving relevant facts from an external vector index.\n\n"
                "## Why It Matters\n"
                "Standard LLMs often hallucinate or lack access to up-to-date data. With RAG, the model can query "
                "private enterprise stores rather than relying solely on frozen parametric weights.\n\n"
                "## How It Works\n"
                "The process works by converting documents into high-dimensional dense vector embeddings. "
                "These embeddings are indexed inside an approximate nearest neighbor HNSW vector database. "
                "At query time, the system performs a cosine similarity search across the vector space to retrieve "
                "top-k chunks, concatenates them into the LLM context window, and executes text completion.\n\n"
                "## Summary\n"
                "In conclusion, RAG combines retrieval algorithms with generative models to reduce hallucinations."
            )
        else:
            return (
                "# 🧠 Introduction to RAG: Giving AI an Open-Book Exam\n\n"
                "Imagine you are taking a tough history exam. In a **closed-book exam**, you must rely entirely "
                "on what you memorized months ago. If you forgot a date or facts changed, you might guess or get it wrong.\n\n"
                "Now imagine an **open-book exam**. You have your textbook and notes right beside you. When a question is asked, "
                "you flip to the exact chapter, read the verified fact, and write a clear, accurate answer.\n\n"
                "That is the core idea behind **RAG (Retrieval-Augmented Generation)**. It transforms AI from a student taking a closed-book exam "
                "into a researcher taking an open-book exam.\n\n"
                "---\n\n"
                "## 1. What is RAG?\n\n"
                "**RAG** stands for **Retrieval-Augmented Generation**:\n"
                "- **Retrieval**: Finding the right reference information from an external source (like your company's documents, manuals, or articles).\n"
                "- **Augmented**: Adding those retrieved facts into the AI's question prompt as reference context.\n"
                "- **Generation**: Having the AI read that context and generate a natural, helpful, and accurate response.\n\n"
                "In plain words: **RAG is an AI technique that looks up verified facts in a library before answering your question.**\n\n"
                "---\n\n"
                "## 2. Why Does RAG Matter? (The Problem It Solves)\n\n"
                "Standard AI language models (like ChatGPT or Claude) are incredibly smart, but they suffer from three major limitations:\n\n"
                "1. **Hallucinations (Confident Guesswork)**: When an AI doesn't know something, it can invent plausible-sounding but false answers.\n"
                "2. **Outdated Knowledge (Knowledge Cutoff)**: Models only know what was in their training data. They don't know what happened yesterday or this morning.\n"
                "3. **No Private Company Knowledge**: A generic model knows about world history, but it has never seen your company's internal HR policies, customer support tickets, or confidential product roadmaps.\n\n"
                "**Why not just retrain the AI?**\n"
                "Retraining (or fine-tuning) a large model takes days, costs thousands of dollars, and must be repeated every time a document changes. "
                "**RAG solves this elegantly**: instead of changing the AI's brain, you simply hand it the right page to read on the fly.\n\n"
                "---\n\n"
                "## 3. How RAG Works (The 3-Step Journey)\n\n"
                "Think of RAG as a 3-step assembly line:\n\n"
                "```\n"
                "[Your Documents] ──► 1. Chunk & Index ──► 2. Retrieve Relevant Page ──► 3. Generate Grounded Answer\n"
                "```\n\n"
                "### Step 1: Ingestion & Indexing (Organizing the Library)\n"
                "- **Chunking**: Large documents (like a 200-page PDF) are cut into small, bite-sized index cards (paragraphs of 200–500 words).\n"
                "- **Embeddings (Numerical Meaning Fingerprints)**: Just like humans use keywords, computers convert text into a list of numbers called an *embedding*. Sentences with similar meanings get similar number patterns.\n"
                "- **Vector Database (The Smart Filing Cabinet)**: A specialized database that stores these chunks and their numerical fingerprints so we can find matching ideas in milliseconds.\n\n"
                "### Step 2: Retrieval (Finding the Relevant Index Cards)\n"
                "- When a user asks a question, the system turns the question into a numerical fingerprint.\n"
                "- It scans the vector database and retrieves the top 2 or 3 chunks that best match the question's meaning.\n\n"
                "### Step 3: Generation (Writing the Answer)\n"
                "- The system builds a new prompt for the AI:\n"
                "  > *\"You are a helpful assistant. Use ONLY the following reference notes to answer the user's question:\n"
                "  > [Retrieved Reference Chunks]\n"
                "  > User Question: ...\"*\n"
                "- The AI reads the provided notes and crafts an accurate, grounded reply.\n\n"
                "---\n\n"
                "## 4. Concrete Real-World Walkthrough\n\n"
                "Let's see RAG in action for **NovaTech Smartwatch Customer Support**:\n\n"
                "- **User Question**: *\"Can I wear my NovaWatch X in a heated sauna?\"*\n"
                "- **Without RAG**: The generic model might say: *\"Most smartwatches are water-resistant up to 50 meters, so you should be fine!\"* (Result: The user damages their watch because high heat ruins battery seals!).\n"
                "- **With RAG**:\n"
                "  1. **Retrieve**: The system searches NovaTech's 80-page user manual and retrieves Section 4.2:\n"
                "     > *\"Warning: The NovaWatch X is water-resistant up to 5 ATM, but MUST NOT be worn in steam rooms, saunas, or hot tubs exceeding 45°C as steam damages the protective acoustic membranes.\"*\n"
                "  2. **Augment**: The system constructs the prompt:\n"
                "     > *\"Reference Manual: [Section 4.2 Warning]\n"
                "     > Question: Can I wear my NovaWatch X in a heated sauna?\"*\n"
                "  3. **Generate**: The AI responds:\n"
                "     > *\"No, you should not wear your NovaWatch X in a sauna. According to Section 4.2 of the manual, steam and temperatures exceeding 45°C can damage the watch's protective seals.\"*\n\n"
                "The answer is accurate, safe, and directly cites company policy!\n\n"
                "---\n\n"
                "## 5. Key Takeaways & Summary\n\n"
                "- 📚 **RAG is an open-book exam for AI**: It retrieves real facts before generating an answer.\n"
                "- 🛡️ **Prevents hallucinations**: The AI answers based on verified documents, not guesswork.\n"
                "- ⚡ **Fresh & private data**: You can update your knowledge base instantly without expensive model retraining.\n"
                "- 🔄 **The 3 steps**: Chunk & Index (library) ➔ Retrieve (search) ➔ Generate (answer)."
            )

    def _generate_embeddings(self, is_retry: bool, topic: str) -> str:
        if not is_retry:
            return (
                f"# {topic}\n\n"
                "## What Are Vector Embeddings?\n"
                "Vector embeddings are dense mathematical representations projected into a high-dimensional latent space.\n\n"
                "## Why It Matters\n"
                "They allow mathematical operations over unstructured textual corpora using cosine similarity matrices.\n\n"
                "## How It Works\n"
                "A transformer encoder processes token sequences through multi-head self-attention layers to compute vector coordinates.\n\n"
                "## Summary\n"
                "Vectors represent data in metric space."
            )
        else:
            return (
                f"# 🧭 {topic}: Giving Words a Digital GPS Location\n\n"
                "Imagine opening a giant map of the world. On this map, cities that are close to each other geographically "
                "(like Paris and Brussels) are located just a few inches apart, while Tokyo is much farther away.\n\n"
                "Now, imagine a map where instead of cities, we map **ideas and words**. On this map, words with similar meanings "
                "(like *dog*, *puppy*, and *canine*) sit right next to each other, while an unrelated word like *refrigerator* sits far across the room.\n\n"
                "That is the intuitive secret behind **Vector Embeddings**: they give every word, sentence, or document a digital GPS coordinate so computers can measure how related they are.\n\n"
                "---\n\n"
                "## 1. What Are Vector Embeddings?\n\n"
                "Computers cannot understand the feeling or meaning behind words—they only understand numbers. "
                "An **embedding** is simply a translation: it turns human text into a list of numbers (a numerical fingerprint).\n\n"
                "- Example: The word *\"king\"* might become `[0.82, -0.14, 0.45, ...]`.\n"
                "- The word *\"queen\"* will have a nearly identical list of numbers because their meanings are so closely connected.\n\n"
                "---\n\n"
                "## 2. Why Do Embeddings Matter? (The Problem It Solves)\n\n"
                "Before embeddings, computer search was dumb. If you searched an online store for *\"comfy footwear for cold rain\"*, "
                "an old search engine would look for those exact keywords. If a product was titled *\"Waterproof Winter Boots\"*, it would fail to find it!\n\n"
                "**Embeddings solve this completely**: because *\"footwear\"* and *\"boots\"* have similar GPS coordinates, the computer knows they mean the exact same concept, even with zero matching words.\n\n"
                "---\n\n"
                "## 3. How Embeddings Work (The 3-Step Process)\n\n"
                "1. **Input Text**: You feed a sentence like *\"I love warm tomato soup\"* into an embedding model.\n"
                "2. **Calculate Meaning Fingerprint**: The model analyzes context and calculates a coordinates vector (like latitude and longitude for meaning).\n"
                "3. **Store in a Smart Filing Cabinet (Vector Database)**: These coordinates are saved so the computer can instantly find neighboring concepts in milliseconds.\n\n"
                "---\n\n"
                "## 4. Concrete Real-World Walkthrough\n\n"
                "Let's see embeddings in action in an **E-Commerce Search Engine**:\n\n"
                "- **User Query**: *\"warm woolen pullover\"*\n"
                "- **Without Embeddings**: The store searches exact keywords and returns 0 results because the product catalog uses different words.\n"
                "- **With Embeddings**:\n"
                "  1. The system turns *\"warm woolen pullover\"* into coordinate `[0.72, 0.44]`.\n"
                "  2. It scans the catalog coordinates and finds Product #302: *\"Cozy Knitted Winter Sweater\"* located at `[0.71, 0.46]`.\n"
                "  3. Result: The user gets the exact sweater they wanted, even though the seller never used the word 'pullover'!\n\n"
                "---\n\n"
                "## 5. Key Takeaways & Summary\n\n"
                "- 📍 **Embeddings are GPS coordinates for meaning**: Concepts with similar ideas live close together.\n"
                "- 🔢 **Bridges words and numbers**: Allows computers to do math on human thoughts and language.\n"
                "- 🔍 **Powers modern AI**: Semantic search, recommendation feeds, and RAG systems all rely on embeddings."
            )

    def _generate_prompt_engineering(self, is_retry: bool, topic: str) -> str:
        if not is_retry:
            return (
                f"# {topic}\n\n"
                "## What is Prompt Engineering?\n"
                "Prompt engineering is the optimization of input context strings to maximize inference probability distributions.\n\n"
                "## Why It Matters\n"
                "It adjusts parametric generation behavior without backpropagation gradient updates.\n\n"
                "## How It Works\n"
                "Developers use few-shot in-context exemplars, system metaprompts, and chain-of-thought token sequences.\n\n"
                "## Summary\n"
                "Prompts guide transformer sampling."
            )
        else:
            return (
                f"# ✍️ {topic}: Giving Clear Recipes to an AI Chef\n\n"
                "Imagine you hired a world-class chef to cook dinner for your guests. If you walk into the kitchen and vaguely say, "
                "*\"Make something delicious!\"*, the chef might prepare raw sushi, while your guests were hoping for warm pasta.\n\n"
                "The chef didn't fail because they lacked skill—they failed because your instructions were vague.\n\n"
                "Now imagine handing the chef a clear request: *\"Please prepare a comforting vegetarian Italian pasta dish for 4 people, mild spice, ready in 30 minutes.\"* "
                "Now the chef delivers a masterpiece.\n\n"
                "That is **Prompt Engineering**: the craft of communicating with AI models clearly, giving them context, constraints, and examples so they produce exactly what you need on the first try.\n\n"
                "---\n\n"
                "## 1. What is Prompt Engineering?\n\n"
                "A **prompt** is simply the text or question you type into an AI (like ChatGPT or Claude). "
                "**Prompt Engineering** is the art and science of structuring that prompt with clarity, role definitions, and rules so the AI gives accurate, high-quality answers.\n\n"
                "---\n\n"
                "## 2. Why Does Prompt Engineering Matter?\n\n"
                "Modern AI models have read almost the entire internet, but they cannot read your mind. Without good prompts, you get:\n"
                "- Generic, bland answers.\n"
                "- AI guessing what tone or format you wanted.\n"
                "- Confident mistakes or hallucinations.\n\n"
                "Mastering prompt engineering saves hours of editing and unlocks the true power of AI tools.\n\n"
                "---\n\n"
                "## 3. How Great Prompts Work (The 4-Part Formula)\n\n"
                "Every high-performing prompt includes 4 building blocks:\n"
                "1. **Role**: Who should the AI pretend to be? (*\"Act as a patient senior software engineer...\"*)\n"
                "2. **Task**: What specific action should it perform? (*\"Review this code for security vulnerabilities...\"*)\n"
                "3. **Context**: What background information does it need? (*\"This is an internal web tool used by employees...\"*)\n"
                "4. **Constraints**: What are the strict rules? (*\"Keep it under 3 bullet points, use plain English, no jargon.\"*)\n\n"
                "---\n\n"
                "## 4. Concrete Real-World Walkthrough\n\n"
                "Let's see prompt engineering in action for **Customer Support Email Writing**:\n\n"
                "- **Vague Prompt (Bad)**: *\"Write an email to a customer whose order was delayed.\"*\n"
                "- **Result**: A robotic, formal 4-paragraph corporate email that sounds cold and unhelpful.\n\n"
                "- **Engineered Prompt (Good)**:\n"
                "  > *\"Role: Empathetic customer support lead at an artisan coffee brand.\n"
                "  > Task: Write an apology email to customer Sarah whose coffee grinder order #489 is delayed by 3 days.\n"
                "  > Constraints: Friendly tone, apologize sincerely, offer a free $10 gift card, keep it under 100 words.\"*\n\n"
                "- **Grounded Output**:\n"
                "  > *\"Hi Sarah, We are so sorry! Your coffee grinder (Order #489) is running about 3 days behind schedule due to unexpected courier delays. We know you're eager for fresh morning brew, so we've added a $10 gift card to your account as our apology. Tracking updates will hit your inbox tomorrow morning! Warmly, The Coffee Team\"*\n\n"
                "---\n\n"
                "## 5. Key Takeaways & Summary\n\n"
                "- 🍳 **Clear recipe, great meal**: Specific prompts yield exceptional results.\n"
                "- 🧱 **Use the 4 blocks**: Role, Task, Context, and Constraints.\n"
                "- 💡 **Show, don't just tell**: Giving 1 or 2 examples inside your prompt dramatically improves accuracy."
            )

    def _generate_fine_tuning(self, is_retry: bool, topic: str) -> str:
        if not is_retry:
            return (
                f"# {topic}\n\n"
                "## What is Fine-Tuning?\n"
                "Fine-tuning adapts model parameters via supervised gradient descent on domain-specific datasets.\n\n"
                "## Why It Matters\n"
                "It modifies internal attention weights rather than appending in-context token prompts.\n\n"
                "## How It Works\n"
                "Model weights are updated using Low-Rank Adaptation (LoRA) matrices over target task loss functions.\n\n"
                "## Summary\n"
                "Fine-tuning modifies weights."
            )
        else:
            return (
                f"# 🎓 {topic}: Medical School vs. Open-Book Reference\n\n"
                "Imagine you want an assistant to help review hospital patient records. You have two options:\n\n"
                "- **Option A (Fine-Tuning)**: Send your assistant through intensive medical residency training for 6 months. They internalize clinical instincts, medical vocabulary, and bedside manner directly into their brain.\n"
                "- **Option B (RAG - Retrieval)**: Hand a smart general assistant a medical reference textbook to look up facts during an open-book exam.\n\n"
                "This comparison explains the essential choice in modern AI: **Fine-Tuning changes how the AI behaves and thinks internally**, while RAG gives the AI external reference notes to read.\n\n"
                "---\n\n"
                "## 1. What is Fine-Tuning?\n\n"
                "**Fine-Tuning** is the process of taking an already trained, smart AI model and training it further on a specialized set of examples so it learns a specific style, tone, or skill.\n\n"
                "---\n\n"
                "## 2. Why Does Fine-Tuning Matter? (When to Use It)\n\n"
                "Fine-tuning is ideal when you need to teach the AI:\n"
                "- **A specialized style or voice** (e.g., writing like a patent attorney or a Shakespearean poet).\n"
                "- **Strict output structures** (e.g., always returning precise JSON with zero conversational filler).\n"
                "- **Complex domain reasoning** (e.g., medical triage logic or financial auditing).\n\n"
                "---\n\n"
                "## 3. How Fine-Tuning Works (The 3 Steps)\n\n"
                "1. **Curate High-Quality Examples**: Collect 500–2,000 pairs of ideal inputs and outputs.\n"
                "2. **Train the Weights**: Run training algorithms that gently update the AI's internal mathematical connections.\n"
                "3. **Deploy the Custom Specialist**: The model now responds in the desired format automatically without needing lengthy instructions in every prompt.\n\n"
                "---\n\n"
                "## 4. Concrete Real-World Walkthrough\n\n"
                "Let's see Fine-Tuning in action for a **Legal Contract Analyzer**:\n\n"
                "- **User Scenario**: A law firm needs an AI that extracts indemnification clauses into strict tabular format.\n"
                "- **Standard AI**: Writes polite small talk: *\"Certainly! Here is an analysis of your contract clauses...\"* (Breaks automated software pipelines!).\n"
                "- **Fine-Tuned AI**:\n"
                "  1. Trained on 1,000 legal contracts with zero conversational fluff.\n"
                "  2. Input: Raw legal paragraph.\n"
                "  3. Output: Immediately outputs the exact clause classification code without small talk.\n\n"
                "---\n\n"
                "## 5. Key Takeaways & Summary\n\n"
                "- 🧠 **Fine-tuning alters the AI's internal instincts**: It teaches style, tone, and formatting.\n"
                "- 📚 **RAG teaches knowledge; Fine-tuning teaches behavior**.\n"
                "- ⚖️ **Combine both for best results**: Fine-tune for form and tone; use RAG for fresh, verified facts."
            )

    def _generate_generic_topic(self, is_retry: bool, topic: str) -> str:
        clean_name = topic.replace("Introduction to", "").replace("Introduction", "").strip() or topic
        if not is_retry:
            return (
                f"# Introduction to {clean_name}\n\n"
                f"## What is {clean_name}?\n"
                f"{clean_name} is an advanced technological architecture designed to optimize computational state spaces.\n\n"
                f"## Why It Matters\n"
                f"Standard systems suffer from high latency and non-deterministic error propagation across distributed nodes.\n\n"
                f"## How It Works\n"
                f"The mechanism executes through multi-layer abstraction pipelines using specialized algorithmic protocols and parameter matrices.\n\n"
                f"## Summary\n"
                f"In conclusion, {clean_name} provides robust systemic optimization."
            )
        else:
            return (
                f"# 💡 Introduction to {clean_name}: A Beginner's Guide\n\n"
                f"Imagine you are trying to assemble a complex piece of furniture with 200 parts. If the instructions are written "
                f"in dense engineering blueprints, you will get frustrated and make mistakes.\n\n"
                f"Now imagine having a friendly expert standing beside you, handing you the exact bolt you need and explaining "
                f"step-by-step why each piece matters in simple, everyday language.\n\n"
                f"That is the goal of this guide: to demystify **{clean_name}** and show you what it is, why it matters, and how it works—even if you are starting from zero.\n\n"
                f"---\n\n"
                f"## 1. What is {clean_name}?\n\n"
                f"At its core, **{clean_name}** is a solution to a fundamental problem. In plain English:\n"
                f"It is a modern technique that simplifies complex tasks by organizing information, automating tedious steps, and ensuring high accuracy.\n\n"
                f"---\n\n"
                f"## 2. Why Does {clean_name} Matter? (The Problems It Solves)\n\n"
                f"Before {clean_name}, people and systems faced three major roadblocks:\n\n"
                f"1. **High Inefficiency**: Tasks that should take seconds took hours of manual guesswork.\n"
                f"2. **Inconsistent Quality**: Without a clear system, errors and inconsistencies slipped through unnoticed.\n"
                f"3. **Scalability Bottlenecks**: As demand grew, traditional methods broke down under the workload.\n\n"
                f"**{clean_name} solves these roadblocks** by introducing standardized, repeatable workflows that anyone can rely on.\n\n"
                f"---\n\n"
                f"## 3. How {clean_name} Works (The 3-Step Journey)\n\n"
                f"Think of {clean_name} as a simple 3-stage assembly line:\n\n"
                f"1. **Input & Preparation**: The system receives the initial request or raw information and breaks it down into clear, structured components.\n"
                f"2. **Processing & Alignment**: It applies core rules and intelligence to match the input with the best possible approach.\n"
                f"3. **Output & Delivery**: It generates a polished, verified result ready for immediate practical use.\n\n"
                f"---\n\n"
                f"## 4. Concrete Real-World Walkthrough\n\n"
                f"Let's look at a concrete real-world scenario showing {clean_name} in action:\n\n"
                f"- **The Scenario**: A growing team needs to handle hundreds of incoming client requests every day.\n"
                f"- **Without {clean_name}**: Team members manually read every ticket, guess the best response, and often give conflicting answers.\n"
                f"- **With {clean_name}**:\n"
                f"  1. **User Request**: A client submits an urgent support inquiry.\n"
                f"  2. **System Processing**: {clean_name} automatically identifies the request intent, consults verified documentation, and crafts a grounded draft.\n"
                f"  3. **Verified Result**: The client receives a fast, consistent, and accurate resolution within minutes.\n\n"
                f"---\n\n"
                f"## 5. Key Takeaways & Summary\n\n"
                f"- 🎯 **{clean_name} solves real problems**: It replaces manual confusion with structured simplicity.\n"
                f"- 🪜 **Follows 3 simple steps**: Input Preparation ➔ Intelligent Processing ➔ Verified Output.\n"
                f"- 🚀 **Accessible to anyone**: You don't need a computer science degree to understand and benefit from it."
            )

    def generate_structured(
        self,
        prompt: str,
        system_prompt: str,
        response_model: Type[BaseModel],
        temperature: float = 0.1
    ) -> BaseModel:
        from src.models import RubricEvaluation, CheckpointResult, CheckpointID

        lesson_text = prompt
        text_lower = lesson_text.lower()
        
        # 1. Check presence of jargon without explanation
        dense_jargon_found = any(j in text_lower for j in [
            "high-dimensional dense", "hnsw", "cosine similarity matrix", "latent space", 
            "token distribution", "few-shot in-context exemplars", "gradient descent", 
            "parametric generation behavior", "computational state spaces"
        ])
        has_analogy_support = any(a in text_lower for a in [
            "fingerprint", "cabinet", "recipe", "exam", "chef", "gps", "library", "intern", "blueprint", "map"
        ])
        jargon_failed = dense_jargon_found and not has_analogy_support

        # 2. Check concrete example
        has_concrete_example = any(ex in text_lower for ex in [
            "novawatch", "walkthrough", "scenario", "user query", "user question", "sarah", "product #302", "case study"
        ]) and any(res in text_lower for res in ["result", "grounded", "answer", "output", "resolution"])

        # 3. Check beginner friendly
        beginner_passed = any(b in text_lower for b in [
            "imagine", "open-book", "recipe", "gps", "chef", "library", "exam", "think of", "friendly", "furniture"
        ])

        # 4. Check key points (What, Why, How)
        covers_what = any(w in text_lower for w in ["what is", "what are", "## 1."])
        covers_why = any(w in text_lower for w in ["why", "problem", "matters", "## 2."])
        covers_how = any(h in text_lower for h in ["how", "step", "journey", "process", "## 3."])
        key_points_passed = covers_what and covers_why and covers_how

        # 5. Check accuracy & grounding
        accurate_passed = len(lesson_text.strip()) > 150 and "#" in lesson_text

        # 6. Check coherent flow
        coherent_passed = ("#" in lesson_text) and any(s in text_lower for s in ["summary", "takeaway", "key takeaways", "conclusion"])

        checkpoints = [
            CheckpointResult(
                checkpoint_id=CheckpointID.ACCURACY,
                name="Accurate & Grounded",
                passed=accurate_passed,
                reason="Concepts are accurately grounded and technically sound." if accurate_passed else "Contains inaccurate statements or lacks substance.",
                evidence="Accurate technical structure." if accurate_passed else "Insufficient factual grounding.",
                remediation="" if accurate_passed else "Ensure core principles are accurately represented."
            ),
            CheckpointResult(
                checkpoint_id=CheckpointID.BEGINNER,
                name="Beginner-Friendly Language",
                passed=beginner_passed,
                reason="Uses intuitive real-world analogies accessible to complete novices." if beginner_passed else "Too academic and dry; assumes prior familiarity with technical terminology.",
                evidence="Analogy grounds reader intuition." if beginner_passed else "Relies on technical terminology without beginner framing.",
                remediation="" if beginner_passed else "Introduce an everyday metaphor early to ground intuition."
            ),
            CheckpointResult(
                checkpoint_id=CheckpointID.EXAMPLE,
                name="Teaches by Concrete Example",
                passed=has_concrete_example,
                reason="Includes a full end-to-end real-world walkthrough showing context, action, and verified outcome." if has_concrete_example else "Lacks an end-to-end real-world walkthrough demonstrating practical application.",
                evidence="Concrete walkthrough scenario included." if has_concrete_example else "No concrete user inquiry with specific scenario provided.",
                remediation="" if has_concrete_example else "Add a realistic scenario showing: (1) Real problem/question, (2) Step-by-step resolution, (3) Practical outcome."
            ),
            CheckpointResult(
                checkpoint_id=CheckpointID.NO_JARGON,
                name="Clear, No Unexplained Jargon",
                passed=not jargon_failed,
                reason="All technical terms are introduced with everyday plain-English analogies." if not jargon_failed else "Introduced technical jargon without defining them for a beginner.",
                evidence="All terms defined simply." if not jargon_failed else "Found technical domain jargon without layman explanation.",
                remediation="" if not jargon_failed else "Define all technical concepts using intuitive everyday analogies before or upon use."
            ),
            CheckpointResult(
                checkpoint_id=CheckpointID.KEY_POINTS,
                name="Covers All Key Pillars",
                passed=key_points_passed,
                reason="Thoroughly covers What it is, Why it matters (problems it solves), and How it works step-by-step." if key_points_passed else "Fails to cover all three foundational pillars adequately.",
                evidence="Distinct coverage of What, Why, and How." if key_points_passed else "Missing clear coverage of all three pillars.",
                remediation="" if key_points_passed else "Ensure distinct sections for What it is, Why it matters, and How it works."
            ),
            CheckpointResult(
                checkpoint_id=CheckpointID.COHERENT_FLOW,
                name="Coherent Pedagogical Flow",
                passed=coherent_passed,
                reason="Clear hierarchical progression: Hook -> Definition -> Problem -> Steps -> Example -> Key Takeaways." if coherent_passed else "Teaching flow is disjointed or lacks structure.",
                evidence="Markdown headings structure the learning path from simple to practical." if coherent_passed else "Needs better structural progression.",
                remediation="" if coherent_passed else "Organize with clear numbered section headers and a concluding summary."
            )
        ]

        failed = [c.checkpoint_id for c in checkpoints if not c.passed]
        all_passed = len(failed) == 0
        pass_rate = (len(checkpoints) - len(failed)) / len(checkpoints)

        overall_feedback = (
            "Congratulations! The lesson clears all 6 hard checkpoints and is ready to ship."
            if all_passed else
            f"The lesson failed {len(failed)} critical checkpoints: {', '.join(failed)}. Must address specific remediation feedback."
        )

        return RubricEvaluation(
            checkpoints=checkpoints,
            all_passed=all_passed,
            failed_checkpoints=failed,
            overall_feedback=overall_feedback,
            pass_rate=pass_rate
        )


def get_llm_provider(provider_type: Optional[str] = None) -> BaseLLMProvider:
    """Factory to get the appropriate LLM provider."""
    provider_name = (provider_type or Config.get_effective_provider()).lower()
    
    if provider_name == "gemini":
        try:
            return GeminiProvider()
        except Exception as e:
            logger.warning(f"Could not initialize GeminiProvider ({e}); falling back to MockProvider.")
            return MockProvider()
    elif provider_name == "openai":
        try:
            return OpenAIProvider()
        except Exception as e:
            logger.warning(f"Could not initialize OpenAIProvider ({e}); falling back to MockProvider.")
            return MockProvider()
    else:
        return MockProvider()
