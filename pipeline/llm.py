"""
LLM module using Groq API for response generation.
Generates educational tutor responses based on user queries,
retrieved context, and conversation history.
"""
import asyncio
from typing import List, Dict, Optional, AsyncGenerator
from groq import Groq, AsyncGroq
from config.settings import get_settings
from utils.logger import get_logger
logger = get_logger(__name__)
TUTOR_SYSTEM_PROMPT = """You are an expert AI tutor. Your role is to educate, explain, and guide the student clearly and patiently. Always base your answers on the provided context documents when relevant. If the context does not contain the answer, use your general knowledge but clearly state that you are doing so. Keep your responses concise, clear, and educational. Use examples and analogies where helpful. Never provide false information. Context from documents: {context}"""
TUTOR_SYSTEM_PROMPT_ARABIC = """تادنتسملا نم قايسلا .ةئطاخ تامولعم اًدبأ مدقت لا .ةدئافلا دنع تاهيبشتلاو ةلثملأا مدختسا .ةيميلعتو ةحضاوو ةزجوم كتاباجإ ىلع ظفاح .كلذ حضو نكلو ةماعلا كتفرعم مدختسا ،ةباجلإا ىلع قايسلا يوتحي مل اذإ .ءاضتقلاا دنع ةمدقملا قايسلا تادنتسم ىلع ءًانب ةباجلإاب اًمئاد مق .ربصو حوضوب بلاطلا هيجوتو حرشلاو ميلعتلا وه كرود .ريبخ يعانطصا ءاكذ ملعم تنأ: {context}"""
SUMMARIZATION_PROMPT = """Summarize the following conversation history into a brief, coherent summary that captures the key topics discussed and any important context needed for continuing the conversation. Keep it under 200 words.
Conversation:
{conversation}"""
class LLMService:
    """Groq LLM service for generating tutor responses."""
    def __init__(self) -> None:
        """Initialize the LLM service with Groq API credentials."""
        self.settings = get_settings()
        self.client = Groq(api_key=self.settings.groq_api_key)
        self.async_client = AsyncGroq(api_key=self.settings.groq_api_key)
        self.model = self.settings.groq_model
        self.max_history = self.settings.max_conversation_history
    async def generate_response(
        self,
        query: str,
        context: str,
        conversation_history: List[Dict[str, str]],
        language: str = "en",
    ) -> str:
        """
        Generate a tutor response using Groq LLM.
        Args:
            query: The user's current question.
            context: Retrieved RAG context from documents.
            conversation_history: List of previous conversation messages.
            language: Language code ('en' for English, 'ar' for Arabic).
        Returns:
            Generated response text from the LLM.
        """
        logger.info(f"Generating LLM response for query: '{query[:60]}...'")
        system_prompt = self._build_system_prompt(context, language)
        messages = self._build_messages(system_prompt, conversation_history, query)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1024,
                    top_p=0.9,
                )
                response_text = response.choices[0].message.content.strip()
                logger.info(f"LLM response generated ({len(response_text)} chars)")
                return response_text
            except Exception as e:
                error_str = str(e)
                if "rate_limit" in error_str.lower() or "429" in error_str:
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited. Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                    continue
                elif attempt < max_retries - 1:
                    logger.warning(f"LLM error on attempt {attempt + 1}: {e}")
                    await asyncio.sleep(1)
                    continue
                else:
                    logger.error(f"LLM generation failed after {max_retries} attempts: {e}")
                    raise RuntimeError(f"Failed to generate LLM response: {e}")
        raise RuntimeError("LLM generation failed after all retries.")
    async def generate_response_stream(
        self,
        query: str,
        context: str,
        conversation_history: List[Dict[str, str]],
        language: str = "en",
    ) -> AsyncGenerator[str, None]:
        """
        Stream a tutor response token by token using Groq LLM.
        Args:
            query: The user's current question.
            context: Retrieved RAG context from documents.
            conversation_history: List of previous conversation messages.
            language: Language code.
        Yields:
            Response text tokens as they are generated.
        """
        logger.info(f"Streaming LLM response for query: '{query[:60]}...'")
        system_prompt = self._build_system_prompt(context, language)
        messages = self._build_messages(system_prompt, conversation_history, query)
        try:
            stream = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
                top_p=0.9,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"LLM streaming failed: {e}")
            yield f"I apologize, but I encountered an error generating the response. Please try again."
    async def summarize_history(
        self, conversation_history: List[Dict[str, str]]
    ) -> str:
        """
        Summarize conversation history when it exceeds the maximum length.
        Args:
            conversation_history: List of conversation messages to summarize.
        Returns:
            Summarized conversation text.
        """
        conversation_text = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in conversation_history]
        )
        prompt = SUMMARIZATION_PROMPT.format(conversation=conversation_text)
        try:
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes conversations."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=300,
            )
            summary = response.choices[0].message.content.strip()
            logger.info("Conversation history summarized successfully.")
            return summary
        except Exception as e:
            logger.error(f"History summarization failed: {e}")
            return "Previous conversation context unavailable."
    def manage_conversation_history(
        self,
        conversation_history: List[Dict[str, str]],
        new_user_message: str,
        new_assistant_message: str,
    ) -> List[Dict[str, str]]:
        """
        Add new messages to history and manage length.
        Args:
            conversation_history: Current conversation history.
            new_user_message: The user's latest message.
            new_assistant_message: The assistant's latest response.
        Returns:
            Updated conversation history list.
        """
        conversation_history.append({"role": "user", "content": new_user_message})
        conversation_history.append({"role": "assistant", "content": new_assistant_message})
        return conversation_history
    async def maybe_summarize_history(
        self, conversation_history: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Check if history needs summarization and do it if necessary.
        Args:
            conversation_history: Current conversation history.
        Returns:
            Potentially condensed conversation history.
        """
        if len(conversation_history) > self.max_history * 2:
            older_messages = conversation_history[:-4]
            recent_messages = conversation_history[-4:]
            summary = await self.summarize_history(older_messages)
            condensed = [
                {"role": "system", "content": f"Summary of earlier conversation: {summary}"}
            ]
            condensed.extend(recent_messages)
            logger.info(
                f"Condensed history from {len(conversation_history)} to {len(condensed)} messages"
            )
            return condensed
        return conversation_history
    def _build_system_prompt(self, context: str, language: str) -> str:
        """Build the system prompt with injected context."""
        if language == "ar":
            return TUTOR_SYSTEM_PROMPT_ARABIC.format(
                context=context if context else "تادنتسملا نم حاتم قايس دجوي لا."
            )
        return TUTOR_SYSTEM_PROMPT.format(
            context=context if context else "No document context available."
        )
    def _build_messages(
        self,
        system_prompt: str,
        conversation_history: List[Dict[str, str]],
        current_query: str,
    ) -> List[Dict[str, str]]:
        """Build the complete message list for the API call."""
        messages = [{"role": "system", "content": system_prompt}]
        for msg in conversation_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": current_query})
        return messages