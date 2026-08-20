import os

from dotenv import load_dotenv
from openai import OpenAI


class AnswerGenerator:
    """
    Erstellt aus Anfrage und Graph-Kontext einen Prompt und generiert eine Antwort.
    """

    def __init__(self, model_name: str | None = None) -> None:
        load_dotenv()
        self.model_name = model_name or os.getenv("OPENAI_LLM_MODEL", "gpt-4.1-mini")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def build_prompt(self, query: str, graph_context: str) -> str:
        """
        Baut den Augmentation-Prompt aus Nutzeranfrage und Knowledge Graph.
        """
        return f"""Beantworte die Frage ausschliesslich auf Basis des folgenden Knowledge-Graph-Kontexts.
Nutze die Entitaeten und Beziehungen zwischen den Konzepten.
Wenn die Antwort nicht im Kontext enthalten ist, sage, dass die Information im Kontext nicht vorhanden ist.

{graph_context}

Frage:
{query}

Antwort:"""

    def generate_answer(self, query: str, graph_context: str) -> tuple[str, int]:
        """
        Uebergibt den Prompt an das Large Language Model und gibt die Antwort zurueck.
        """
        prompt = self.build_prompt(query, graph_context)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        answer = response.choices[0].message.content or "" 
        used_tokens = ( 
            response.usage.total_tokens 
            if response.usage is not None 
            else 0 
            ) 

        return answer, used_tokens
        

        # return response.choices[0].message.content or ""

