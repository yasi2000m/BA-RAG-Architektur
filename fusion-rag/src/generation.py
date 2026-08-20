import os

from dotenv import load_dotenv
from openai import OpenAI


class AnswerGenerator:
    """
    Erstellt aus Anfrage und Kontext einen Prompt und generiert eine Antwort.
    """

    def __init__(self, model_name: str | None = None) -> None:
        load_dotenv()
        self.model_name = model_name or os.getenv("OPENAI_LLM_MODEL", "gpt-4.1-mini")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def build_prompt(self, query: str, relevant_chunks: list[str]) -> str:
        """
        Baut den Augmentation-Prompt aus Nutzeranfrage und abgerufenen Chunks.
        """
        context = "\n\n".join(
            f"Kontext {index + 1}:\n{chunk}"
            for index, chunk in enumerate(relevant_chunks)
        )

        return f"""Beantworte die Frage ausschliesslich auf Basis des folgenden Kontexts.
Wenn die Antwort nicht im Kontext enthalten ist, sage, dass die Information im Kontext nicht vorhanden ist.

{context}

Frage:
{query}

Antwort:"""

    # def generate_answer(self, query: str, relevant_chunks: list[str]) -> str:
    #     """
    #     Uebergibt den Prompt an das Large Language Model und gibt die Antwort zurueck.
    #     """
    #     prompt = self.build_prompt(query, relevant_chunks)

    #     response = self.client.chat.completions.create(
    #         model=self.model_name,
    #         messages=[
    #             {"role": "user", "content": prompt},
    #         ],
    #         temperature=0.3,
    #     )

    #     return response.choices[0].message.content or ""

    def generate_answer(
        self,
        query: str,
        relevant_chunks: list[str],
    ) -> tuple[str, int]:
        """
        Gibt die Antwort und den gesamten Tokenverbrauch zurück.
        """
        prompt = self.build_prompt(query, relevant_chunks)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )

        answer = response.choices[0].message.content or ""
        total_tokens = response.usage.total_tokens if response.usage else 0

        return answer, total_tokens