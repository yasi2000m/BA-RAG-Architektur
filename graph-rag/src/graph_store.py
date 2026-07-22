import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


@dataclass
class GraphEntity:
    """Eine Entitaet im lokalen Knowledge Graph."""

    name: str
    entity_type: str
    description: str
    segment_ids: list[int]


@dataclass
class GraphRelationship:
    """Eine gerichtete Beziehung zwischen zwei Entitaeten."""

    source: str
    target: str
    relation: str
    description: str
    segment_id: int


@dataclass
class GraphQueryResult:
    """Das aus dem Knowledge Graph abgerufene Kontextwissen."""

    context: str
    entity_names: list[str]
    relationships: list[GraphRelationship]


class KnowledgeGraphExtractor:
    """
    Extrahiert Entitaeten und Beziehungen aus Textabschnitten.

    Graph-RAG erweitert Standard-RAG um strukturierte Beziehungen. Dafuer wird
    jeder Textabschnitt in eine kleine Menge von Knoten und Kanten ueberfuehrt.
    """

    def __init__(self, model_name: str | None = None) -> None:
        load_dotenv()
        self.model_name = model_name or os.getenv("OPENAI_LLM_MODEL", "gpt-4.1-mini")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def extract(self, text_segment: str) -> dict[str, list[dict[str, str]]]:
        prompt = f"""Extrahiere aus dem folgenden Text einen Knowledge Graph.

Gib ausschliesslich valides JSON in diesem Schema zurueck:
{{
  "entities": [
    {{"name": "kurzer eindeutiger Name", "type": "Konzept|Person|Organisation|Ort|Bauteil|Regel|Messwert|Sonstiges", "description": "kurze Beschreibung"}}
  ],
  "relationships": [
    {{"source": "Name einer Entitaet", "target": "Name einer Entitaet", "relation": "kurze Beziehung", "description": "kurze Begruendung aus dem Text"}}
  ]
}}

Regeln:
- Nutze nur Informationen aus dem Text.
- Erfinde keine Entitaeten oder Beziehungen.
- Verwende stabile, kurze Namen.
- Extrahiere hoechstens 12 Entitaeten und 18 Beziehungen.

Text:
{text_segment}"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content or "{}"
        return _parse_graph_json(content)


class KnowledgeGraphStore:
    """
    Lokaler Knowledge Graph fuer Graph-RAG.

    Der Store speichert Entitaeten, Beziehungen und die Zuordnung zwischen
    Textabschnitten und Entitaeten als JSON. Dadurch bleibt die Architektur
    lokal und gut nachvollziehbar.
    """

    def __init__(self, storage_dir: str = "graph_db") -> None:
        self.storage_dir = Path(storage_dir)
        self.graph_path = self.storage_dir / "knowledge_graph.json"
        self.entities: dict[str, GraphEntity] = {}
        self.relationships: list[GraphRelationship] = []
        self.segment_entities: dict[int, list[str]] = {}

    def build_from_segments(
        self,
        text_segments: list[str],
        extractor: KnowledgeGraphExtractor,
    ) -> None:
        """Erstellt den Graphen aus allen Textabschnitten."""
        self.entities = {}
        self.relationships = []
        self.segment_entities = {}

        for segment_id, text_segment in enumerate(text_segments):
            extracted_graph = extractor.extract(text_segment)
            segment_entity_names: list[str] = []

            for entity in extracted_graph.get("entities", []):
                name = _normalize_name(entity.get("name", ""))
                if not name:
                    continue

                segment_entity_names.append(name)
                entity_type = entity.get("type", "Sonstiges").strip() or "Sonstiges"
                description = entity.get("description", "").strip()

                if name in self.entities:
                    existing = self.entities[name]
                    if segment_id not in existing.segment_ids:
                        existing.segment_ids.append(segment_id)
                    if description and description not in existing.description:
                        existing.description = f"{existing.description}; {description}".strip("; ")
                else:
                    self.entities[name] = GraphEntity(
                        name=name,
                        entity_type=entity_type,
                        description=description,
                        segment_ids=[segment_id],
                    )

            self.segment_entities[segment_id] = sorted(set(segment_entity_names))

            known_entities = set(self.entities)
            for relationship in extracted_graph.get("relationships", []):
                source = _normalize_name(relationship.get("source", ""))
                target = _normalize_name(relationship.get("target", ""))
                relation = relationship.get("relation", "").strip()

                if not source or not target or source not in known_entities or target not in known_entities:
                    continue

                self.relationships.append(
                    GraphRelationship(
                        source=source,
                        target=target,
                        relation=relation or "steht in Beziehung zu",
                        description=relationship.get("description", "").strip(),
                        segment_id=segment_id,
                    )
                )

    def save(self) -> None:
        """Persistiert den Graphen lokal."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "entities": [
                {
                    "name": entity.name,
                    "type": entity.entity_type,
                    "description": entity.description,
                    "segment_ids": entity.segment_ids,
                }
                for entity in self.entities.values()
            ],
            "relationships": [
                {
                    "source": relationship.source,
                    "target": relationship.target,
                    "relation": relationship.relation,
                    "description": relationship.description,
                    "segment_id": relationship.segment_id,
                }
                for relationship in self.relationships
            ],
            "segment_entities": {
                str(segment_id): entity_names
                for segment_id, entity_names in self.segment_entities.items()
            },
        }
        self.graph_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self) -> None:
        """Laedt einen gespeicherten Knowledge Graph."""
        if not self.graph_path.exists():
            raise FileNotFoundError("Kein gespeicherter Knowledge Graph gefunden.")

        payload = json.loads(self.graph_path.read_text(encoding="utf-8"))
        self.entities = {
            item["name"]: GraphEntity(
                name=item["name"],
                entity_type=item.get("type", "Sonstiges"),
                description=item.get("description", ""),
                segment_ids=item.get("segment_ids", []),
            )
            for item in payload.get("entities", [])
        }
        self.relationships = [
            GraphRelationship(
                source=item["source"],
                target=item["target"],
                relation=item.get("relation", "steht in Beziehung zu"),
                description=item.get("description", ""),
                segment_id=item.get("segment_id", -1),
            )
            for item in payload.get("relationships", [])
        ]
        stored_segment_entities = payload.get("segment_entities", {})
        self.segment_entities = {
            int(segment_id): entity_names
            for segment_id, entity_names in stored_segment_entities.items()
        }

    def context_for_query(
        self,
        query: str,
        seed_segment_ids: list[int] | None = None,
        max_entities: int = 10,
        max_relationships: int = 15,
    ) -> str:
        """
        Erzeugt spezifisches Graph-Kontextwissen fuer eine Anfrage.

        Der Kontext wird aus direkt passenden Entitaeten und deren
        ein-Hops-Beziehungen gebildet.
        """
        if not self.entities:
            return ""

        seed_segment_ids = seed_segment_ids or []
        selected_entities: list[str] = []
        query_terms = _query_terms(query)

        for segment_id in seed_segment_ids:
            for entity_name in self.segment_entities.get(segment_id, []):
                _append_unique(selected_entities, entity_name)

        for entity_name in self.entities:
            entity = self.entities[entity_name]
            entity_terms = _query_terms(f"{entity.name} {entity.description} {entity.entity_type}")
            if query_terms.intersection(entity_terms):
                _append_unique(selected_entities, entity_name)

        if not selected_entities:
            scored_entities = sorted(
                self.entities.values(),
                key=lambda entity: len(
                    query_terms.intersection(
                        _query_terms(f"{entity.name} {entity.description} {entity.entity_type}")
                    )
                ),
                reverse=True,
            )
            for entity in scored_entities:
                if query_terms.intersection(_query_terms(f"{entity.name} {entity.description} {entity.entity_type}")):
                    _append_unique(selected_entities, entity.name)

        selected_entities = selected_entities[:max_entities]
        selected_set = set(selected_entities)

        matching_relationships = [
            relationship
            for relationship in self.relationships
            if relationship.source in selected_set
            or relationship.target in selected_set
            or relationship.segment_id in seed_segment_ids
        ]
        if not matching_relationships:
            matching_relationships = [
                relationship
                for relationship in self.relationships
                if query_terms.intersection(
                    _query_terms(
                        f"{relationship.source} {relationship.target} "
                        f"{relationship.relation} {relationship.description}"
                    )
                )
            ]
        selected_relationships = matching_relationships[:max_relationships]

        if not selected_entities and not selected_relationships:
            return ""

        lines = ["Knowledge-Graph-Kontext:"]

        if selected_entities:
            lines.append("Entitaeten:")
            for entity_name in selected_entities:
                entity = self.entities[entity_name]
                lines.append(f"- {entity.name} ({entity.entity_type}): {entity.description}")

        if selected_relationships:
            lines.append("Beziehungen:")
            for relationship in selected_relationships:
                description = f" - {relationship.description}" if relationship.description else ""
                lines.append(
                    f"- {relationship.source} --{relationship.relation}--> "
                    f"{relationship.target}{description}"
                )

        return "\n".join(lines)

    def query_subgraph(
        self,
        query: str,
        seed_segment_ids: list[int] | None = None,
        max_entities: int = 10,
        max_relationships: int = 15,
    ) -> GraphQueryResult:
        """
        Ruft den fuer eine Frage relevanten Teilgraphen ab.

        Diese Methode liefert dieselben Inhalte wie der Prompt-Kontext, aber
        zusaetzlich strukturiert als Entitaets- und Beziehungsliste fuer die UI.
        """
        seed_segment_ids = seed_segment_ids or []
        selected_entities = self._select_entities_for_query(
            query=query,
            seed_segment_ids=seed_segment_ids,
            max_entities=max_entities,
        )
        selected_relationships = self._select_relationships_for_query(
            query=query,
            selected_entities=selected_entities,
            seed_segment_ids=seed_segment_ids,
            max_relationships=max_relationships,
        )
        context = self._format_context(selected_entities, selected_relationships)

        return GraphQueryResult(
            context=context,
            entity_names=selected_entities,
            relationships=selected_relationships,
        )

    def _select_entities_for_query(
        self,
        query: str,
        seed_segment_ids: list[int],
        max_entities: int,
    ) -> list[str]:
        selected_entities: list[str] = []
        query_terms = _query_terms(query)

        for segment_id in seed_segment_ids:
            for entity_name in self.segment_entities.get(segment_id, []):
                _append_unique(selected_entities, entity_name)

        for entity_name in self.entities:
            entity = self.entities[entity_name]
            entity_terms = _query_terms(f"{entity.name} {entity.description} {entity.entity_type}")
            if query_terms.intersection(entity_terms):
                _append_unique(selected_entities, entity_name)

        return selected_entities[:max_entities]

    def _select_relationships_for_query(
        self,
        query: str,
        selected_entities: list[str],
        seed_segment_ids: list[int],
        max_relationships: int,
    ) -> list[GraphRelationship]:
        query_terms = _query_terms(query)
        selected_set = set(selected_entities)

        matching_relationships = [
            relationship
            for relationship in self.relationships
            if relationship.source in selected_set
            or relationship.target in selected_set
            or relationship.segment_id in seed_segment_ids
        ]

        if not matching_relationships:
            matching_relationships = [
                relationship
                for relationship in self.relationships
                if query_terms.intersection(
                    _query_terms(
                        f"{relationship.source} {relationship.target} "
                        f"{relationship.relation} {relationship.description}"
                    )
                )
            ]

        return matching_relationships[:max_relationships]

    def _format_context(
        self,
        selected_entities: list[str],
        selected_relationships: list[GraphRelationship],
    ) -> str:
        if not selected_entities and not selected_relationships:
            return ""

        lines = ["Knowledge-Graph-Kontext:"]

        if selected_entities:
            lines.append("Entitaeten:")
            for entity_name in selected_entities:
                entity = self.entities[entity_name]
                lines.append(f"- {entity.name} ({entity.entity_type}): {entity.description}")

        if selected_relationships:
            lines.append("Beziehungen:")
            for relationship in selected_relationships:
                description = f" - {relationship.description}" if relationship.description else ""
                lines.append(
                    f"- {relationship.source} --{relationship.relation}--> "
                    f"{relationship.target}{description}"
                )

        return "\n".join(lines)


def _parse_graph_json(content: str) -> dict[str, list[dict[str, str]]]:
    try:
        payload: Any = json.loads(content)
    except json.JSONDecodeError:
        return {"entities": [], "relationships": []}

    if not isinstance(payload, dict):
        return {"entities": [], "relationships": []}

    entities = payload.get("entities", [])
    relationships = payload.get("relationships", [])

    return {
        "entities": entities if isinstance(entities, list) else [],
        "relationships": relationships if isinstance(relationships, list) else [],
    }


def _normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip()


def _query_terms(text: str) -> set[str]:
    return {
        term.lower()
        for term in re.findall(r"[A-Za-zÄÖÜäöüß0-9_-]{3,}", text)
    }


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)
