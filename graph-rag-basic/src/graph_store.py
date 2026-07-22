import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


STOP_TERMS = set(
    "aber alle als auf aus bei das den der des die ein eine einer eines fuer für "
    "ist mit nenne oder sind und von was welche zu zum zur".split()
)


@dataclass
class GraphEntity:
    name: str
    entity_type: str
    description: str
    chunk_ids: list[int]


@dataclass
class GraphRelationship:
    source: str
    target: str
    relation: str
    description: str
    chunk_id: int


@dataclass
class GraphQueryResult:
    context: str
    entity_names: list[str]
    relationships: list[GraphRelationship]


class KnowledgeGraphExtractor:
    """Extrahiert Entitaeten und Beziehungen aus Text-Chunks."""

    def __init__(self, model_name: str | None = None) -> None:
        load_dotenv()
        self.model_name = model_name or os.getenv("OPENAI_LLM_MODEL", "gpt-4.1-mini")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def extract(self, chunk: str) -> dict[str, list[dict[str, str]]]:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": self._build_prompt(chunk)}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        return _parse_graph_json(response.choices[0].message.content or "{}")

    def _build_prompt(self, chunk: str) -> str:
        return f"""Extrahiere aus dem Text einen einfachen Knowledge Graph.
Antworte nur als valides JSON:
{{
  "entities": [
    {{"name": "kurzer Name", "type": "Konzept|Person|Organisation|Ort|Bauteil|Regel|Messwert|Sonstiges", "description": "kurze Beschreibung aus dem Text"}}
  ],
  "relationships": [
    {{"source": "Entitaet A", "target": "Entitaet B", "relation": "kurze Beziehung", "description": "kurze Begruendung aus dem Text"}}
  ]
}}
Regeln:
- Nur Informationen aus dem Text nutzen, nichts erfinden.
- Stabile, kurze Namen verwenden.
- Tabellen, nummerierte Listen und Aufzaehlungen vollstaendig abbilden:
  Oberbegriff als Entitaet, sichtbare Eintraege als eigene Entitaeten,
  verbunden durch eine Beziehung wie "besteht aus".
- Maximal 20 Entitaeten und 30 Beziehungen extrahieren.
- Source und Target muessen in der Entity-Liste vorkommen.

Text:
{chunk}"""


class KnowledgeGraphStore:
    """Speichert Entitaeten, Beziehungen und Chunk-Zuordnungen als lokalen JSON-Graph."""

    def __init__(self, storage_dir: str = "graph_db") -> None:
        self.storage_dir = Path(storage_dir)
        self.graph_path = self.storage_dir / "knowledge_graph.json"
        self.entities: dict[str, GraphEntity] = {}
        self.relationships: list[GraphRelationship] = []
        self.chunk_entities: dict[int, list[str]] = {}

    def build_from_chunks(self, chunks: list[str], extractor: KnowledgeGraphExtractor) -> None:
        self.entities = {}
        self.relationships = []
        self.chunk_entities = {}

        for chunk_id, chunk in enumerate(chunks):
            extracted = extractor.extract(chunk)
            chunk_entity_names = self._merge_entities(extracted.get("entities", []), chunk_id)
            self.chunk_entities[chunk_id] = sorted(set(chunk_entity_names))
            self._add_relationships(extracted.get("relationships", []), chunk_id)

    def save(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "entities": [asdict(entity) for entity in self.entities.values()],
            "relationships": [asdict(relationship) for relationship in self.relationships],
            "chunk_entities": {str(key): value for key, value in self.chunk_entities.items()},
        }
        self.graph_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self) -> None:
        if not self.graph_path.exists():
            raise FileNotFoundError("Kein gespeicherter Knowledge Graph gefunden.")

        payload = json.loads(self.graph_path.read_text(encoding="utf-8"))
        self.entities = {
            item["name"]: GraphEntity(
                name=item["name"],
                entity_type=item.get("entity_type", item.get("type", "Sonstiges")),
                description=item.get("description", ""),
                chunk_ids=item.get("chunk_ids", []),
            )
            for item in payload.get("entities", [])
        }
        self.relationships = [
            GraphRelationship(
                source=item["source"],
                target=item["target"],
                relation=item.get("relation", "steht in Beziehung zu"),
                description=item.get("description", ""),
                chunk_id=item.get("chunk_id", -1),
            )
            for item in payload.get("relationships", [])
        ]
        self.chunk_entities = {
            int(chunk_id): entity_names
            for chunk_id, entity_names in payload.get("chunk_entities", {}).items()
        }

    def query_subgraph(self, query: str, max_depth: int = 2, max_entities: int = 12, max_relationships: int = 20) -> GraphQueryResult:
        selected_entities = self._traverse_entities(
            start_entities=self._find_entities_for_query(query),
            max_depth=max_depth,
            max_entities=max_entities,
        )
        relationships = self._relationships_for_entities(selected_entities, max_relationships)
        return GraphQueryResult(
            context=self._format_context(selected_entities, relationships),
            entity_names=selected_entities,
            relationships=relationships,
        )

    def _merge_entities(self, entities: list[dict[str, str]], chunk_id: int) -> list[str]:
        names: list[str] = []
        for item in entities:
            name = _normalize_name(item.get("name", ""))
            if not name:
                continue

            names.append(name)
            description = item.get("description", "").strip()
            entity_type = item.get("type", item.get("entity_type", "Sonstiges")).strip() or "Sonstiges"

            if name not in self.entities:
                self.entities[name] = GraphEntity(name, entity_type, description, [chunk_id])
                continue

            existing = self.entities[name]
            if chunk_id not in existing.chunk_ids:
                existing.chunk_ids.append(chunk_id)
            if description and description not in existing.description:
                existing.description = f"{existing.description}; {description}".strip("; ")

        return names

    def _add_relationships(self, relationships: list[dict[str, str]], chunk_id: int) -> None:
        known_entities = set(self.entities)
        for item in relationships:
            source = _normalize_name(item.get("source", ""))
            target = _normalize_name(item.get("target", ""))
            if not source or not target or source == target:
                continue
            if source not in known_entities or target not in known_entities:
                continue

            self.relationships.append(
                GraphRelationship(
                    source=source,
                    target=target,
                    relation=item.get("relation", "").strip() or "steht in Beziehung zu",
                    description=item.get("description", "").strip(),
                    chunk_id=chunk_id,
                )
            )

    def _find_entities_for_query(self, query: str) -> list[str]:
        query_terms = _terms(query)
        scored = []
        for entity in self.entities.values():
            entity_terms = _terms(f"{entity.name} {entity.entity_type} {entity.description}")
            score = len(query_terms & entity_terms)
            if score:
                scored.append((score, entity.name))
        return [name for _score, name in sorted(scored, reverse=True)]

    def _traverse_entities(self, start_entities: list[str], max_depth: int, max_entities: int) -> list[str]:
        selected: list[str] = []
        frontier = _unique(start_entities)

        for name in frontier:
            _append_unique(selected, name)

        for _ in range(max_depth):
            next_frontier: list[str] = []
            for relationship in self.relationships:
                if relationship.source in frontier:
                    _append_unique(next_frontier, relationship.target)
                if relationship.target in frontier:
                    _append_unique(next_frontier, relationship.source)

            frontier = [name for name in next_frontier if name not in selected]
            for name in frontier:
                _append_unique(selected, name)
                if len(selected) >= max_entities:
                    return selected[:max_entities]
            if not frontier:
                break

        return selected[:max_entities]

    def _relationships_for_entities(self, selected_entities: list[str], max_relationships: int) -> list[GraphRelationship]:
        selected = set(selected_entities)
        return [
            relationship
            for relationship in self.relationships
            if relationship.source in selected and relationship.target in selected
        ][:max_relationships]

    def _format_context(self, entity_names: list[str], relationships: list[GraphRelationship]) -> str:
        if not entity_names and not relationships:
            return ""

        lines = ["Knowledge-Graph-Kontext:"]
        if entity_names:
            lines.append("Entitaeten:")
            for name in entity_names:
                entity = self.entities[name]
                description = f": {entity.description}" if entity.description else ""
                lines.append(f"- {entity.name} ({entity.entity_type}){description}")

        if relationships:
            lines.append("Beziehungen:")
            for relationship in relationships:
                description = f" - {relationship.description}" if relationship.description else ""
                lines.append(
                    f"- {relationship.source} --{relationship.relation}-- "
                    f"{relationship.target}{description}"
                )

        return "\n".join(lines)


def _parse_graph_json(content: str) -> dict[str, list[dict[str, str]]]:
    try:
        payload = json.loads(content)
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


def _terms(text: str) -> set[str]:
    return {
        term.lower()
        for term in re.findall(r"[A-Za-zÄÖÜäöüß0-9_-]{3,}", text)
        if term.lower() not in STOP_TERMS
    }


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        _append_unique(result, value)
    return result
