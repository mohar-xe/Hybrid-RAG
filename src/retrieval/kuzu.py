import kuzu
from typing import List, Dict, Any

class KuzuGraphRetriever:
    def __init__(self, db_path: str = "./kuzu_db"):
        self.db_path = db_path
        self.db = kuzu.Database(self.db_path)
        self.conn = kuzu.Connection(self.db)
        
    def upsert_node(self, label: str, primary_key: str, properties: Dict[str, Any]) -> None:
        if primary_key not in properties:
            raise ValueError(f"Primary key '{primary_key}' must be included in properties.")
            
        pk_val = properties[primary_key]
        other_props = {k: v for k, v in properties.items() if k != primary_key}
        
        set_statements = ", ".join([f"n.{k} = $prop_{k}" for k in other_props.keys()])
        set_clause = f"SET {set_statements}" if set_statements else ""
        
        query = f"""
        MERGE (n:{label} {{{primary_key}: $pk_val}})
        {set_clause}
        """

        params = {"pk_val": pk_val}
        for k, v in other_props.items():
            params[f"prop_{k}"] = v
            
        try:
            self.conn.execute(query, parameters=params)
        except Exception as e:
            print(f"Failed to upsert node {label} ({pk_val}): {e}")

    def upsert_relationship(self, src_label: str, src_pk: str, src_val: Any,
                                  dst_label: str, dst_pk: str, dst_val: Any,
                                  rel_label: str, properties: Dict[str, Any] = None) -> None:
        properties = properties or {}
        
        set_statements = ", ".join([f"r.{k} = $prop_{k}" for k in properties.keys()])
        set_clause = f"SET {set_statements}" if set_statements else ""
        
        query = f"""
        MATCH (a:{src_label} {{{src_pk}: $src_val}})
        MATCH (b:{dst_label} {{{dst_pk}: $dst_val}})
        MERGE (a)-[r:{rel_label}]->(b)
        {set_clause}
        """
        
        params = {"src_val": src_val, "dst_val": dst_val}
        for k, v in properties.items():
            params[f"prop_{k}"] = v
            
        try:
            self.conn.execute(query, parameters=params)
        except Exception as e:
            print(f"Failed to upsert relationship {rel_label}: {e}")
    
    def execute_query(self, cypher_query: str) -> List[Any]:
        try:
            query_result = self.conn.execute(cypher_query)
            
            results = []
            while query_result.has_next():
                results.append(query_result.get_next())
                
            return results
        except Exception as e:
            print(f"Failed to execute query: {e}")
            return []
            
    def get_entity_neighborhood(self, entity_name: str) -> List[Any]:
        query = f"""
        MATCH (a)-[r]->(b) 
        WHERE a.name = '{entity_name}' OR b.name = '{entity_name}'
        RETURN a, r, b;
        """
        return self.execute_query(query)

    def search_entities_by_label(self, label: str) -> List[Any]:
        query = f"MATCH (n:{label}) RETURN n;"
        return self.execute_query(query)
