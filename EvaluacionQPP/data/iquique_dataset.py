from typing import Dict, Iterator, Any
import pandas as pd

class IquiqueDataset:
    """Un conjunto de datos sobre Iquique compatible con la interfaz de datasets de PyTerrier"""
    
    def __init__(self):
        # Documentos de muestra con docno y texto
        self.documents = {
            "doc1": "Iquique es una ciudad portuaria y comuna del norte de Chile, capital de la provincia homónima y de la región de Tarapacá",
            "doc2": "La Zona Franca de Iquique (ZOFRI) es uno de los centros comerciales más importantes del norte de Chile y Sudamérica",
            "doc3": "Playa Cavancha es la playa urbana más popular de Iquique, ideal para el surf y deportes acuáticos",
            "doc4": "El Museo Regional de Iquique exhibe la historia de la cultura Chinchorro y la época del salitre",
            "doc5": "El clima de Iquique es desértico costero con abundante nubosidad y temperaturas moderadas durante todo el año",
            "doc6": "La Guerra del Pacífico tuvo un impacto significativo en Iquique, con el combate naval de 1879 entre Chile y Perú",
            "doc7": "La industria del salitre transformó Iquique en el siglo XIX, atrayendo inmigrantes europeos y chinos a la región",
            "doc8": "El patrimonio cultural de Iquique incluye edificios históricos de la época salitrera y tradiciones pampinas"
        }
        
        # Consultas de muestra con qid y texto de consulta
        self.topics = pd.DataFrame([
            {"qid": "q1", "query": "playa cavancha iquique"},
            {"qid": "q2", "query": "zona franca zofri"},
            {"qid": "q3", "query": "museo historia iquique"},
            {"qid": "q4", "query": "historia salitre guerra pacifico"}  # Consulta difícil y ambigua
        ])
        
        # Juicios de relevancia de muestra (qrels)
        self.qrels = pd.DataFrame([
            {"qid": "q1", "docno": "doc3", "relevance": 1},
            {"qid": "q2", "docno": "doc2", "relevance": 1},
            {"qid": "q3", "docno": "doc4", "relevance": 1},
            {"qid": "q3", "docno": "doc1", "relevance": 1},
            {"qid": "q4", "docno": "doc4", "relevance": 1},  # Relevante por mención del salitre
            {"qid": "q4", "docno": "doc6", "relevance": 1},  # Relevante por la Guerra del Pacífico
            {"qid": "q4", "docno": "doc7", "relevance": 2},  # Muy relevante por combinar salitre e historia
            {"qid": "q4", "docno": "doc8", "relevance": 1}   # Relevante por contexto histórico
        ])

    def get_corpus_iter(self) -> Iterator[Dict[str, Any]]:
        """Devuelve un iterador sobre la colección de documentos"""
        for docno, text in self.documents.items():
            yield {"docno": docno, "text": text}

    def get_topics(self) -> pd.DataFrame:
        """Devuelve los temas/consultas"""
        return self.topics

    def get_qrels(self) -> pd.DataFrame:
        """Devuelve los juicios de relevancia"""
        return self.qrels

    def get_corpus_lang(self) -> str:
        """Devuelve el código de idioma ISO 639-1 para el corpus"""
        return 'es'

    def get_topics_lang(self) -> str:
        """Devuelve el código de idioma ISO 639-1 para los temas"""
        return 'es'

    def info_url(self) -> str:
        """Devuelve una URL que proporciona más información sobre este conjunto de datos"""
        return None

    def text_loader(self) -> Dict[str, str]:
        """
        Método requerido por PyTerrier para cargar el texto de los documentos.
        Returns:
            Dict[str, str]: Diccionario que mapea docno a texto del documento
        """
        return self.documents