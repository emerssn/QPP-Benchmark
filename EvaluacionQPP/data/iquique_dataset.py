from typing import Dict, Iterator, Any
import pandas as pd

class IquiqueDataset:
    """Un conjunto de datos sobre Iquique compatible con la interfaz de datasets de PyTerrier"""
    
    def __init__(self):
        # Documentos de muestra con docno y texto
        self.documents = {
            "doc0": "Iquique es una ciudad portuaria y comuna del norte de Chile, capital de la provincia homónima y de la región de Tarapacá",
            "doc1": "La Zona Franca de Iquique (ZOFRI) es uno de los centros comerciales más importantes del norte de Chile y Sudamérica",
            "doc2": "Playa Cavancha es la playa urbana más popular de Iquique, ideal para el surf y deportes acuáticos",
            "doc3": "El Museo Regional de Iquique exhibe la historia de la cultura Chinchorro y la época del salitre",
            "doc4": "El clima de Iquique es desértico costero con abundante nubosidad y temperaturas moderadas durante todo el año",
            "doc5": "La Guerra del Pacífico tuvo un impacto significativo en Iquique, con el combate naval de 1879 entre Chile y Perú",
            "doc6": "La industria del salitre transformó Iquique en el siglo XIX, atrayendo inmigrantes europeos y chinos a la región",
            "doc7": "El patrimonio cultural de Iquique incluye edificios históricos de la época salitrera y tradiciones pampinas"
        }
        
        # Create a mapping for docno to internal ID (now using the same IDs)
        self.docno_to_id = {k: k for k in self.documents.keys()}
        self.id_to_docno = self.docno_to_id
        
        # Consultas de muestra con qid y texto de consulta
        self.topics = pd.DataFrame([
            {"qid": "0", "query": "playa cavancha iquique"},
            {"qid": "1", "query": "zona franca zofri"},
            {"qid": "2", "query": "museo historia iquique"},
            {"qid": "3", "query": "historia salitre guerra pacifico"}
        ])
        
        # Juicios de relevancia de muestra (qrels)
        self.qrels = pd.DataFrame([
            {"qid": "0", "docno": "doc2", "relevance": 1},
            {"qid": "1", "docno": "doc1", "relevance": 1},
            {"qid": "2", "docno": "doc3", "relevance": 1},
            {"qid": "2", "docno": "doc0", "relevance": 1},
            {"qid": "3", "docno": "doc3", "relevance": 1},
            {"qid": "3", "docno": "doc5", "relevance": 1},
            {"qid": "3", "docno": "doc6", "relevance": 2},
            {"qid": "3", "docno": "doc7", "relevance": 1}
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

    def text_loader(self, metadata=None, verbose=False, **kwargs) -> Dict[str, str]:
        """
        Método requerido por PyTerrier para cargar el texto de los documentos.
        """
        if verbose:
            print("Loading text from IquiqueDataset...")
        return self.documents

    def map_docno(self, docno: str) -> str:
        """Maps between internal and external document IDs"""
        return docno  # Since we're using the same IDs now, just return the input