from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter


def test_long_text_is_split_into_multiple_chunks():
    text = " ".join(f"Sentence number {i} talks about retrieval." for i in range(400))
    nodes = SentenceSplitter(chunk_size=512, chunk_overlap=50).get_nodes_from_documents(
        [Document(text=text)]
    )
    assert len(nodes) > 1


def test_short_text_stays_one_chunk():
    nodes = SentenceSplitter(chunk_size=512, chunk_overlap=50).get_nodes_from_documents(
        [Document(text="A short note.")]
    )
    assert len(nodes) == 1
