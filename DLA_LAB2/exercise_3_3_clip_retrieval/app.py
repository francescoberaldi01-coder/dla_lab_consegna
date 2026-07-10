"""Applicazione Gradio per retrieval text-to-image con CLIP su Flickr8k."""

from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr
from omegaconf import OmegaConf


LAB2_DIR = Path(__file__).resolve().parents[1]
EXERCISE_DIR = Path(__file__).resolve().parent
if str(LAB2_DIR) not in sys.path:
    sys.path.insert(0, str(LAB2_DIR))

from src.clip_retrieval import ClipRetrievalSystem, build_clip_retrieval_system
from src.utils import get_device


CONFIG_PATH = EXERCISE_DIR / "config.yaml"
cfg = OmegaConf.load(CONFIG_PATH)
_retrieval_system: ClipRetrievalSystem | None = None
APP_CSS = """
.gradio-container { max-width: 1180px !important; }
#query-row { align-items: end; }
#status-box textarea { font-size: 0.92rem; }
"""
APP_THEME = gr.themes.Soft(primary_hue="teal", neutral_hue="slate")


def get_cache_path() -> Path:
    """Restituisce il percorso della cache locale degli embedding visuali."""
    return EXERCISE_DIR / str(cfg.output.dir) / str(cfg.output.cache_file)


def get_retrieval_system() -> ClipRetrievalSystem:
    """Costruisce il sistema CLIP una sola volta e lo riusa per le query successive."""
    global _retrieval_system
    if _retrieval_system is None:
        _retrieval_system = build_clip_retrieval_system(
            dataset_name=str(cfg.dataset.name),
            split=str(cfg.dataset.split),
            model_name=str(cfg.model.name),
            max_images=int(cfg.dataset.max_images),
            seed=int(cfg.dataset.seed),
            cache_path=get_cache_path(),
            device=get_device(),
            image_batch_size=int(cfg.model.image_batch_size),
        )
    return _retrieval_system


def retrieve_images(query: str, top_k: int) -> tuple[list[tuple[object, str]], str]:
    """Esegue la ricerca e restituisce una gallery Gradio con caption e score."""
    if not query.strip():
        return [], "Inserisci una descrizione testuale per avviare la ricerca."

    try:
        system = get_retrieval_system()
        results = system.search(query=query, top_k=top_k)
    except Exception as error:
        return [], f"Impossibile preparare il sistema di retrieval: {error}"

    gallery_items = [
        (
            result.record.image,
            f"#{result.rank} | score {result.score:.3f} | {result.record.caption}",
        )
        for result in results
    ]
    status = (
        f"Indicizzate {len(system.index.records)} immagini da "
        f"{system.index.metadata['dataset_name']} ({system.index.metadata['split']})."
    )
    return gallery_items, status


def build_demo() -> gr.Blocks:
    """Definisce l'interfaccia utente dell'applicazione."""
    with gr.Blocks(title="CLIP Text-to-image Retrieval") as demo:
        gr.Markdown(
            """
            # CLIP Text-to-image Retrieval
            Cerca immagini di Flickr8k usando una descrizione in linguaggio naturale.
            """
        )

        with gr.Row(elem_id="query-row"):
            query = gr.Textbox(
                label="Prompt testuale",
                value="a dog running through water",
                lines=2,
                scale=5,
            )
            top_k = gr.Slider(
                label="Risultati",
                minimum=1,
                maximum=10,
                step=1,
                value=int(cfg.retrieval.top_k),
                scale=1,
            )
            submit = gr.Button("Cerca", variant="primary", scale=1)

        status = gr.Textbox(label="Stato indice", interactive=False, elem_id="status-box")
        gallery = gr.Gallery(
            label="Top immagini",
            columns=5,
            rows=2,
            height="auto",
            object_fit="cover",
            show_label=True,
        )

        examples = gr.Examples(
            examples=[
                ["a group of people playing football", 10],
                ["a child in a red shirt", 10],
                ["a dog jumping in the air", 10],
                ["people walking near a lake", 10],
            ],
            inputs=[query, top_k],
            outputs=[gallery, status],
            fn=retrieve_images,
            cache_examples=False,
        )

        submit.click(fn=retrieve_images, inputs=[query, top_k], outputs=[gallery, status])
        query.submit(fn=retrieve_images, inputs=[query, top_k], outputs=[gallery, status])

    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name=str(cfg.app.server_name),
        server_port=int(cfg.app.server_port),
        theme=APP_THEME,
        css=APP_CSS,
    )
