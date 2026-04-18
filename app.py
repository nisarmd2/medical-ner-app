import dash
from dash import dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import base64
import io
import os

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = "nisarmd2/ner-model"  



tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32
)

model.eval()

ENTITY_COLORS = {
    "Pathogen":         {"bg": "#fde8e8", "border": "#e74c3c", "text": "#c0392b", "badge": "#e74c3c"},
    "Medicine":         {"bg": "#e8f4fd", "border": "#2980b9", "text": "#1a5276", "badge": "#2980b9"},
    "MedicalCondition": {"bg": "#e8fdf0", "border": "#27ae60", "text": "#1e8449", "badge": "#27ae60"},
}

ENTITY_LABELS = {
    "Pathogen":         "🦠 Pathogen",
    "Medicine":         "💊 Medicine",
    "MedicalCondition": "🏥 Medical Condition",
}

# Predict
def predict(text):
    encoding = tokenizer(
        text, return_offsets_mapping=True,
        return_tensors="pt", truncation=True, max_length=512
    )
    offsets  = encoding["offset_mapping"][0]
    word_ids = encoding.word_ids()
    encoding.pop("offset_mapping")

    with torch.no_grad():
        outputs = model(**encoding)
    predictions = outputs.logits.argmax(dim=2)[0]

    word_preds, word_starts, word_ends = {}, {}, {}
    for idx, w_id in enumerate(word_ids):
        if w_id is None:
            continue
        label = model.config.id2label[predictions[idx].item()]
        t_start, t_end = offsets[idx].tolist()
        if w_id not in word_preds:
            word_preds[w_id]  = label
            word_starts[w_id] = t_start
        word_ends[w_id] = t_end

    entities      = []
    current_label = None
    entity_start  = None
    entity_end    = None

    for w_id in sorted(word_preds.keys()):
        label        = word_preds[w_id]
        w_char_start = word_starts[w_id]
        w_char_end   = word_ends[w_id]

        if label.startswith("B-"):
            if current_label is not None:
                span = text[entity_start:entity_end].strip()
                if len(span) > 2:
                    entities.append((span, entity_start, entity_end, current_label))
            current_label = label[2:]
            entity_start  = w_char_start
            entity_end    = w_char_end

        elif label.startswith("I-") and current_label == label[2:]:
            entity_end = w_char_end

        else:
            if current_label is not None:
                gap = text[entity_end:w_char_start]
                if gap in ("-", "/", ".") and label != "O":
                    entity_end = w_char_end
                    continue
                span = text[entity_start:entity_end].strip()
                if len(span) > 2:
                    entities.append((span, entity_start, entity_end, current_label))
            current_label = None
            entity_start  = None
            entity_end    = None

    if current_label is not None:
        span = text[entity_start:entity_end].strip()
        if len(span) > 2:
            entities.append((span, entity_start, entity_end, current_label))

    # Merge adjacent same-type entities
    if entities:
        merged = [list(entities[0])]
        for curr in entities[1:]:
            prev = merged[-1]
            gap  = text[prev[2]:curr[1]]
            if curr[3] == prev[3] and gap in (" ", "-", " - "):
                prev[0] = text[prev[1]:curr[2]].strip()
                prev[2] = curr[2]
            else:
                merged.append(list(curr))
        entities = [tuple(e) for e in merged]

    return entities


def build_highlighted_html(text, entities):
    """Build list of Dash html components with colored entity highlights."""
    if not entities:
        return [html.Span(text, style={"lineHeight": "2"})]

    parts   = []
    cursor  = 0
    for (span, start, end, label) in sorted(entities, key=lambda x: x[1]):
        # plain text before entity
        if cursor < start:
            parts.append(html.Span(text[cursor:start], style={"lineHeight": "2"}))

        colors = ENTITY_COLORS.get(label, {"bg": "#f5f5f5", "border": "#999", "text": "#333", "badge": "#999"})
        parts.append(
            html.Span(
                [
                    html.Span(span, style={"fontWeight": "600"}),
                    html.Sup(
                        label.replace("MedicalCondition", "Condition"),
                        style={
                            "fontSize": "0.6em",
                            "marginLeft": "3px",
                            "background": colors["badge"],
                            "color": "white",
                            "padding": "1px 5px",
                            "borderRadius": "4px",
                            "fontWeight": "700",
                            "letterSpacing": "0.03em",
                        }
                    )
                ],
                style={
                    "background":    colors["bg"],
                    "border":        f"1.5px solid {colors['border']}",
                    "borderRadius":  "5px",
                    "padding":       "1px 4px",
                    "color":         colors["text"],
                    "lineHeight":    "2.2",
                    "display":       "inline",
                    "margin":        "0 1px",
                }
            )
        )
        cursor = end

    if cursor < len(text):
        parts.append(html.Span(text[cursor:], style={"lineHeight": "2"}))

    return parts


def entity_summary_cards(entities):
    if not entities:
        return []

    from collections import defaultdict
    grouped = defaultdict(list)
    for (span, _, _, label) in entities:
        grouped[label].append(span)

    cards = []
    for label, spans in grouped.items():
        colors = ENTITY_COLORS[label]
        unique_spans = list(dict.fromkeys(spans))   # preserve order, deduplicate
        cards.append(
            html.Div([
                html.Div(ENTITY_LABELS[label], style={
                    "fontWeight": "700",
                    "fontSize": "0.8rem",
                    "color": colors["badge"],
                    "marginBottom": "8px",
                    "textTransform": "uppercase",
                    "letterSpacing": "0.08em",
                }),
                html.Div([
                    html.Span(s, style={
                        "display":       "inline-block",
                        "background":    colors["bg"],
                        "border":        f"1px solid {colors['border']}",
                        "color":         colors["text"],
                        "borderRadius":  "20px",
                        "padding":       "3px 10px",
                        "margin":        "3px",
                        "fontSize":      "0.82rem",
                        "fontWeight":    "500",
                    }) for s in unique_spans
                ])
            ], style={
                "background":   "#fff",
                "border":       f"1.5px solid {colors['border']}",
                "borderRadius": "10px",
                "padding":      "14px 16px",
                "marginBottom": "12px",
            })
        )
    return cards


# layout
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.title = "Medical NER"

FONT_URL = "https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap"

app.layout = html.Div([

    html.Link(rel="stylesheet", href=FONT_URL),
    
    html.Div([
        html.Div([
            html.Div("⚕", style={"fontSize": "2.2rem", "marginBottom": "4px"}),
            html.H1("Medical NER", style={
                "fontFamily":   "'DM Serif Display', serif",
                "fontSize":     "2.4rem",
                "color":        "#0f172a",
                "margin":       "0",
                "letterSpacing": "-0.02em",
            }),
            html.P("Named Entity Recognition for Pathogens · Medicines · Medical Conditions", style={
                "fontFamily": "'DM Sans', sans-serif",
                "color":      "#64748b",
                "fontSize":   "0.95rem",
                "margin":     "6px 0 0",
            }),
        ], style={"textAlign": "center", "padding": "40px 20px 30px"}),

        html.Div([
            html.Div([
                html.Span(style={
                    "display":      "inline-block",
                    "width":        "10px",
                    "height":       "10px",
                    "borderRadius": "50%",
                    "background":   c["badge"],
                    "marginRight":  "6px",
                }),
                html.Span(ENTITY_LABELS[lbl], style={
                    "fontFamily": "'DM Sans', sans-serif",
                    "fontSize":   "0.82rem",
                    "color":      "#475569",
                    "fontWeight": "500",
                })
            ], style={"display": "inline-flex", "alignItems": "center", "margin": "0 14px"})
            for lbl, c in ENTITY_COLORS.items()
        ], style={"textAlign": "center", "paddingBottom": "28px"}),

    ], style={
        "background":   "linear-gradient(135deg, #f8fafc 0%, #e8f4fd 100%)",
        "borderBottom": "1px solid #e2e8f0",
    }),

    # Main content 
    html.Div([
        dbc.Row([

            dbc.Col([
                html.Div([
                    html.Label("Input Text", style={
                        "fontFamily":    "'DM Sans', sans-serif",
                        "fontWeight":    "600",
                        "fontSize":      "0.85rem",
                        "color":         "#374151",
                        "textTransform": "uppercase",
                        "letterSpacing": "0.07em",
                        "marginBottom":  "10px",
                        "display":       "block",
                    }),
                    dcc.Textarea(
                        id="text-input",
                        placeholder="Paste medical text here…",
                        style={
                            "width":       "100%",
                            "height":      "220px",
                            "border":      "1.5px solid #cbd5e1",
                            "borderRadius":"10px",
                            "padding":     "14px",
                            "fontFamily":  "'DM Sans', sans-serif",
                            "fontSize":    "0.92rem",
                            "color":       "#1e293b",
                            "resize":      "vertical",
                            "outline":     "none",
                            "lineHeight":  "1.7",
                        }
                    ),

                    html.Div([
                        html.Label("Or upload a .txt / .pdf file", style={
                            "fontFamily":    "'DM Sans', sans-serif",
                            "fontWeight":    "600",
                            "fontSize":      "0.85rem",
                            "color":         "#374151",
                            "textTransform": "uppercase",
                            "letterSpacing": "0.07em",
                            "marginBottom":  "10px",
                            "display":       "block",
                        }),
                        dcc.Upload(
                            id="file-upload",
                            children=html.Div([
                                html.Span("📂 ", style={"fontSize": "1.4rem"}),
                                html.Span(" Drag & drop or ", style={"color": "#64748b"}),
                                html.Span("browse", style={"color": "#2980b9", "fontWeight": "600"}),
                            ], style={"fontFamily": "'DM Sans', sans-serif", "fontSize": "0.9rem"}),
                            style={
                                "border":        "2px dashed #cbd5e1",
                                "borderRadius":  "10px",
                                "padding":       "24px",
                                "textAlign":     "center",
                                "cursor":        "pointer",
                                "background":    "#f8fafc",
                                "transition":    "border-color 0.2s",
                            },
                            accept=".txt,.pdf"
                        ),
                        html.Div(id="upload-status", style={
                            "fontFamily": "'DM Sans', sans-serif",
                            "fontSize":   "0.8rem",
                            "color":      "#64748b",
                            "marginTop":  "6px",
                        }),
                    ], style={"marginTop": "20px"}),

                    html.Button("Analyse Text →", id="analyse-btn", n_clicks=0, style={
                        "marginTop":     "20px",
                        "width":         "100%",
                        "padding":       "13px",
                        "background":    "#0f172a",
                        "color":         "white",
                        "border":        "none",
                        "borderRadius":  "10px",
                        "fontFamily":    "'DM Sans', sans-serif",
                        "fontWeight":    "600",
                        "fontSize":      "0.95rem",
                        "cursor":        "pointer",
                        "letterSpacing": "0.02em",
                        "transition":    "background 0.2s",
                    }),

                    html.Div(id="error-msg", style={
                        "color":      "#e74c3c",
                        "fontFamily": "'DM Sans', sans-serif",
                        "fontSize":   "0.85rem",
                        "marginTop":  "8px",
                    }),

                ], style={
                    "background":   "#fff",
                    "borderRadius": "14px",
                    "padding":      "24px",
                    "boxShadow":    "0 1px 3px rgba(0,0,0,0.08), 0 4px 16px rgba(0,0,0,0.04)",
                })
            ], md=5),

            dbc.Col([

                html.Div([
                    html.Label("Highlighted Entities", style={
                        "fontFamily":    "'DM Sans', sans-serif",
                        "fontWeight":    "600",
                        "fontSize":      "0.85rem",
                        "color":         "#374151",
                        "textTransform": "uppercase",
                        "letterSpacing": "0.07em",
                        "marginBottom":  "12px",
                        "display":       "block",
                    }),
                    html.Div(
                        id="highlighted-output",
                        children=html.Span("Results will appear here after analysis.", style={
                            "color":      "#94a3b8",
                            "fontFamily": "'DM Sans', sans-serif",
                            "fontStyle":  "italic",
                        }),
                        style={
                            "minHeight":   "120px",
                            "lineHeight":  "2.2",
                            "fontFamily":  "'DM Sans', sans-serif",
                            "fontSize":    "0.93rem",
                            "color":       "#1e293b",
                            "background":  "#f8fafc",
                            "border":      "1.5px solid #e2e8f0",
                            "borderRadius":"10px",
                            "padding":     "16px",
                        }
                    ),
                ], style={
                    "background":   "#fff",
                    "borderRadius": "14px",
                    "padding":      "24px",
                    "boxShadow":    "0 1px 3px rgba(0,0,0,0.08), 0 4px 16px rgba(0,0,0,0.04)",
                    "marginBottom": "16px",
                }),

                html.Div([
                    html.Label("Detected Entities", style={
                        "fontFamily":    "'DM Sans', sans-serif",
                        "fontWeight":    "600",
                        "fontSize":      "0.85rem",
                        "color":         "#374151",
                        "textTransform": "uppercase",
                        "letterSpacing": "0.07em",
                        "marginBottom":  "12px",
                        "display":       "block",
                    }),
                    html.Div(id="entity-summary", children=html.Span(
                        "No entities detected yet.",
                        style={"color": "#94a3b8", "fontFamily": "'DM Sans', sans-serif", "fontStyle": "italic", "fontSize": "0.88rem"}
                    )),
                ], style={
                    "background":   "#fff",
                    "borderRadius": "14px",
                    "padding":      "24px",
                    "boxShadow":    "0 1px 3px rgba(0,0,0,0.08), 0 4px 16px rgba(0,0,0,0.04)",
                }),

            ], md=7),
        ], className="g-4"),
    ], style={"maxWidth": "1200px", "margin": "0 auto", "padding": "36px 24px"}),

], style={"minHeight": "100vh", "background": "#f1f5f9"})


#Callbacks

@app.callback(
    Output("text-input",    "value"),
    Output("upload-status", "children"),
    Input("file-upload",    "contents"),
    State("file-upload",    "filename"),
    prevent_initial_call=True,
)
def load_file(contents, filename):
    if contents is None:
        return no_update, ""

    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)

    if filename.endswith(".txt"):
        text = decoded.decode("utf-8", errors="ignore")
        return text, f"✅ Loaded: {filename}"

    elif filename.endswith(".pdf"):
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(decoded))
            text   = "\n".join(page.extract_text() or "" for page in reader.pages)
            return text, f"✅ Loaded: {filename} ({len(reader.pages)} pages)"
        except Exception as e:
            return no_update, f"❌ Could not read PDF: {str(e)}"

    return no_update, "❌ Unsupported file type. Use .txt or .pdf"


@app.callback(
    Output("highlighted-output", "children"),
    Output("entity-summary",     "children"),
    Output("error-msg",          "children"),
    Input("analyse-btn",         "n_clicks"),
    State("text-input",          "value"),
    prevent_initial_call=True,
)
def analyse(n_clicks, text):
    if not text or not text.strip():
        return no_update, no_update, "⚠️ Please enter or upload some text first."

    try:
        entities = predict(text.strip())

        highlighted = build_highlighted_html(text.strip(), entities)
        summary     = entity_summary_cards(entities)

        if not summary:
            summary = html.Span(
                "No entities detected in this text.",
                style={"color": "#94a3b8", "fontFamily": "'DM Sans', sans-serif", "fontStyle": "italic", "fontSize": "0.88rem"}
            )

        return highlighted, summary, ""

    except Exception as e:
        return no_update, no_update, f"❌ Error: {str(e)}"

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080))
    )