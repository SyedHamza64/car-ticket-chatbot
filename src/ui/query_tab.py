"""Query tab rendering logic."""


def render_query_tab(
    st,
    datetime,
    time,
    initialize_pipeline,
    reset_pipeline,
    selected_model,
    provider_name,
):
    """Render the Ask Question tab."""
    st.markdown("### 💬 **What would you like to know?**")

    query = st.text_area(
        "Enter your question",
        height=100,
        placeholder="Es: Come posso rimuovere i graffi dalla carrozzeria?\nEs: Quale prodotto usare per lucidare l'auto?",
        label_visibility="collapsed",
    )

    col1, col2, col3 = st.columns([1.5, 1.5, 3])
    with col1:
        n_tickets = st.selectbox(
            "📋 Ticket Sources",
            [1, 2, 3, 4, 5],
            index=2,
            help="Number of relevant tickets to retrieve",
        )
    with col2:
        n_guides = st.selectbox(
            "📚 Guide Sources",
            [1, 2, 3, 4, 5],
            index=2,
            help="Number of relevant guide sections to retrieve",
        )
    with col3:
        generate_btn = st.button("✨ Generate Response", type="primary", use_container_width=True)

    if generate_btn and query.strip():
        with st.spinner("🤔 Thinking..."):
            try:
                start_time = time.time()
                result = st.session_state.pipeline.answer(
                    query,
                    top_k_tickets=n_tickets,
                    top_k_guides=n_guides,
                )
                elapsed = time.time() - start_time

                st.session_state.current_response = result["answer"]
                st.session_state.current_responses = None
                st.session_state.num_drafts = 1
                st.session_state.current_context = result["context"]
                st.session_state.current_sources = result["sources"]
                st.session_state.response_time = elapsed

                st.session_state.query_history.append(
                    {
                        "time": datetime.now().strftime("%H:%M"),
                        "query": query[:50],
                        "response": result["answer"],
                    }
                )

            except Exception as e:
                err_text = str(e)
                if (
                    "error executing plan" in err_text.lower()
                    or "error finding id" in err_text.lower()
                    or "internal error" in err_text.lower()
                ):
                    try:
                        reset_pipeline(recycle_client=True)
                        retry_model = st.session_state.get("current_model") or selected_model
                        retry_provider = st.session_state.get("current_provider") or provider_name
                        pipeline, init_error = initialize_pipeline(retry_model, provider=retry_provider)
                        if init_error:
                            raise RuntimeError(init_error)
                        st.session_state.pipeline = pipeline
                        st.session_state.initialized = True
                        st.session_state.current_model = retry_model
                        st.session_state.current_provider = retry_provider

                        start_time = time.time()
                        result = st.session_state.pipeline.answer(
                            query,
                            top_k_tickets=n_tickets,
                            top_k_guides=n_guides,
                        )
                        elapsed = time.time() - start_time

                        st.session_state.current_response = result["answer"]
                        st.session_state.current_responses = None
                        st.session_state.num_drafts = 1
                        st.session_state.current_context = result["context"]
                        st.session_state.current_sources = result["sources"]
                        st.session_state.response_time = elapsed
                        st.session_state.query_history.append(
                            {
                                "time": datetime.now().strftime("%H:%M"),
                                "query": query[:50],
                                "response": result["answer"],
                            }
                        )
                        st.info("Refreshed vector index state and retried successfully.")
                    except Exception as retry_e:
                        st.error(f"❌ Error: {str(retry_e)}")
                else:
                    st.error(f"❌ Error: {err_text}")

    if st.session_state.current_response:
        st.markdown("---")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("### 💡 AI Response")
        with col2:
            st.caption(f"⏱️ {st.session_state.get('response_time', 0):.1f}s")

        if st.session_state.get("num_drafts", 1) > 1 and st.session_state.get("current_responses"):
            draft_tabs = st.tabs([f"Draft {i + 1}" for i in range(st.session_state.num_drafts)])
            for i, tab in enumerate(draft_tabs):
                with tab:
                    st.markdown(
                        f"""
                    <div class="response-box">
                        <div class="response-text">{st.session_state.current_responses[i]['text']}</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
        else:
            import re

            response_text = st.session_state.current_response
            response_html = re.sub(
                r"\[([^\]]+)\]\(([^)]+)\)",
                r'<a href="\2" target="_blank" style="color: #60A5FA; text-decoration: underline;">\1</a>',
                response_text,
            )
            response_html = response_html.replace("\n", "<br>")
            st.markdown(
                f"""
            <div class="response-box">
                <div class="response-text">{response_html}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            if st.button("📋 Copy to Clipboard", use_container_width=True):
                st.toast("✅ Copied!", icon="📋")
        with col2:
            if st.button("👍 Helpful", use_container_width=True):
                st.toast("Thanks for feedback!", icon="👍")
        with col3:
            if st.button("👎 Not helpful", use_container_width=True):
                st.toast("We'll improve!", icon="📝")

        with st.expander("📚 View Sources", expanded=False):
            src_tab1, src_tab2 = st.tabs(["Tickets", "Guides"])

            with src_tab1:
                if st.session_state.get("current_sources"):
                    tickets = st.session_state.current_sources.get("tickets", {})
                    if tickets.get("ids") and tickets["ids"][0]:
                        docs = tickets.get("documents", [[]])[0] or []
                        metas = tickets.get("metadatas", [[]])[0] or []
                        dists = tickets.get("distances", [[]])[0] or []

                        for i, (doc, meta) in enumerate(zip(docs, metas), 1):
                            subject = meta.get("subject", "N/A")
                            status = meta.get("status", "N/A")
                            ticket_id = meta.get("ticket_id") or meta.get("orig_ticket_id", "N/A")
                            priority = meta.get("priority", "N/A")
                            created_at = meta.get("created_at", "N/A")

                            relevance_badge = ""
                            try:
                                dist = float(dists[i - 1]) if len(dists) >= i else None
                            except Exception:
                                dist = None

                            if dist is not None:
                                if dist <= 0.5:
                                    relevance_badge = "🟢 **High relevance**"
                                elif dist <= 0.75:
                                    relevance_badge = "🟡 **Medium relevance**"
                                else:
                                    relevance_badge = "🔴 **Low relevance**"

                            st.markdown(f"**{i}. {subject}**")
                            caption_line = f"Status: {status} • Ticket ID: {ticket_id}"
                            if relevance_badge:
                                caption_line += f" • {relevance_badge}"
                            st.caption(caption_line)

                            if st.checkbox("Show details", key=f"ticket_src_{i}"):
                                st.markdown(f"- **Ticket ID**: `{ticket_id}`")
                                st.markdown(f"- **Status**: `{status}`")
                                if priority and priority != "N/A":
                                    st.markdown(f"- **Priority**: `{priority}`")
                                if created_at and created_at != "N/A":
                                    st.markdown(f"- **Created at**: `{created_at}`")

                                from html import unescape

                                subject_text = ""
                                description_text = ""
                                messages = []
                                for line in (doc or "").splitlines():
                                    raw = line.strip()
                                    if not raw:
                                        continue
                                    if raw.startswith("Subject:"):
                                        subject_text = raw[len("Subject:") :].strip()
                                    elif raw.startswith("Description:"):
                                        description_text = raw[len("Description:") :].strip()
                                    elif ": " in raw:
                                        author, msg = raw.split(": ", 1)
                                        messages.append((author.strip(), unescape(msg.strip())))

                                st.markdown("---")
                                if subject_text:
                                    st.markdown(f"**Subject**: {subject_text}")
                                if description_text:
                                    st.markdown(f"**Description**: {description_text}")

                                if messages:
                                    st.markdown("**Conversation:**")
                                    for author, msg in messages:
                                        preview = msg if len(msg) <= 500 else msg[:500] + " [...]"
                                        st.markdown(f"- **{author}**: {preview}")

                                if st.checkbox("Show raw searchable text", key=f"ticket_raw_{i}"):
                                    st.text(doc if len(doc) <= 2000 else doc[:2000] + "\n...\n[truncated]")

                            st.markdown("---")
                    else:
                        st.info("No tickets found")
                else:
                    st.info("No tickets found")

            with src_tab2:
                if st.session_state.get("current_sources"):
                    guides = st.session_state.current_sources.get("guides", {})
                    if guides.get("ids") and guides["ids"][0]:
                        docs = guides.get("documents", [[]])[0] or []
                        metas = guides.get("metadatas", [[]])[0] or []
                        dists = guides.get("distances", [[]])[0] or []

                        for i, (doc, meta) in enumerate(zip(docs, metas), 1):
                            guide_title = meta.get("guide_title", "N/A")
                            section_title = meta.get("section_title", "N/A")
                            guide_url = meta.get("url", "")
                            guide_number = meta.get("guide_number", "")

                            relevance_badge = ""
                            try:
                                dist = float(dists[i - 1]) if len(dists) >= i else None
                            except Exception:
                                dist = None

                            if dist is not None:
                                if dist <= 0.5:
                                    relevance_badge = "🟢 **High relevance**"
                                elif dist <= 0.75:
                                    relevance_badge = "🟡 **Medium relevance**"
                                else:
                                    relevance_badge = "🔴 **Low relevance**"

                            if guide_url and guide_url != "N/A" and guide_url.strip():
                                st.markdown(f"**{i}. [{guide_title}]({guide_url})** 🔗")
                            else:
                                st.markdown(f"**{i}. {guide_title}**")

                            caption_parts = []
                            if guide_number and guide_number != "N/A":
                                caption_parts.append(f"Guide: {guide_number}")
                            if section_title and section_title != "N/A":
                                caption_parts.append(f"Section: {section_title}")
                            if relevance_badge:
                                caption_parts.append(relevance_badge)
                            if caption_parts:
                                st.caption(" • ".join(caption_parts))

                            if st.checkbox("Show content", key=f"guide_src_{i}"):
                                st.text(doc if len(doc) <= 2000 else doc[:2000] + "\n...\n[truncated]")

                            st.markdown("---")
                    else:
                        st.info("No guides found")
                else:
                    st.info("No guides found")

    if st.session_state.query_history:
        st.markdown("---")
        st.markdown("### 📜 **Recent Queries**")

        for item in reversed(st.session_state.query_history[-3:]):
            with st.expander(f"🕐 {item['time']} — {item['query']}..."):
                st.text(item["response"][:300] + "..." if len(item["response"]) > 300 else item["response"])
