"""Knowledge base management tab rendering."""


def render_manage_tab(
    st,
    project_root,
    sys,
    subprocess,
    Path,
    json,
    time,
    datetime,
    reset_pipeline,
):
    """Render Manage Knowledge Base tab."""
    st.markdown("### 🗄️ **Knowledge Base Management**")
    st.markdown("Upload tickets, refresh guides, and rebuild the vector database.")

    st.markdown("---")

    st.markdown("#### 🗑️ **Delete KB Data**")
    st.markdown(
        "*Deletes ticket embeddings, QA pair embeddings, BM25 index, processed tickets, and QA pairs file. **Guides are preserved**.*"
    )

    confirm_delete = st.checkbox("I understand this action cannot be undone", key="confirm_delete_kb")

    if st.button("🗑️ Delete All KB Data", type="secondary", disabled=not confirm_delete, use_container_width=True):
        with st.spinner("Deleting KB data..."):
            deleted_items = []

            try:
                from config.settings import CHROMA_DB_DIR, BM25_INDEX_PATH, PROCESSED_DIR
                import shutil

                reset_pipeline()
                st.session_state.stats = {"tickets": 0, "guides": 0}

                import gc

                gc.collect()
                gc.collect()
                time.sleep(2)

                try:
                    import chromadb

                    _client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
                    _coll = _client.get_collection("rag_v2")
                    ticket_data = _coll.get(where={"type": "ticket"}, limit=None)
                    if ticket_data["ids"]:
                        _coll.delete(ids=ticket_data["ids"])
                        deleted_items.append(f"Ticket embeddings ({len(ticket_data['ids'])})")
                    qa_data = _coll.get(where={"type": "qa_pair"}, limit=None)
                    if qa_data["ids"]:
                        _coll.delete(ids=qa_data["ids"])
                        deleted_items.append(f"QA embeddings ({len(qa_data['ids'])})")
                except Exception:
                    if CHROMA_DB_DIR.exists():
                        try:
                            shutil.rmtree(CHROMA_DB_DIR)
                            deleted_items.append("ChromaDB (full)")
                        except Exception:
                            st.warning("⚠️ ChromaDB files are locked. Refresh the page and try again.")

                if BM25_INDEX_PATH.exists():
                    try:
                        BM25_INDEX_PATH.unlink()
                        deleted_items.append("BM25 index")
                    except Exception as be:
                        st.warning(f"⚠️ BM25 index is locked: {be}")

                processed_tickets = PROCESSED_DIR / "processed_tickets.json"
                if processed_tickets.exists():
                    processed_tickets.unlink()
                    deleted_items.append("Processed tickets")

                qa_pairs_file = PROCESSED_DIR / "qa_pairs.json"
                if qa_pairs_file.exists():
                    qa_pairs_file.unlink()
                    deleted_items.append("QA pairs")

                if deleted_items:
                    st.success(f"✅ Deleted: {', '.join(deleted_items)}")
                    st.info("ℹ️ Guides were preserved. Please refresh the page.")
                else:
                    st.info("No KB data found to delete.")

            except Exception as e:
                st.error(f"❌ Error deleting KB data: {e}")

    st.markdown("---")

    st.markdown("#### 📤 **Upload Zendesk Export**")
    st.markdown("*Upload raw Zendesk export file (JSON or NDJSON format)*")

    from config.settings import PROCESSED_TICKETS_FILE

    if PROCESSED_TICKETS_FILE.exists():
        import os

        last_modified = datetime.fromtimestamp(os.path.getmtime(PROCESSED_TICKETS_FILE))
        st.caption(f"📅 Last uploaded: {last_modified.strftime('%Y-%m-%d %H:%M')}")
    else:
        st.caption("📅 No tickets uploaded yet")

    uploaded = st.file_uploader("Upload JSON", type=["json"], label_visibility="collapsed")

    if uploaded:
        try:
            content = uploaded.read().decode("utf-8")
            uploaded_tickets = []

            try:
                for line in content.strip().split("\n"):
                    if line.strip():
                        ticket = json.loads(line)
                        uploaded_tickets.append(ticket)
            except json.JSONDecodeError:
                uploaded_tickets = json.loads(content)
                if not isinstance(uploaded_tickets, list):
                    uploaded_tickets = [uploaded_tickets]

            st.success(f"📊 Found **{len(uploaded_tickets)}** tickets in uploaded file")

            clear_existing = st.checkbox(
                "🗑️ Clear existing tickets from knowledge base before checking",
                value=False,
                help=(
                    "If checked, all existing tickets will be removed from ChromaDB before checking for duplicates. "
                    "All uploaded tickets will be treated as new."
                ),
            )

            st.info("🔍 Checking against existing knowledge base...")

            try:
                existing_kb_ids = set()
                _coll = None

                try:
                    import chromadb as _chroma
                    from config.settings import CHROMA_DB_DIR as _chroma_dir

                    if _chroma_dir.exists():
                        _client = _chroma.PersistentClient(path=str(_chroma_dir))
                        _coll = _client.get_collection("rag_v2")
                except Exception:
                    _coll = None

                if clear_existing and _coll:
                    with st.spinner("🗑️ Clearing existing tickets from knowledge base..."):
                        try:
                            ticket_data = _coll.get(where={"type": "ticket"}, limit=None)
                            ticket_ids = ticket_data.get("ids", [])
                            if ticket_ids:
                                _coll.delete(ids=ticket_ids)
                                st.success(f"✅ Cleared {len(ticket_ids)} existing tickets from knowledge base")
                            qa_data = _coll.get(where={"type": "qa_pair"}, limit=None)
                            qa_ids = qa_data.get("ids", [])
                            if qa_ids:
                                _coll.delete(ids=qa_ids)
                                st.success(f"✅ Cleared {len(qa_ids)} existing QA pairs from knowledge base")
                        except Exception as e:
                            st.warning(f"⚠️ Could not clear tickets: {e}")

                if _coll and not clear_existing:
                    try:
                        existing_data = _coll.get(where={"type": "ticket"}, limit=None)
                        for id_ in existing_data.get("ids", []):
                            if id_.startswith("ticket_"):
                                parts = id_.split("__idx")
                                ticket_part = parts[0]
                                if "_" in ticket_part:
                                    try:
                                        ticket_id = int(ticket_part.split("_")[-1])
                                        existing_kb_ids.add(ticket_id)
                                    except ValueError:
                                        continue
                    except Exception:
                        pass

                with st.expander("🔍 Debug: Duplicate Detection Info", expanded=False):
                    st.write(f"**Existing ticket IDs in KB:** {len(existing_kb_ids)}")
                    if existing_kb_ids:
                        sample_ids = sorted(list(existing_kb_ids))[:10]
                        st.write(f"**Sample IDs (first 10):** {sample_ids}")
                    uploaded_ids = [t.get("id") for t in uploaded_tickets[:10]]
                    st.write(f"**Uploaded ticket IDs (first 10):** {uploaded_ids}")
                    st.write(f"**Collection used:** {'rag_v2' if _coll else 'none (empty KB)'}")

                unique_tickets = []
                duplicate_count = 0
                for ticket in uploaded_tickets:
                    ticket_id = ticket.get("id")
                    if ticket_id and ticket_id not in existing_kb_ids:
                        unique_tickets.append(ticket)
                    else:
                        duplicate_count += 1

                col_stat1, col_stat2 = st.columns(2)
                with col_stat1:
                    st.metric("✅ Unique Tickets", len(unique_tickets), delta=f"{len(unique_tickets)} new")
                with col_stat2:
                    st.metric("⚠️ Duplicates", duplicate_count, delta=f"{duplicate_count} already in KB")

                if len(unique_tickets) == 0:
                    st.warning("⚠️ All tickets already exist in the knowledge base. No new tickets to import.")
                else:
                    with st.expander("📄 Preview First Unique Ticket", expanded=False):
                        if unique_tickets:
                            preview = {
                                "id": unique_tickets[0].get("id"),
                                "subject": unique_tickets[0].get("subject"),
                                "status": unique_tickets[0].get("status"),
                                "created_at": unique_tickets[0].get("created_at"),
                            }
                            st.json(preview)

                    st.session_state.unique_tickets_to_process = unique_tickets
                    st.session_state.unique_tickets_count = len(unique_tickets)

                    st.success(f"💾 Ready to process **{len(unique_tickets)}** unique tickets")

                    st.markdown("---")
                    if st.button("🚀 Process & Update Knowledge Base", type="primary", use_container_width=True):
                        unique_tickets = st.session_state.get("unique_tickets_to_process", [])
                        if not unique_tickets:
                            st.error("No unique tickets to process. Please upload a file first.")
                            st.stop()

                        with st.status("Processing and updating knowledge base...", expanded=True) as status:
                            reset_pipeline()

                            status.write(f"📝 Step 1/3: Processing {len(unique_tickets)} unique tickets...")
                            try:
                                import tempfile
                                import os

                                with tempfile.NamedTemporaryFile(
                                    mode="w", suffix=".json", delete=False, encoding="utf-8"
                                ) as tmp_file:
                                    json.dump(unique_tickets, tmp_file, ensure_ascii=False, indent=2)
                                    tmp_path = Path(tmp_file.name)

                                env = os.environ.copy()
                                env["ZENDESK_EXPORT_FILE"] = str(tmp_path)

                                process = subprocess.Popen(
                                    [sys.executable, "scripts/process_tickets_only.py"],
                                    cwd=project_root,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    text=True,
                                    bufsize=1,
                                    universal_newlines=True,
                                    env=env,
                                )

                                for line in iter(process.stdout.readline, ""):
                                    if line:
                                        line = line.strip()
                                        if line:
                                            status.write(line)

                                process.wait()

                                if tmp_path.exists():
                                    tmp_path.unlink()

                                if process.returncode != 0:
                                    status.update(label="❌ Processing failed", state="error")
                                    st.error("Processing failed. Check logs above.")
                                    st.stop()

                            except Exception as e:
                                status.write(f"❌ Error processing: {e}")
                                st.error(f"Processing failed: {e}")
                                import traceback

                                st.code(traceback.format_exc())
                                st.stop()

                            status.write("")
                            status.write("🔍 Step 2/3: Generating ticket embeddings...")
                            try:
                                process = subprocess.Popen(
                                    [sys.executable, "scripts/update_tickets_only.py"],
                                    cwd=project_root,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    text=True,
                                    bufsize=1,
                                    universal_newlines=True,
                                )

                                for line in iter(process.stdout.readline, ""):
                                    if line:
                                        line = line.strip()
                                        if line:
                                            status.write(line)

                                process.wait()

                                if process.returncode != 0:
                                    status.update(label="❌ Ticket embedding failed", state="error")
                                    st.error("Failed to embed tickets. Check logs above.")
                                    st.stop()

                            except Exception as e:
                                status.write(f"❌ Error: {e}")
                                st.error(f"Ticket embedding failed: {e}")
                                import traceback

                                st.code(traceback.format_exc())
                                st.stop()

                            status.write("")
                            status.write("💬 Step 3/3: Extracting QA pairs and embedding...")
                            try:
                                process = subprocess.Popen(
                                    [sys.executable, "scripts/update_qa_only.py"],
                                    cwd=project_root,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    text=True,
                                    bufsize=1,
                                    universal_newlines=True,
                                )

                                for line in iter(process.stdout.readline, ""):
                                    if line:
                                        line = line.strip()
                                        if line:
                                            status.write(line)

                                process.wait()

                                if process.returncode != 0:
                                    status.write("⚠️ QA extraction had issues, but tickets are saved.")

                            except Exception as e:
                                status.write(f"⚠️ QA extraction error: {e} (tickets are still saved)")

                            status.update(label="✅ Complete! Tickets + QA pairs embedded", state="complete")
                            unique_count = st.session_state.get("unique_tickets_count", len(unique_tickets))
                            st.success(f"🎉 Successfully imported **{unique_count}** tickets + QA pairs!")
                            st.balloons()
                            if "unique_tickets_to_process" in st.session_state:
                                del st.session_state.unique_tickets_to_process
                            reset_pipeline(recycle_client=True)
                            time.sleep(1)
                            st.rerun()

            except Exception as e:
                st.error(f"❌ Error checking knowledge base: {e}")
                import traceback

                with st.expander("🔍 Error Details", expanded=False):
                    st.code(traceback.format_exc())

        except json.JSONDecodeError as e:
            st.error(f"❌ Invalid JSON format: {e}")
        except Exception as e:
            st.error(f"❌ Error: {e}")
            import traceback

            with st.expander("🔍 Error Details", expanded=False):
                st.code(traceback.format_exc())

    st.markdown("---")

    st.markdown("#### 🌐 **Guides**")

    from config.settings import GUIDES_CHUNKS_FILE

    guides_chunks_exist = GUIDES_CHUNKS_FILE.exists()

    if guides_chunks_exist:
        import os as _os

        last_modified = datetime.fromtimestamp(_os.path.getmtime(GUIDES_CHUNKS_FILE))
        st.caption(f"📅 Last refreshed: {last_modified.strftime('%Y-%m-%d %H:%M')}")

        st.markdown("*Processed guide chunks found. Embed them into the knowledge base.*")
        if st.button("📦 Embed Existing Guides", use_container_width=True, type="primary"):
            reset_pipeline()
            with st.status("Creating guide embeddings...", expanded=True) as status:
                try:
                    process = subprocess.Popen(
                        [sys.executable, "scripts/update_guides_incremental.py"],
                        cwd=project_root,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        universal_newlines=True,
                    )

                    for line in iter(process.stdout.readline, ""):
                        if line:
                            line = line.strip()
                            if line:
                                status.write(line)

                    process.wait()

                    if process.returncode == 0:
                        reset_pipeline(recycle_client=True)
                        status.update(label="✅ Guide embeddings created!", state="complete")
                        st.balloons()
                        st.rerun()
                    else:
                        status.update(label="❌ Process failed", state="error")
                        st.error("Failed to embed guides")

                except Exception as e:
                    status.update(label="❌ Error", state="error")
                    st.error(f"Error: {e}")

        with st.expander("🔄 Re-scrape guides from website", expanded=False):
            st.markdown("*Scrape latest guides, chunk, and update embeddings*")
            if st.button("🔄 Refresh Guides", use_container_width=True):
                reset_pipeline()
                with st.status("Processing guides...", expanded=True) as status:
                    try:
                        status.write("📥 Step 1/3: Scraping guides from website...")
                        process = subprocess.Popen(
                            [sys.executable, "-m", "src.phase3.scrape_guides_fast"],
                            cwd=project_root,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1,
                            universal_newlines=True,
                        )

                        for line in iter(process.stdout.readline, ""):
                            if line:
                                line = line.strip()
                                if line:
                                    status.write(line)

                        process.wait()

                        if process.returncode != 0:
                            status.update(label="❌ Scraping failed", state="error")
                            st.stop()

                        status.write("✂️  Step 2/3: Chunking guides...")
                        chunk_process = subprocess.run(
                            [sys.executable, "-m", "src.phase1.semantic_chunker"],
                            cwd=project_root,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            timeout=120,
                        )

                        for line in chunk_process.stdout.split("\n"):
                            if line.strip():
                                status.write(line.strip())

                        if chunk_process.returncode != 0:
                            status.update(label="❌ Chunking failed", state="error")
                            st.stop()

                        status.write("🔧 Step 3/3: Updating guide embeddings...")
                        update_process = subprocess.run(
                            [sys.executable, "scripts/update_guides_incremental.py"],
                            cwd=project_root,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            timeout=300,
                        )

                        for line in update_process.stdout.split("\n"):
                            if line.strip():
                                status.write(line.strip())

                        if update_process.returncode == 0:
                            reset_pipeline(recycle_client=True)
                            status.update(label="✅ Guides refreshed!", state="complete")
                            st.balloons()
                            st.rerun()
                        else:
                            status.update(label="❌ Update failed", state="error")
                            st.error("Failed to update guide embeddings")
                            st.code(update_process.stdout)

                    except Exception as e:
                        status.update(label="❌ Error", state="error")
                        st.error(f"Error: {e}")
                        import traceback

                        st.code(traceback.format_exc())
    else:
        st.caption("📅 No processed guides found")
        st.markdown("*Scrape guides from website, chunk, and create embeddings*")
        if st.button("📦 Scrape & Process Guides", use_container_width=True, type="primary"):
            reset_pipeline()
            with st.status("Processing guides...", expanded=True) as status:
                try:
                    status.write("📥 Step 1/3: Scraping guides from website...")
                    process = subprocess.Popen(
                        [sys.executable, "-m", "src.phase3.scrape_guides_fast"],
                        cwd=project_root,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        universal_newlines=True,
                    )

                    for line in iter(process.stdout.readline, ""):
                        if line:
                            line = line.strip()
                            if line:
                                status.write(line)

                    process.wait()

                    if process.returncode != 0:
                        status.update(label="❌ Scraping failed", state="error")
                        st.stop()

                    status.write("✂️  Step 2/3: Chunking guides...")
                    chunk_process = subprocess.run(
                        [sys.executable, "-m", "src.phase1.semantic_chunker"],
                        cwd=project_root,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=120,
                    )

                    for line in chunk_process.stdout.split("\n"):
                        if line.strip():
                            status.write(line.strip())

                    if chunk_process.returncode != 0:
                        status.update(label="❌ Chunking failed", state="error")
                        st.stop()

                    status.write("🔧 Step 3/3: Creating guide embeddings...")
                    update_process = subprocess.run(
                        [sys.executable, "scripts/update_guides_incremental.py"],
                        cwd=project_root,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=300,
                    )

                    for line in update_process.stdout.split("\n"):
                        if line.strip():
                            status.write(line.strip())

                    if update_process.returncode == 0:
                        reset_pipeline(recycle_client=True)
                        status.update(label="✅ Guides processed!", state="complete")
                        st.balloons()
                        st.rerun()
                    else:
                        status.update(label="❌ Update failed", state="error")
                        st.error("Failed to create guide embeddings")
                        st.code(update_process.stdout)

                except Exception as e:
                    status.update(label="❌ Error", state="error")
                    st.error(f"Error: {e}")
                    import traceback

                    st.code(traceback.format_exc())

    st.markdown("---")

    st.markdown("#### 💬 **Extract QA Pairs**")
    st.markdown("*Extract question-answer pairs from processed tickets*")

    from config.settings import PROCESSED_DIR

    qa_pairs_file = PROCESSED_DIR / "qa_pairs.json"
    if qa_pairs_file.exists():
        try:
            with open(qa_pairs_file, "r", encoding="utf-8") as f:
                qa_count = len(json.load(f))
            st.caption(f"📊 Current QA pairs: {qa_count:,}")
        except Exception:
            pass

    if st.button("💬 Extract QA Pairs + Embed", use_container_width=True):
        reset_pipeline()
        with st.status("Extracting QA pairs and creating embeddings...", expanded=True) as status:
            try:
                process = subprocess.Popen(
                    [sys.executable, "scripts/update_qa_only.py"],
                    cwd=project_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                )

                for line in iter(process.stdout.readline, ""):
                    if line:
                        line = line.strip()
                        if line:
                            status.write(line)

                process.wait()

                if process.returncode == 0:
                    reset_pipeline(recycle_client=True)
                    status.update(label="✅ QA pairs extracted and embedded!", state="complete")
                    st.success("Successfully extracted and embedded QA pairs!")
                    st.balloons()
                    st.rerun()
                else:
                    status.update(label="❌ Process failed", state="error")
                    st.error("Failed to extract/embed QA pairs")

            except Exception as e:
                status.update(label="❌ Error", state="error")
                st.error(f"Error: {e}")

    st.markdown("---")

    st.markdown("#### 📊 **Database Info**")

    info_col1, info_col2 = st.columns(2)
    with info_col1:
        st.markdown(
            f"""
        <div class="stat-box">
            <div class="stat-value">{st.session_state.stats['tickets']}</div>
            <div class="stat-label">Tickets</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with info_col2:
        st.markdown(
            f"""
        <div class="stat-box">
            <div class="stat-value">{st.session_state.stats['guides']}</div>
            <div class="stat-label">Guide Chunks</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
